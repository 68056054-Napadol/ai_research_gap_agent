"""
The 7 graph nodes — see architecture doc §2.2 for the rationale behind each one. Every node
function takes the current AgentState and returns a dict of the fields it updates; LangGraph
merges that into the running state (standard LangGraph node signature).
"""
import datetime
import json
import logging
import re
import time

import conditions
import embeddings
import quality
import evidence
import links
import llm_client
import metrics
import privacy
import prompts
import rag_store
import tools
from config import CONFIG
from contracts import GapProposalOutput, KeywordOutput, PaperExtractOutput, VerifierOutput
from sources import search_all, pdf_extract, _dedupe
from sources import net as source_net
from state import AgentState

logger = logging.getLogger(__name__)


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_whitespace(s: str) -> str:
    """Collapse all whitespace runs (spaces, newlines, tabs — PDF-to-Markdown conversion in
    particular reflows these unpredictably) to a single space, and strip. Used only to compare
    a quote against source text; nothing else about the text is altered (still case-sensitive —
    "verbatim" is the whole point of this check)."""
    return _WHITESPACE_RE.sub(" ", s or "").strip()


def _quote_verified(quote: str, source_text: str) -> bool:
    """Extraction-quote grounding (README "Fact-checking"): a cheap, deterministic check that
    catches an extraction hallucinating a claim never in the source, without another LLM call.
    Checks the LLM-reported supporting_quote is an exact substring of the text it was actually
    shown (whitespace-normalized only — still requires a verbatim, not paraphrased, match)."""
    quote = (quote or "").strip()
    if not quote:
        return False
    return _normalize_whitespace(quote) in _normalize_whitespace(source_text)


# ---------------------------------------------------------------------------
# Node 0 — Input privacy check (LLM extraction + decision model)
# ---------------------------------------------------------------------------
def sanitize_input(state: AgentState) -> dict:
    """Runs before anything else: the topic the user typed goes through regex + LLM entity
    extraction, then the decision model (decision_model/sensitive_info.dmn.json) decides per
    entity (entity_action) and for the whole input (input_action). Only the redacted topic is
    kept in state."""
    global _RUN_STARTED
    _RUN_STARTED = time.monotonic()
    privacy.reset_run()
    source_net.reset_run()
    result = privacy.sanitize_topic(state["research_topic"])
    d = result["decision"]
    logger.info("Input decision: %s (rule %s) - %d entity decision(s)%s", d["action"], d["rule"],
                len(result["records"]), f" - {d['message']}" if d.get("message") else "")
    cond, cond_records = _sanitize_conditions(state.get("search_conditions"))
    records = result["records"] + cond_records
    for r in records:
        logger.info("  [privacy] %s %s role=%s by=%s conf=%.2f -> %s (rule %s)", r["type"], r["masked"],
                    r["role"], r["detected_by"], r["confidence"], r["action"], r["rule"])
    if not conditions.is_empty(cond):
        logger.info("User conditions: %s", " · ".join(conditions.describe(cond)))
    return {
        "research_topic": result["topic"] if d["action"] != "REJECT" else "[REJECTED INPUT]",
        "input_decision": d,
        "search_conditions": cond,
        "privacy_log": records,
        "tool_trace": [],
    }


def _sanitize_conditions(raw) -> tuple:
    """User conditions are user input too: the focus text gets the same regex + LLM + decision-model
    check as the topic, include/exclude terms the regex detectors. Only the redacted form is kept."""
    if not raw:
        return conditions.normalize(), []
    cond = raw if {"focus", "include", "exclude"} <= set(raw) else conditions.normalize(
        raw.get("focus", ""), raw.get("include"), raw.get("exclude"))
    cond = conditions.normalize(cond.get("focus", ""), cond.get("include"), cond.get("exclude"))
    records = []
    if cond["focus"]:
        text, recs = privacy.sanitize(cond["focus"], channel="user_input", where="search_conditions.focus", use_llm=True)
        cond["focus"] = privacy.strip_placeholders(text)
        records += recs
    for key in ("include", "exclude"):
        clean = []
        for item in cond[key]:
            alts = item if isinstance(item, list) else [item]
            kept = []
            for a in alts:
                text, recs = privacy.sanitize(a, channel="user_input", where=f"search_conditions.{key}", use_llm=False)
                records += recs
                text = privacy.strip_placeholders(text).strip()
                if len(text) >= 2:
                    kept.append(text.lower())
            if kept:
                clean.append(kept if isinstance(item, list) else kept[0])
        cond[key] = clean
    return cond, records


def route_after_input(state: AgentState) -> str:
    return "reject" if state.get("input_decision", {}).get("action") == "REJECT" else "proceed"


# ---------------------------------------------------------------------------
# Node 1 — Topic Understanding & Keyword Generation
# ---------------------------------------------------------------------------
def generate_keywords(state: AgentState) -> dict:
    prior = state.get("prior_queries_tried", [])
    retry_context, mode, update = "", "initial", {}
    prior_list = "\n".join(f"- {q}" for q in prior)
    if prior and state.get("search_verdict") == "too_broad":
        # Node 4 "too broad" rule: more specific queries. The loosely related candidates are dropped
        # from the pool (only those that passed the cut-off stay), so the new, specific results fit.
        mode = "narrow"
        scope = state.get("last_round_scope") or {}
        on_topic = [p.get("title") for p in (state.get("ranked_papers") or [])[:5] if p.get("title")]
        logger.info("Regenerating keywords (narrow #%d) — previous search too broad (%d/%d passed the cut-off)",
                    state.get("narrow_count", 0), scope.get("n_passed", 0), scope.get("n_candidates", 0))
        retry_context = prompts.NARROW_CONTEXT_TEMPLATE.format(
            n_candidates=scope.get("n_candidates", 0), n_passed=scope.get("n_passed", 0),
            pass_rate=scope.get("pass_rate", 0.0), prior_queries=prior_list,
            on_topic_titles="\n".join(f"- {t}" for t in on_topic) or "- (none passed)")
        update["candidate_papers"] = list(state.get("ranked_papers") or [])
    elif prior:
        mode = "broaden"
        logger.info("Regenerating keywords (retry #%d) — avoiding %d previously tried queries",
                    state.get("retry_count", 0), len(prior))
        retry_context = prompts.RETRY_CONTEXT_TEMPLATE.format(prior_queries=prior_list)
        # Keep only the candidates that passed the cut-off. Before, the pool kept every earlier
        # candidate and was cut at MAX_CANDIDATE_PAPERS, so a full pool of off-topic papers left no
        # room for the new round's results (live run 21 Nov: 0/80 passed in all 3 rounds).
        update["candidate_papers"] = list(state.get("ranked_papers") or [])
    else:
        logger.info("Generating search keywords for topic: %r", state["research_topic"])

    prompt = prompts.KEYWORD_GEN_PROMPT.format(topic=privacy.strip_placeholders(state["research_topic"]),
                                               conditions=conditions.prompt_block(state.get("search_conditions")),
                                               retry_context=retry_context)
    result = llm_client.call_json(prompt, tier="fast", schema=KeywordOutput, purpose="generate-search-queries")
    queries = result["queries"]
    logger.info("Generated %d search queries: %s", len(queries), queries)

    rounds = list(state.get("search_rounds") or [])
    rounds.append({"round": len(rounds) + 1, "mode": mode, "planned_queries": queries})
    return {
        **update,
        "search_queries": queries,
        "prior_queries_tried": prior + queries,
        "search_rounds": rounds,   # one entry per search round - evaluation/eval_query_adaptation.py
    }


# ---------------------------------------------------------------------------
# Node 2 — Multi-Source Search
# ---------------------------------------------------------------------------
def search_sources(state: AgentState) -> dict:
    """Search AGENT: the LLM decides which search_papers calls to make (query, source, year range,
    limit) via real tool calling, sees each call's returned context, and adapts. The harness
    (tools.SearchToolContext) validates every argument against contracts.SearchPapersArgs and
    enforces policy (enabled sources, the user's year range, a call budget)."""
    queries = state["search_queries"]
    ctx = tools.SearchToolContext(year_from=state.get("year_from"), year_to=state.get("year_to"),
                                  enabled_sources=CONFIG.enabled_sources)
    logger.info("Search agent starting: %d suggested queries, sources=%s, budget=%d tool calls",
                len(queries), ", ".join(ctx.enabled_sources), ctx.budget)
    prompt = prompts.SEARCH_AGENT_PROMPT.format(
        topic=privacy.strip_placeholders(state["research_topic"]), year_from=state.get("year_from") or "any",
        conditions=conditions.prompt_block(state.get("search_conditions")),
        year_to=state.get("year_to") or "any", sources=", ".join(ctx.enabled_sources) + " (or 'all')",
        queries="\n".join(f"- {q}" for q in queries), budget=ctx.budget, pool=CONFIG.max_candidate_papers,
    )
    result = llm_client.run_tool_loop(prompts.SEARCH_AGENT_SYSTEM, prompt, tools.SEARCH_TOOLS, ctx.execute,
                                      tier="fast", agent_name="search_agent",
                                      trace_metadata={"budget": ctx.budget, "sources": ctx.enabled_sources})
    trace = result.trace
    used_fallback = False
    if ctx.calls_made == 0:
        # The model answered without ever calling the tool (some small local models do this).
        # Run the suggested queries through the same tool executor so the run can continue, and
        # label these trace entries as harness fallback - they were NOT chosen by the LLM.
        used_fallback = True
        logger.warning("Search agent made no successful tool calls - harness fallback: running suggested queries")
        for q in queries:
            text, entry = ctx.execute("search_papers", {"query": q, "source": "all",
                                                         "limit": max(5, CONFIG.max_candidate_papers // max(len(queries), 1))})
            trace.append({"agent": "harness_fallback", "step": 0, "tool_name": "search_papers",
                          "arguments": {"query": q, "source": "all"}, "result_preview": text[:1500],
                          "result_chars": len(text), **entry})

    # accumulate across retry loops instead of throwing earlier finds away
    merged = _dedupe((state.get("candidate_papers") or []) + ctx.papers)[: CONFIG.max_candidate_papers]
    logger.info("Search agent done: %d tool call(s), %d unique papers this round, %d total candidates",
                ctx.calls_made, len(ctx.papers), len(merged))
    health = source_net.source_health()
    for src in ctx.enabled_sources:
        h = health.get(src)
        if h and h["ok"] == 0:
            logger.warning("Source %s returned nothing usable this run: %s", src, h["last_error"] or "no response")
    rounds = [dict(r) for r in (state.get("search_rounds") or [{"round": 1}])]
    rounds[-1].update({"searched_queries": [t["arguments"].get("query") for t in trace
                                            if t.get("tool_name") == "search_papers" and t.get("ok", True)],
                       "n_found_this_round": len(ctx.papers), "n_candidates_total": len(merged)})
    return {
        "candidate_papers": merged,
        "search_rounds": rounds,
        "enabled_sources": list(ctx.enabled_sources),
        "source_health": health,
        "search_summary": result.final_text.strip(),
        "search_used_fallback": used_fallback,
        "tool_trace": (state.get("tool_trace") or []) + trace,
        "privacy_log": (state.get("privacy_log") or []) + ctx.privacy_records,
    }


# ---------------------------------------------------------------------------
# Node 3 — Relevance Filtering & Ranking
# ---------------------------------------------------------------------------
def filter_and_rank(state: AgentState) -> dict:
    cond = state.get("search_conditions")
    candidates, low_quality = quality.apply(state["candidate_papers"])
    if low_quality:
        logger.info("Quality filter dropped %d/%d candidates (e.g. %s)", len(low_quality),
                    len(state["candidate_papers"]), low_quality[0]["reason"])
    candidates, excluded = conditions.apply(candidates, cond)
    if excluded:
        logger.info("User conditions dropped %d/%d candidates (e.g. %s)", len(excluded),
                    len(state["candidate_papers"]), excluded[0]["reason"])
    logger.info("Ranking %d candidate papers by embedding similarity to the topic", len(candidates))
    ranked = embeddings.rank_by_similarity(state["research_topic"], candidates)
    above_threshold = embeddings.filter_by_threshold(ranked)
    logger.info("%d/%d papers passed similarity threshold %.2f (%s)",
                len(above_threshold), len(ranked), embeddings.ranking_threshold(), CONFIG.ranking_embedding)

    # NOTE (architecture doc §7.2 stage 2 - optional LLM re-rank): for course-scope speed this
    # scaffold ships with embedding-only ranking. To add the semantic re-rank pass, take the
    # top CONFIG.llm_rerank_top_n from `above_threshold`, ask the LLM to judge true relevance
    # per-paper, and re-sort. Left as the highest-value TODO per the README's "what's next".

    budget = state.get("max_papers") or CONFIG.max_deep_analysis_papers   # CLI --max-papers / UI slider
    selected = above_threshold[:budget]
    logger.info("Selected top %d papers for deep analysis (budget: %d)", len(selected), budget)
    rounds = [dict(r) for r in (state.get("search_rounds") or [{"round": 1}])]
    rounds[-1].update({"n_passed_cutoff": len(above_threshold), "n_selected": len(selected),
                       "n_excluded_by_conditions": len(excluded)})
    return {"ranked_papers": above_threshold, "selected_papers": selected, "search_rounds": rounds,
            "excluded_by_conditions": excluded, "excluded_by_quality": low_quality}


# ---------------------------------------------------------------------------
# Node 4 — Sufficiency Check (conditional edge, no LLM call)
# ---------------------------------------------------------------------------
def check_sufficiency(state: AgentState) -> dict:
    """Judges the search round from counts only (no LLM): "sufficient", "too_few" or "too_broad".

    too_broad - the candidate pool is full (>= MAX_CANDIDATE_PAPERS) but fewer than BROAD_MAX_PASS_RATE
                of the candidates passed the relevance cut-off: the queries were too general, so
                Node 1 is asked for more specific ones (at most MAX_NARROW_LOOPS times). Checked first.
    too_few   - fewer selected papers than min(MIN_PAPERS_BEFORE_RETRY, max_papers), or fewer
                candidates than MIN_CANDIDATE_PAPERS while the max_papers budget is not filled:
                Node 1 is asked for broader queries.
    """
    retry_count = state.get("retry_count", 0)
    narrow_count = state.get("narrow_count", 0)
    n_selected = len(state.get("selected_papers", []))
    n_candidates = len(state.get("candidate_papers") or [])
    n_after_conditions = (n_candidates - len(state.get("excluded_by_conditions") or [])
                          - len(state.get("excluded_by_quality") or []))
    n_passed = len(state.get("ranked_papers") or [])
    pass_rate = n_passed / n_after_conditions if n_after_conditions else 0.0
    # Live-run bug: with --max-papers 4 and MIN_PAPERS_BEFORE_RETRY=5 the check could never pass,
    # so every run burned all retry loops. You can't need more papers than you're allowed to select.
    need = min(CONFIG.min_papers_before_retry, state.get("max_papers") or CONFIG.max_deep_analysis_papers)
    min_pool = min(CONFIG.min_candidate_papers, CONFIG.max_candidate_papers)
    budget = state.get("max_papers") or CONFIG.max_deep_analysis_papers

    too_broad = (n_candidates >= CONFIG.max_candidate_papers and pass_rate < CONFIG.broad_max_pass_rate
                 and narrow_count < CONFIG.max_narrow_loops)
    if too_broad:
        verdict = "too_broad"
    elif n_selected >= need and (n_candidates >= min_pool or n_selected >= budget):
        # a short candidate pool only matters while the paper budget is not filled yet
        verdict = "sufficient"
    else:
        verdict = "too_few"
    logger.info("Sufficiency check: %d candidates (pool %d, need >= %d), %d passed the cut-off (%.0f%%), "
                "%d selected (need >= %d) -> %s", n_candidates, CONFIG.max_candidate_papers, min_pool,
                n_passed, pass_rate * 100, n_selected, need, verdict.upper())
    rounds = [dict(r) for r in (state.get("search_rounds") or [{"round": 1}])]
    rounds[-1]["verdict"] = verdict
    return {
        "sufficient": verdict == "sufficient",
        "search_verdict": verdict,
        "search_rounds": rounds,
        "last_round_scope": {"n_candidates": n_candidates, "n_passed": n_passed, "pass_rate": round(pass_rate, 3)},
        "retry_count": retry_count + (1 if verdict == "too_few" else 0),
        "narrow_count": narrow_count + (1 if verdict == "too_broad" else 0),
    }


_RUN_STARTED = time.monotonic()   # reset by sanitize_input (Node 0) at the start of every run


def _run_tokens() -> float:
    return metrics.get("input_tokens") + metrics.get("output_tokens")


def run_seconds() -> float:
    return time.monotonic() - _RUN_STARTED


def token_budget_used_up() -> bool:
    """MAX_RUN_TOKENS stop condition (0 = off) - see README "Decision rules and stop conditions"."""
    return CONFIG.max_run_tokens > 0 and _run_tokens() >= CONFIG.max_run_tokens


def run_budget_exceeded() -> str:
    """Why the run's budget is used up ("" = not used up): tokens (MAX_RUN_TOKENS) or wall-clock
    seconds (MAX_RUN_SECONDS). Checked before every optional, repeatable step."""
    if token_budget_used_up():
        return f"token budget MAX_RUN_TOKENS={CONFIG.max_run_tokens} used up ({int(_run_tokens())} tokens)"
    if CONFIG.max_run_seconds > 0 and run_seconds() >= CONFIG.max_run_seconds:
        return f"time budget MAX_RUN_SECONDS={CONFIG.max_run_seconds} used up ({run_seconds():.0f} s)"
    return ""


def route_after_sufficiency(state: AgentState) -> str:
    """Conditional-edge function passed to graph.add_conditional_edges (see graph.py)."""
    verdict = state.get("search_verdict") or ("sufficient" if state.get("sufficient") else "too_few")
    if verdict == "sufficient":
        return "extract"
    over = run_budget_exceeded()
    if over:
        metrics.inc("token_budget_stopped_retry" if "token" in over else "time_budget_stopped_retry")
        logger.warning("%s - no further search loop; proceeding with %d papers.", over,
                       len(state.get("selected_papers", [])))
        return "extract"
    if verdict == "too_broad":
        logger.info("Search too broad -> looping back for more specific queries (narrow %d/%d)",
                    state.get("narrow_count", 0), CONFIG.max_narrow_loops)
        return "narrow"
    if state.get("retry_count", 0) >= CONFIG.max_retry_loops:
        logger.warning("Hit max retry loops (%d) with only %d papers; proceeding anyway.",
                        CONFIG.max_retry_loops, len(state.get("selected_papers", [])))
        return "extract"
    logger.info("Not enough papers yet -> looping back to generate broader keywords (retry %d/%d)",
                state.get("retry_count", 0), CONFIG.max_retry_loops)
    return "retry"


# ---------------------------------------------------------------------------
# Node 5 — Per-Paper Extraction
# ---------------------------------------------------------------------------
# Character caps for the extraction prompt's {text} slot. Abstracts are short already, so their
# cap is just a safety net; full text gets a much bigger budget so results/limitations sections
# (usually well past the intro) actually make it into the prompt — otherwise fetching the PDF at
# all would be pointless. llm_client sizes Ollama's context window (num_ctx) off the prompt it's
# actually given, so raising this doesn't risk the model silently dropping the extra text.
ABSTRACT_CHAR_LIMIT = 4000
FULL_TEXT_CHAR_LIMIT = 12000


# What Node 5 asks the paper's own chunks for, instead of cutting the text at the first 12k chars
# (which mostly kept intro + related work and dropped results/limitations near the end).
EXTRACT_RETRIEVAL_QUERIES = {
    "method": "proposed method approach model architecture",
    "dataset": "datasets and experimental setup used for evaluation",
    "results": "main results performance compared with baselines",
    "limitations": "limitations weaknesses future work",
}


def _rag_view(store, paper: dict, full_text: str) -> str:
    """Text shown to the extraction LLM for a full-text paper: its opening (title + abstract)
    plus the chunks most relevant to each extraction field, retrieved from the knowledge base."""
    parts, seen = [store.paper_opening(paper["id"], 2500)], set()
    for field_name, query in EXTRACT_RETRIEVAL_QUERIES.items():
        for hit in store.search(query, paper_ids=[paper["id"]], k=2):
            if hit["chunk"] in seen or hit["text"][:200] in parts[0]:
                continue
            seen.add(hit["chunk"])
            parts.append(f"[{hit['section']}]\n{hit['text']}")
    # chunks stored by an earlier run were scrubbed with that run's author list only
    return privacy.scrub_known("\n\n".join(parts)[:FULL_TEXT_CHAR_LIMIT], f"{paper['id']}:rag_view")[0]


def _extract_cache_key(prompt: str) -> str:
    import hashlib
    model = CONFIG.cloud_model_fast if CONFIG.llm_backend == "cloud" else CONFIG.local_model_fast
    return hashlib.sha1(f"{model}\n{prompt}".encode("utf-8")).hexdigest()[:20]


def extract_paper_data(state: AgentState) -> dict:
    total = len(state["selected_papers"])
    logger.info("Extracting structured data (method/dataset/results) from %d papers", total)
    store = rag_store.get_store() if CONFIG.rag_enabled else None
    extracts = []
    skipped = []   # papers selected for analysis that produced no extract - shown in "Search scope"
    privacy_records = []
    for i, paper in enumerate(state["selected_papers"], 1):
        title = (paper.get("title") or "")[:80]
        full_text = paper.get("full_text")

        # Knowledge base first (KM): a paper indexed with full text in an earlier run is reused -
        # no PDF download, no conversion, no re-redaction.
        if full_text is None and store is not None and store.indexed_kind(paper["id"]) == "full_text":
            full_text, recs = privacy.scrub_known(store.paper_text(paper["id"]), f"{paper['id']}:kb_full_text")
            privacy_records.extend(recs)
            paper["full_text"] = full_text
            metrics.inc("rag_full_text_reused")
            logger.info("  [%d/%d] Full text reused from knowledge base (%d chars): %s", i, total, len(full_text), title)

        # Fetch + convert the open-access PDF to Markdown if we don't already have full text for
        # this paper (see sources/pdf_extract.py for why Markdown, not raw PDF text). Off switch:
        # CONFIG.fetch_full_text (sidebar checkbox / FETCH_FULL_TEXT=false in .env) for a faster,
        # abstract-only run. Any failure here (no OA link, 404, paywall page, oversized PDF) just
        # falls back to the abstract below — never aborts the paper.
        if full_text is None and CONFIG.fetch_full_text and paper.get("oa_pdf_url"):
            logger.info("  [%d/%d] Fetching PDF full text: %s", i, total, title)
            pdf_start = time.monotonic()
            full_text = pdf_extract.fetch_and_convert(paper["oa_pdf_url"], paper["id"])
            if full_text:
                # Privacy: author names / emails / affiliations are redacted BEFORE the text reaches
                # the extraction LLM, the verifier tool, the knowledge base, the UI or state.json -
                # and the on-disk PDF cache is overwritten with the redacted version.
                full_text, recs = privacy.scrub_full_text(full_text, paper["id"],
                                                           has_author_metadata=paper.get("n_authors", 0) > 0,
                                                           title=paper.get("title", ""))
                privacy_records.extend(recs)
                pdf_extract.overwrite_cache(paper["id"], full_text)
                paper["full_text"] = full_text  # cache on the paper dict for the rest of this run
                logger.info("  [%d/%d] Got PDF full text (%d chars, %.1fs)",
                            i, total, len(full_text), time.monotonic() - pdf_start)
            else:
                logger.info("  [%d/%d] No usable PDF (%.1fs) — falling back to abstract",
                            i, total, time.monotonic() - pdf_start)

        text = full_text or paper.get("abstract", "")
        if not text:
            logger.info("  [%d/%d] SKIPPED (no abstract or text available): %s", i, total, title)
            skipped.append({"id": paper["id"], "title": paper.get("title", ""), "reason": "no abstract or full text available"})
            continue

        # Index into the knowledge base (abstract-only papers too, so search_corpus covers them).
        if store is not None:
            try:
                store.index_paper(paper, text, "full_text" if full_text else "abstract")
            except Exception as e:   # an embedding hiccup must not cost us the paper
                logger.warning("  [%d/%d] RAG indexing failed (%s) - using plain truncation", i, total, e)

        logger.info("  [%d/%d] Extracting (%s): %s", i, total,
                    "full text" if full_text else "abstract only", title)
        paper_start = time.monotonic()
        if full_text and store is not None and store.indexed_kind(paper["id"]) == "full_text":
            text_shown_to_llm = _rag_view(store, paper, full_text)
        else:
            text_shown_to_llm = text[:FULL_TEXT_CHAR_LIMIT if full_text else ABSTRACT_CHAR_LIMIT]
        prompt = prompts.EXTRACT_PROMPT.format(title=paper.get("title", ""), year=paper.get("year", "?"), text=text_shown_to_llm)

        cache_key = _extract_cache_key(prompt)
        result = store.get_extract(paper["id"], cache_key) if (store is not None and CONFIG.rag_reuse_extracts) else None
        if result is not None:
            metrics.inc("extract_cache_hits")
            logger.info("  [%d/%d] Extraction reused from knowledge base (same text, prompt and model)", i, total)
        else:
            try:
                result = llm_client.call_json(prompt, tier="fast", schema=PaperExtractOutput, purpose="extract-paper-facts")
            except llm_client.LLMError as e:
                logger.warning("  [%d/%d] Extraction FAILED for %s (%.1fs): %s",
                                i, total, paper["id"], time.monotonic() - paper_start, e)
                skipped.append({"id": paper["id"], "title": paper.get("title", ""),
                                "reason": f"extraction failed ({type(e).__name__})"})
                continue
            if store is not None:
                store.put_extract(paper["id"], cache_key, result)

        result = dict(result)
        result["paper_id"] = paper["id"]
        result["extraction_confidence"] = "full_text" if full_text else "abstract_only"

        # Extraction-quote grounding (README "Fact-checking"): a deterministic, no-LLM check
        # that the model's supporting_quote actually appears in the text it was shown — catches
        # a fabricated extraction (something never in the source) that a same-data verifier
        # (Node 7) can't, since Node 7 only ever sees this summary, not the original text.
        quote = result.get("supporting_quote", "")
        verified = _quote_verified(quote, text_shown_to_llm)
        result["quote_verified"] = verified
        if not quote:
            logger.warning("  [%d/%d] No supporting_quote returned for %s — quote_verified=False", i, total, title)
        elif not verified:
            logger.warning("  [%d/%d] Quote NOT found verbatim in source for %s (possible hallucination): %r",
                            i, total, title, quote[:120])
        else:
            logger.info("  [%d/%d] Quote verified against source text for %s", i, total, title)

        extracts.append(result)
        # Distinct "done" line (vs. "Extracting" above) so the UI can tell a paper's LLM call
        # finished without waiting for the whole node to end — see app.py's step cards.
        logger.info("  [%d/%d] Done (%.1fs): %s", i, total, time.monotonic() - paper_start, title)

    logger.info("Extraction complete: %d/%d papers successfully extracted", len(extracts), total)
    out = {"paper_extracts": extracts, "skipped_papers": skipped, "selected_papers": state["selected_papers"],
           "privacy_log": (state.get("privacy_log") or []) + privacy_records}
    if store is not None:
        out["knowledge_base"] = store.stats()
    return out


# ---------------------------------------------------------------------------
# Node 6 — Cross-Paper Comparison & Trend Synthesis
# ---------------------------------------------------------------------------
def short_title(title: str, max_words: int = 6) -> str:
    """'Grade Guard: A Smart System for ...' -> 'Grade Guard'; long titles cut to a few words."""
    t = " ".join((title or "").split()).strip('"\u201c\u201d ')
    head = re.split(r"\s*[:\u2013\u2014]\s+|\s+-{1,2}\s+", t, maxsplit=1)[0]
    words = head.split()
    if len(words) > max_words:
        head = " ".join(words[:max_words]) + "..."
    return head or "untitled"


def paper_labels(papers: list) -> dict:
    """{paper_id: 'Short title (year)'}; duplicate short names get a letter so they stay distinct."""
    labels, used = {}, {}
    for p in papers:
        base = f"{short_title(p.get('title', ''))} ({p.get('year', 'n.d.')})"
        n = used.get(base, 0)
        used[base] = n + 1
        labels[p["id"]] = base if n == 0 else f"{base[:-1]}{chr(97 + n)})"
    return labels


def replace_ids_with_labels(text: str, labels: dict) -> str:
    """Safety net for the user-facing summary: any raw id the model still wrote becomes its name."""
    for pid in sorted(labels, key=len, reverse=True):
        text = text.replace(pid, labels[pid])
    return text


def _cell(value) -> str:
    """A table cell: extractions cached before the 'problem' field existed have none."""
    return " ".join(str(value or "not stated").split()).replace("|", "/")


def compare_papers(state: AgentState) -> dict:
    logger.info("Building comparison table from %d extracted papers", len(state["paper_extracts"]))
    by_id = {p["id"]: p for p in state["selected_papers"]}
    labels = paper_labels([by_id[ex["paper_id"]] for ex in state["paper_extracts"] if ex["paper_id"] in by_id])
    # Two views of the same rows: the id table is for gap analysis (gaps must cite ids) and the UI;
    # the trend synthesis gets short names, so the summary a person reads names papers, not ids.
    rows = ["| Paper | Year | Problem | Method | Dataset | Key findings | Quote verified |",
            "|---|---|---|---|---|---|---|"]
    named = ["| Paper | Problem | Method | Dataset | Key findings |", "|---|---|---|---|---|"]
    for ex in state["paper_extracts"]:
        p = by_id.get(ex["paper_id"], {})
        # "Quote verified" = extraction-quote grounding (README "Fact-checking"): whether the
        # supporting_quote the LLM gave for key_findings was actually found verbatim in the
        # source text (Node 5).
        quote_flag = "✅ yes" if ex.get("quote_verified") else "⚠️ no"
        rows.append(
            f"| {ex['paper_id']} | {p.get('year', '?')} | {_cell(ex.get('problem'))} | {ex.get('method', '')} "
            f"| {ex.get('dataset', '')} | {ex.get('key_findings', '')} | {quote_flag} |"
        )
        findings = ex.get("key_findings", "") + ("" if ex.get("quote_verified") else " (not confirmed in the text)")
        named.append(f"| {labels.get(ex['paper_id'], ex['paper_id'])} | {_cell(ex.get('problem'))} | {ex.get('method', '')} "
                     f"| {ex.get('dataset', '')} | {findings} |")
    table = "\n".join(rows)

    logger.info("Calling strong-tier LLM to synthesize trends across papers...")
    prompt = prompts.COMPARE_PROMPT.format(topic=state["research_topic"], comparison_table="\n".join(named))
    trend_summary = llm_client.call(prompt, tier="strong", purpose="synthesize-trends")
    trend_summary = replace_ids_with_labels(trend_summary, labels)
    # cheap redaction now (regex + this run's author list) so no intermediate view - UI card, log -
    # ever shows a name; the LLM-based pass on the final answer happens in privacy_guard
    trend_summary, recs = privacy.sanitize(trend_summary, "final_answer", "trend_summary(draft)", use_llm=False)
    logger.info("Trend synthesis complete (%d characters)", len(trend_summary))

    return {"comparison_table": table, "trend_summary": trend_summary, "paper_labels": labels,
            "privacy_log": (state.get("privacy_log") or []) + recs}


# ---------------------------------------------------------------------------
# Node 7 — Gap Analysis + Verifier -> confidence-scored gap candidates
# ---------------------------------------------------------------------------
MIN_COUNTEREXAMPLE_QUOTE_CHARS = 30   # a few words like "short answer grading" prove nothing


def verifier_decision(facts: dict, shown_text) -> tuple:
    """Deterministic verdict from the verifier's reported facts (the LLM extracts, code decides -
    same pattern as the privacy decision model). Returns (supported, reason).

    shown_text: {paper_id: [text the tools returned for it]} (a plain set of ids is accepted too).
    A counterexample only counts when the model names a paper the tools actually showed it AND
    copies a sentence from that paper's text verbatim. A live run lost a gap to a "counterexample"
    the verifier itself described as a title that "may be relevant"."""
    if not isinstance(shown_text, dict):
        shown_text = {pid: [] for pid in (shown_text or ())}
    if not facts.get("cited_papers_relevant"):
        return False, "cited papers do not concern the claim (topic or qualifier)"
    if facts.get("counterexample_found"):
        cid = facts.get("counterexample_paper_id")
        if not cid or cid not in shown_text:
            # the model named a counterexample paper the tools never returned - don't trust it
            return True, f"counterexample {cid or '(no id)'} was not among the passages the tools returned - ignored"
        quote = (facts.get("counterexample_quote") or "").strip()
        if len(quote) < MIN_COUNTEREXAMPLE_QUOTE_CHARS or not _quote_verified(quote, " \n".join(shown_text[cid])):
            return True, f"counterexample {cid} claimed without a verbatim sentence from its text - ignored"
        return False, f"counterexample found in the corpus ({cid}), quote verified against its text"
    if facts.get("overstated"):
        return False, "claim is overstated relative to the source text"
    return True, "cited papers relevant, no counterexample in the corpus, not overstated"


def _verify_with_tools(gap: dict, supporting_ids: list, extracts_by_id: dict, selected_papers: list):
    """Verifier AGENT: reads the cited papers' source text through the get_paper_text tool, then
    returns a contract-validated verdict. Returns (verdict_dict, trace, read_ids)."""
    store = rag_store.get_store() if CONFIG.rag_enabled else None
    ctx = tools.VerifierToolContext(selected_papers, store=store)
    toolset = tools.VERIFIER_TOOLS if store is not None else tools.VERIFIER_TOOLS_NO_RAG
    cited_data = "\n".join(f"- {pid}: {json.dumps({k: v for k, v in extracts_by_id[pid].items() if k != 'paper_id'}, ensure_ascii=False)}"
                           for pid in supporting_ids)
    prompt = prompts.VERIFIER_PROMPT.format(claim=gap["claim"], rationale=gap.get("rationale", ""),
                                            cited_ids=", ".join(supporting_ids), cited_paper_data=cited_data)
    loop = llm_client.run_tool_loop(prompts.VERIFIER_SYSTEM, prompt, toolset, ctx.execute,
                                    tier="strong", agent_name="verifier_agent",
                                    trace_metadata={"claim": gap["claim"][:300], "cited_paper_ids": supporting_ids})
    try:
        verdict = llm_client.parse_with_contract(loop.final_text, VerifierOutput)
    except Exception as e:
        # The final answer of a tool loop can't use JSON prefill; if it broke the contract, ask once
        # more (no tools) with the tool contexts it already gathered pasted in.
        logger.warning("  Verifier final answer broke the contract (%s) - re-asking for JSON", type(e).__name__)
        metrics.inc("contract_violations")
        contexts = "\n".join(t["result_preview"] for t in loop.trace if t.get("ok"))
        verdict = llm_client.call_json(prompt + "\n\nSource text already retrieved with get_paper_text:\n"
                                       + contexts, tier="strong", schema=VerifierOutput, purpose="repair-verifier-answer")
    llm_opinion = verdict.get("supported")
    verdict["supported"], verdict["decision_reason"] = verifier_decision(verdict, ctx.shown_text)
    verdict["counterexample_verified"] = bool(verdict.get("counterexample_found")) and not verdict["supported"] \
        and "quote verified" in verdict["decision_reason"]
    if verdict.get("counterexample_found") and not verdict["counterexample_verified"]:
        metrics.inc("verifier_counterexample_rejected")
    if verdict["counterexample_verified"]:
        cid = verdict["counterexample_paper_id"]
        meta = next((p for p in selected_papers if p["id"] == cid), None) or {"id": cid, **ctx.seen_meta.get(cid, {})}
        verdict["counterexample_paper"] = {"id": cid, "title": meta.get("title") or cid, "year": meta.get("year"),
                                           "url": meta.get("url") or links.paper_url(meta)}
    verdict["llm_opinion"] = llm_opinion
    if llm_opinion is not None and llm_opinion != verdict["supported"]:
        metrics.inc("verifier_llm_disagreed_with_rule")
        logger.info("    verifier facts -> supported=%s (%s); the model's own opinion was %s",
                    verdict["supported"], verdict["decision_reason"], llm_opinion)
    return verdict, loop.trace, sorted(ctx.read_ids), list(ctx.corpus_searches)


CONFIDENCE_CAP_SCORE = 70.0   # just below the High band (>= 75)


def _confidence_caps(state: dict) -> list:
    """Reasons the search behind this run was too narrow to call an absence claim 'Strong'
    (evidence.evidence_level) - and to give the legacy score more than CONFIDENCE_CAP_SCORE."""
    caps = []
    n_analyzed = len(state.get("paper_extracts") or [])
    if n_analyzed < CONFIG.high_confidence_min_papers:
        caps.append(f"only {n_analyzed} papers analysed (Strong needs >= {CONFIG.high_confidence_min_papers})")
    health = state.get("source_health")
    if health:   # empty = no source went through sources/net.py (offline fakes) - nothing to judge
        answered = sorted(s for s, h in health.items() if h.get("ok", 0) > 0)
        if len(answered) < CONFIG.high_confidence_min_sources:
            failed = sorted(s for s, h in health.items() if h.get("ok", 0) == 0)
            caps.append(f"papers came from {', '.join(answered) or 'no source'} only"
                        + (f" ({', '.join(failed)} did not answer)" if failed else "")
                        + f" - Strong needs >= {CONFIG.high_confidence_min_sources} sources")
    return caps


def analyze_gaps(state: AgentState) -> dict:
    logger.info("Proposing candidate research gaps from the comparison table + trends...")
    prompt = prompts.GAP_ANALYSIS_PROMPT.format(
        topic=state["research_topic"],
        conditions=conditions.prompt_block(state.get("search_conditions")),
        comparison_table=state["comparison_table"],
        trend_summary=state["trend_summary"],
    )
    proposal = llm_client.call_json(prompt, tier="strong", schema=GapProposalOutput, purpose="propose-gaps")
    raw_candidates = proposal["gap_candidates"]
    logger.info("Proposer returned %d candidate gap(s)", len(raw_candidates))
    early_privacy = []
    for gap in raw_candidates:   # same cheap redaction as above, before claims hit the log/UI
        for k in ("claim", "rationale"):
            gap[k], recs = privacy.sanitize(gap.get(k) or "", "final_answer", f"gap.{k}(draft)", use_llm=False)
            early_privacy += recs

    extracts_by_id = {ex["paper_id"]: ex for ex in state["paper_extracts"]}
    papers_by_id = {p["id"]: p for p in state["selected_papers"]}
    trace = list(state.get("tool_trace") or [])
    current_year = datetime.date.today().year
    verified = []
    for i, gap in enumerate(raw_candidates, 1):
        claim_preview = (gap.get("claim") or "")[:70]
        # Bug fix: only ids that are really among the extracted papers count as evidence. Before,
        # a made-up id passed the ">= 2 supporting papers" rule and earned evidence points.
        cited = list(dict.fromkeys(gap.get("supporting_paper_ids", [])))
        supporting_ids = [pid for pid in cited if pid in extracts_by_id]
        dropped = [pid for pid in cited if pid not in extracts_by_id]
        if dropped:
            metrics.inc("gap_ids_hallucinated", len(dropped))
            logger.warning("  [%d/%d] dropped %d cited id(s) not among analyzed papers: %s",
                           i, len(raw_candidates), len(dropped), dropped)
        if len(supporting_ids) < CONFIG.min_supporting_papers_for_gap:
            logger.info("  [%d/%d] REJECTED (only %d valid supporting paper(s), need >= %d): %r",
                        i, len(raw_candidates), len(supporting_ids), CONFIG.min_supporting_papers_for_gap, claim_preview)
            continue  # structural rule from architecture doc §7.3 — reject weakly-evidenced gaps

        over = run_budget_exceeded()
        if over:
            metrics.inc("token_budget_skipped_verifier" if "token" in over else "time_budget_skipped_verifier")
            logger.warning("  [%d/%d] %s - gap kept but NOT verified: %r", i, len(raw_candidates), over, claim_preview)
            verdict = {"supported": False, "notes": f"not verified: run {over.split(' MAX_')[0]} used up",
                       "unsupported_aspects": "unknown",
                       "decision_reason": f"{over} - not verified"}
            read_ids, corpus_searches = [], []
        else:
            logger.info("  [%d/%d] Verifying with tools (%d supporting papers): %r",
                        i, len(raw_candidates), len(supporting_ids), claim_preview)
            try:
                verdict, vtrace, read_ids, corpus_searches = _verify_with_tools(
                    gap, supporting_ids, extracts_by_id, state["selected_papers"])
                for t in vtrace:
                    t["gap_index"] = i
                trace += vtrace
            except llm_client.LLMError as e:
                logger.warning("  [%d/%d] Verifier failed, treating as unsupported: %s", i, len(raw_candidates), e)
                verdict = {"supported": False, "notes": "verifier call failed", "unsupported_aspects": "unknown",
                           "decision_reason": "verifier call failed"}
                read_ids, corpus_searches = [], []
        coverage = len([r for r in read_ids if r in supporting_ids]) / len(supporting_ids)
        if coverage == 0 and verdict.get("supported"):
            # live run with thinking on: the verifier made no tool call at all, yet its notes said
            # "Checked both cited papers ... no counterexample in corpus search". Facts it never
            # looked up are not facts.
            metrics.inc("verifier_answered_without_reading")
            verdict["supported"] = False
            verdict["decision_reason"] = "verifier did not read any cited paper (its notes claim otherwise) - unverified"

        # --- legacy 0-100 score (architecture doc §7.3) - kept in state for comparison with older
        # runs, NOT shown as the main answer: the evidence level below replaces it.
        evidence_count_score = min(len(supporting_ids) / 4, 1.0) * 40       # up to 40 pts

        cited_years = [papers_by_id.get(pid, {}).get("year") for pid in supporting_ids]
        cited_years = [y for y in cited_years if isinstance(y, int)]
        most_recent = max(cited_years) if cited_years else None
        if most_recent is None:
            recency_score = 10  # no year data at all -> can't claim it's recent
        else:
            age = max(current_year - most_recent, 0)
            recency_score = max(30 - age * 5, 5)  # up to 30 pts, decays with age of newest evidence

        # up to 30 pts: full marks only if the verifier said "supported" AND actually read the
        # source text of at least half the cited papers via the tool
        if verdict.get("supported"):
            verifier_score = 30 if coverage >= 0.5 else 15
        else:
            verifier_score = 5
        confidence_score = evidence_count_score + recency_score + verifier_score
        caps = _confidence_caps(state)
        if verdict.get("counterexample_found") and not verdict.get("counterexample_verified"):
            # not enough to refute the gap, but not nothing either: a live run had 3 of 4 gaps where
            # the verifier pointed at a paper it could not quote - keep the gap, flag it, no Strong
            caps = caps + [f"verifier named a possible counterexample ({verdict.get('counterexample_paper_id') or 'no id'}) "
                           "but could not quote it - check that paper"]
        if caps and confidence_score > CONFIDENCE_CAP_SCORE:
            confidence_score = CONFIDENCE_CAP_SCORE

        # --- evidence level: Strong / Moderate / Limited from countable facts (config.py, README
        # "Evidence levels"); the same caps keep a gap below Strong as they kept it below High.
        ev = evidence.gap_evidence(
            supporting_ids, extracts_by_id, papers_by_id, verifier_supported=bool(verdict.get("supported")),
            counterexample_verified=bool(verdict.get("counterexample_verified")),
            counterexample_unquoted=bool(verdict.get("counterexample_found")) and not verdict.get("counterexample_verified"),
            coverage=coverage)
        level, level_reasons = evidence.evidence_level(ev, caps, current_year)
        logger.info("    -> verdict supported=%s, read %d/%d sources, evidence=%s (%d supporting, %d full text, "
                    "%d contradicting)", verdict.get("supported"), len(read_ids), len(supporting_ids), level,
                    ev["n_supporting"], ev["n_supporting_full_text"], ev["n_contradicting"])

        verified.append({
            "claim": gap["claim"],
            "supporting_paper_ids": supporting_ids,
            "supporting_papers": links.supporting_papers(supporting_ids, papers_by_id),
            # verified sentence from each supporting paper (None = no quote found verbatim in its text)
            "evidence_quotes": evidence.evidence_quotes(supporting_ids, extracts_by_id, papers_by_id),
            "rationale": gap.get("rationale", ""),
            "evidence_level": level,                 # Strong | Moderate | Limited - the level shown to the user
            "evidence": ev,                          # the countable facts it was decided from
            "evidence_level_reasons": level_reasons, # which criterion kept it from the next level up
            "confidence_level": level,               # same value, for readers of older field names
            "confidence_score": round(confidence_score, 1),   # legacy 0-100 score, state only
            "confidence_caps": caps,
            "category": gap.get("category") or "Other",
            # the paper that refutes the gap, with a link from metadata - only when its quote was verified
            "counterexample_paper": verdict.get("counterexample_paper"),
            "verifier_supported": bool(verdict.get("supported")),
            "verifier_facts": {k: verdict.get(k) for k in ("cited_papers_relevant", "counterexample_found",
                                                          "counterexample_paper_id", "counterexample_quote",
                                                          "counterexample_verified", "overstated")},
            "verifier_decision_reason": verdict.get("decision_reason", ""),
            "verifier_notes": verdict.get("notes", ""),
            "unsupported_aspects": verdict.get("unsupported_aspects", ""),
            "evidence_read_ids": read_ids,
            "evidence_coverage": round(coverage, 2),
            "counterexample_searches": corpus_searches,
            "dropped_ids": dropped,
        })

    caps = _confidence_caps(state)
    if caps:
        logger.info("Evidence capped below Strong for this run: %s", "; ".join(caps))
    # what a person reads names papers; the ids stay in supporting_paper_ids / supporting_papers
    labels = state.get("paper_labels") or paper_labels(state["selected_papers"])
    for g in verified:
        for k in ("rationale", "verifier_notes", "unsupported_aspects"):
            g[k] = replace_ids_with_labels(g.get(k) or "", labels)
    logger.info("Gap analysis complete: %d/%d candidates passed verification", len(verified), len(raw_candidates))
    return {"gap_candidates": verified, "tool_trace": trace,
            "privacy_log": (state.get("privacy_log") or []) + early_privacy}


# ---------------------------------------------------------------------------
# Node 7b — Citation check before the report (citations.py, no LLM)
# ---------------------------------------------------------------------------
def verify_citations(state: AgentState, get=None) -> dict:
    """Looks up every paper the gaps cite in Crossref (and OpenAlex for non-OpenAlex papers) and
    compares title / year / DOI. A gap whose supporting citations do not all check out cannot be
    Strong. `get` is the HTTP getter (tests pass a fake one)."""
    import citations
    gaps = [dict(g) for g in state.get("gap_candidates") or []]
    papers = {p["id"]: p for p in (state.get("selected_papers") or []) if p.get("id")}
    for g in gaps:
        cp = g.get("counterexample_paper") or {}
        if cp.get("id") and cp["id"] not in papers:
            papers[cp["id"]] = cp
    ids = citations.cited_paper_ids(gaps)
    checks = {}
    if not CONFIG.verify_citations:
        logger.info("Citation check off (VERIFY_CITATIONS=false) - %d cited papers left unchecked", len(ids))
        checks = {pid: {"status": citations.UNCHECKED, "checks": [], "summary": "VERIFY_CITATIONS=false"} for pid in ids}
    else:
        get = get or citations.http_getter(CONFIG.citation_timeout)
        logger.info("Checking %d cited papers against Crossref/OpenAlex", len(ids))
        for pid in ids:
            over = run_budget_exceeded()
            if over and "time" in over:
                checks[pid] = {"status": citations.UNCHECKED, "checks": [], "summary": over}
                continue
            paper = papers.get(pid) or {"id": pid}
            checks[pid] = citations.check_paper(paper, citations.lookups_for(paper, get, CONFIG), CONFIG.citation_title_sim)
            metrics.inc(f"citation_{checks[pid]['status']}")
            logger.info("  %s %s - %s", citations.STATUS_MARK[checks[pid]["status"]], pid, checks[pid]["summary"])
    # remote titles are not kept (they may carry names); only status, counts and the short summary
    for c in checks.values():
        for r in c.get("checks", []):
            r.pop("remote_title", None)

    for g in gaps:
        summary = citations.gap_citation_summary(g, checks)
        g["citation_check"] = summary
        g["supporting_papers"] = [{**sp, "citation_status": (checks.get(sp.get("id")) or {}).get("status", citations.UNCHECKED)}
                                  for sp in g.get("supporting_papers") or []]
        reason = citations.cap_reason(summary)
        if reason:
            if g.get("evidence_level") == evidence.STRONG:
                logger.info("  gap capped below Strong: %s", reason)
                g["evidence_level"] = g["confidence_level"] = evidence.MODERATE
            if g.get("evidence_level") != evidence.LIMITED:
                g["evidence_level_reasons"] = list(g.get("evidence_level_reasons") or []) + [reason]
            g["confidence_caps"] = list(g.get("confidence_caps") or []) + [reason]
    n_ok = sum(1 for c in checks.values() if c["status"] in citations.OK_STATUSES)
    logger.info("Citation check: %d/%d cited papers checked out", n_ok, len(checks))
    return {"gap_candidates": gaps, "citation_checks": checks}


# ---------------------------------------------------------------------------
# Node 8 — Privacy guard on the final answer (LLM extraction + decision model)
# ---------------------------------------------------------------------------
def privacy_guard(state: AgentState) -> dict:
    """Last line of defence: everything the user is about to read (trend summary, gap claims,
    rationales, verifier notes) goes through the same detectors + decision model with
    channel=final_answer. One LLM extraction call over the combined text, then each field is
    redacted with the decided entities."""
    fields = [("trend_summary", state.get("trend_summary") or "")]
    for i, g in enumerate(state.get("gap_candidates") or []):
        for k in ("claim", "rationale", "verifier_notes", "unsupported_aspects"):
            fields.append((f"gap{i}.{k}", g.get(k) or ""))
    combined = "\n\n".join(f"[{k}] {v}" for k, v in fields if v)
    llm_ents = privacy.llm_detect(combined, "final_answer") if combined.strip() else []

    records, out, total = [], {}, 0
    for key, text in fields:
        ents = privacy.regex_detect(text) + privacy.metadata_detect(text) + [e for e in llm_ents if e["text"] in text]
        new_text, recs = privacy.decide_and_redact(text, ents, "final_answer", key)
        out[key] = new_text
        records += recs
        total += sum(r["occurrences"] for r in recs if r["action"] == "REDACT")

    gaps = []
    for i, g in enumerate(state.get("gap_candidates") or []):
        g = dict(g)
        for k in ("claim", "rationale", "verifier_notes", "unsupported_aspects"):
            g[k] = out.get(f"gap{i}.{k}", g.get(k, ""))
        gaps.append(g)
    logger.info("Privacy guard on final answer: %d decision(s), %d redaction(s)", len(records), total)
    return {
        "trend_summary": out.get("trend_summary", state.get("trend_summary", "")),
        "gap_candidates": gaps,
        "final_answer_redactions": total,
        "privacy_log": (state.get("privacy_log") or []) + records,
        "run_metrics": metrics.snapshot(),
    }
