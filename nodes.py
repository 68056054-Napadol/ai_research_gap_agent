"""
The 7 graph nodes — see architecture doc §2.2 for the rationale behind each one. Every node
function takes the current AgentState and returns a dict of the fields it updates; LangGraph
merges that into the running state (standard LangGraph node signature).
"""
import logging
import re
import time

import embeddings
import llm_client
import prompts
from config import CONFIG
from sources import search_all, REGISTRY, pdf_extract
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
# Node 1 — Topic Understanding & Keyword Generation
# ---------------------------------------------------------------------------
def generate_keywords(state: AgentState) -> dict:
    prior = state.get("prior_queries_tried", [])
    retry_context = ""
    if prior:
        logger.info("Regenerating keywords (retry #%d) — avoiding %d previously tried queries",
                    state.get("retry_count", 0), len(prior))
        retry_context = prompts.RETRY_CONTEXT_TEMPLATE.format(prior_queries="\n".join(f"- {q}" for q in prior))
    else:
        logger.info("Generating search keywords for topic: %r", state["research_topic"])

    prompt = prompts.KEYWORD_GEN_PROMPT.format(topic=state["research_topic"], retry_context=retry_context)
    result = llm_client.call_json(prompt, tier="fast")
    queries = result.get("queries", [state["research_topic"]])
    logger.info("Generated %d search queries: %s", len(queries), queries)

    return {
        "search_queries": queries,
        "prior_queries_tried": prior + queries,
    }


# ---------------------------------------------------------------------------
# Node 2 — Multi-Source Search
# ---------------------------------------------------------------------------
def search_sources(state: AgentState) -> dict:
    logger.info("Searching %d source(s) [%s] for %d quer%s",
                len(CONFIG.enabled_sources), ", ".join(CONFIG.enabled_sources),
                len(state["search_queries"]), "y" if len(state["search_queries"]) == 1 else "ies")
    all_papers = []
    for query in state["search_queries"]:
        papers = search_all(
            query,
            sources=list(CONFIG.enabled_sources),
            year_from=state.get("year_from"),
            year_to=state.get("year_to"),
            limit_per_source=max(10, CONFIG.max_candidate_papers // len(state["search_queries"])),
        )
        logger.info("  query %r -> %d papers", query, len(papers))
        all_papers.extend(papers)

    # dedupe across queries too, and cap to the candidate-pool budget (architecture doc §7.4)
    from sources import _dedupe
    deduped = _dedupe(all_papers)[: CONFIG.max_candidate_papers]
    logger.info("Total: %d raw results -> %d after dedup (capped at %d)",
                len(all_papers), len(deduped), CONFIG.max_candidate_papers)

    return {"candidate_papers": deduped}


# ---------------------------------------------------------------------------
# Node 3 — Relevance Filtering & Ranking
# ---------------------------------------------------------------------------
def filter_and_rank(state: AgentState) -> dict:
    logger.info("Ranking %d candidate papers by embedding similarity to the topic",
                len(state["candidate_papers"]))
    ranked = embeddings.rank_by_similarity(state["research_topic"], state["candidate_papers"])
    above_threshold = embeddings.filter_by_threshold(ranked)
    logger.info("%d/%d papers passed similarity threshold %.2f",
                len(above_threshold), len(ranked), CONFIG.similarity_threshold)

    # NOTE (architecture doc §7.2 stage 2 - optional LLM re-rank): for course-scope speed this
    # scaffold ships with embedding-only ranking. To add the semantic re-rank pass, take the
    # top CONFIG.llm_rerank_top_n from `above_threshold`, ask the LLM to judge true relevance
    # per-paper, and re-sort. Left as the highest-value TODO per the README's "what's next".

    selected = above_threshold[: CONFIG.max_deep_analysis_papers]
    logger.info("Selected top %d papers for deep analysis (budget: %d)",
                len(selected), CONFIG.max_deep_analysis_papers)
    return {"ranked_papers": above_threshold, "selected_papers": selected}


# ---------------------------------------------------------------------------
# Node 4 — Sufficiency Check (conditional edge, no LLM call)
# ---------------------------------------------------------------------------
def check_sufficiency(state: AgentState) -> dict:
    retry_count = state.get("retry_count", 0)
    n_selected = len(state.get("selected_papers", []))
    sufficient = n_selected >= CONFIG.min_papers_before_retry
    logger.info("Sufficiency check: %d selected papers (need >= %d) -> %s",
                n_selected, CONFIG.min_papers_before_retry, "sufficient" if sufficient else "INSUFFICIENT")
    return {"sufficient": sufficient, "retry_count": retry_count + (0 if sufficient else 1)}


def route_after_sufficiency(state: AgentState) -> str:
    """Conditional-edge function passed to graph.add_conditional_edges (see graph.py)."""
    if state.get("sufficient"):
        return "extract"
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


def extract_paper_data(state: AgentState) -> dict:
    total = len(state["selected_papers"])
    logger.info("Extracting structured data (method/dataset/results) from %d papers", total)
    extracts = []
    for i, paper in enumerate(state["selected_papers"], 1):
        title = (paper.get("title") or "")[:80]
        full_text = paper.get("full_text")

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
                paper["full_text"] = full_text  # cache on the paper dict for the rest of this run
                logger.info("  [%d/%d] Got PDF full text (%d chars, %.1fs)",
                            i, total, len(full_text), time.monotonic() - pdf_start)
            else:
                logger.info("  [%d/%d] No usable PDF (%.1fs) — falling back to abstract",
                            i, total, time.monotonic() - pdf_start)

        text = full_text or paper.get("abstract", "")
        if not text:
            logger.info("  [%d/%d] SKIPPED (no abstract or text available): %s", i, total, title)
            continue

        logger.info("  [%d/%d] Extracting (%s): %s", i, total,
                    "full text" if full_text else "abstract only", title)
        paper_start = time.monotonic()
        char_limit = FULL_TEXT_CHAR_LIMIT if full_text else ABSTRACT_CHAR_LIMIT
        text_shown_to_llm = text[:char_limit]
        prompt = prompts.EXTRACT_PROMPT.format(title=paper.get("title", ""), year=paper.get("year", "?"), text=text_shown_to_llm)
        try:
            result = llm_client.call_json(prompt, tier="fast")
        except llm_client.LLMError as e:
            logger.warning("  [%d/%d] Extraction FAILED for %s (%.1fs): %s",
                            i, total, paper["id"], time.monotonic() - paper_start, e)
            continue

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
    return {"paper_extracts": extracts}


# ---------------------------------------------------------------------------
# Node 6 — Cross-Paper Comparison & Trend Synthesis
# ---------------------------------------------------------------------------
def compare_papers(state: AgentState) -> dict:
    logger.info("Building comparison table from %d extracted papers", len(state["paper_extracts"]))
    by_id = {p["id"]: p for p in state["selected_papers"]}
    rows = ["| Paper | Year | Method | Dataset | Key findings | Quote verified |",
            "|---|---|---|---|---|---|"]
    for ex in state["paper_extracts"]:
        p = by_id.get(ex["paper_id"], {})
        # "Quote verified" = extraction-quote grounding (README "Fact-checking"): whether the
        # supporting_quote the LLM gave for key_findings was actually found verbatim in the
        # source text (Node 5). Included in the table itself (not just the UI) so the strong-tier
        # LLM synthesizing trends below can see which findings are on solid footing.
        quote_flag = "✅ yes" if ex.get("quote_verified") else "⚠️ no"
        rows.append(
            f"| {ex['paper_id']} | {p.get('year', '?')} | {ex.get('method', '')} "
            f"| {ex.get('dataset', '')} | {ex.get('key_findings', '')} | {quote_flag} |"
        )
    table = "\n".join(rows)

    logger.info("Calling strong-tier LLM to synthesize trends across papers...")
    prompt = prompts.COMPARE_PROMPT.format(topic=state["research_topic"], comparison_table=table)
    trend_summary = llm_client.call(prompt, tier="strong")
    logger.info("Trend synthesis complete (%d characters)", len(trend_summary))

    return {"comparison_table": table, "trend_summary": trend_summary}


# ---------------------------------------------------------------------------
# Node 7 — Gap Analysis + Verifier -> confidence-scored gap candidates
# ---------------------------------------------------------------------------
def analyze_gaps(state: AgentState) -> dict:
    logger.info("Proposing candidate research gaps from the comparison table + trends...")
    prompt = prompts.GAP_ANALYSIS_PROMPT.format(
        topic=state["research_topic"],
        comparison_table=state["comparison_table"],
        trend_summary=state["trend_summary"],
    )
    proposal = llm_client.call_json(prompt, tier="strong")
    raw_candidates = proposal.get("gap_candidates", [])
    logger.info("Proposer returned %d candidate gap(s)", len(raw_candidates))

    extracts_by_id = {ex["paper_id"]: ex for ex in state["paper_extracts"]}
    papers_by_id = {p["id"]: p for p in state["selected_papers"]}
    verified = []
    for i, gap in enumerate(raw_candidates, 1):
        supporting_ids = gap.get("supporting_paper_ids", [])
        claim_preview = (gap.get("claim") or "")[:70]
        if len(supporting_ids) < CONFIG.min_supporting_papers_for_gap:
            logger.info("  [%d/%d] REJECTED (only %d supporting paper(s), need >= %d): %r",
                        i, len(raw_candidates), len(supporting_ids), CONFIG.min_supporting_papers_for_gap, claim_preview)
            continue  # structural rule from architecture doc §7.3 — reject weakly-evidenced gaps

        logger.info("  [%d/%d] Verifying (%d supporting papers): %r",
                    i, len(raw_candidates), len(supporting_ids), claim_preview)
        cited_data = "\n".join(
            f"- {pid}: {extracts_by_id.get(pid, {})}" for pid in supporting_ids
        )
        verifier_prompt = prompts.VERIFIER_PROMPT.format(
            claim=gap["claim"], rationale=gap.get("rationale", ""), cited_paper_data=cited_data,
        )
        try:
            verdict = llm_client.call_json(verifier_prompt, tier="strong")
        except llm_client.LLMError as e:
            logger.warning("  [%d/%d] Verifier call failed, treating as unsupported: %s", i, len(raw_candidates), e)
            verdict = {"supported": False, "notes": "verifier call failed", "unsupported_aspects": "unknown"}

        # --- confidence scoring formula (architecture doc §7.3) ---
        # A simple, explainable composite — tune the weights once you have real outputs to
        # calibrate against (architecture doc §8, "confidence calibration" eval step).
        # TODO(Member C): this is a first-pass heuristic, not a validated formula. Before the
        # demo, run the calibration check in the architecture doc §8 and adjust weights/cutoffs.
        evidence_count_score = min(len(supporting_ids) / 4, 1.0) * 40       # up to 40 pts

        cited_years = [papers_by_id.get(pid, {}).get("year") for pid in supporting_ids]
        cited_years = [y for y in cited_years if isinstance(y, int)]
        most_recent = max(cited_years) if cited_years else None
        current_year_estimate = 2026
        if most_recent is None:
            recency_score = 10  # no year data at all -> can't claim it's recent
        else:
            age = max(current_year_estimate - most_recent, 0)
            recency_score = max(30 - age * 5, 5)  # up to 30 pts, decays with age of newest evidence

        verifier_score = 30 if verdict.get("supported") else 5               # up to 30 pts
        confidence_score = evidence_count_score + recency_score + verifier_score
        confidence_level = "High" if confidence_score >= 75 else "Medium" if confidence_score >= 45 else "Low"
        logger.info("    -> verdict supported=%s, confidence=%s (%.1f/100)",
                    verdict.get("supported"), confidence_level, confidence_score)

        verified.append({
            **gap,
            "confidence_score": round(confidence_score, 1),
            "confidence_level": confidence_level,
            "verifier_notes": verdict.get("notes", ""),
        })

    logger.info("Gap analysis complete: %d/%d candidates passed verification", len(verified), len(raw_candidates))
    return {"gap_candidates": verified}
