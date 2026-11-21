"""
Offline end-to-end tests. Run:  python -m pytest -q tests
No network, no API key: the LLM, the paper sources and the embedding model are replaced by the
fakes in tests/fakes.py (fictional papers and names only).
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import decision_model  # noqa: E402
import embeddings  # noqa: E402
import llm_client  # noqa: E402
import main  # noqa: E402
import privacy  # noqa: E402
import sources  # noqa: E402
from config import CONFIG  # noqa: E402
from tests.fakes import FAKE_AUTHORS, FakeLLM, FakeSource, fake_embed  # noqa: E402


@pytest.fixture
def offline(monkeypatch, tmp_path):
    fake = FakeLLM()
    monkeypatch.setattr(llm_client, "_call_anthropic", fake.call)   # real call()/call_json() + tracing still run
    monkeypatch.setattr(llm_client, "_tool_turn", fake.tool_turn)
    for name in ("semantic_scholar", "arxiv", "openalex"):
        monkeypatch.setitem(sources.REGISTRY, name, FakeSource(name))
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    monkeypatch.setattr(embeddings, "embed_for_rag", fake_embed)
    monkeypatch.setattr(embeddings, "embed_for_ranking", fake_embed)
    import rag_store
    monkeypatch.setattr(CONFIG, "rag_store_dir", str(tmp_path / "rag_store"))
    monkeypatch.setattr(rag_store, "_store", None)
    monkeypatch.setattr(CONFIG, "llm_backend", "cloud")
    monkeypatch.setattr(CONFIG, "fetch_full_text", False)
    monkeypatch.setattr(CONFIG, "similarity_threshold", 0.1)
    monkeypatch.setattr(CONFIG, "min_papers_before_retry", 3)
    monkeypatch.setattr(CONFIG, "output_dir", str(tmp_path))
    # hermetic: never inherit run settings from the shell / .env (a .bat that sets
    # RAG_REUSE_EXTRACTS=false for a model comparison must not change test outcomes)
    monkeypatch.setattr(CONFIG, "rag_enabled", True)
    monkeypatch.setattr(CONFIG, "rag_reuse_extracts", True)
    monkeypatch.setattr(CONFIG, "rag_embedding", "sentence_transformers:BAAI/bge-small-en-v1.5")
    monkeypatch.setattr(CONFIG, "llm_temperature", 0.0)
    monkeypatch.setattr(CONFIG, "privacy_llm_on_papers", "auto")
    monkeypatch.setattr(CONFIG, "ranking_embedding", "sentence_transformers:BAAI/bge-small-en-v1.5")
    monkeypatch.setattr(CONFIG, "high_confidence_min_papers", 8)
    monkeypatch.setattr(CONFIG, "ollama_think", False)
    monkeypatch.setattr(CONFIG, "ollama_think_purposes", frozenset())
    monkeypatch.setattr(CONFIG, "high_confidence_min_sources", 2)
    monkeypatch.setattr(CONFIG, "tracing_enabled", True)
    monkeypatch.setattr(CONFIG, "max_run_tokens", 0)
    monkeypatch.setattr(CONFIG, "max_run_seconds", 0)
    # the tests exercise multi-source behaviour; the shipped default is the single-source MVP
    monkeypatch.setattr(CONFIG, "enabled_sources", ("semantic_scholar", "arxiv", "openalex"))
    monkeypatch.setattr(CONFIG, "min_candidate_papers", 1)       # the fake corpus has only 6 papers
    monkeypatch.setattr(CONFIG, "verify_citations", False)       # no network; tests below fake the lookups
    return fake


def test_full_run_uses_real_tool_calls(offline):
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    trace = state["tool_trace"]
    search = [t for t in trace if t["agent"] == "search_agent"]
    verify = [t for t in trace if t["agent"] == "verifier_agent"]
    assert search and all(t["tool_name"] == "search_papers" for t in search)
    assert all("query" in t["arguments"] for t in search)          # tool name + tool arguments chosen by the LLM
    assert verify and {t["tool_name"] for t in verify} == {"search_paper_chunks", "search_corpus"}
    assert not state["search_used_fallback"]
    # the invalid call (query="x") is rejected by the argument contract and reported back to the LLM
    assert any(t.get("contract_error") for t in search)
    # the user's year range is enforced as policy even though the LLM asked for year_from=2015
    assert any("clamped" in " ".join(t.get("policy_notes") or []) for t in search)
    assert state["gap_candidates"], "at least one gap should survive"


def test_no_author_names_anywhere_in_outputs(offline):
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    dumped = json.dumps(state, default=str, ensure_ascii=False) + main.render_report(state)
    for full in FAKE_AUTHORS:
        for part in full.split():
            if part[0].isupper() and len(part) > 4:
                assert part not in dumped, f"author name part {part!r} leaked into outputs"
    assert "alex.testauthor@example.org" not in dumped
    assert all("authors" not in p for p in state["candidate_papers"])


def test_hallucinated_ids_are_dropped(offline):
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    for g in state["gap_candidates"]:
        assert "arxiv:9999.99999" not in g["supporting_paper_ids"]
    assert any("arxiv:9999.99999" in g["dropped_ids"] for g in state["gap_candidates"])
    # the 1-real-id + 1-fake-id gap must now be rejected by the >=2 supporting papers rule
    assert all("weak gap" not in g["claim"] for g in state["gap_candidates"])


def test_contract_violation_is_retried(offline):
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    m = state["run_metrics"]
    assert m["contract_violations"] >= 1          # the empty-queries answer was caught...
    assert m.get("json_calls_failed", 0) == 0     # ...and fixed by the retry, not a crash
    assert len(state["search_queries"]) == 3


def test_person_in_topic_is_redacted_but_research_kept(offline):
    state = main.run("few-shot NER papers by Dr. Pat Realperson for low-resource languages", 2020, 2026, 5)
    assert "Realperson" not in state["research_topic"]
    assert state["input_decision"]["action"] == "PROCEED_REDACTED"
    assert "few-shot NER" in state["research_topic"]


def test_person_only_topic_is_rejected(offline):
    state = main.run("งานวิจัยของ ดร.สมมติ ไม่มีจริง", None, None, 5)
    assert state["input_decision"]["action"] == "REJECT"
    assert "gap_candidates" not in state


def test_eponym_is_kept_names_are_redacted():
    text = "We apply a Kalman filter. As Testauthor et al. showed, it helps. Mail x.y@uni.example.edu"
    privacy.reset_run()
    out, recs = privacy.decide_and_redact(
        text, privacy.regex_detect(text) + [{"text": "Kalman", "type": "PERSON", "role": "eponym",
                                             "detected_by": "llm", "confidence": 0.9}],
        "paper_text", "t")
    assert "Kalman filter" in out and "Testauthor" not in out and "@" not in out
    assert {r["rule"] for r in recs} >= {"R1", "R2", "R4"}


@pytest.mark.parametrize("inputs,expected", [
    ({"type": "EMAIL", "role": "contact", "detected_by": "regex", "confidence": 1, "channel": "paper_text"}, "R1"),
    ({"type": "PERSON", "role": "eponym", "detected_by": "llm", "confidence": 0.9, "channel": "paper_text"}, "R2"),
    ({"type": "PERSON", "role": "author", "detected_by": "metadata", "confidence": 1, "channel": "paper_text"}, "R3"),
    ({"type": "PERSON", "role": "other", "detected_by": "llm", "confidence": 0.5, "channel": "user_input"}, "R5"),
    ({"type": "PERSON", "role": "other", "detected_by": "llm", "confidence": 0.5, "channel": "final_answer"}, "R8"),
    ({"type": "ORGANIZATION", "role": "affiliation", "detected_by": "llm", "confidence": 0.9, "channel": "paper_text"}, "R7"),
])
def test_decision_table_rules(inputs, expected):
    assert decision_model.evaluate("entity_action", inputs)["rule"] == expected


def test_model_without_tool_support_falls_back_and_is_labelled(offline):
    offline.use_tools = False
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    assert state["search_used_fallback"]
    assert all(t["agent"] == "harness_fallback" for t in state["tool_trace"] if t["tool_name"] == "search_papers")


def test_pdf_header_quirks_are_redacted():
    """Regressions from a live run: grouped e-mails and superscript-glued affiliations."""
    privacy.reset_run()
    text = "1Example AI Lab 2Dept. of CS, Sample University\n{alpha, beta, gamma}@lab.example.com"
    ents = privacy.regex_detect(text) + [
        {"text": "Example AI Lab", "type": "ORGANIZATION", "role": "affiliation", "detected_by": "llm", "confidence": 0.9},
        {"text": "Dept. of CS, Sample University", "type": "ORGANIZATION", "role": "affiliation", "detected_by": "llm", "confidence": 0.9}]
    out, _ = privacy.decide_and_redact(text, ents, "paper_text", "t")
    assert "Example AI Lab" not in out and "Sample University" not in out
    assert "alpha" not in out and "@lab.example.com" not in out


def test_placeholders_never_reach_search_apis(offline, monkeypatch):
    seen = []
    orig = sources.REGISTRY["arxiv"].search_papers
    monkeypatch.setattr(sources.REGISTRY["arxiv"], "search_papers", lambda q, **kw: (seen.append(q), orig(q, **kw))[1])
    import tools
    ctx = tools.SearchToolContext(enabled_sources=["arxiv"])
    ctx.execute("search_papers", {"query": "few-shot NER in Thai by [PERSON]", "source": "arxiv"})
    assert seen == ["few-shot NER in Thai"]
    assert privacy.strip_placeholders("งานวิจัยของ [PERSON] เรื่อง NER") == "งานวิจัย เรื่อง NER"


def test_content_words_any_script():
    assert privacy.content_words_left("[PERSON]") == 0
    assert privacy.content_words_left("งานวิจัยของ ดร.[PERSON]") == 0
    assert privacy.content_words_left("少样本命名实体识别") >= 1          # Chinese topic is not "empty"
    assert privacy.content_words_left("papers by [PERSON] on few-shot NER") >= 2


def test_knowledge_base_is_reused_across_runs(offline):
    first = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    assert first["knowledge_base"]["papers"] >= 5
    assert first["run_metrics"].get("rag_papers_indexed", 0) >= 5
    second = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    m = second["run_metrics"]
    assert m.get("rag_papers_indexed", 0) == 0               # nothing re-embedded
    assert m.get("extract_cache_hits", 0) >= 5               # no extraction LLM calls repeated
    assert second["knowledge_base"]["papers"] == first["knowledge_base"]["papers"]


def test_verifier_searches_for_counterexamples(offline):
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    for g in state["gap_candidates"]:
        assert g["counterexample_searches"], "verifier should search the corpus before accepting an absence claim"
        assert g["evidence_coverage"] == 1.0


def test_links_come_from_metadata_only(offline):
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    analyzed = {p["id"] for p in state["selected_papers"]}
    for g in state["gap_candidates"]:
        assert g["supporting_papers"]
        for sp in g["supporting_papers"]:
            assert sp["id"] in analyzed                          # never a link to a paper we didn't analyze
            assert sp["url"] == "https://arxiv.org/abs/" + sp["id"].split(":", 1)[1] or sp["url"].startswith("https://")
    report = main.render_report(state)
    assert "https://arxiv.org/abs/0000.00001" in report or "https://openalex.org/" in report


def test_chunking_skips_references_and_labels_sections():
    import rag_store
    md = ("# A paper\n\n## Abstract\nWe study X.\n\n## 3 Method\nOur approach uses Y.\n\n"
          "## 5 Limitations\nOnly English.\n\n## References\nSomeone, A. 2020. A title.\n")
    chunks = rag_store.chunk_text(md, "full_text", size=200, overlap=20)
    sections = [s for s, _ in chunks]
    assert "method" in sections and "limitations" in sections
    assert not any("Someone" in c for _, c in chunks)


@pytest.fixture
def traced(offline):
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    import tracing
    exporter = InMemorySpanExporter()
    tracing.reset()
    import uuid
    tracing.init(span_exporter=exporter, force=True, public_key=f"pk-lf-test-{uuid.uuid4().hex[:8]}")
    yield exporter
    tracing.flush()
    tracing.reset()


def _spans(exporter):
    import tracing
    tracing.flush()
    return [{"name": s.name, "attrs": {k: str(v) for k, v in (s.attributes or {}).items()},
             "parent": s.parent.span_id if s.parent else None, "id": s.context.span_id}
            for s in exporter.get_finished_spans()]


def test_trace_has_prompts_tool_calls_and_context(traced):
    main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    spans = _spans(traced)
    names = [s["name"] for s in spans]
    assert "run-research-gap-agent" in names
    assert {"check-input-privacy", "search-literature", "propose-and-verify-gaps", "redact-final-answer"} <= set(names)
    assert {"search-agent", "verifier-agent", "search-agent-turn", "verifier-agent-turn"} <= set(names)
    assert {"generate-search-queries", "extract-paper-facts", "propose-gaps"} <= set(names)
    # audit fix: no duplicate/noisy spans - a retriever TOOL already records query + context,
    # so its internal chunk search and the query embedding are not emitted separately
    ids_type = {s["id"]: s["attrs"].get("langfuse.observation.type") for s in spans}
    assert not any(s["name"] == "retrieve-chunks" and ids_type.get(s["parent"]) == "retriever" for s in spans)
    assert sum(1 for s in spans if s["name"] == "embed-texts") <= 3
    types = {s["name"]: s["attrs"].get("langfuse.observation.type") for s in spans}
    assert types["search-agent"] == "agent" and types["search-papers"] == "retriever"
    assert types["search-agent-turn"] == "generation" and types["check-input-privacy"] == "guardrail"
    # requested tool call appears in the assistant message as an OpenAI-format tool_call card
    turn = next(s for s in spans if s["name"] == "search-agent-turn" and "tool_calls" in s["attrs"].get("langfuse.observation.output", ""))
    assert '"type": "function"' in turn["attrs"]["langfuse.observation.output"]
    tool = [s for s in spans if s["name"] == "search-papers"]
    assert tool, names
    blob = " ".join(tool[0]["attrs"].values())
    assert "few-shot NER low-resource" in blob          # arguments chosen by the LLM
    assert "total_unique_so_far" in blob               # context returned by the tool
    gen = next(s for s in spans if s["name"] == "generate-search-queries")
    gin = gen["attrs"]["langfuse.observation.input"]
    assert '"role": "system"' in gin and '"role": "user"' in gin                   # system + user prompt, role-labelled
    assert gen["attrs"].get("langfuse.observation.model.name")                      # model on the generation
    assert "langfuse.observation.usage_details" in gen["attrs"]                     # token usage
    root = next(s for s in spans if s["name"] == "run-research-gap-agent")
    assert "few-shot named entity recognition" in root["attrs"]["langfuse.observation.input"]   # trace input = redacted request
    assert "final_answer" in root["attrs"]["langfuse.observation.output"]
    assert "research-gap" in str(root["attrs"].get("langfuse.trace.tags"))
    assert all(s["parent"] is not None for s in spans if s["id"] != root["id"]), "all spans belong to one trace"


def test_trace_outbound_contract_no_pii(traced):
    main.run("few-shot NER papers by Dr. Pat Realperson for low-resource languages", 2020, 2026, 5)
    spans = _spans(traced)
    everything = " ".join(" ".join(s["attrs"].values()) for s in spans)
    assert "Realperson" not in everything                      # raw topic never traced
    for full in FAKE_AUTHORS:
        for part in full.split():
            if len(part) > 4:
                assert part not in everything, f"{part!r} leaked into the trace"
    assert "alex.testauthor@example.org" not in everything
    sanitize = next(s for s in spans if s["name"] == "check-input-privacy")
    assert "withheld" in " ".join(sanitize["attrs"].values())


@pytest.mark.parametrize("facts,seen,expected", [
    ({"cited_papers_relevant": True, "counterexample_found": False, "overstated": False}, set(), True),
    ({"cited_papers_relevant": False, "counterexample_found": False, "overstated": False}, set(), False),
    # a counterexample without a verbatim sentence from that paper is ignored (live-run bug)
    ({"cited_papers_relevant": True, "counterexample_found": True, "counterexample_paper_id": "arxiv:1", "overstated": False}, {"arxiv:1"}, True),
    ({"cited_papers_relevant": True, "counterexample_found": True, "counterexample_paper_id": "arxiv:404", "overstated": False}, {"arxiv:1"}, True),
    ({"cited_papers_relevant": True, "counterexample_found": False, "overstated": True}, set(), False),
])
def test_verifier_decision_rule(facts, seen, expected):
    import nodes
    assert nodes.verifier_decision(facts, seen)[0] is expected


def test_absence_claim_with_no_counterexample_is_supported(offline):
    """Live-run regression: the model said supported=False because 'no paper does X' - for an
    'X is missing' claim that is evidence FOR it. Code now decides from the reported facts."""
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    assert state["gap_candidates"] and all(g["verifier_supported"] for g in state["gap_candidates"])
    assert state["run_metrics"].get("verifier_llm_disagreed_with_rule", 0) >= 1


def test_full_text_extractions_are_reused_on_warm_run(offline, monkeypatch):
    """Live-run regression: warm runs re-read full text from the store, and the extraction input
    must be byte-identical to the cold run's, or the extraction cache never hits."""
    from sources import pdf_extract
    md = ("# {t}\n\n## Abstract\nWe study few-shot NER in low-resource languages.\n\n## 3 Method\n"
          "Prototype networks with label semantics.\n\n## 4 Experiments\nWe use a ten-language benchmark.\n\n"
          "## 5 Results\nF1 improves by 4 points.\n\n## 6 Limitations\nNo code-switched data.\n\n## References\nX. 2020.\n")
    monkeypatch.setattr(pdf_extract, "fetch_and_convert", lambda url, pid: md.format(t=pid))
    monkeypatch.setattr(pdf_extract, "overwrite_cache", lambda pid, text: None)
    monkeypatch.setattr(CONFIG, "fetch_full_text", True)
    for src in sources.REGISTRY.values():
        orig = src.search_papers
        monkeypatch.setattr(src, "search_papers", lambda q, _o=orig, **kw: [{**p, "oa_pdf_url": "https://x/p.pdf"} for p in _o(q, **kw)])
    cold = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    assert all(e["extraction_confidence"] == "full_text" for e in cold["paper_extracts"])
    warm = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    m = warm["run_metrics"]
    assert m.get("rag_full_text_reused") == 5
    assert m.get("extract_cache_hits") == 5, m


def test_embedding_formats_follow_model_cards():
    q = embeddings.format_texts("ollama:embeddinggemma:300m", ["what is X?"], "query")[0]
    d = embeddings.format_texts("ollama:embeddinggemma:300m", ["passage"], "document")[0]
    assert q == "task: search result | query: what is X?" and d == "title: none | text: passage"
    assert embeddings.format_texts("ollama:bge-m3", ["a"], "query") == ["a"]       # bge-m3 needs no prefix


def test_ollama_profiles():
    """gpt-oss can't disable thinking and returns empty content under format=json (live finding)."""
    assert llm_client.ollama_profile("gpt-oss:20b") == {"think": "low", "json_format": False}
    assert llm_client.ollama_profile("qwen3:14b")["json_format"] is True


def test_sufficiency_never_asks_for_more_than_the_budget(offline, monkeypatch):
    """Live-run regression: --max-papers 4 with MIN_PAPERS_BEFORE_RETRY=5 retried forever."""
    import nodes
    monkeypatch.setattr(CONFIG, "min_papers_before_retry", 5)
    out = nodes.check_sufficiency({"selected_papers": [{}] * 4, "max_papers": 4, "retry_count": 0})
    assert out["sufficient"] is True


def test_markdown_link_email_is_redacted_everywhere():
    """Live-run leak: pymupdf4llm turns e-mails into [a@b.c](mailto:a@b.c); the '[' before the
    address defeated the replacement boundary - at the source AND at the outbound boundary."""
    import outbound
    privacy.reset_run()
    text = "**Correspondence:** [first.last@uni-name.ac](mailto:first.last@uni-name.ac)\n"
    out, _ = privacy.decide_and_redact(text, privacy.regex_detect(text), "paper_text", "t")
    assert "@" not in out
    cleaned = outbound.enforce({"x": [text]}, "test")
    assert "@" not in json.dumps(cleaned)
    assert outbound.violations(cleaned) == []


# ---------------------------------------------------------------------------
# Tool-argument repair (harness): repair what is unambiguous, reject what is a real mistake
# ---------------------------------------------------------------------------
def _verifier_ctx():
    import tools
    papers = [{"id": "arxiv:0000.00001", "title": "T", "year": 2024, "abstract": "An abstract about few-shot NER."}]
    return tools.VerifierToolContext(papers, store=None)


def test_live_regression_max_chars_no_longer_costs_a_turn():
    """qwen3:14b sent get_paper_text(max_chars=10000) in 18/183 live calls -> rejected (le=8000)
    -> a wasted model turn each time. The knob is gone; the call now succeeds with a note."""
    import metrics
    metrics.reset()
    text, entry = _verifier_ctx().execute("get_paper_text", {"paper_id": "arxiv:0000.00001", "max_chars": 10000})
    assert entry["ok"] is True
    assert "max_chars" in json.loads(text)["harness_notes"][0]
    assert metrics.get("tool_args_invalid") == 0 and metrics.get("tool_args_repaired") == 1
    assert "max_chars" not in json.dumps(__import__("tools").GET_PAPER_TEXT_SPEC)   # not offered to the model


def test_out_of_range_and_string_numbers_are_repaired(offline):
    import tools
    ctx = tools.SearchToolContext(enabled_sources=["arxiv"])
    text, entry = ctx.execute("search_papers", {"query": "few-shot NER", "source": "arxiv", "limit": "500"})
    assert entry["ok"] is True
    notes = " ".join(json.loads(text)["policy_notes"])
    assert "converted string to integer" in notes and "capped at 50" in notes
    assert entry["effective_args"]["sources"] == ["arxiv"]


def test_real_mistakes_are_still_rejected(offline):
    import tools
    ctx = tools.SearchToolContext(enabled_sources=["arxiv"])
    _, e1 = ctx.execute("search_papers", {"query": "x"})                                 # too short: no safe repair
    _, e2 = ctx.execute("search_papers", {"query": "few-shot NER", "source": "scopus"})   # source doesn't exist
    _, e3 = ctx.execute("search_papers", {"query": "few-shot NER", "year_from": 2024, "year_to": 2020})
    assert all(e.get("contract_error") for e in (e1, e2, e3))
    _, e4 = _verifier_ctx().execute("get_paper_text", {"paper_id": "arxiv:9999.99999"})  # made-up id
    assert e4.get("unknown_id") and not e4["ok"]
    _, e5 = _verifier_ctx().execute("get_paper_text", {"paper_id": "not-an-id"})          # malformed id
    assert e5.get("contract_error")


# ---------------------------------------------------------------------------
# Quality fixes from the live app run (2026-11-19)
# ---------------------------------------------------------------------------
class _Resp:
    def __init__(self, status, headers=None, text=""):
        self.status_code, self.headers, self.text = status, headers or {}, text

    def json(self):
        return {"data": []}

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError(str(self.status_code), request=None, response=self)


@pytest.fixture
def net_env(monkeypatch):
    from sources import net
    net.reset_run()
    monkeypatch.setattr(net, "_last_call", {})
    sleeps = []
    monkeypatch.setattr(net.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(CONFIG, "source_max_retries", 2)
    monkeypatch.setattr(CONFIG, "source_retry_max_wait", 10.0)
    monkeypatch.setattr(CONFIG, "source_disable_after_failures", 2)
    yield net, sleeps
    net.reset_run()


def test_429_is_retried_honouring_retry_after(net_env, monkeypatch):
    net, sleeps = net_env
    replies = iter([_Resp(429, {"Retry-After": "3"}), _Resp(200)])
    monkeypatch.setattr(net.httpx, "get", lambda *a, **k: next(replies))
    assert net.get("openalex", "https://x").status_code == 200
    assert 3.0 in sleeps
    h = net.source_health()["openalex"]
    assert h["ok"] == 1 and h["retries"] == 1 and not h["disabled"]


def test_source_that_keeps_failing_is_switched_off_and_reported(net_env, monkeypatch):
    net, _ = net_env
    calls = []
    monkeypatch.setattr(net.httpx, "get", lambda *a, **k: calls.append(1) or _Resp(429))
    monkeypatch.setattr(CONFIG, "semantic_scholar_api_key", "")
    for _ in range(2):
        net.get("semantic_scholar", "https://x")
    h = net.source_health()["semantic_scholar"]
    assert h["disabled"] and "SEMANTIC_SCHOLAR_API_KEY" in h["last_error"]
    n = len(calls)
    with pytest.raises(net.SourceUnavailable):
        net.get("semantic_scholar", "https://x")
    assert len(calls) == n                       # no more network calls this run
    assert net.answered_sources() == []


def test_example_values_from_env_example_are_not_sent(monkeypatch):
    import config
    for v in ("you@example.com    # OpenAlex polite pool", "your-key-here", "", "none"):
        monkeypatch.setenv("X_TEST", v)
        assert config._secret("X_TEST") == ""
    monkeypatch.setenv("X_TEST", "real-key-123   # comment")
    assert config._secret("X_TEST") == "real-key-123"


def test_arxiv_query_requires_every_topic_word_and_filters_years_server_side():
    from sources import arxiv
    q = arxiv.build_query("automatic short answer grading using large language models", 2020, 2026)
    assert "all:grading AND" in q and " OR " not in q and "using" not in q
    assert "submittedDate:[202001010000 TO 202612312359]" in q
    assert arxiv.build_query("few-shot NER", None, None) == '(all:"few shot" AND all:ner)'
    relaxed = arxiv.build_query("automatic short answer grading using large language models", strict=False)
    assert "language" not in relaxed and "all:grading" in relaxed


def test_arxiv_relaxes_query_only_when_strict_finds_too_little(monkeypatch):
    from sources import arxiv
    seen = []

    def fake_fetch(search_query, limit):
        seen.append(search_query)
        return [{"id": f"arxiv:{len(seen)}.{i}", "year": 2024} for i in range(2 if len(seen) == 1 else 6)]
    monkeypatch.setattr(arxiv, "_fetch", fake_fetch)
    out = arxiv.search_papers("automatic short answer grading using large language models", 2020, 2026)
    assert len(seen) == 2 and len(out) == 8


def test_ranking_uses_the_ranking_model_with_query_and_document_formats(monkeypatch):
    kinds, n_docs = [], []

    def spy(texts, kind="document"):
        kinds.append(kind)
        n_docs.append(len(texts))
        return fake_embed(texts, kind)
    monkeypatch.setattr(embeddings, "embed_for_ranking", spy)
    papers = [{"id": "a", "title": "Few-shot NER for Thai", "abstract": "few-shot named entity recognition"},
              {"id": "b", "title": "Quantum photonics engine", "abstract": ""}]
    ranked = embeddings.rank_by_similarity("few-shot named entity recognition", papers)
    assert kinds == ["query", "document"] and ranked[0]["id"] == "a"
    assert n_docs == [1, 2]          # the title-only paper is embedded and scored too, not zeroed


def test_ranking_threshold_is_per_model_and_overridable(monkeypatch):
    monkeypatch.setattr(CONFIG, "similarity_threshold", None)
    monkeypatch.setattr(CONFIG, "ranking_embedding", "ollama:embeddinggemma:300m")
    assert embeddings.ranking_threshold() == embeddings.RANKING_THRESHOLDS["embeddinggemma"]
    monkeypatch.setattr(CONFIG, "similarity_threshold", 0.3)
    assert embeddings.ranking_threshold() == 0.3


@pytest.mark.parametrize("quote,expected", [
    ("Grade Guard grades short answers in Spanish and Hindi using the same rubric prompts.", False),   # verbatim
    ("Grade Guard supports many languages.", True),                                                     # not in text
    ("Grade Guard", True),                                                                              # too short
    (None, True),
])
def test_counterexample_needs_a_verbatim_sentence_from_that_paper(quote, expected):
    import nodes
    shown = {"arxiv:2504.01253v1": ["Intro. Grade Guard grades short answers in Spanish and Hindi using the "
                                    "same rubric prompts. Results follow."]}
    facts = {"cited_papers_relevant": True, "counterexample_found": True,
             "counterexample_paper_id": "arxiv:2504.01253v1", "counterexample_quote": quote, "overstated": False}
    assert nodes.verifier_decision(facts, shown)[0] is expected


def test_verifier_context_records_every_text_it_showed(offline):
    text, _ = _verifier_ctx().execute("get_paper_text", {"paper_id": "arxiv:0000.00001"})
    ctx = _verifier_ctx()
    ctx.execute("get_paper_text", {"paper_id": "arxiv:0000.00001"})
    assert "few-shot NER" in " ".join(ctx.shown_text["arxiv:0000.00001"])


def test_narrow_search_caps_confidence_below_high(monkeypatch):
    import nodes
    monkeypatch.setattr(CONFIG, "high_confidence_min_sources", 2)   # default is 1 for the single-source MVP
    state = {"paper_extracts": [{}] * 5,
             "source_health": {"arxiv": {"ok": 5}, "semantic_scholar": {"ok": 0, "last_error": "rate limited (429)"},
                               "openalex": {"ok": 0, "last_error": "rate limited (429)"}}}
    caps = nodes._confidence_caps(state)
    assert any("only 5 papers" in c for c in caps) and any("arxiv only" in c for c in caps)
    state = {"paper_extracts": [{}] * 8, "source_health": {"arxiv": {"ok": 3}, "openalex": {"ok": 2}}}
    assert nodes._confidence_caps(state) == []


def test_full_run_with_few_papers_never_reports_high(offline):
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    assert state["gap_candidates"]
    for g in state["gap_candidates"]:
        assert g["evidence_level"] != "Strong" and g["confidence_caps"]
        assert any("only" in r and "papers analysed" in r for r in g["evidence_level_reasons"])


def test_trend_summary_names_papers_not_ids(offline):
    import nodes
    papers = [{"id": "arxiv:2504.01253v1", "title": "Grade Guard: A Smart System for Short Answer Grading", "year": 2025},
              {"id": "arxiv:2205.15219v3", "title": "Automatic Short Math Answer Grading via In-context Meta-learning",
               "year": 2022}]
    labels = nodes.paper_labels(papers)
    assert labels["arxiv:2504.01253v1"] == "Grade Guard (2025)"
    assert nodes.replace_ids_with_labels("as arxiv:2504.01253v1 shows", labels) == "as Grade Guard (2025) shows"
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    ids = [p["id"] for p in state["selected_papers"]]
    assert not any(pid in state["trend_summary"] for pid in ids)
    synth = [p for p in offline.prompts if "Write a short synthesis" in p]
    table = synth[0].split("Here is a structured comparison table", 1)[1].split("Write a short synthesis", 1)[0]
    assert not any(pid in table for pid in ids)
    assert "Quote verified" not in table


HEADER = ("# Automatic Grading of Short Answers with LLMs\n\n"
          "**Ann Example**[1] **, Bob Sample**[2]\n\n"
          "1Department of Computer Science, Example State University\n"
          "Cambridge, MA 02139, USA\n{ann, bob}@cs.example.edu\n\n**Abstract**\nWe study grading. ") + "Body text. " * 50


@pytest.mark.parametrize("authors,mode,llm_expected", [
    (["Ann Example", "Bob Sample"], "auto", False),     # metadata + affiliation lines cover the header
    (["Ann Example"], "auto", True),                    # a name the source did not list survives -> LLM
    ([], "auto", True),                                 # no author list at all -> LLM
    (["Ann Example", "Bob Sample"], "always", True),
    ([], "never", False),
])
def test_header_llm_only_when_deterministic_detectors_cannot_vouch(monkeypatch, authors, mode, llm_expected):
    calls = []
    monkeypatch.setattr(privacy, "llm_detect", lambda text, ch: calls.append(1) or [])
    monkeypatch.setattr(CONFIG, "privacy_llm_on_papers", mode)
    privacy.reset_run()
    privacy.register_authors(authors)
    out, _ = privacy.scrub_full_text(HEADER, "arxiv:1", has_author_metadata=bool(authors),
                                     title="Automatic Grading of Short Answers with LLMs")
    assert bool(calls) is llm_expected
    assert "Example State University" not in out and "Cambridge, MA" not in out and "@cs.example.edu" not in out
    assert "Automatic Grading of Short Answers with LLMs" in out          # the title is not redacted
    privacy.reset_run()


def test_breaker_trips_on_consecutive_failures_even_after_a_success(net_env, monkeypatch):
    net, _ = net_env
    replies = iter([_Resp(200)] + [_Resp(429)] * 20)
    monkeypatch.setattr(net.httpx, "get", lambda *a, **k: next(replies))
    net.get("semantic_scholar", "https://x")                 # answered once
    for _ in range(2):
        net.get("semantic_scholar", "https://x")             # then 2 failed calls in a row
    assert net.source_health()["semantic_scholar"]["disabled"]
    assert net.answered_sources() == ["semantic_scholar"]    # still counted as having answered


def test_title_only_papers_get_slack_at_the_cut_off():
    ranked = [{"id": "a", "abstract": "x", "similarity_score": 0.48},
              {"id": "b", "abstract": "", "similarity_score": 0.48},
              {"id": "c", "abstract": "", "similarity_score": 0.40}]
    assert [p["id"] for p in embeddings.filter_by_threshold(ranked, 0.50)] == ["b"]


def test_same_paper_from_two_sources_is_kept_once():
    """Live run: the same paper came from arXiv (no DOI) and OpenAlex (with a DOI) and was
    extracted twice."""
    a = {"id": "arxiv:1", "title": "Automated Essay Scoring Using Efficient Transformer-Based Language Models",
         "doi": None, "abstract": "", "oa_pdf_url": "https://arxiv.org/pdf/1"}
    b = {"id": "openalex:W1", "title": "Automated essay scoring using efficient transformer-based language models.",
         "doi": "10.1000/xyz", "abstract": "We study AES.", "oa_pdf_url": None, "citation_count": 7}
    out = sources._dedupe([a, b])
    assert [p["id"] for p in out] == ["arxiv:1"]
    assert out[0]["abstract"] == "We study AES." and out[0]["doi"] == "10.1000/xyz" and out[0]["citation_count"] == 7


@pytest.mark.parametrize("a_authors,b_authors,a_year,b_year,expected", [
    (["Wei Wang", "Ann Example"], ["Wang Wei", "A. Example"], 2025, 2025, True),    # order + initials
    (["Wei Wang", "Ann Example"], ["Bob Other", "Cara Third"], 2025, 2025, False),  # same title, other people
    ([], ["Bob Other"], 2025, 2025, True),                                          # no list: same year -> same
    ([], ["Bob Other"], 2019, 2025, False),                                         # no list: years far apart
])
def test_same_title_needs_matching_authors(a_authors, b_authors, a_year, b_year, expected):
    a = {"id": "arxiv:1", "title": "LCES: Zero-shot Automated Essay Scoring", "doi": None,
         "authors": a_authors, "year": a_year}
    b = {"id": "openalex:W2", "title": "LCES: zero-shot automated essay scoring.", "doi": "10.1/x",
         "authors": b_authors, "year": b_year}
    assert (len(sources._dedupe([a, b])) == 1) is expected


def test_author_fingerprint_survives_privacy_scrub_and_still_dedupes():
    a = {"id": "arxiv:1", "title": "Same Title Here", "doi": None, "authors": ["Wei Wang", "Ann Example"], "year": 2025}
    b = {"id": "openalex:W2", "title": "Same title here", "doi": "10.1/x", "authors": ["Wang Wei"], "year": 2025}
    c = {"id": "openalex:W3", "title": "Same title here", "doi": "10.1/y", "authors": ["Zed Nobody"], "year": 2025}
    privacy.reset_run()
    for p in (a, b, c):
        privacy.scrub_paper(p)
        assert "authors" not in p and p["author_keys"]
        assert not any("wang" in k.lower() for k in p["author_keys"])          # hashes, not names
    out = sources._dedupe([a, b, c])
    assert [p["id"] for p in out] == ["arxiv:1", "openalex:W3"]
    privacy.reset_run()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def _saved_state(tmp_path, name, topic, gaps, papers=None):
    st = {"research_topic": topic, "gap_candidates": gaps, "selected_papers": papers or [], "paper_extracts": [{}] * 8,
          "source_health": {"arxiv": {"ok": 2}, "openalex": {"ok": 1}, "semantic_scholar": {"ok": 0}},
          "run_info": {"started_at": name, "seconds": 300, "model_strong": "qwen3:14b"}}
    (tmp_path / f"{name}_state.json").write_text(json.dumps(st), encoding="utf-8")


def _gap(claim, supported=True, category=None, ce=None, verified=None, quote=None):
    g = {"claim": claim, "supporting_paper_ids": ["arxiv:1", "arxiv:2"], "confidence_level": "High" if supported else "Medium",
         "confidence_score": 75 if supported else 50, "verifier_supported": supported,
         "verifier_facts": {"counterexample_found": bool(ce), "counterexample_paper_id": ce, "counterexample_quote": quote},
         "supporting_papers": [{"id": "arxiv:1", "title": "P1", "year": 2024, "url": "https://arxiv.org/abs/1"}]}
    if verified is not None:
        g["verifier_facts"]["counterexample_verified"] = verified
    if category:
        g["category"] = category
    return g


def test_dashboard_statuses_categories_and_refuting_paper(tmp_path):
    import dashboard
    papers = [{"id": "openalex:W9", "title": "Grade Guard", "year": 2025, "doi": "10.1/gg"}]
    _saved_state(tmp_path, "20261119_100000", "short answer grading", [
        _gap("Non-English short answer grading is under-explored", category="Language & population"),
        _gap("Human feedback loops are missing in LLM grading", supported=False, ce="openalex:W9", verified=True,
             quote="a model evaluation mechanism composed of human verifiers"),
        _gap("Old-format gap refuted without a quote", supported=False, ce="openalex:W9"),       # pre-quote run
        _gap("Claims more than the sources say about datasets", supported=False, verified=False),
        _gap("Unquoted counterexample keeps the gap but flags it", ce="openalex:W9", verified=False),
    ], papers)
    rows = dashboard.gap_rows(dashboard.load_runs(str(tmp_path)))
    by = {r["claim"][:20]: r for r in rows}
    assert by["Non-English short an"]["category_source"] == "llm"
    assert by["Human feedback loops"]["status"] == dashboard.STATUS_REFUTED
    ref = by["Human feedback loops"]["counterexample_paper"]
    assert ref["title"] == "Grade Guard" and ref["url"] == "https://doi.org/10.1/gg"
    assert by["Human feedback loops"]["counterexample_quote"].startswith("a model evaluation")
    assert by["Old-format gap refut"]["status"] == dashboard.STATUS_REFUTED_OLD
    assert by["Claims more than the"]["status"] == dashboard.STATUS_NOT_SUPPORTED
    assert by["Old-format gap refut"]["category_source"] == "rules"
    assert by["Unquoted counterexam"]["status"] == dashboard.STATUS_CONTESTED
    assert dashboard.summary(rows, dashboard.load_runs(str(tmp_path)))["refuted"] == 2


def test_dashboard_flags_gaps_that_recur_across_topics(tmp_path):
    import dashboard
    _saved_state(tmp_path, "20261119_100000", "essay scoring", [_gap("Lack of research on essay scoring in non-English languages", category="Language & population")])
    _saved_state(tmp_path, "20261119_110000", "short answer grading", [_gap("Lack of research on short answer grading in non-English languages", category="Language & population")])
    _saved_state(tmp_path, "20261119_120000", "rag hallucination", [_gap("Combining retrieval graphs with entity verification is unexplored", category="Method combination")])
    rec = dashboard.recurring_gaps(dashboard.gap_rows(dashboard.load_runs(str(tmp_path))))
    assert len(rec) == 1 and rec[0]["n_topics"] == 2 and rec[0]["theme"] == "Non-English / low-resource languages"
    assert dashboard.rule_category("Large language models lack calibration for grading") != "Language & population"


def test_dashboard_page_renders(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    _saved_state(tmp_path, "20261119_100000", "short answer grading", [
        _gap("Non-English short answer grading is under-explored", category="Language & population"),
        _gap("Human feedback loops are missing", supported=False, ce="arxiv:1", verified=True, quote="x" * 40)])
    monkeypatch.setattr(CONFIG, "output_dir", str(tmp_path))

    def page():
        import dashboard_page
        dashboard_page.render()
    at = AppTest.from_function(page).run(timeout=60)
    assert not at.exception
    assert any("Gaps refuted by a paper (1)" in h.value for h in at.subheader)
    assert [m.value for m in at.metric][:4] == ["1", "1", "8", "2"]


def test_gap_category_is_normalised_not_rejected():
    from contracts import GapItem
    assert GapItem(claim="x" * 20, supporting_paper_ids=[], rationale="r", category="language").category == "Language & population"
    assert GapItem(claim="x" * 20, supporting_paper_ids=[], rationale="r", category="whatever").category == "Other"
    assert GapItem(claim="x" * 20, supporting_paper_ids=[], rationale="r").category == "Other"


def test_rejected_key_is_reported_as_such(net_env, monkeypatch):
    net, _ = net_env
    monkeypatch.setattr(net.httpx, "get", lambda *a, **k: _Resp(403))
    net.get("semantic_scholar", "https://x")
    assert "SEMANTIC_SCHOLAR_API_KEY" in net.source_health()["semantic_scholar"]["last_error"]


def test_text_stored_by_an_earlier_run_is_rescrubbed_with_this_runs_authors():
    """Live run: 90 author names reached state.json (caught only by the outbound contract) from
    full texts reused out of the knowledge base, scrubbed back when those authors were unknown."""
    import tools

    class Store:
        def search(self, query, paper_ids=None, k=3):
            return [{"paper_id": "arxiv:0000.00001", "title": "T", "year": 2024, "section": "References",
                     "score": 0.9, "text": "[12] Ann Example and Bob Sample. Grading with LLMs. 2024."}]
    privacy.reset_run()
    privacy.register_authors(["Ann Example", "Bob Sample"])           # found by THIS run's search
    out, recs = privacy.scrub_known("As shown by Ann Example and Bob Sample (2024).", "x")
    assert "Ann Example" not in out and recs
    ctx = tools.VerifierToolContext([{"id": "arxiv:0000.00001", "title": "T", "year": 2024, "abstract": "a"}], store=Store())
    text, _ = ctx.execute("search_corpus", {"query": "grading with large language models"})
    assert "Ann Example" not in text and "Bob Sample" not in text
    privacy.reset_run()


def test_verifier_that_reads_nothing_cannot_support_a_gap(offline, monkeypatch):
    """Live run (thinking on): the verifier made no tool call but its notes said it had checked
    both papers and searched the corpus. Without a single read, the gap is unverified."""
    from tests.fakes import FakeLLM
    lazy = FakeLLM(use_tools=False)
    monkeypatch.setattr(llm_client, "_call_anthropic", lazy.call)
    monkeypatch.setattr(llm_client, "_tool_turn", lazy.tool_turn)
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    assert state["gap_candidates"]
    for g in state["gap_candidates"]:
        assert g["evidence_coverage"] == 0 and not g["verifier_supported"]
        assert "did not read any cited paper" in g["verifier_decision_reason"]
        assert g["evidence_level"] == "Limited" and g["evidence"]["evidence_coverage"] == 0


def test_thinking_can_be_switched_on_for_chosen_steps_only(monkeypatch):
    monkeypatch.setattr(CONFIG, "ollama_think", False)
    monkeypatch.setattr(CONFIG, "ollama_think_purposes", frozenset({"propose-gaps", "synthesize-trends"}))
    assert llm_client.ollama_profile("qwen3:14b")["think"] is False
    with llm_client.purpose_scope("propose-gaps-retry"):
        assert llm_client.ollama_profile("qwen3:14b")["think"] is True
        assert llm_client._reply_reserve("qwen3:14b") == 2048 + 6144
    with llm_client.purpose_scope("extract-paper-facts"):
        assert llm_client.ollama_profile("qwen3:14b")["think"] is False
    with llm_client.purpose_scope("propose-gaps"):
        assert llm_client.ollama_profile("gpt-oss:20b")["think"] == "low"      # model profile still wins


def test_hybrid_thinking_reaches_only_the_chosen_steps_in_a_full_run(offline, monkeypatch):
    """Local backend, OLLAMA_THINK=false, OLLAMA_THINK_PURPOSES=propose-gaps,synthesize-trends:
    the proposer and the trend synthesis think; extraction, search and verifier agents do not."""
    from tests.fakes import FakeLLM
    fake, seen = FakeLLM(), []

    def local_call(prompt, system, tier, json_mode=False):
        seen.append((getattr(llm_client._current, "purpose", None), llm_client.ollama_profile("qwen3:14b")["think"]))
        return fake.call(prompt, system, tier, json_mode)

    def local_turn(*a, **k):
        seen.append((getattr(llm_client._current, "purpose", None), llm_client.ollama_profile("qwen3:14b")["think"]))
        return fake.tool_turn(*a, **k)
    monkeypatch.setattr(CONFIG, "llm_backend", "local")
    monkeypatch.setattr(llm_client, "_call_ollama", local_call)
    monkeypatch.setattr(llm_client, "_tool_turn", local_turn)
    monkeypatch.setattr(CONFIG, "ollama_think_purposes", frozenset({"propose-gaps", "synthesize-trends"}))
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    thinking = {p for p, t in seen if t}
    assert thinking == {"propose-gaps", "synthesize-trends"}
    assert {"search-agent", "verifier-agent", "extract-paper-facts"} <= {p for p, t in seen if not t}
    assert state["run_info"]["think_purposes"] == ["propose-gaps", "synthesize-trends"]


# ---------------------------------------------------------------------------
# Teacher feedback: potential (not discovered) gaps, search scope, evidence levels
# ---------------------------------------------------------------------------
def test_answer_is_called_potential_research_gaps_with_a_limitation_note(offline):
    import evidence
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    report = main.render_report(state)
    assert report.startswith("# Potential Research Gaps Report:")
    assert "## Final answer: Potential Research Gaps" in report
    assert evidence.LIMITATION_NOTE_TH in report and "candidate Research Gaps" not in report
    gap_prompt = next(p for p in offline.prompts if "Potential Research Gaps" in p and "Comparison table" in p)
    assert "not a systematic literature review" in gap_prompt


def test_search_scope_is_reported_with_skipped_papers(offline, monkeypatch):
    import evidence
    import sources as src
    real_search = src.REGISTRY["arxiv"].search_papers

    def with_empty_paper(query, year_from=None, year_to=None, limit=30):
        # a paper with no abstract that still ranks (title shares every topic word) -> must be skipped
        out = real_search(query, year_from, year_to, limit)
        return out + [{"id": "arxiv:0000.00099", "source": "arxiv", "abstract": "", "year": 2024, "authors": [],
                       "title": "few-shot named entity recognition in low-resource languages", "doi": None,
                       "oa_pdf_url": None, "citation_count": 0}]
    monkeypatch.setattr(src.REGISTRY["arxiv"], "search_papers", with_empty_paper)
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 6)
    assert {"id": "arxiv:0000.00099", "title": "few-shot named entity recognition in low-resource languages",
            "reason": "no abstract or full text available"} in state["skipped_papers"]
    scope = evidence.search_scope(state)
    assert scope["enabled_sources"] == ["semantic_scholar", "arxiv", "openalex"]
    assert scope["year_from"] == 2020 and scope["year_to"] == 2026
    assert "few-shot NER low-resource" in scope["queries_searched"] and scope["queries_planned"]
    assert scope["n_analyzed"] == scope["n_full_text"] + scope["n_abstract_only"] == len(state["paper_extracts"])
    assert scope["n_candidates"] >= scope["n_passed_cutoff"] >= scope["n_selected"] >= scope["n_analyzed"]
    report = main.render_report(state)
    assert "## Search scope" in report and "arxiv:0000.00099" in report and "from abstract only" in report


def test_search_scope_names_sources_that_failed():
    import evidence
    scope = evidence.search_scope({"source_health": {"arxiv": {"ok": 3}, "semantic_scholar": {"ok": 0, "last_error": "HTTP 403"}},
                                   "enabled_sources": ["arxiv", "semantic_scholar"], "year_from": 2021})
    assert scope["sources_answered"] == ["arxiv"]
    assert scope["sources_failed"] == [{"source": "semantic_scholar", "error": "HTTP 403"}]
    text = "\n".join(evidence.search_scope_lines(scope))
    assert "semantic_scholar (HTTP 403)" in text and "2021–any" in text


def test_trace_output_carries_the_search_scope(traced):
    main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    root = next(s for s in _spans(traced) if s["name"] == "run-research-gap-agent")
    out = root["attrs"]["langfuse.observation.output"]
    assert "search_scope" in out and "queries_searched" in out and "n_full_text" in out and "skipped_papers" in out



def _ev(**kw):
    ev = {"n_supporting": 3, "n_supporting_full_text": 2, "n_supporting_quote_verified": 3, "n_contradicting": 0,
          "n_unquoted_counterexamples": 0, "year_min": 2023, "year_max": 2025, "evidence_coverage": 1.0,
          "verifier_supported": True}
    ev.update(kw)
    return ev


@pytest.mark.parametrize("kw,caps,expected", [
    ({}, [], "Strong"),
    ({}, ["only 5 papers analysed (Strong needs >= 8)"], "Moderate"),     # old ceiling kept
    ({"n_supporting": 2}, [], "Moderate"),
    ({"n_supporting_full_text": 1}, [], "Moderate"),
    ({"evidence_coverage": 0.5}, [], "Moderate"),
    ({"year_max": 2020}, [], "Moderate"),
    ({"year_min": None, "year_max": None}, [], "Moderate"),
    ({"n_contradicting": 1}, [], "Limited"),
    ({"verifier_supported": False}, [], "Limited"),
    ({"n_supporting": 1}, [], "Limited"),
    ({"evidence_coverage": 0.25}, [], "Limited"),
])
def test_evidence_level_follows_the_written_criteria(kw, caps, expected):
    import evidence
    level, reasons = evidence.evidence_level(_ev(**kw), caps, 2026, CONFIG)
    assert level == expected
    assert (reasons == []) == (expected == "Strong")


def test_every_gap_shows_its_countable_evidence(offline):
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    report = main.render_report(state)
    by_id = {p["id"]: p for p in state["selected_papers"]}
    for g in state["gap_candidates"]:
        ev = g["evidence"]
        assert g["evidence_level"] in ("Strong", "Moderate", "Limited") and g["confidence_level"] == g["evidence_level"]
        years = [by_id[pid]["year"] for pid in g["supporting_paper_ids"]]
        assert ev["n_supporting"] == len(g["supporting_paper_ids"]) and (ev["year_min"], ev["year_max"]) == (min(years), max(years))
        assert ev["n_supporting_full_text"] == 0          # fetch_full_text is off in the offline fixture
        assert isinstance(g["confidence_score"], float)     # legacy score still in state
    assert "**Evidence: " in report and "contradicting papers (verified quote)" in report
    assert "/100" not in report                              # score not shown as the main answer


def test_dashboard_recomputes_evidence_level_for_runs_saved_before_it(tmp_path):
    import dashboard
    old = _gap("An old run's gap with only a High/75 score", category="Other")
    old["evidence_coverage"] = 1.0
    papers = [{"id": "arxiv:1", "year": 2024}, {"id": "arxiv:2", "year": 2025}]
    _saved_state(tmp_path, "20261119_100000", "old topic", [old], papers)
    row = dashboard.gap_rows(dashboard.load_runs(str(tmp_path)))[0]
    assert row["evidence_source"] == "recomputed" and row["confidence_level"] == "High"
    assert row["evidence_level"] == "Moderate"           # 2 supporting papers < Strong's 3
    assert row["n_supporting"] == 2 and row["evidence_years"] == "2024–2025" and row["n_contradicting"] == 0


def test_problem_is_extracted_and_shown_in_the_comparison_table(offline):
    from contracts import PaperExtractOutput
    assert PaperExtractOutput(method="m", key_findings="k").problem == "not stated"   # old cached extracts still load
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    assert all(e["problem"] == "few-shot NER in low-resource languages" for e in state["paper_extracts"])
    header = state["comparison_table"].splitlines()[0]
    assert header.startswith("| Paper | Year | Problem | Method |")
    assert "few-shot NER in low-resource languages" in state["comparison_table"]
    extract_prompt = next(p for p in offline.prompts if "Extract structured information" in p)
    assert '"problem"' in extract_prompt
    import nodes
    assert nodes._cell(None) == "not stated" and nodes._cell("a | b\nc") == "a / b c"


def test_gaps_show_only_verified_evidence_sentences(offline, monkeypatch):
    import evidence
    import nodes
    real = nodes._quote_verified
    calls = {"n": 0}

    def first_quote_unverified(quote, text):
        calls["n"] += 1
        return False if calls["n"] == 1 else real(quote, text)
    monkeypatch.setattr(nodes, "_quote_verified", first_quote_unverified)
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    extracts = {e["paper_id"]: e for e in state["paper_extracts"]}
    seen_none = False
    for g in state["gap_candidates"]:
        assert [q["id"] for q in g["evidence_quotes"]] == g["supporting_paper_ids"]
        for q in g["evidence_quotes"]:
            ex = extracts[q["id"]]
            assert q["quote"] == (ex["supporting_quote"] if ex["quote_verified"] else None)
            seen_none |= q["quote"] is None
    report = main.render_report(state)
    assert "Evidence sentences" in report
    assert seen_none and evidence.NO_VERIFIED_QUOTE in report
    assert evidence.quote_lines([{"id": "a:1", "title": "T", "year": 2024, "quote": None}]) == [f"- T (2024): _{evidence.NO_VERIFIED_QUOTE}_"]


def test_token_budget_stops_retry_and_verifier(offline, monkeypatch):
    import nodes
    monkeypatch.setattr(CONFIG, "max_run_tokens", 0)
    monkeypatch.setattr(nodes, "_run_tokens", lambda: 10 ** 9)
    assert not nodes.token_budget_used_up()                       # 0 = no limit
    assert nodes.route_after_sufficiency({"sufficient": False, "retry_count": 0}) == "retry"
    monkeypatch.setattr(CONFIG, "max_run_tokens", 1000)
    assert nodes.route_after_sufficiency({"sufficient": False, "retry_count": 0, "selected_papers": []}) == "extract"
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    assert state["gap_candidates"]
    for g in state["gap_candidates"]:
        assert "MAX_RUN_TOKENS" in g["verifier_decision_reason"] and g["evidence_level"] == "Limited"
    assert not any(t["agent"] == "verifier_agent" for t in state["tool_trace"])


def test_api_keys_never_reach_the_run_log(caplog):
    """Live run: the UI's Full run log (and its download) showed httpx's
    'HTTP Request: GET https://api.openalex.org/works?...&api_key=...' lines in full."""
    import logging
    import queue
    import logging_utils
    q = queue.Queue()
    handler = logging_utils.QueueLogHandler(q)
    root = logging.getLogger()
    root.addHandler(handler)
    prev = root.level
    root.setLevel(logging.INFO)
    logging_utils.quiet_noisy_third_party_loggers()
    try:
        logging.getLogger("httpx").info('HTTP Request: %s %s "%s %d %s"', "GET",
                                        "https://api.openalex.org/works?search=x&api_key=SECRETKEY123&per-page=25",
                                        "HTTP/1.1", 200, "OK")
        logging.getLogger("nodes").info("calling %s", "https://x.org/?token=tok_abc&mailto=me@uni.ac.th")
        logging.getLogger("nodes").warning("key sk-ant-api03-ABCDEFGHIJKLMNOP leaked?")
    finally:
        root.removeHandler(handler)
        root.setLevel(prev)
    lines = "\n".join(q.get_nowait() for _ in range(q.qsize()))
    assert "SECRETKEY123" not in lines and "tok_abc" not in lines and "me@uni.ac.th" not in lines
    assert "sk-ant-api03" not in lines
    assert "api_key=***&per-page=25" in lines and "search=x" in lines      # the rest of the URL stays readable
    assert logging_utils.mask_secrets("https://a.org/w?api_key=***") == "https://a.org/w?api_key=***"


def test_outbound_contract_masks_api_keys():
    import outbound
    out = outbound.enforce({"error": "Client error '403' for url 'https://api.openalex.org/works?api_key=K3Y&x=1'"}, "test")
    assert "K3Y" not in out["error"] and "api_key=***" in out["error"]


# ---------------------------------------------------------------------------
# Evaluation scaffolding (the human answers themselves are never written by code)
# ---------------------------------------------------------------------------
def _eval_module(name):
    import importlib
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation"))
    return importlib.import_module(name)


def test_extraction_gold_template_is_empty_and_scoring_works(tmp_path):
    ev = _eval_module("eval_extraction")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template = open(os.path.join(root, "evaluation", "data", "extraction_gold_TEMPLATE.csv"), encoding="utf-8").read()
    assert template.strip() == "paper_id,title,problem,method,dataset,findings,annotator"   # header only, no answers
    assert ev.token_f1("prompt-based few-shot NER", "few-shot NER with prompts") > 0.5
    assert ev.token_f1("not stated", "Not stated.") == 1.0 and ev.token_f1("CoNLL", "not stated") == 0.0
    state = {"paper_extracts": [{"paper_id": "arxiv:1", "problem": "few-shot NER", "method": "prompting",
                                 "dataset": "not stated", "key_findings": "F1 improves by 4 points",
                                 "extraction_confidence": "abstract_only"}]}
    (tmp_path / "20261120_000000_state.json").write_text(json.dumps(state), encoding="utf-8")
    gold = tmp_path / "gold.csv"
    gold.write_text("paper_id,title,problem,method,dataset,findings,annotator\n"
                    "arxiv:1,T,few-shot NER,prompting,not stated,F1 improves 4 points,A\n"
                    "arxiv:1,T,few-shot NER,prompt tuning,,F1 +4,B\n"
                    "arxiv:2,T2,x,y,z,w,A\n"
                    "arxiv:3,,,,,,A\n", encoding="utf-8")                       # empty row = not filled in yet
    rows = ev.load_gold([str(gold)])
    assert len(rows) == 3
    res = ev.evaluate(rows, ev.load_extracts(str(tmp_path / "*_state.json")))
    assert res["missing_in_runs"] == ["arxiv:2"] and res["n_scored"] == 2
    assert res["fields"]["problem"]["mean_f1"] == 1.0 and res["fields"]["dataset"]["n"] == 1
    assert res["inter_annotator"]["method"]["pairs"] == 1
    assert "| problem | 2 | 1.0 |" in ev.to_markdown(res, [str(gold)])


def test_citation_check_compares_title_year_and_doi():
    ec = _eval_module("eval_citations")
    paper = {"run_id": "r", "id": "arxiv:1", "role": "supporting", "title": "Few-Shot NER: A Survey", "year": 2023,
             "doi": "https://doi.org/10.1/ABC"}
    calls = []

    def get(url, params):
        calls.append((url, dict(params)))
        if url.endswith("/doi:10.1/abc"):
            return {"id": "W1", "display_name": "Few-shot NER - a survey", "publication_year": 2024,
                    "doi": "https://doi.org/10.1/abc"}
        return None
    rec, how = ec.lookup_openalex(paper, get, api_key="K")
    assert how == "by DOI" and calls[0][1] == {"api_key": "K"}
    cmp = ec.compare(paper, rec, 0.9)
    assert cmp["title_match"] and cmp["doi_match"] and cmp["year_match"] is False and cmp["year_within_1"]
    res = ec.evaluate([paper, {**paper, "id": "arxiv:2", "doi": None, "title": ""}],
                      {"openalex": lambda p: ec.lookup_openalex(p, get)})
    s = res["sources"]["openalex"]["summary"]
    assert s["found"]["rate"] == 0.5 and s["title_match"]["rate"] == 1.0
    assert "mismatches to check" in ec.to_markdown(res)
    cr = ec._crossref_record({"title": ["<i>Few</i>-shot NER"], "issued": {"date-parts": [[2022, 5]]}, "DOI": "10.1/X"})
    assert cr == {"title": "<i>Few</i>-shot NER", "year": 2022, "doi": "10.1/x"}
    assert ec.title_similarity(cr["title"], "Few-shot NER") == 1.0


def test_query_adaptation_is_measured_from_search_rounds(offline, monkeypatch):
    qa = _eval_module("eval_query_adaptation")
    monkeypatch.setattr(CONFIG, "min_papers_before_retry", 7)      # fake source has only 6 papers -> retries
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 8)
    rounds = state["search_rounds"]
    assert len(rounds) == CONFIG.max_retry_loops and [r["round"] for r in rounds] == [1, 2, 3]
    assert all(r["planned_queries"] and "n_selected" in r and "n_candidates_total" in r for r in rounds)
    r = qa.adaptation(state)
    assert r["retry_happened"] and r["stopped_by"] == "MAX_RETRY_LOOPS" and r["n_rounds"] == 3
    assert r["papers_increased"] is False             # fakes never change with the query - by construction
    assert "Query adaptation" in qa.to_markdown([r], offline=True)


def test_precision_at_10():
    cr = _eval_module("calibrate_ranking")
    labels = [1] * 7 + [0] * 5
    scores = list(range(12, 0, -1))           # first 7 (on topic) score highest
    assert cr.precision_at(labels, scores, 10) == 0.7
    assert cr.precision_at(labels[:4], scores[:4], 10) == 1.0     # fewer than 10 candidates: use them all


def test_readme_matches_the_code():
    """README drift found in review: old observation names (node:<name>, llm:fast, tool:<name>) and a
    stale test count."""
    import re
    import tracing
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    readme = open(os.path.join(root, "README.md"), encoding="utf-8").read()
    for obs_name, obs_type in tracing.NODE_OBSERVATIONS.values():
        assert f"`{obs_name}` ({obs_type})" in readme, obs_name
    assert "`node:<name>`" not in readme and "`llm:fast`" not in readme
    with open(__file__, encoding="utf-8") as f:
        n_defs = sum(1 for line in f if line.startswith("def test_"))
    counts = {int(n) for n in re.findall(r"(\d+) offline tests", readme)}
    assert counts and all(n >= n_defs for n in counts), (counts, n_defs)   # parametrized tests count more than once


def test_extraction_that_copies_the_template_is_re_asked(offline, monkeypatch):
    """Live run (qwen3:14b): 4 of 8 extractions returned the prompt's own field descriptions."""
    from contracts import EXTRACT_FIELD_DESCRIPTIONS, PaperExtractOutput
    with pytest.raises(Exception):
        PaperExtractOutput(problem="the research problem or question the paper addresses, 1 sentence",
                           method="the approach/technique used, 1-2 sentences", key_findings="k")
    real, echoed = offline.call, {"n": 0}

    def echo_once(prompt, *a, **k):
        if "Extract structured information" in prompt and echoed["n"] == 0:
            echoed["n"] += 1
            return json.dumps({f: d + ", 1-2 sentences" for f, d in EXTRACT_FIELD_DESCRIPTIONS.items()})
        return real(prompt, *a, **k)
    monkeypatch.setattr(llm_client, "_call_anthropic", echo_once)
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    assert echoed["n"] == 1 and len(state["paper_extracts"]) == 5
    for e in state["paper_extracts"]:
        assert not any(str(e.get(f, "")).lower().startswith(d) for f, d in EXTRACT_FIELD_DESCRIPTIONS.items())
    assert state["run_metrics"].get("json_calls_needed_retry", 0) >= 1


# ---------------------------------------------------------------------------
# Teacher feedback, Nov 2026: single-source MVP, 20-30 -> 8-12 papers, budgets, user conditions,
# "too broad" rule, citation check before the report
# ---------------------------------------------------------------------------
def test_mvp_defaults(monkeypatch):
    import config
    for k in ("ENABLED_SOURCES", "MAX_CANDIDATE_PAPERS", "MAX_DEEP_ANALYSIS_PAPERS", "MAX_RUN_TOKENS",
              "MAX_RUN_SECONDS", "HIGH_CONFIDENCE_MIN_SOURCES", "MIN_CANDIDATE_PAPERS"):
        monkeypatch.delenv(k, raising=False)
    cfg = config.Config.__dataclass_fields__
    assert cfg["enabled_sources"].default == ("openalex",)
    assert config._sources("openalex, arxiv ,bogus") == ("openalex", "arxiv")
    assert config._sources("") == ("openalex",)
    assert 20 <= cfg["max_candidate_papers"].default <= 30 and cfg["min_candidate_papers"].default == 20
    assert 8 <= cfg["max_deep_analysis_papers"].default <= 12
    assert (cfg["mvp_min_papers"].default, cfg["mvp_max_papers"].default) == (8, 12)
    assert cfg["max_run_tokens"].default > 0 and cfg["max_run_seconds"].default > 0
    assert cfg["high_confidence_min_sources"].default == 1      # Strong must stay reachable with one source


def test_condition_terms_match_words_and_thai():
    import conditions
    c = conditions.normalize("", "NER, Thai|Vietnamese", "survey, review")
    assert c["include"] == [["ner"], ["thai", "vietnamese"]] and c["exclude"] == ["survey", "review"]
    ok = {"title": "Few-shot NER for Vietnamese", "abstract": "We study named entities."}
    assert conditions.check_paper(ok, c) == (True, "")
    inner = {"title": "An inner product method for Thai text", "abstract": ""}   # "ner" only inside "inner"
    assert conditions.check_paper(inner, c)[0] is False
    surv = {"title": "A survey of NER for Thai", "abstract": ""}
    keep, why = conditions.check_paper(surv, c)
    assert not keep and "survey" in why
    th = conditions.normalize("", "ภาษาไทย")
    assert conditions.check_paper({"title": "การรู้จำชื่อเฉพาะภาษาไทยแบบ few-shot"}, th)[0]
    assert conditions.is_empty(conditions.normalize()) and conditions.prompt_block({}) == ""


def test_user_conditions_reach_prompts_filter_papers_and_report(offline):
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5,
                     conditions={"focus": "low-resource Asian languages, compare with Dr. Pat Realperson",
                                 "include": "", "exclude": "crop, satellite"})
    cond = state["search_conditions"]
    assert "Pat Realperson" not in cond["focus"] and "Asian languages" in cond["focus"]   # privacy-checked
    assert cond["exclude"] == ["crop", "satellite"]
    dropped = state["excluded_by_conditions"]
    assert dropped and all("crop" in (d["title"] or "").lower() for d in dropped)
    assert all("crop" not in p["title"].lower() for p in state["selected_papers"])
    planning = [p for p in offline.prompts if "Generate 3-5 different search queries" in p]
    gap = [p for p in offline.prompts if "Propose 2-4 Potential Research Gaps" in p]
    assert planning and all("Papers must NOT mention: crop, satellite" in p for p in planning)
    assert gap and "Asian languages" in gap[0]
    report = main.render_report(state)
    assert "User conditions:" in report and "dropped by them" in report and "Pat Realperson" not in report


def test_too_broad_search_narrows_once_with_specific_queries(offline, monkeypatch):
    import nodes
    monkeypatch.setattr(CONFIG, "max_candidate_papers", 6)       # the fake corpus fills the pool
    monkeypatch.setattr(CONFIG, "broad_max_pass_rate", 1.01)     # every full pool counts as too broad
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    rounds = state["search_rounds"]
    assert [r["mode"] for r in rounds][:2] == ["initial", "narrow"]
    assert rounds[0]["verdict"] == "too_broad" and state["narrow_count"] == 1
    assert sum(1 for r in rounds if r.get("verdict") == "too_broad") == 1      # capped by MAX_NARROW_LOOPS
    narrow = [p for p in offline.prompts if "TOO BROAD" in p]
    assert len(narrow) == 1 and "Queries already tried" in narrow[0]
    assert "2: narrow" in main.render_report(state)
    # routing rules on their own
    assert nodes.route_after_sufficiency({"search_verdict": "too_broad", "narrow_count": 1}) == "narrow"
    monkeypatch.setattr(CONFIG, "max_run_seconds", 1)
    monkeypatch.setattr(nodes, "run_seconds", lambda: 99)
    assert "MAX_RUN_SECONDS" in nodes.run_budget_exceeded()
    assert nodes.route_after_sufficiency({"search_verdict": "too_broad", "selected_papers": []}) == "extract"


def test_too_few_candidates_broadens_until_the_budget_is_filled(offline, monkeypatch):
    import nodes
    monkeypatch.setattr(CONFIG, "min_candidate_papers", 20)
    monkeypatch.setattr(CONFIG, "min_papers_before_retry", 3)
    st = {"candidate_papers": [{}] * 6, "ranked_papers": [{}] * 6, "selected_papers": [{}] * 4, "max_papers": 8}
    assert nodes.check_sufficiency(st)["search_verdict"] == "too_few"          # pool < 20, budget not filled
    st["max_papers"] = 4
    assert nodes.check_sufficiency(st)["search_verdict"] == "sufficient"       # budget filled - more adds nothing


def _cite_state():
    papers = [{"id": "openalex:W1", "source": "openalex", "title": "Prompt-based few-shot NER", "year": 2024, "doi": "10.1/a"},
              {"id": "openalex:W2", "source": "openalex", "title": "Cross-lingual transfer for NER", "year": 2023, "doi": "10.1/b"},
              {"id": "arxiv:3", "source": "arxiv", "title": "Meta-learning for NER", "year": 2025, "doi": None}]
    gap = {"claim": "c", "supporting_paper_ids": [p["id"] for p in papers], "evidence_level": "Strong",
           "confidence_level": "Strong", "evidence_level_reasons": [],
           "supporting_papers": [{"id": p["id"], "title": p["title"], "year": p["year"]} for p in papers]}
    return {"selected_papers": papers, "gap_candidates": [gap]}


def _fake_get(url, params):
    if "crossref" in url and url.endswith("10.1/a"):
        return {"message": {"title": ["Prompt-based few-shot NER"], "issued": {"date-parts": [[2024]]}, "DOI": "10.1/a"}}
    if "crossref" in url and url.endswith("10.1/b"):
        return {"message": {"title": ["Cross-lingual transfer for NER"], "issued": {"date-parts": [[2019]]}, "DOI": "10.1/b"}}
    if "crossref" in url:
        return {"message": {"items": []}}
    if "openalex" in url and "search" in params:
        return {"results": []}
    return None


def test_citation_check_runs_before_the_report_and_caps_strong(monkeypatch):
    import citations
    import nodes
    monkeypatch.setattr(CONFIG, "verify_citations", True)
    monkeypatch.setattr(CONFIG, "max_run_seconds", 0)
    out = nodes.verify_citations(_cite_state(), get=_fake_get)
    checks = out["citation_checks"]
    assert checks["openalex:W1"]["status"] == citations.VERIFIED
    assert checks["openalex:W2"]["status"] == citations.MISMATCH and "year 2019" in checks["openalex:W2"]["summary"]
    assert checks["arxiv:3"]["status"] == citations.NOT_FOUND
    assert {c["service"] for c in checks["arxiv:3"]["checks"]} == {"crossref", "openalex"}   # non-OpenAlex paper: both
    assert {c["service"] for c in checks["openalex:W1"]["checks"]} == {"crossref"}          # never OpenAlex vs itself
    g = out["gap_candidates"][0]
    assert g["citation_check"]["n_ok"] == 1 and g["evidence_level"] == "Moderate"
    assert any("did not check out" in r for r in g["evidence_level_reasons"])
    state = {**_cite_state(), **out, "research_topic": "t"}
    report = main.render_report(state)
    assert "citation verified" in report and "citation mismatch" in report and "Citation check (3 cited papers" in report


def test_citation_check_survives_network_errors(monkeypatch):
    import citations
    import nodes
    monkeypatch.setattr(CONFIG, "verify_citations", True)

    def boom(url, params):
        raise ConnectionError("proxy")
    out = nodes.verify_citations(_cite_state(), get=boom)
    assert {c["status"] for c in out["citation_checks"].values()} == {citations.UNCHECKED}
    assert out["gap_candidates"][0]["evidence_level"] == "Moderate"          # unchecked is not verified
    monkeypatch.setattr(CONFIG, "verify_citations", False)
    out = nodes.verify_citations(_cite_state())
    assert all(c["summary"] == "VERIFY_CITATIONS=false" for c in out["citation_checks"].values())


def test_search_scope_reports_limits_basis_and_rounds(offline):
    import evidence
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    text = "\n".join(evidence.search_scope_lines(evidence.search_scope(state)))
    for needle in ("**Limits:**", "**Analysed from:** abstract only", "**Search rounds:**", "**User conditions:** none",
                   "not a systematic literature review"):
        assert needle in text, needle


def test_grounding_flags_invented_references_numbers_and_names(offline, tmp_path):
    eg = _eval_module("eval_grounding")
    state = main.run("few-shot named entity recognition in low-resource languages", 2020, 2026, 5)
    clean = eg.evaluate_state(state, "fake")
    assert clean["flag_counts"]["reference"] == 0
    state = dict(state)
    state["trend_summary"] = ("Most papers report gains of 4 points over fine-tuning. Smith et al. (2017) reached 97.3% "
                              "on MasakhaNER with XLM-R, see 10.9999/fake.1 for details.")
    res = eg.evaluate_state(state, "fake")
    flags = [f for r in res["rows"] for f in r["flags"]]
    kinds = {k for k, _ in flags}
    assert {"reference", "year", "number", "name"} <= kinds
    assert ("number", "4 points") not in flags                  # "4 points" is in a fake abstract ("by 4 points")
    assert any(v.startswith("DOI 10.9999") for k, v in flags) and ("number", "97.3%") in flags
    sheet = tmp_path / "audit.csv"
    n = eg.make_audit(res, state, str(sheet), n_random=2)
    assert n >= 1
    import csv
    rows = list(csv.reader(sheet.open(encoding="utf-8")))
    for i, r in enumerate(rows[1:]):
        r[-3:] = ["no", "invented", "A"] if i == 0 else ["yes", "", "A"]
    with sheet.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    score = eg.score_audit(str(sheet))
    assert score["n_labelled"] == n and score["unsupported_rate"] == round(1 / n, 3)
