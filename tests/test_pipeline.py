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
    monkeypatch.setattr(CONFIG, "privacy_llm_on_papers", True)
    monkeypatch.setattr(CONFIG, "tracing_enabled", True)
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
    ({"cited_papers_relevant": True, "counterexample_found": True, "counterexample_paper_id": "arxiv:1", "overstated": False}, {"arxiv:1"}, False),
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
