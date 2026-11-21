"""Shared helpers for the evaluation scripts in this folder."""
import itertools
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def enable_offline_fakes() -> None:
    """--offline: swap in tests/fakes.py (fictional papers, scripted LLM). Only proves the script
    runs end to end - the numbers it produces are NOT an evaluation of any real model."""
    import embeddings
    import llm_client
    import sources
    from config import CONFIG
    from tests.fakes import FakeLLM, FakeSource, fake_embed

    fake = FakeLLM(hallucinate_id=False, break_contract_once=False)
    llm_client._call_anthropic = fake.call
    llm_client._call_ollama = fake.call
    llm_client._tool_turn = fake.tool_turn
    for n in ("semantic_scholar", "arxiv", "openalex"):
        sources.REGISTRY[n] = FakeSource(n)
    embeddings.embed = fake_embed
    embeddings.embed_for_rag = fake_embed
    embeddings.embed_for_ranking = fake_embed
    CONFIG.fetch_full_text = False
    CONFIG.similarity_threshold = 0.1
    CONFIG.min_papers_before_retry = 3
    CONFIG.min_candidate_papers = 1            # the fake corpus has 6 papers
    CONFIG.enabled_sources = ("semantic_scholar", "arxiv", "openalex")   # the fakes script multi-source calls
    CONFIG.verify_citations = False            # no network in offline mode


def timed_run(topic: str, year_from=None, year_to=None, max_papers=None, session_id=None, tags=None) -> tuple:
    """session_id groups every run of one evaluation into one Langfuse session."""
    import main
    start = time.monotonic()
    state = main.run(topic, year_from, year_to, max_papers, session_id=session_id, entrypoint="evaluation", tags=tags)
    return state, time.monotonic() - start


def run_summary(state: dict, seconds: float) -> dict:
    """The per-run numbers both scripts report."""
    extracts = state.get("paper_extracts") or []
    gaps = state.get("gap_candidates") or []
    trace = state.get("tool_trace") or []
    m = state.get("run_metrics") or {}
    return {
        "seconds": round(seconds, 1),
        "input_action": (state.get("input_decision") or {}).get("action"),
        "n_candidates": len(state.get("candidate_papers") or []),
        "selected_ids": [p["id"] for p in state.get("selected_papers") or []],
        "n_extracted": len(extracts),
        "quote_verified_rate": round(sum(1 for e in extracts if e.get("quote_verified")) / len(extracts), 3) if extracts else None,
        "search_tool_calls_by_llm": sum(1 for t in trace if t.get("agent") == "search_agent"),
        "search_used_fallback": bool(state.get("search_used_fallback")),
        "verifier_tool_calls": sum(1 for t in trace if t.get("agent") == "verifier_agent"),
        "gaps": [g["claim"] for g in gaps],
        "gap_supporting_ids": [g.get("supporting_paper_ids", []) for g in gaps],
        "n_gaps": len(gaps),
        "n_gaps_supported": sum(1 for g in gaps if g.get("verifier_supported")),
        "avg_evidence_coverage": round(float(np.mean([g.get("evidence_coverage", 0) for g in gaps])), 3) if gaps else None,
        "gap_ids_hallucinated": m.get("gap_ids_hallucinated", 0),
        "contract_first_try_rate": m.get("contract_first_try_rate"),
        "contract_final_pass_rate": m.get("contract_final_pass_rate"),
        "tool_args_valid_rate": m.get("tool_args_valid_rate"),
        "tool_args_repaired": m.get("tool_args_repaired", 0),
        "llm_calls": m.get("llm_calls", 0),
        "json_empty_responses": m.get("json_empty_response", 0),
        "input_tokens": m.get("input_tokens", 0),
        "output_tokens": m.get("output_tokens", 0),
        "final_answer_redactions": state.get("final_answer_redactions", 0),
    }


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def gap_set_similarity(claims_a: list, claims_b: list) -> float:
    """Symmetric best-match cosine similarity between two runs' gap claims (embedding model):
    for each claim, how close is the most similar claim from the other run? Averaged both ways.
    1.0 = same ideas, ~0.5 = unrelated text."""
    import embeddings
    if not claims_a or not claims_b:
        return 1.0 if not claims_a and not claims_b else 0.0
    va, vb = embeddings.embed(claims_a), embeddings.embed(claims_b)
    sims = va @ vb.T
    return float((sims.max(axis=1).mean() + sims.max(axis=0).mean()) / 2)


def consistency(runs: list) -> dict:
    pairs = list(itertools.combinations(range(len(runs)), 2))
    if not pairs:
        return {}
    gap_sims = [gap_set_similarity(runs[i]["gaps"], runs[j]["gaps"]) for i, j in pairs]
    paper_j = [jaccard(runs[i]["selected_ids"], runs[j]["selected_ids"]) for i, j in pairs]
    cited_j = [jaccard(sum(runs[i]["gap_supporting_ids"], []), sum(runs[j]["gap_supporting_ids"], [])) for i, j in pairs]
    return {
        "n_runs": len(runs),
        "gap_similarity_mean": round(float(np.mean(gap_sims)), 3),
        "gap_similarity_min": round(float(np.min(gap_sims)), 3),
        "selected_papers_jaccard_mean": round(float(np.mean(paper_j)), 3),
        "cited_papers_jaccard_mean": round(float(np.mean(cited_j)), 3),
        "n_gaps_per_run": [r["n_gaps"] for r in runs],
    }


def save(name: str, payload: dict, markdown: str) -> tuple:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    jp = os.path.join(RESULTS_DIR, f"{name}_{ts}.json")
    mp = os.path.join(RESULTS_DIR, f"{name}_{ts}.md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(markdown)
    return jp, mp
