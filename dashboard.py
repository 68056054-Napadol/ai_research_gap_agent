"""
Dashboard data: every saved run (output/*_state.json) turned into tables the Dashboard page
(app.py) draws - gaps with their category and verdict, the paper that refuted a gap (with a link
and the verified quote), runs, papers, and gaps that keep coming back across unrelated topics.

Pure functions over the saved state files, so they are tested offline (tests/test_pipeline.py).
Nothing here calls an LLM. Older runs saved before gaps had a category get one from keyword rules
(category_source = "rules"); newer runs carry the category the gap proposer chose ("llm").
"""
import datetime
import glob
import json
import os
import re

from contracts import GAP_CATEGORIES

# Keyword rules for runs saved before the proposer assigned categories. First match wins.
_CATEGORY_RULES = [
    ("Language & population", r"non-english|multilingual|cross-lingual|low-resource language|languages?\b|"
                              r"dialect|region|population|children|primary|k-12|students? from|culture"),
    ("Efficiency & deployment", r"real-time|latency|efficien|lightweight|small(er)? model|on-device|cost|"
                                r"scalab|deploy|compute"),
    ("Reliability, fairness & explainability", r"bias|fair|robust|adversarial|explain|interpret|transparen|"
                                               r"trust|reliab|hallucinat|safety|privacy"),
    ("Evaluation & metrics", r"evaluat|metric|benchmark(ing)? method|human judg|agreement|validity"),
    ("Data & benchmarks", r"dataset|corpus|corpora|annotat|benchmark|data scarcity|labeled data"),
    ("Method combination", r"combin|integrat|hybrid|together with|joint|fusion|multimodal"),
    ("Application domain & setting", r"domain|setting|field|clinical|medical|legal|education|industry|"
                                     r"classroom|tutoring|application"),
]

STATUS_SUPPORTED = "Supported"
STATUS_REFUTED = "Refuted by a paper"
STATUS_REFUTED_OLD = "Refuted (older rule, no quote)"
STATUS_NOT_SUPPORTED = "Not supported"

_STOP = {"the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "with", "is", "are", "be", "by",
         "there", "lack", "limited", "research", "exploration", "explored", "underexplored", "remains",
         "insufficient", "use", "using", "systems", "system", "models", "model", "large", "language",
         "llm", "llms", "based", "particularly", "such", "as", "has", "not", "been", "that", "this"}


def rule_category(claim: str) -> str:
    text = (claim or "").lower()
    for category, pattern in _CATEGORY_RULES:
        if re.search(pattern, text):
            return category
    return "Other"


def _run_time(path: str, state: dict) -> str:
    info = state.get("run_info") or {}
    if info.get("started_at"):
        return info["started_at"]
    m = re.search(r"(\d{8})_(\d{6})", os.path.basename(path))
    if m:
        return datetime.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").isoformat(timespec="seconds")
    return ""


def load_runs(output_dir: str) -> list:
    """Every completed, non-rejected run, newest first: {run_id, path, time, state}."""
    runs = []
    for path in glob.glob(os.path.join(output_dir, "*_state.json")):
        try:
            state = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if "gap_candidates" not in state or (state.get("input_decision") or {}).get("action") == "REJECT":
            continue
        runs.append({"run_id": os.path.basename(path).replace("_state.json", ""), "path": path,
                     "time": _run_time(path, state), "state": state})
    return sorted(runs, key=lambda r: r["time"], reverse=True)


def gap_status(g: dict) -> str:
    facts = g.get("verifier_facts") or {}
    if facts.get("counterexample_verified"):
        return STATUS_REFUTED
    if not g.get("verifier_supported"):
        if facts.get("counterexample_found") and "counterexample_verified" not in facts:
            return STATUS_REFUTED_OLD
        return STATUS_NOT_SUPPORTED
    return STATUS_SUPPORTED


def _counterexample(g: dict, papers_by_id: dict) -> dict | None:
    if g.get("counterexample_paper"):
        return g["counterexample_paper"]
    facts = g.get("verifier_facts") or {}
    cid = facts.get("counterexample_paper_id")
    if not cid or gap_status(g) == STATUS_SUPPORTED:
        return None
    import links
    p = papers_by_id.get(cid, {"id": cid})
    return {"id": cid, "title": p.get("title") or cid, "year": p.get("year"), "url": p.get("url") or links.paper_url(p)}


def gap_rows(runs: list) -> list:
    rows = []
    for run in runs:
        st = run["state"]
        papers_by_id = {p["id"]: p for p in (st.get("selected_papers") or [])}
        info = st.get("run_info") or {}
        for i, g in enumerate(st.get("gap_candidates") or [], 1):
            category = g.get("category")
            source = "llm" if category in GAP_CATEGORIES else "rules"
            if source == "rules":
                category = rule_category(g.get("claim"))
            facts = g.get("verifier_facts") or {}
            status = gap_status(g)
            rows.append({
                "run_id": run["run_id"], "time": run["time"], "topic": st.get("research_topic", ""),
                "gap_no": i, "claim": g.get("claim", ""), "category": category, "category_source": source,
                "status": status, "confidence_level": g.get("confidence_level", ""),
                "confidence_score": g.get("confidence_score"), "confidence_caps": g.get("confidence_caps") or [],
                "verdict_reason": g.get("verifier_decision_reason", ""),
                "rationale": g.get("rationale", ""),
                "supporting_papers": g.get("supporting_papers") or [],
                "counterexample_paper": _counterexample(g, papers_by_id) if status != STATUS_SUPPORTED else None,
                "counterexample_quote": facts.get("counterexample_quote") if status in (STATUS_REFUTED, STATUS_REFUTED_OLD) else None,
                "model": info.get("model_strong", ""),
            })
    return rows


def run_rows(runs: list) -> list:
    out = []
    for run in runs:
        st = run["state"]
        info, m = st.get("run_info") or {}, st.get("run_metrics") or {}
        health = st.get("source_health") or {}
        gaps = st.get("gap_candidates") or []
        statuses = [gap_status(g) for g in gaps]
        out.append({
            "time": run["time"], "topic": st.get("research_topic", ""), "run_id": run["run_id"],
            "minutes": round(info["seconds"] / 60, 1) if info.get("seconds") else None,
            "model": info.get("model_strong", ""),
            "candidates": len(st.get("candidate_papers") or []), "analyzed": len(st.get("paper_extracts") or []),
            "sources_answered": ", ".join(sorted(s for s, h in health.items() if h.get("ok"))) or ("?" if not health else "none"),
            "sources_failed": ", ".join(sorted(s for s, h in health.items() if not h.get("ok"))),
            "gaps": len(gaps), "supported": statuses.count(STATUS_SUPPORTED),
            "refuted": statuses.count(STATUS_REFUTED) + statuses.count(STATUS_REFUTED_OLD),
            "high": sum(1 for g in gaps if g.get("confidence_level") == "High"),
            "tokens": (m.get("input_tokens") or 0) + (m.get("output_tokens") or 0),
            "redactions": sum(r.get("occurrences", 0) for r in (st.get("privacy_log") or []) if r.get("action") == "REDACT"),
            "trace_url": st.get("trace_url", ""),
        })
    return out


def paper_rows(runs: list) -> list:
    """Analysed papers across runs (one row per paper per run) + how many gaps cite each."""
    out = []
    for run in runs:
        st = run["state"]
        cited = {}
        for g in st.get("gap_candidates") or []:
            for pid in g.get("supporting_paper_ids") or []:
                cited[pid] = cited.get(pid, 0) + 1
        extracts = {e.get("paper_id"): e for e in st.get("paper_extracts") or []}
        for p in st.get("selected_papers") or []:
            ex = extracts.get(p["id"], {})
            out.append({"topic": st.get("research_topic", ""), "run_id": run["run_id"], "id": p["id"],
                        "title": p.get("title", ""), "year": p.get("year"), "source": p.get("source") or p["id"].split(":")[0],
                        "url": p.get("url"), "full_text": bool(ex.get("extraction_confidence") == "full_text"),
                        "quote_verified": ex.get("quote_verified"), "cited_by_gaps": cited.get(p["id"], 0)})
    return out


def _signature(claim: str) -> set:
    return {w for w in re.findall(r"[a-z][a-z-]+", (claim or "").lower()) if w not in _STOP and len(w) > 2}


def recurring_gaps(gap_list: list, min_topics: int = 2, threshold: float = 0.3) -> list:
    """Gaps that come back almost word for word under DIFFERENT topics - a sign the model is
    proposing a stock gap ("non-English settings are under-explored") rather than one these
    papers motivate. Groups by word overlap (Jaccard) within the same category, after removing the
    run's own topic words from each claim - what is left is the "stock" part of the gap."""
    groups = []
    for row in gap_list:
        sig = _signature(row["claim"]) - _signature(row["topic"])
        for grp in groups:
            if grp["category"] == row["category"] and _jaccard(sig, grp["sig"]) >= threshold:
                grp["rows"].append(row)
                grp["sig"] |= sig
                break
        else:
            groups.append({"category": row["category"], "sig": set(sig), "rows": [row]})
    out = []
    for grp in groups:
        topics = sorted({r["topic"] for r in grp["rows"]})
        if len(topics) >= min_topics:
            out.append({"category": grp["category"], "example": grp["rows"][0]["claim"],
                        "n_gaps": len(grp["rows"]), "n_topics": len(topics), "topics": topics})
    return sorted(out, key=lambda g: (-g["n_topics"], -g["n_gaps"]))


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def summary(gap_list: list, runs: list) -> dict:
    statuses = [g["status"] for g in gap_list]
    scores = [g["confidence_score"] for g in gap_list if isinstance(g["confidence_score"], (int, float))]
    return {
        "runs": len(runs), "topics": len({r["state"].get("research_topic") for r in runs}),
        "papers_analyzed": sum(len(r["state"].get("paper_extracts") or []) for r in runs),
        "gaps": len(gap_list), "supported": statuses.count(STATUS_SUPPORTED),
        "refuted": statuses.count(STATUS_REFUTED) + statuses.count(STATUS_REFUTED_OLD),
        "not_supported": statuses.count(STATUS_NOT_SUPPORTED),
        "high": sum(1 for g in gap_list if g["confidence_level"] == "High"),
        "mean_confidence": round(sum(scores) / len(scores), 1) if scores else None,
    }


def count_by(rows: list, key: str) -> dict:
    out = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
