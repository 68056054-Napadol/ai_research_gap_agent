"""
Query adaptation: when a narrow topic finds too few papers, does the agent actually adapt?

Runs topics chosen to be narrow (few papers expected) and measures, from each run's
`search_rounds` (one entry per search round, written by nodes 1-3):

  - retry_happened   : did the sufficiency check send the run back to keyword generation?
  - queries_changed  : did later rounds plan / search queries that earlier rounds had not used?
                       (share of new queries per round, and query-set Jaccard between rounds)
  - papers_increased : did the candidate pool and the selected set grow after a retry?
  - stopped_by       : sufficient / MAX_RETRY_LOOPS / MAX_RUN_TOKENS

    python evaluation/eval_query_adaptation.py                      # real run (LLM + sources)
    python evaluation/eval_query_adaptation.py --topics "..." "..." --max-papers 8
    python evaluation/eval_query_adaptation.py --offline            # fakes: proves the script runs

--offline swaps in tests/fakes.py: fictional papers that never change with the query, so
"papers_increased" stays False there by construction - only a real run measures adaptation.
Whether the new queries were *sensible* is a human judgment: read the query lists in the report.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import enable_offline_fakes, jaccard, save, timed_run  # noqa: E402

DEFAULT_TOPICS = [
    "large language models for automated code review of COBOL programs in Thai banks",
    "few-shot named entity recognition for Lao legal contracts",
    "automated essay scoring for Khmer-language secondary school exams",
]


def adaptation(state: dict) -> dict:
    from config import CONFIG
    rounds = state.get("search_rounds") or []
    retry = state.get("retry_count", 0) > 0 or len(rounds) > 1
    seen, per_round = set(), []
    for r in rounds:
        qs = [q for q in (r.get("planned_queries") or []) + (r.get("searched_queries") or []) if q]
        uniq = list(dict.fromkeys(q.strip().lower() for q in qs))
        new = [q for q in uniq if q not in seen]
        per_round.append({"round": r.get("round"), "planned_queries": r.get("planned_queries") or [],
                          "searched_queries": r.get("searched_queries") or [],
                          "share_new_queries": round(len(new) / len(uniq), 3) if uniq else None,
                          "n_found_this_round": r.get("n_found_this_round"), "n_candidates_total": r.get("n_candidates_total"),
                          "n_passed_cutoff": r.get("n_passed_cutoff"), "n_selected": r.get("n_selected")})
        seen.update(uniq)
    jac = [round(jaccard([q.lower() for q in a["planned_queries"] + a["searched_queries"]],
                         [q.lower() for q in b["planned_queries"] + b["searched_queries"]]), 3)
           for a, b in zip(per_round, per_round[1:])]
    first, last = (per_round[0], per_round[-1]) if per_round else ({}, {})
    m = state.get("run_metrics") or {}
    if state.get("sufficient"):
        stopped = "sufficient"
    elif m.get("token_budget_stopped_retry"):
        stopped = "MAX_RUN_TOKENS"
    elif state.get("retry_count", 0) >= CONFIG.max_retry_loops:
        stopped = "MAX_RETRY_LOOPS"
    else:
        stopped = "unknown"
    return {
        "topic": state.get("research_topic"), "n_rounds": len(rounds), "retry_happened": retry,
        "queries_changed": any((r["share_new_queries"] or 0) > 0 for r in per_round[1:]),
        "query_jaccard_between_rounds": jac,
        "papers_increased": bool(per_round[1:]) and (
            (last.get("n_candidates_total") or 0) > (first.get("n_candidates_total") or 0)
            or (last.get("n_selected") or 0) > (first.get("n_selected") or 0)),
        "candidates_first_last": [first.get("n_candidates_total"), last.get("n_candidates_total")],
        "selected_first_last": [first.get("n_selected"), last.get("n_selected")],
        "stopped_by": stopped, "rounds": per_round,
    }


def to_markdown(results: list, offline: bool) -> str:
    lines = ["# Query adaptation on narrow topics", ""]
    if offline:
        lines += ["> **--offline**: fictional fake papers that do not depend on the query. This only shows the "
                  "script runs; the numbers are not an evaluation.", ""]
    n = len(results)
    lines += [f"{n} topic(s). Retry happened in {sum(r['retry_happened'] for r in results)}/{n}; queries changed "
              f"after a retry in {sum(r['queries_changed'] for r in results)}/{n}; papers increased after a retry in "
              f"{sum(r['papers_increased'] for r in results)}/{n}.", "",
              "| topic | rounds | retry | queries changed | candidates first→last | selected first→last | stopped by |",
              "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['topic']} | {r['n_rounds']} | {r['retry_happened']} | {r['queries_changed']} "
                     f"| {r['candidates_first_last'][0]}→{r['candidates_first_last'][1]} "
                     f"| {r['selected_first_last'][0]}→{r['selected_first_last'][1]} | {r['stopped_by']} |")
    for r in results:
        lines += ["", f"## {r['topic']}", ""]
        for rd in r["rounds"]:
            lines.append(f"- round {rd['round']}: share of new queries {rd['share_new_queries']}, {rd['n_found_this_round']} "
                         f"papers found this round (pool {rd['n_candidates_total']}, passed cut-off {rd['n_passed_cutoff']}, selected {rd['n_selected']})")
            lines.append(f"  - planned: " + "; ".join(f"“{q}”" for q in rd["planned_queries"]))
            lines.append(f"  - searched: " + "; ".join(f"“{q}”" for q in rd["searched_queries"]))
    lines += ["", "Human check: were the new queries broader / adjacent wording rather than repeats or drift "
                  "off the topic? Note it in evaluation/eval_plan.md §6."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topics", nargs="+", default=DEFAULT_TOPICS)
    ap.add_argument("--max-papers", type=int, default=8)
    ap.add_argument("--year-from", type=int, default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    from config import CONFIG
    if args.offline:
        enable_offline_fakes()
        import tempfile
        CONFIG.rag_store_dir = tempfile.mkdtemp(prefix="offline_rag_")
        CONFIG.tracing_enabled = False
        # the fake source always returns the same 6 papers: asking for more than that forces retries
        CONFIG.min_papers_before_retry = 7
    results = []
    for topic in args.topics:
        print(f"Running: {topic}")
        state, secs = timed_run(topic, args.year_from, None, args.max_papers, session_id="eval-query-adaptation",
                                tags=["eval:query-adaptation"])
        if (state.get("input_decision") or {}).get("action") == "REJECT":
            print("  input rejected by the decision model - skipped")
            continue
        r = adaptation(state)
        r["seconds"] = round(secs, 1)
        results.append(r)
        print(f"  rounds={r['n_rounds']} retry={r['retry_happened']} queries_changed={r['queries_changed']} "
              f"papers_increased={r['papers_increased']} stopped_by={r['stopped_by']}")
    md = to_markdown(results, args.offline)
    print(md)
    print("Saved:", *save("eval_query_adaptation" + ("_offline" if args.offline else ""),
                          {"offline": args.offline, "results": results}, md))


if __name__ == "__main__":
    main()
