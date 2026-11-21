"""
Run the fixed test set (evaluation/data/test_topics.json) once per topic, with its conditions, and save
each run as output/testset_<topic id>_<timestamp>_{report.md,state.json}. Every later evaluation step
(ranking labels, extraction gold, reference gaps, gap ratings, grounding audit, citations) works from
these saved runs, so all annotators judge the same papers. See evaluation/eval_plan.md §0.

    python evaluation/run_test_set.py                    # all topics
    python evaluation/run_test_set.py --ids T3 T4        # some topics
    python evaluation/run_test_set.py --split dev        # only dev (tuning) topics
    python evaluation/run_test_set.py --offline          # smoke test on the fake data (no network)

Settings: the script warns when a setting in .env differs from `frozen_settings` in the topic file;
report results only from runs made with the frozen settings (or say what differed).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, save  # noqa: E402

TOPICS_FILE = os.path.join(ROOT, "evaluation", "data", "test_topics.json")

_CONFIG_FIELDS = {"ENABLED_SOURCES": "enabled_sources", "MAX_CANDIDATE_PAPERS": "max_candidate_papers",
                  "MAX_DEEP_ANALYSIS_PAPERS": "max_deep_analysis_papers", "MAX_RETRY_LOOPS": "max_retry_loops",
                  "MAX_NARROW_LOOPS": "max_narrow_loops", "MAX_RUN_TOKENS": "max_run_tokens",
                  "LLM_TEMPERATURE": "llm_temperature", "FETCH_FULL_TEXT": "fetch_full_text"}


def load_topics(path: str = TOPICS_FILE) -> dict:
    return json.load(open(path, encoding="utf-8"))


def settings_diff(frozen: dict, cfg) -> list:
    diffs = []
    for env, attr in _CONFIG_FIELDS.items():
        if env not in frozen:
            continue
        want, have = frozen[env], getattr(cfg, attr)
        if attr == "enabled_sources":
            want, have = tuple(str(want).split(",")), tuple(have)
        if want != have:
            diffs.append(f"{env}: frozen {want!r}, current {have!r}")
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--split", choices=["dev", "test"], default=None)
    ap.add_argument("--offline", action="store_true", help="fake LLM + fake papers (checks the script only)")
    args = ap.parse_args()

    if args.offline:
        from common import enable_offline_fakes
        enable_offline_fakes()
    from config import CONFIG
    import main as agent

    data = load_topics()
    topics = [t for t in data["topics"] if (not args.ids or t["id"] in args.ids)
              and (not args.split or t["split"] == args.split)]
    diffs = [] if args.offline else settings_diff(data.get("frozen_settings") or {}, CONFIG)
    for d in diffs:
        print("WARNING setting differs from the frozen test settings -", d)

    rows = []
    for t in topics:
        print(f"\n=== {t['id']} ({t['split']}): {t['topic']} ===")
        state = agent.run(t["topic"], t.get("year_from"), t.get("year_to"), None,
                          entrypoint="test_set", tags=[f"testset:{t['id']}"], conditions=t.get("conditions"))
        report_path, state_path = (None, None) if args.offline else agent.save_outputs(state, prefix=f"testset_{t['id']}_")
        rounds = state.get("search_rounds") or []
        rm = state.get("run_metrics") or {}
        rows.append({"id": t["id"], "split": t["split"], "topic": t["topic"], "expected": t.get("expected_path"),
                     "path": " → ".join(f"{r.get('mode')}({r.get('verdict')})" for r in rounds),
                     "n_candidates": len(state.get("candidate_papers") or []),
                     "n_analysed": len(state.get("paper_extracts") or []),
                     "n_excluded_by_conditions": len(state.get("excluded_by_conditions") or []),
                     "n_gaps": len(state.get("gap_candidates") or []),
                     "seconds": (state.get("run_info") or {}).get("seconds"),
                     "tokens": int((rm.get("input_tokens") or 0) + (rm.get("output_tokens") or 0)),
                     "state_file": os.path.relpath(state_path, ROOT) if state_path else None})
        print("  saved:", report_path, state_path)

    md = ["# Test-set runs", "", f"Settings differing from the frozen ones: {'; '.join(diffs) or 'none'}", "",
          "| id | split | expected path | actual path | candidates | analysed | dropped by conditions | gaps | s | tokens | state file |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['id']} | {r['split']} | {r['expected']} | {r['path']} | {r['n_candidates']} | {r['n_analysed']} "
                  f"| {r['n_excluded_by_conditions']} | {r['n_gaps']} | {r['seconds']} | {r['tokens']} | {r['state_file']} |")
    md = "\n".join(md) + "\n"
    print(md)
    if not args.offline:
        print("Saved:", *save("test_set_runs", {"settings_diff": diffs, "runs": rows}, md))


if __name__ == "__main__":
    main()
