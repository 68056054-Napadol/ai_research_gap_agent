"""
Harness tuning check: ask the SAME question N times (default 5) and measure whether the substance
of the answer stays the same.

    python evaluation/consistency_test.py --topic "few-shot named entity recognition in low-resource languages" \
        --runs 5 --max-papers 6 --year-from 2021

What makes answers stable (the "tuning"): temperature=0 + fixed seed (config.py), contracts that
force the same output shape (contracts.py), deterministic ranking/threshold, PDF cache, and the
>= 2 real supporting papers rule. This script measures the result:

  gap_similarity_mean   symmetric best-match cosine between runs' gap claims (embedding model)
  selected_papers_jaccard_mean / cited_papers_jaccard_mean   overlap of the evidence used

PASS criterion used here: gap_similarity_mean >= 0.80 and selected_papers_jaccard_mean >= 0.80.
Results: evaluation/results/consistency_<timestamp>.md/.json
"""
import argparse
import logging

import common  # sets sys.path to the project root

from config import CONFIG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="few-shot named entity recognition in low-resource languages")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-papers", type=int, default=6)
    ap.add_argument("--year-from", type=int, default=2021)
    ap.add_argument("--year-to", type=int, default=None)
    ap.add_argument("--backend", choices=["cloud", "local"], default=None)
    ap.add_argument("--temperature", type=float, default=None, help="override LLM_TEMPERATURE (e.g. 1.0 to see the untuned baseline)")
    ap.add_argument("--offline", action="store_true", help="smoke-test with fictional fake data (not a real evaluation)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    if args.offline:
        common.enable_offline_fakes()
    if args.backend:
        CONFIG.llm_backend = args.backend
    if args.temperature is not None:
        CONFIG.llm_temperature = args.temperature

    import time as _t
    session = f"consistency-{_t.strftime('%Y%m%d-%H%M%S')}"
    runs = []
    for i in range(1, args.runs + 1):
        state, secs = common.timed_run(args.topic, args.year_from, args.year_to, args.max_papers,
                                       session_id=session, tags=["eval:consistency"])
        s = common.run_summary(state, secs)
        runs.append(s)
        print(f"run {i}/{args.runs}: {secs:.0f}s, {s['n_gaps']} gap(s)")
        for g in s["gaps"]:
            print(f"    - {g[:110]}")

    c = common.consistency(runs)
    passed = c.get("gap_similarity_mean", 0) >= 0.80 and c.get("selected_papers_jaccard_mean", 0) >= 0.80
    model = CONFIG.cloud_model_strong if CONFIG.llm_backend == "cloud" else CONFIG.local_model_strong
    md = [f"# Consistency test ({'OFFLINE FAKE DATA - not a real evaluation' if args.offline else 'live'})", "",
          f"- Topic: {args.topic}", f"- Backend: {CONFIG.llm_backend} ({model}), temperature={CONFIG.llm_temperature}, seed={CONFIG.llm_seed}",
          f"- Runs: {args.runs}, max papers: {args.max_papers}, year_from: {args.year_from}", "",
          "| Metric | Value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in c.items()]
    md += ["", f"**Result: {'PASS' if passed else 'FAIL'}** (criterion: gap similarity >= 0.80 and selected-paper overlap >= 0.80)", "",
           "## Gaps per run", ""]
    for i, r in enumerate(runs, 1):
        md.append(f"**Run {i}** ({r['seconds']}s)")
        md += [f"- {g}" for g in r["gaps"]] or ["- (no gaps)"]
        md.append("")
    jp, mp = common.save("consistency", {"args": vars(args), "consistency": c, "passed": passed, "runs": runs}, "\n".join(md))
    print("\n".join(md[:14]))
    print(f"\nSaved: {mp}")


if __name__ == "__main__":
    main()
