"""
Open-source vs commercial model: run the same topics on the cloud backend (Claude) and the local
backend (an open-source model via Ollama, default qwen3) and compare them on the same measurable
dimensions.

    python evaluation/compare_models.py --max-papers 5 --runs 3
    python evaluation/compare_models.py --local-fast qwen3:4b --local-strong qwen3:8b

Dimensions (all computed by the pipeline itself, no manual rating needed):
  correctness / grounding  quote_verified_rate   extraction quotes found verbatim in the source
                           avg_evidence_coverage verifier actually read the cited sources via tool
                           gap_ids_hallucinated  cited paper ids that don't exist
  reliability              contract_first_try_rate, tool_args_valid_rate, search_used_fallback
  consistency (runs >= 2)  gap_similarity_mean over repeated runs of the same topic
  cost                     seconds, tokens

A dimension counts as "open-source matches commercial" when the local value is at least 95% of
the cloud value (or within 0.05 absolute for rates). Results: evaluation/results/compare_models_*.md
"""
import argparse
import logging

import common  # sets sys.path to the project root
import numpy as np

from config import CONFIG  # noqa: E402

import time as _time
SESSION_TS = _time.strftime("%Y%m%d-%H%M%S")

DEFAULT_TOPICS = [
    "few-shot named entity recognition in low-resource languages",
    "retrieval-augmented generation hallucination detection",
    "graph neural networks for traffic forecasting",
]
# (metric, higher_is_better)
DIMENSIONS = [
    ("quote_verified_rate", True),
    ("avg_evidence_coverage", True),
    ("contract_first_try_rate", True),
    ("tool_args_valid_rate", True),
    ("gap_similarity_mean", True),
    ("gap_ids_hallucinated", False),
    ("n_gaps_supported", True),
    ("seconds", False),
]


def _mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(float(np.mean(vals)), 3) if vals else None


def evaluate_backend(backend: str, topics: list, runs: int, args, fast: str = None, strong: str = None,
                     label: str = None) -> dict:
    CONFIG.llm_backend = backend
    if backend == "local" and fast:
        CONFIG.local_model_fast, CONFIG.local_model_strong = fast, strong or fast
    per_topic = {}
    failures = []
    for topic in topics:
        rs = []
        for i in range(runs):
            try:
                state, secs = common.timed_run(topic, args.year_from, None, args.max_papers,
                                               session_id=f"compare-{label or backend}-{SESSION_TS}",
                                               tags=["eval:compare-models", f"config:{label or backend}"])
            except Exception as e:   # a model that crashes the pipeline is a result, not a reason to lose the others
                print(f"[{label or backend}] {topic[:50]!r} run {i + 1}/{runs}: FAILED {type(e).__name__}: {str(e)[:200]}")
                failures.append(f"{topic[:40]}: {type(e).__name__}")
                continue
            rs.append(common.run_summary(state, secs))
            print(f"[{label or backend}] {topic[:50]!r} run {i + 1}/{runs}: {secs:.0f}s, {rs[-1]['n_gaps']} gaps")
        per_topic[topic] = {"runs": rs, "consistency": common.consistency(rs)}
    all_runs = [r for t in per_topic.values() for r in t["runs"]]
    agg = {k: _mean([r.get(k) for r in all_runs]) for k, _ in DIMENSIONS if k != "gap_similarity_mean"}
    agg["gap_similarity_mean"] = _mean([t["consistency"].get("gap_similarity_mean") for t in per_topic.values()])
    agg["search_used_fallback_runs"] = sum(1 for r in all_runs if r["search_used_fallback"])
    agg["json_empty_responses"] = int(sum(r.get("json_empty_responses", 0) for r in all_runs))
    agg["total_tokens"] = int(sum(r["input_tokens"] + r["output_tokens"] for r in all_runs))
    agg["failed_runs"] = len(failures)
    agg["completed_runs"] = len(all_runs)
    return {"per_topic": per_topic, "aggregate": agg, "failures": failures}


def matches(local, cloud, higher_is_better: bool):
    if local is None or cloud is None:
        return None
    if higher_is_better:
        return local >= cloud * 0.95 or local >= cloud - 0.05
    return local <= cloud * 1.05 or local <= cloud + 0.05


def parse_config(spec: str) -> tuple:
    """'label=backend|fast_model|strong_model'  e.g.  'qwen14=local|qwen3:14b|qwen3:14b'"""
    label, _, rest = spec.partition("=")
    parts = rest.split("|")
    return label, parts[0], (parts[1] if len(parts) > 1 else None), (parts[2] if len(parts) > 2 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="*", default=DEFAULT_TOPICS)
    ap.add_argument("--runs", type=int, default=2, help="runs per topic per config (>=2 to measure consistency)")
    ap.add_argument("--max-papers", type=int, default=5)
    ap.add_argument("--year-from", type=int, default=2021)
    ap.add_argument("--configs", nargs="*", default=None,
                    help="compare any set of models: label=backend|fast|strong, e.g. "
                         "'qwen14=local|qwen3:14b|qwen3:14b' 'gptoss=local|gpt-oss:20b|gpt-oss:20b' 'claude=cloud'")
    ap.add_argument("--local-fast", default=None)
    ap.add_argument("--local-strong", default=None)
    ap.add_argument("--only", choices=["cloud", "local"], default=None, help="run just one backend")
    ap.add_argument("--offline", action="store_true", help="smoke-test with fictional fake data (not a real evaluation)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    if args.offline:
        common.enable_offline_fakes()
    if args.configs:
        configs = [parse_config(c) for c in args.configs]
    else:
        configs = [("cloud", "cloud", None, None),
                   ("local", "local", args.local_fast or CONFIG.local_model_fast, args.local_strong or CONFIG.local_model_strong)]
        if args.only:
            configs = [c for c in configs if c[0] == args.only]

    results, labels = {}, []
    for label, backend, fast, strong in configs:
        results[label] = evaluate_backend(backend, args.topics, args.runs, args, fast, strong, label)
        results[label]["model"] = (f"{CONFIG.cloud_model_fast} / {CONFIG.cloud_model_strong}" if backend == "cloud"
                                   else f"{CONFIG.local_model_fast} / {CONFIG.local_model_strong}")
        labels.append(label)
        common.save("compare_models_partial", {"args": vars(args), "results": results}, f"partial after {label}")

    md = [f"# Model comparison ({'OFFLINE FAKE DATA - not a real evaluation' if args.offline else 'live'})", ""]
    md += [f"- {l}: {results[l]['model']}" for l in labels]
    md += [f"- temperature={CONFIG.llm_temperature}, topics={len(args.topics)}, runs/topic={args.runs}, max papers={args.max_papers}", "",
           "| Dimension | better | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    for k, hib in [("completed_runs", True), ("failed_runs", False)] + DIMENSIONS + [("search_used_fallback_runs", False), ("json_empty_responses", False), ("total_tokens", False)]:
        vals = [results[l]["aggregate"].get(k) for l in labels]
        md.append(f"| {k} | {'higher' if hib else 'lower'} | " + " | ".join(str(v) for v in vals) + " |")
    matched = []
    if "cloud" in results and "local" in results:
        for k, hib in DIMENSIONS:
            if matches(results["local"]["aggregate"].get(k), results["cloud"]["aggregate"].get(k), hib):
                matched.append(k)
        md += ["", f"**Dimensions where the open-source model is as good as the commercial one:** "
                   f"{', '.join(matched) if matched else 'none'}"]
    md.append("")
    jp, mp = common.save("compare_models", {"args": vars(args), "results": results, "matched": matched}, "\n".join(md))
    print("\n".join(md))
    print(f"\nSaved: {mp}")


if __name__ == "__main__":
    main()
