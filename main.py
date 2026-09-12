"""
CLI entrypoint. Usage:

    python main.py --topic "few-shot NER in low-resource languages" --year-from 2021 --max-papers 20
"""
import argparse
import datetime
import json
import os

from config import CONFIG
from graph import build_graph
from logging_utils import configure_console_logging


def render_report(state: dict) -> str:
    lines = [
        f"# Research Gap Report: {state['research_topic']}",
        "",
        f"Papers analyzed: {len(state.get('selected_papers', []))} "
        f"(from {len(state.get('candidate_papers', []))} candidates found)",
        "",
        "## Papers analyzed",
        "",
    ]
    for p in state.get("selected_papers", []):
        authors = ", ".join(p.get("authors") or []) or "unknown authors"
        lines.append(f"- **{p.get('title')}** ({p.get('year', '?')}) — {authors}. "
                      f"[{p.get('source')}:{p.get('id')}] doi:{p.get('doi') or 'n/a'}")

    lines += ["", "## Comparison table", "", state.get("comparison_table", "(none)"), ""]
    lines += ["## Trends across papers", "", state.get("trend_summary", "(none)"), ""]
    lines += ["## Candidate Research Gaps", ""]
    for i, gap in enumerate(state.get("gap_candidates", []), 1):
        lines += [
            f"### {i}. {gap['claim']}",
            f"**Confidence: {gap['confidence_level']}** (score: {gap['confidence_score']}/100)",
            "",
            gap.get("rationale", ""),
            "",
            f"*Supporting papers:* {', '.join(gap.get('supporting_paper_ids', []))}",
            "",
            f"*Verifier notes:* {gap.get('verifier_notes', '')}",
            "",
        ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", required=True, help="Research topic / question")
    parser.add_argument("--year-from", type=int, default=None)
    parser.add_argument("--year-to", type=int, default=None)
    parser.add_argument("--max-papers", type=int, default=CONFIG.max_deep_analysis_papers)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_console_logging(verbose=args.verbose)

    app = build_graph()
    initial_state = {
        "research_topic": args.topic,
        "year_from": args.year_from,
        "year_to": args.year_to,
        "max_papers": args.max_papers,
        "prior_queries_tried": [],
        "retry_count": 0,
    }

    print(f"Running agent for topic: {args.topic!r} ...")
    final_state = app.invoke(initial_state)

    os.makedirs(CONFIG.output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    report = render_report(final_state)
    report_path = os.path.join(CONFIG.output_dir, f"{ts}_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    state_path = os.path.join(CONFIG.output_dir, f"{ts}_state.json")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(final_state, f, indent=2, default=str, ensure_ascii=False)

    print(f"\nDone. Report: {report_path}\nFull state (for debugging/eval): {state_path}")


if __name__ == "__main__":
    main()
