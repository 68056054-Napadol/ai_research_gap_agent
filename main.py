"""
CLI entrypoint. Usage:

    python main.py --topic "few-shot NER in low-resource languages" --year-from 2021 --max-papers 20
"""
import argparse
import datetime
import json
import os

import links
import metrics
import outbound
import tracing
from config import CONFIG
from graph import build_graph
from logging_utils import configure_console_logging


def source_health_md(health: dict) -> str:
    parts = []
    for src, h in sorted(health.items()):
        parts.append(f"{src}: answered" + (f" ({h['retries']} retries)" if h.get("retries") else "")
                     if h.get("ok") else f"{src}: no results - {h.get('last_error') or 'no answer'}")
    return "Sources: " + "; ".join(parts)


def render_report(state: dict) -> str:
    decision = state.get("input_decision") or {}
    lines = [f"# Research Gap Report: {state['research_topic']}", ""]
    if decision.get("action") == "REJECT":
        return "\n".join(lines + [f"**Input rejected by the decision model (rule {decision.get('rule')}):** "
                                   f"{decision.get('message')}", ""])
    if decision.get("action") == "PROCEED_REDACTED":
        lines += [f"> Privacy: {decision.get('message')} (decision rule {decision.get('rule')})", ""]
    lines += [
        f"Papers analyzed: {len(state.get('selected_papers', []))} "
        f"(from {len(state.get('candidate_papers', []))} candidates found)",
        "",
        *([source_health_md(state["source_health"]), ""] if state.get("source_health") else []),
        "## Papers analyzed",
        "",
    ]
    for p in state.get("selected_papers", []):
        # author names are deliberately not shown (privacy.scrub_paper) - only the count
        title = links.markdown_link({"title": p.get("title"), "year": p.get("year"), "url": p.get("url") or links.paper_url(p)})
        lines.append(f"- **{title}** — {p.get('n_authors', '?')} author(s). `{p.get('id')}`")

    if state.get("trace_url"):
        lines += ["", f"Langfuse trace: {state['trace_url']}"]
    lines += ["", "## Comparison table", "", state.get("comparison_table", "(none)"), ""]
    lines += ["## Trends across papers", "", state.get("trend_summary", "(none)"), ""]
    lines += ["## Final answer: candidate Research Gaps", ""]
    for i, gap in enumerate(state.get("gap_candidates", []), 1):
        lines += [
            f"### {i}. {gap['claim']}",
            f"**Confidence: {gap['confidence_level']}** (score: {gap['confidence_score']}/100) · "
            f"verifier read {len(gap.get('evidence_read_ids', []))}/{len(gap.get('supporting_paper_ids', []))} "
            f"cited sources via tools · verdict: {gap.get('verifier_decision_reason', '')}",
            *([f"*Capped below High:* {'; '.join(gap['confidence_caps'])}"] if gap.get("confidence_caps") else []),
            "",
            gap.get("rationale", ""),
            "",
            "*Supporting papers:*",
            *[f"- {links.markdown_link(sp)} `{sp['id']}`" for sp in gap.get("supporting_papers", [])],
            "",
            f"*Verifier notes:* {gap.get('verifier_notes', '')}",
            "",
        ]
        if gap.get("dropped_ids"):
            lines += [f"*Dropped (not among analyzed papers):* {', '.join(gap['dropped_ids'])}", ""]

    trace = state.get("tool_trace") or []
    lines += ["## Tool calls (context returned to the agents)", "",
              "| # | Agent | Tool | Arguments | Result |", "|---|---|---|---|---|"]
    for i, t in enumerate(trace, 1):
        res = (f"{t.get('n_items', '?')} item(s), {t.get('result_chars', 0)} chars" if t.get("ok", True)
               else f"ERROR: {t.get('error', '')}")
        args = json.dumps(t.get("arguments", {}), ensure_ascii=False).replace("|", "\\|")
        lines.append(f"| {i} | {t.get('agent')} | `{t.get('tool_name')}` | `{args}` | {res} |")

    plog = state.get("privacy_log") or []
    redacted = [r for r in plog if r["action"] == "REDACT"]
    lines += ["", "## Privacy decisions (decision model)", "",
              f"{len(plog)} entity decision(s), {len(redacted)} REDACT, {len(plog) - len(redacted)} KEEP. "
              f"Redactions in the final answer: {state.get('final_answer_redactions', 0)}.", ""]
    by_rule: dict = {}
    for r in plog:
        by_rule[(r["rule"], r["action"])] = by_rule.get((r["rule"], r["action"]), 0) + 1
    for (rule, action), n in sorted(by_rule.items()):
        lines.append(f"- rule {rule} -> {action}: {n}")
    return "\n".join(lines)


def run_info(started: float, max_papers: int, entrypoint: str) -> dict:
    """What the Dashboard needs to compare runs: when, how long, with which models and sources."""
    import time
    local = CONFIG.llm_backend == "local"
    return {
        "started_at": datetime.datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "seconds": round(time.time() - started, 1),
        "entrypoint": entrypoint,
        "backend": CONFIG.llm_backend,
        "model_fast": CONFIG.local_model_fast if local else CONFIG.cloud_model_fast,
        "model_strong": CONFIG.local_model_strong if local else CONFIG.cloud_model_strong,
        "max_papers": max_papers,
        "enabled_sources": list(CONFIG.enabled_sources),
        "ranking_embedding": CONFIG.ranking_embedding,
        "think": bool(CONFIG.ollama_think) if local else None,
        "think_purposes": sorted(CONFIG.ollama_think_purposes) if local else [],
    }


def run(topic: str, year_from=None, year_to=None, max_papers=None, session_id: str = None,
        entrypoint: str = "cli", tags=None) -> dict:
    """Run the whole graph once and return the final state (used by main() and the evaluation
    scripts in evaluation/)."""
    import time
    started = time.time()
    metrics.reset()
    app = build_graph()
    initial_state = {
        "research_topic": topic,
        "year_from": year_from,
        "year_to": year_to,
        "max_papers": max_papers or CONFIG.max_deep_analysis_papers,
        "prior_queries_tried": [],
        "retry_count": 0,
    }
    tracing.init()
    with tracing.run_trace(entrypoint, session_id=session_id, extra_tags=tags) as trace:
        final_state = app.invoke(initial_state, {"recursion_limit": 60})
        final_state.setdefault("run_metrics", metrics.snapshot())
        final_state["run_info"] = run_info(started, initial_state["max_papers"], entrypoint)
        trace.finish(final_state)
    if trace.trace_id:
        final_state["trace_id"] = trace.trace_id
        if trace.trace_url:
            final_state["trace_url"] = trace.trace_url
    tracing.flush()
    return final_state


def save_outputs(final_state: dict, prefix: str = "") -> tuple:
    os.makedirs(CONFIG.output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_state = outbound.enforce(final_state, "state.json")       # outbound contract: no PII leaves
    report_path = os.path.join(CONFIG.output_dir, f"{prefix}{ts}_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(outbound.enforce(render_report(final_state), "report"))
    state_path = os.path.join(CONFIG.output_dir, f"{prefix}{ts}_state.json")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(final_state, f, indent=2, default=str, ensure_ascii=False)
    return report_path, state_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", required=True, help="Research topic / question")
    parser.add_argument("--year-from", type=int, default=None)
    parser.add_argument("--year-to", type=int, default=None)
    parser.add_argument("--max-papers", type=int, default=CONFIG.max_deep_analysis_papers)
    parser.add_argument("--backend", choices=["cloud", "local"], default=None, help="override LLM_BACKEND")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_console_logging(verbose=args.verbose)
    if args.backend:
        CONFIG.llm_backend = args.backend

    print(f"Running agent for topic: {args.topic!r} ...")
    final_state = run(args.topic, args.year_from, args.year_to, args.max_papers)
    decision = final_state.get("input_decision") or {}
    if decision.get("action") == "REJECT":
        print(f"\nInput rejected (decision rule {decision.get('rule')}): {decision.get('message')}")
    report_path, state_path = save_outputs(final_state)
    print(f"\nDone. Report: {report_path}\nFull state (for debugging/eval): {state_path}")
    if final_state.get("trace_url"):
        print(f"Langfuse trace: {final_state['trace_url']}")


if __name__ == "__main__":
    main()
