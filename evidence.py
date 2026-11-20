"""
What the user is told about the evidence behind an answer - shared by the Markdown report
(main.render_report), the web UI (app.py), the Dashboard and the Langfuse trace, so all four say
the same thing.

Teacher feedback this module answers: the system must not claim to have *discovered* a research
gap. It proposes *Potential Research Gaps* from the papers it happened to find and analyse, and
says how wide that search was.
"""

ANSWER_TITLE = "Potential Research Gaps"

# Shown under every "Potential Research Gaps" heading (report, UI, Dashboard).
LIMITATION_NOTE_TH = ("เสนอจากหลักฐานในชุด paper ที่ระบบค้นและวิเคราะห์ได้เท่านั้น "
                      "ไม่ใช่การทบทวนวรรณกรรมอย่างเป็นระบบ ผู้ใช้ควรตรวจสอบต่อ")
LIMITATION_NOTE_EN = ("Proposed only from the papers this run found and analysed - not a systematic "
                      "literature review. Check each one before relying on it.")
LIMITATION_NOTE = f"{LIMITATION_NOTE_TH} ({LIMITATION_NOTE_EN})"


# ---------------------------------------------------------------------------
# Search scope: how wide the search behind this answer was
# ---------------------------------------------------------------------------
def search_scope(state: dict) -> dict:
    """Everything a reader needs to judge how far an absence claim ("X is under-explored") can go:
    which sources were asked and which answered, every query used, the year range, and how many
    papers were found / passed the cut-off / analysed (from full text or abstract) / skipped.
    Pure function of the run state - no LLM, no network."""
    health = state.get("source_health") or {}
    enabled = list(state.get("enabled_sources") or (state.get("run_info") or {}).get("enabled_sources") or [])
    answered = sorted(s for s, h in health.items() if h.get("ok", 0) > 0)
    failed = [{"source": s, "error": h.get("last_error") or "no answer"}
              for s, h in sorted(health.items()) if not h.get("ok", 0)]
    searched = []
    for t in state.get("tool_trace") or []:
        if t.get("tool_name") == "search_papers" and t.get("ok", True):
            q = (t.get("arguments") or {}).get("query")
            if q and q not in searched:
                searched.append(q)
    extracts = state.get("paper_extracts") or []
    n_full = sum(1 for e in extracts if e.get("extraction_confidence") == "full_text")
    return {
        "enabled_sources": enabled,
        "sources_answered": answered,
        "sources_failed": failed,
        "source_health_recorded": bool(health),
        "queries_planned": list(state.get("prior_queries_tried") or []),
        "queries_searched": searched,
        "year_from": state.get("year_from"),
        "year_to": state.get("year_to"),
        "retry_loops": state.get("retry_count", 0),
        "n_candidates": len(state.get("candidate_papers") or []),
        "n_passed_cutoff": len(state.get("ranked_papers") or []),
        "n_selected": len(state.get("selected_papers") or []),
        "n_analyzed": len(extracts),
        "n_full_text": n_full,
        "n_abstract_only": len(extracts) - n_full,
        "skipped_papers": list(state.get("skipped_papers") or []),
    }


def _year_range(scope: dict) -> str:
    return f"{scope.get('year_from') or 'any'}–{scope.get('year_to') or 'any'}"


def search_scope_lines(scope: dict) -> list:
    """Markdown bullet lines (report and UI use the same text)."""
    if scope["source_health_recorded"]:
        failed = "; ".join(f"{f['source']} ({f['error']})" for f in scope["sources_failed"]) or "none"
        sources = (f"**Sources enabled:** {', '.join(scope['enabled_sources']) or '?'} · "
                   f"**answered:** {', '.join(scope['sources_answered']) or 'none'} · **failed / no answer:** {failed}")
    else:
        sources = f"**Sources enabled:** {', '.join(scope['enabled_sources']) or '?'} · answered/failed: not recorded"
    skipped = scope["skipped_papers"]
    lines = [
        f"- {sources}",
        f"- **Years:** {_year_range(scope)} · **retry loops:** {scope['retry_loops']}",
        f"- **Queries planned ({len(scope['queries_planned'])}):** "
        + ("; ".join(f"“{q}”" for q in scope["queries_planned"]) or "none"),
        f"- **Queries actually searched by the search agent ({len(scope['queries_searched'])}):** "
        + ("; ".join(f"“{q}”" for q in scope["queries_searched"]) or "none"),
        f"- **Papers:** {scope['n_candidates']} candidates found → {scope['n_passed_cutoff']} passed the relevance "
        f"cut-off → {scope['n_selected']} selected → **{scope['n_analyzed']} analysed** "
        f"({scope['n_full_text']} from full text, {scope['n_abstract_only']} from abstract only)",
        f"- **Skipped papers ({len(skipped)}):** "
        + ("; ".join(f"{s.get('title') or s.get('id')} (`{s.get('id')}`) - {s.get('reason')}" for s in skipped) or "none"),
    ]
    return lines
