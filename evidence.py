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


# ---------------------------------------------------------------------------
# Evidence level: Strong / Moderate / Limited from countable facts only
# ---------------------------------------------------------------------------
STRONG, MODERATE, LIMITED = "Strong", "Moderate", "Limited"
LEVELS = (STRONG, MODERATE, LIMITED)
LEVEL_TH = {STRONG: "หลักฐานมาก", MODERATE: "หลักฐานปานกลาง", LIMITED: "หลักฐานจำกัด"}
LEVEL_COLOR = {STRONG: "green", MODERATE: "orange", LIMITED: "red"}


def gap_evidence(supporting_ids: list, extracts_by_id: dict, papers_by_id: dict, *, verifier_supported: bool,
                 counterexample_verified: bool, counterexample_unquoted: bool, coverage: float) -> dict:
    """The countable facts behind one gap. Every number here can be checked by hand in state.json."""
    years = [papers_by_id.get(pid, {}).get("year") for pid in supporting_ids]
    years = [y for y in years if isinstance(y, int)]
    return {
        "n_supporting": len(supporting_ids),
        "n_supporting_full_text": sum(1 for pid in supporting_ids
                                      if (extracts_by_id.get(pid) or {}).get("extraction_confidence") == "full_text"),
        "n_supporting_quote_verified": sum(1 for pid in supporting_ids if (extracts_by_id.get(pid) or {}).get("quote_verified")),
        "n_contradicting": 1 if counterexample_verified else 0,     # counterexample whose quote was verified
        "n_unquoted_counterexamples": 1 if counterexample_unquoted else 0,
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
        "evidence_coverage": round(float(coverage or 0), 2),       # share of cited papers the verifier read
        "verifier_supported": bool(verifier_supported),
    }


def evidence_level(ev: dict, caps: list, current_year: int, cfg=None) -> tuple:
    """(level, reasons). Reasons say which pre-written criterion kept the gap from the next level up."""
    if cfg is None:
        from config import CONFIG as cfg
    limited = []
    if not ev["verifier_supported"]:
        limited.append("verifier did not support the claim")
    if ev["n_contradicting"]:
        limited.append(f"{ev['n_contradicting']} paper contradicts it (verified quote)")
    if ev["n_supporting"] < cfg.evidence_moderate_min_supporting:
        limited.append(f"{ev['n_supporting']} supporting paper(s), Moderate needs >= {cfg.evidence_moderate_min_supporting}")
    if ev["evidence_coverage"] < cfg.evidence_moderate_min_coverage:
        limited.append(f"verifier read {ev['evidence_coverage']:.0%} of the cited papers, "
                       f"Moderate needs >= {cfg.evidence_moderate_min_coverage:.0%}")
    if limited:
        return LIMITED, limited
    short = []
    if ev["n_supporting"] < cfg.evidence_strong_min_supporting:
        short.append(f"{ev['n_supporting']} supporting papers, Strong needs >= {cfg.evidence_strong_min_supporting}")
    if ev["n_supporting_full_text"] < cfg.evidence_strong_min_full_text:
        short.append(f"{ev['n_supporting_full_text']} supporting paper(s) read in full text, "
                     f"Strong needs >= {cfg.evidence_strong_min_full_text}")
    if ev["evidence_coverage"] < cfg.evidence_strong_min_coverage:
        short.append(f"verifier read {ev['evidence_coverage']:.0%} of the cited papers, "
                     f"Strong needs >= {cfg.evidence_strong_min_coverage:.0%}")
    if ev["year_max"] is None or current_year - ev["year_max"] > cfg.evidence_strong_max_age_years:
        short.append(f"newest supporting paper {ev['year_max'] or 'has no year'}, Strong needs one from the last "
                     f"{cfg.evidence_strong_max_age_years} years")
    short += list(caps or [])
    return (MODERATE, short) if short else (STRONG, [])


def evidence_line(ev: dict, level: str) -> str:
    """One Markdown line of the counts, shown under every gap (report, UI, Dashboard)."""
    years = (f"{ev['year_min']}–{ev['year_max']}" if ev.get("year_min") is not None else "no year data")
    return (f"**Evidence: {level}** ({LEVEL_TH.get(level, level)}) · supporting papers **{ev['n_supporting']}** "
            f"(full text {ev['n_supporting_full_text']}) · contradicting papers (verified quote) "
            f"**{ev['n_contradicting']}** · evidence years {years} · verifier read the cited papers: "
            f"{'yes' if ev['evidence_coverage'] >= 1 else 'partly' if ev['evidence_coverage'] > 0 else 'no'} "
            f"(coverage {ev['evidence_coverage']:.2f})")


def evidence_for_saved_gap(g: dict, state: dict, current_year: int, cfg=None) -> tuple:
    """(evidence, level, reasons) for a gap from a saved run. Runs saved before evidence levels only
    have the old 0-100 score and High/Medium/Low; the counts are rebuilt from what they did store
    (supporting ids, paper_extracts, paper years, evidence_coverage, verifier facts)."""
    if g.get("evidence") and g.get("evidence_level") in LEVELS:
        return g["evidence"], g["evidence_level"], g.get("evidence_level_reasons") or []
    facts = g.get("verifier_facts") or {}
    ev = gap_evidence(g.get("supporting_paper_ids") or [],
                      {e.get("paper_id"): e for e in state.get("paper_extracts") or []},
                      {p["id"]: p for p in state.get("selected_papers") or [] if p.get("id")},
                      verifier_supported=bool(g.get("verifier_supported")),
                      counterexample_verified=bool(facts.get("counterexample_verified")),
                      counterexample_unquoted=bool(facts.get("counterexample_found")) and facts.get("counterexample_verified") is False,
                      coverage=g.get("evidence_coverage") or 0)
    level, reasons = evidence_level(ev, g.get("confidence_caps") or [], current_year, cfg)
    return ev, level, reasons


# ---------------------------------------------------------------------------
# Evidence sentences: the verified quote of each supporting paper
# ---------------------------------------------------------------------------
NO_VERIFIED_QUOTE = "ไม่มีประโยคที่ยืนยันได้"


def evidence_quotes(supporting_ids: list, extracts_by_id: dict, papers_by_id: dict) -> list:
    """[{id, title, year, quote}] for every supporting paper. quote is the extraction's
    supporting_quote ONLY when quote_verified=True (found verbatim in the text the model was shown);
    otherwise None, shown as NO_VERIFIED_QUOTE - an unverified quote may be invented."""
    out = []
    for pid in supporting_ids:
        ex, p = extracts_by_id.get(pid) or {}, papers_by_id.get(pid) or {}
        quote = (ex.get("supporting_quote") or "").strip() if ex.get("quote_verified") else ""
        out.append({"id": pid, "title": p.get("title") or pid, "year": p.get("year"), "quote": quote or None})
    return out


def quotes_for_saved_gap(g: dict, state: dict) -> list:
    if "evidence_quotes" in g:
        return g["evidence_quotes"]
    return evidence_quotes(g.get("supporting_paper_ids") or [],
                           {e.get("paper_id"): e for e in state.get("paper_extracts") or []},
                           {p["id"]: p for p in state.get("selected_papers") or [] if p.get("id")})


def quote_lines(quotes: list) -> list:
    return [f"- {q['title']} ({q.get('year') or 'n.d.'}): " + (f"“{q['quote']}”" if q.get("quote") else f"_{NO_VERIFIED_QUOTE}_")
            for q in quotes]
