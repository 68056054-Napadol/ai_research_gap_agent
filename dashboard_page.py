"""
Dashboard page of the Streamlit app (app.py -> sidebar "Page: Dashboard").
Data comes from dashboard.py (every saved run in output/). Only st.* drawing lives here.
"""
import pandas as pd
import streamlit as st

import dashboard as dash
import evidence
from config import CONFIG

STATUS_ORDER = [dash.STATUS_SUPPORTED, dash.STATUS_CONTESTED, dash.STATUS_NOT_SUPPORTED, dash.STATUS_REFUTED,
                dash.STATUS_REFUTED_OLD]
STATUS_COLOR = {dash.STATUS_SUPPORTED: "#2e9e5b", dash.STATUS_CONTESTED: "#e6c229", dash.STATUS_NOT_SUPPORTED: "#9aa0a6",
                dash.STATUS_REFUTED: "#d64545", dash.STATUS_REFUTED_OLD: "#f0a04b"}
STATUS_ICON = {dash.STATUS_SUPPORTED: "✅", dash.STATUS_CONTESTED: "⚠️", dash.STATUS_NOT_SUPPORTED: "⚪",
               dash.STATUS_REFUTED: "❌", dash.STATUS_REFUTED_OLD: "❔"}


def _link(p: dict | None) -> str:
    if not p:
        return ""
    label = f"{p.get('title') or p.get('id')} ({p.get('year') or 'n.d.'})"
    return f"[{label}]({p['url']})" if p.get("url") else label


def _by_status(df: pd.DataFrame, index: str) -> pd.DataFrame:
    """Counts per index value and verdict, verdict columns in a fixed order (and fixed colours)."""
    pivot = df.pivot_table(index=index, columns="status", values="claim", aggfunc="count", fill_value=0)
    return pivot[[c for c in STATUS_ORDER if c in pivot.columns]]


def render():
    st.title(f"📊 Dashboard — {evidence.ANSWER_TITLE} across runs")
    st.caption(evidence.LIMITATION_NOTE)
    runs = dash.load_runs(CONFIG.output_dir)
    if not runs:
        st.info(f"No saved runs in `{CONFIG.output_dir}/` yet - run the agent first.")
        return
    gaps = dash.gap_rows(runs)

    # ---- filters ----
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
        topics = sorted({g["topic"] for g in gaps})
        f_topic = c1.multiselect("Topic", topics, placeholder="All topics")
        f_cat = c2.multiselect("Category", sorted({g["category"] for g in gaps}), placeholder="All categories")
        f_status = c3.multiselect("Verdict", [s for s in STATUS_ORDER if any(g["status"] == s for g in gaps)],
                                  placeholder="All verdicts")
        f_conf = c4.multiselect("Evidence", list(evidence.LEVELS), placeholder="All")
    sel = [g for g in gaps if (not f_topic or g["topic"] in f_topic) and (not f_cat or g["category"] in f_cat)
           and (not f_status or g["status"] in f_status) and (not f_conf or g["evidence_level"] in f_conf)]
    sel_runs = [r for r in runs if not f_topic or r["state"].get("research_topic") in f_topic]

    # ---- KPIs ----
    s = dash.summary(sel, sel_runs)
    k = st.columns(8)
    for col, (label, value) in zip(k, [("Runs", s["runs"]), ("Topics", s["topics"]), ("Papers analysed", s["papers_analyzed"]),
                                       ("Gaps", s["gaps"]), ("✅ Supported", s["supported"]), ("❌ Refuted", s["refuted"]),
                                       ("Strong evidence", s["strong"]), ("Limited evidence", s["limited"])]):
        col.metric(label, value)

    if not sel:
        st.warning("No gaps match these filters.")
        return
    df = pd.DataFrame(sel)

    # ---- charts ----
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Gaps by category")
        pivot = _by_status(df, "category")
        st.bar_chart(pivot, horizontal=True, height=320, color=[STATUS_COLOR[c] for c in pivot.columns])
    with c2:
        st.subheader("Evidence level by verdict")
        pivot = _by_status(df, "evidence_level")
        pivot = pivot.reindex([x for x in evidence.LEVELS if x in pivot.index])
        st.bar_chart(pivot, height=320, color=[STATUS_COLOR[c] for c in pivot.columns])

    # ---- all gaps ----
    st.subheader(f"All potential research gaps ({len(sel)})")
    table = df.assign(verdict=df["status"].map(lambda x: f"{STATUS_ICON.get(x, '')} {x}"),
                      papers=df["supporting_papers"].map(len),
                      refuted_by=df["counterexample_paper"].map(lambda p: (p or {}).get("title", "")))
    st.caption("Evidence level comes from countable facts only (README \"Evidence levels\"): supporting papers, "
               "how many were read in full text, contradicting papers with a verified quote, evidence years, and "
               "whether the verifier read the cited papers. Runs saved before this had only a 0-100 score; their "
               "level is recomputed from what they stored.")
    st.dataframe(table[["time", "topic", "category", "claim", "verdict", "evidence_level", "n_supporting",
                        "n_full_text", "n_contradicting", "evidence_years", "evidence_coverage", "refuted_by"]],
                 hide_index=True, width="stretch",
                 column_config={"claim": st.column_config.TextColumn("gap", width="large"),
                                "evidence_level": "evidence", "n_supporting": "supporting",
                                "n_full_text": "full text", "n_contradicting": "contradicting",
                                "evidence_years": "years", "evidence_coverage": "verifier read"})
    st.download_button("Download these gaps (.csv)", table.drop(columns=["supporting_papers", "counterexample_paper", "evidence", "evidence_quotes"])
                       .to_csv(index=False).encode("utf-8-sig"), "potential_research_gaps.csv", "text/csv")

    # ---- refuted gaps, with the paper that refutes them ----
    refuted = [g for g in sel if g["status"] in (dash.STATUS_REFUTED, dash.STATUS_REFUTED_OLD)]
    st.subheader(f"❌ Gaps refuted by a paper ({len(refuted)})")
    st.caption("The verifier found a paper that already does what the gap says is missing. Since this version a "
               "refutation only counts when it copies a sentence from that paper verbatim and the harness finds "
               "it in the text the tools returned. ❔ = older run, before quotes were required.")
    if not refuted:
        st.write("None in this selection.")
    for g in refuted:
        with st.container(border=True):
            st.markdown(f"**{STATUS_ICON[g['status']]} {g['claim']}**  \n"
                        f"<small>{g['topic']} · {g['time']} · {g['category']}</small>", unsafe_allow_html=True)
            st.markdown(f"**Refuted by:** {_link(g['counterexample_paper']) or '_(paper id not recorded)_'}")
            if g.get("counterexample_quote"):
                st.markdown(f"> “{g['counterexample_quote']}”")
            st.caption(f"Verdict rule: {g['verdict_reason']}")

    # ---- possible counterexamples the verifier could not quote ----
    contested = [g for g in sel if g["status"] == dash.STATUS_CONTESTED]
    if contested:
        st.subheader(f"⚠️ Possible counterexamples to check ({len(contested)})")
        st.caption("The verifier named a paper that may already do this, but could not quote a sentence that shows it. "
                   "The gap is kept, capped below Strong - open the paper and decide.")
        st.dataframe(pd.DataFrame([{"topic": g["topic"], "gap": g["claim"], "check paper": g["counterexample_id"]}
                                   for g in contested]), hide_index=True, width="stretch",
                     column_config={"gap": st.column_config.TextColumn(width="large")})

    # ---- stock gaps ----
    rec = dash.recurring_gaps(sel)
    st.subheader(f"🔁 Stock gaps - themes that recur across different topics ({len(rec)})")
    st.caption("The same kind of gap proposed for unrelated topics is usually generic - check that the papers "
               "really motivate it before using one.")
    if rec:
        st.dataframe(pd.DataFrame([{"theme": r["theme"], "topics": r["n_topics"], "gaps": r["n_gaps"],
                                    "refuted": r["refuted"], "example": r["example"],
                                    "in topics": "; ".join(r["topics"])} for r in rec]),
                     hide_index=True, width="stretch",
                     column_config={"example": st.column_config.TextColumn(width="large")})
    else:
        st.write("None found.")

    # ---- gap details ----
    st.subheader("Potential research gap details")
    for g in sel:
        with st.expander(f"{STATUS_ICON.get(g['status'], '')} [{g['evidence_level']}] {g['claim'][:140]}"):
            st.markdown(f"**Topic:** {g['topic']} · **Run:** {g['time']} · **Category:** {g['category']}"
                        + (" _(inferred by keyword rules - run saved before categories)_" if g["category_source"] == "rules" else ""))
            st.markdown(evidence.evidence_line(g["evidence"], g["evidence_level"]))
            if g["evidence_level_reasons"]:
                st.caption(f"Why not {'Moderate' if g['evidence_level'] == evidence.LIMITED else 'Strong'}: "
                           + "; ".join(g["evidence_level_reasons"]))
            st.write(g["rationale"])
            st.markdown("**Supporting papers:**  \n" + "  \n".join(f"• {_link(p)}" for p in g["supporting_papers"]))
            st.markdown("**Evidence sentences:**  \n" + "  \n".join(evidence.quote_lines(g["evidence_quotes"])))
            if g["counterexample_paper"]:
                st.markdown(f"**Refuted by:** {_link(g['counterexample_paper'])}")
            st.caption(f"Verdict rule: {g['verdict_reason']}")

    # ---- runs ----
    st.subheader(f"Runs ({len(sel_runs)})")
    rr = pd.DataFrame(dash.run_rows(sel_runs))
    st.dataframe(rr, hide_index=True, width="stretch",
                 column_config={"trace_url": st.column_config.LinkColumn("trace", display_text="open"),
                                "run_id": None})

    # ---- papers ----
    pr = pd.DataFrame(dash.paper_rows(sel_runs))
    if not pr.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Analysed papers by year")
            st.bar_chart(pr.dropna(subset=["year"]).groupby("year").size(), height=260)
        with c2:
            st.subheader("By source")
            st.bar_chart(pr.groupby("source").size(), height=260, horizontal=True)
        st.subheader("Papers cited most by gaps")
        top = (pr[pr["cited_by_gaps"] > 0].groupby(["title", "year", "url"], dropna=False)["cited_by_gaps"].sum()
               .reset_index().sort_values("cited_by_gaps", ascending=False).head(15))
        st.dataframe(top, hide_index=True, width="stretch",
                     column_config={"url": st.column_config.LinkColumn("link", display_text="open"),
                                    "title": st.column_config.TextColumn(width="large")})
