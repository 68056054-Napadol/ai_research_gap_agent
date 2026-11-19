"""
Dashboard page of the Streamlit app (app.py -> sidebar "Page: Dashboard").
Data comes from dashboard.py (every saved run in output/). Only st.* drawing lives here.
"""
import pandas as pd
import streamlit as st

import dashboard as dash
from config import CONFIG

STATUS_ORDER = [dash.STATUS_SUPPORTED, dash.STATUS_NOT_SUPPORTED, dash.STATUS_REFUTED, dash.STATUS_REFUTED_OLD]
STATUS_ICON = {dash.STATUS_SUPPORTED: "✅", dash.STATUS_NOT_SUPPORTED: "⚪",
               dash.STATUS_REFUTED: "❌", dash.STATUS_REFUTED_OLD: "❔"}


def _link(p: dict | None) -> str:
    if not p:
        return ""
    label = f"{p.get('title') or p.get('id')} ({p.get('year') or 'n.d.'})"
    return f"[{label}]({p['url']})" if p.get("url") else label


def render():
    st.title("📊 Dashboard — research gaps across runs")
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
        f_conf = c4.multiselect("Confidence", ["High", "Medium", "Low"], placeholder="All")
    sel = [g for g in gaps if (not f_topic or g["topic"] in f_topic) and (not f_cat or g["category"] in f_cat)
           and (not f_status or g["status"] in f_status) and (not f_conf or g["confidence_level"] in f_conf)]
    sel_runs = [r for r in runs if not f_topic or r["state"].get("research_topic") in f_topic]

    # ---- KPIs ----
    s = dash.summary(sel, sel_runs)
    k = st.columns(8)
    for col, (label, value) in zip(k, [("Runs", s["runs"]), ("Topics", s["topics"]), ("Papers analysed", s["papers_analyzed"]),
                                       ("Gaps", s["gaps"]), ("✅ Supported", s["supported"]), ("❌ Refuted", s["refuted"]),
                                       ("High confidence", s["high"]), ("Mean confidence", s["mean_confidence"] or "-")]):
        col.metric(label, value)

    if not sel:
        st.warning("No gaps match these filters.")
        return
    df = pd.DataFrame(sel)

    # ---- charts ----
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Gaps by category")
        pivot = df.pivot_table(index="category", columns="status", values="claim", aggfunc="count", fill_value=0)
        st.bar_chart(pivot, horizontal=True, height=320)
    with c2:
        st.subheader("Confidence by verdict")
        pivot = df.pivot_table(index="confidence_level", columns="status", values="claim", aggfunc="count", fill_value=0)
        st.bar_chart(pivot.reindex([x for x in ["High", "Medium", "Low"] if x in pivot.index]), height=320)

    # ---- all gaps ----
    st.subheader(f"All gaps ({len(sel)})")
    table = df.assign(verdict=df["status"].map(lambda x: f"{STATUS_ICON.get(x, '')} {x}"),
                      papers=df["supporting_papers"].map(len),
                      refuted_by=df["counterexample_paper"].map(lambda p: (p or {}).get("title", "")))
    st.dataframe(table[["time", "topic", "category", "claim", "verdict", "confidence_level", "confidence_score",
                        "papers", "refuted_by"]],
                 hide_index=True, width="stretch",
                 column_config={"claim": st.column_config.TextColumn("gap", width="large"),
                                "confidence_score": st.column_config.ProgressColumn("score", min_value=0, max_value=100,
                                                                                    format="%.0f"),
                                "confidence_level": "confidence"})
    st.download_button("Download these gaps (.csv)", table.drop(columns=["supporting_papers", "counterexample_paper"])
                       .to_csv(index=False).encode("utf-8-sig"), "research_gaps.csv", "text/csv")

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

    # ---- stock gaps ----
    rec = dash.recurring_gaps(sel)
    st.subheader(f"🔁 Gaps that recur across different topics ({len(rec)})")
    st.caption("Near-identical gaps proposed for unrelated topics are usually generic - check that the papers "
               "really motivate them before using one.")
    if rec:
        st.dataframe(pd.DataFrame([{"category": r["category"], "example": r["example"], "gaps": r["n_gaps"],
                                    "topics": r["n_topics"], "in topics": "; ".join(r["topics"])} for r in rec]),
                     hide_index=True, width="stretch",
                     column_config={"example": st.column_config.TextColumn(width="large")})
    else:
        st.write("None found.")

    # ---- gap details ----
    st.subheader("Gap details")
    for g in sel:
        with st.expander(f"{STATUS_ICON.get(g['status'], '')} [{g['confidence_level']}] {g['claim'][:140]}"):
            st.markdown(f"**Topic:** {g['topic']} · **Run:** {g['time']} · **Category:** {g['category']}"
                        + (" _(inferred by keyword rules - run saved before categories)_" if g["category_source"] == "rules" else ""))
            st.write(g["rationale"])
            st.markdown("**Supporting papers:**  \n" + "  \n".join(f"• {_link(p)}" for p in g["supporting_papers"]))
            if g["counterexample_paper"]:
                st.markdown(f"**Refuted by:** {_link(g['counterexample_paper'])}")
            if g["confidence_caps"]:
                st.caption("Capped below High: " + "; ".join(g["confidence_caps"]))
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
