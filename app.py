"""
Streamlit UI for the agent — a thin interface layer over graph.py/config.py. No agent logic
lives here; this only collects input, runs the agent in a background thread so the UI stays
responsive, and renders live progress as one card per pipeline node (input -> output), plus a
collapsed full raw log for debugging. Also renders the same state that main.py's CLI writes to
output/*.json.

Run:  streamlit run app.py

--- Why cards instead of one scrolling log box (design note) ---
A run can take many minutes on local/CPU hardware, and most of that time is spent inside a
single node waiting on the LLM. A flat log box makes that feel like noise: the same
"still generating locally..." line repeats for minutes with nothing that looks like progress.
Instead, each of the 7 graph nodes (architecture doc §2.2) gets its own card showing: a status
badge (pending/running/done/error), a one-line summary of what went IN to the node, and once
it finishes, a one-line-to-few-line summary of what came OUT. While a node is running, the card
shows a single live-updating sub-status line (the latest thing the agent's logger said) instead
of an ever-growing wall of text — except the per-paper extraction node, which gets a small live
checklist (1 line per paper) since that node's whole point is per-item progress. The full,
unfiltered log is still available in a collapsed expander and as a download, for debugging.
"""
import datetime
import json
import logging
import queue
import os
import re
import threading
import time

import streamlit as st

import embeddings
import evidence
import links
import metrics
import outbound
import tracing
from config import CONFIG
from graph import build_graph
from logging_utils import QueueLogHandler, quiet_noisy_third_party_loggers
from main import render_report, run_info

st.set_page_config(page_title="AI Research Assistant Agent", layout="wide")

CONFIDENCE_COLOR = {"High": "green", "Medium": "orange", "Low": "red"}
LOG_BOX_HEIGHT = 300
PAPER_BOX_HEIGHT = 180

# ---------------------------------------------------------------------------
# Node metadata + graph shape (mirrors graph.py exactly — see that file for the real edges).
# ---------------------------------------------------------------------------
NODE_ORDER = [
    "sanitize_input", "generate_keywords", "search_sources", "filter_and_rank", "check_sufficiency",
    "extract_paper_data", "compare_papers", "analyze_gaps", "privacy_guard",
]
NODE_META = {
    "sanitize_input": ("🛡️", "Privacy check on input (LLM + decision model)"),
    "generate_keywords": ("🔑", "Generate search keywords"),
    "search_sources": ("🔍", "Search agent (LLM tool calling: search_papers)"),
    "filter_and_rank": ("📊", "Rank papers by relevance"),
    "check_sufficiency": ("✅", "Check if enough papers found"),
    "extract_paper_data": ("📄", "Extract data from each paper"),
    "compare_papers": ("🔬", "Compare papers & synthesize trends"),
    "analyze_gaps": ("💡", "Propose potential research gaps + verifier agent (tools: get_paper_text, search_corpus)"),
    "privacy_guard": ("🛡️", "Privacy guard on final answer"),
}
# Linear "what runs next" edges. check_sufficiency is conditional (handled specially below) and
# analyze_gaps has no successor (END) — see graph.py.
NEXT_NODE = {
    "sanitize_input": "generate_keywords",
    "generate_keywords": "search_sources",
    "search_sources": "filter_and_rank",
    "filter_and_rank": "check_sufficiency",
    "extract_paper_data": "compare_papers",
    "compare_papers": "analyze_gaps",
    "analyze_gaps": "privacy_guard",
}
def source_health_line(health: dict) -> str:
    """'Sources: arxiv ✅ 5 · semantic_scholar ⛔ rate limited ...' for the search card and report."""
    if not health:
        return ""
    parts = []
    for src, h in sorted(health.items()):
        if h.get("ok"):
            parts.append(f"{src} ✅" + (f" ({h['retries']} retried)" if h.get("retries") else ""))
        else:
            parts.append(f"{src} ⛔ {h.get('last_error') or 'no answer'}")
    return "Sources: " + " · ".join(parts)


RETRY_RESET_KEYS = ["generate_keywords", "search_sources", "filter_and_rank", "check_sufficiency"]

# logging_utils.LOG_FORMAT is "HH:MM:SS  LEVEL   name             message" (no separator before
# the message) — this strips that prefix so cards show just the message, not log plumbing.
_LOG_PREFIX_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\s+\S+\s+\S+\s+")

# Per-paper progress lines from nodes.extract_paper_data — see that function's logger.info calls.
_EXTRACTING_RE = re.compile(r"\[(\d+)/(\d+)\] Extracting \(([^)]+)\): (.*)")
_SKIPPED_RE = re.compile(r"\[(\d+)/(\d+)\] SKIPPED[^:]*: (.*)")
_FAILED_RE = re.compile(r"\[(\d+)/(\d+)\] Extraction FAILED for (\S+) \(([\d.]+)s\): (.*)")
_DONE_RE = re.compile(r"\[(\d+)/(\d+)\] Done \(([\d.]+)s\): (.*)")


def _new_card_state():
    return {
        "status": "pending",  # pending | running | done | error
        "started_at": None,
        "finished_at": None,
        "attempt": 1,
        "input_summary": "",
        "output_summary": "",
        "sub_status": "",
        "error": "",
        "papers": {},  # extract_paper_data only: {index: {"total", "title", "status", "seconds"}}
    }


def _summarize_input(node_key: str, acc: dict) -> str:
    if node_key == "sanitize_input":
        return "Topic as typed (checked before anything else sees it)"
    if node_key == "privacy_guard":
        return "Trend summary + gap claims/rationales/verifier notes"
    if node_key == "generate_keywords":
        topic = acc.get("research_topic", "")
        prior = acc.get("prior_queries_tried") or []
        if prior:
            return f'Topic: "{topic}" — avoiding {len(prior)} previously tried quer{"y" if len(prior) == 1 else "ies"}'
        return f'Topic: "{topic}"'
    if node_key == "search_sources":
        queries = acc.get("search_queries") or []
        return f"{len(queries)} quer{'y' if len(queries) == 1 else 'ies'} across {', '.join(CONFIG.enabled_sources)}"
    if node_key == "filter_and_rank":
        n = len(acc.get("candidate_papers") or [])
        return f"{n} candidate papers to rank by similarity to the topic"
    if node_key == "check_sufficiency":
        n = len(acc.get("selected_papers") or [])
        return f"{n} papers selected so far (need ≥ {CONFIG.min_papers_before_retry})"
    if node_key == "extract_paper_data":
        n = len(acc.get("selected_papers") or [])
        return f"{n} papers to extract method/dataset/findings from"
    if node_key == "compare_papers":
        n = len(acc.get("paper_extracts") or [])
        return f"{n} extracted papers to compare"
    if node_key == "analyze_gaps":
        return "Comparison table + cross-paper trend summary"
    return ""


def _summarize_output(node_key: str, update: dict, acc: dict) -> str:
    if node_key == "sanitize_input":
        d = update.get("input_decision") or {}
        recs = update.get("privacy_log") or []
        extra = f" — {d.get('message')}" if d.get("message") else ""
        return (f"Decision **{d.get('action')}** (rule {d.get('rule')}), {len(recs)} entity decision(s){extra}  \n"
                f"Topic used: _{update.get('research_topic', '')}_")
    if node_key == "privacy_guard":
        return f"**{update.get('final_answer_redactions', 0)}** redaction(s) applied to the final answer"
    if node_key == "generate_keywords":
        qs = update.get("search_queries") or []
        return "Queries: " + "; ".join(f'"{q}"' for q in qs)
    if node_key == "search_sources":
        n = len(update.get("candidate_papers") or [])
        calls = [t for t in (update.get("tool_trace") or []) if t.get("tool_name") == "search_papers"]
        fb = " ⚠️ model made no tool calls — harness fallback used" if update.get("search_used_fallback") else ""
        return (f"**{len(calls)}** search_papers call(s) chosen by the LLM → **{n}** unique candidate papers{fb}  \n"
                + source_health_line(update.get("source_health") or {}) +
                f"  \nAgent summary: _{(update.get('search_summary') or '')[:300]}_")
    if node_key == "filter_and_rank":
        sel = update.get("selected_papers") or []
        ranked = update.get("ranked_papers") or []
        return (f"**{len(ranked)}** passed the relevance cut-off ({embeddings.ranking_threshold():.2f}, "
                f"`{CONFIG.ranking_embedding}`) — top **{len(sel)}** selected for deep analysis")
    if node_key == "check_sufficiency":
        if update.get("sufficient"):
            return "Sufficient papers found — proceeding to extraction"
        if update.get("retry_count", 0) >= CONFIG.max_retry_loops:
            return f"Still short on papers after {CONFIG.max_retry_loops} retries — proceeding anyway"
        return "Not enough papers yet — retrying with broader keywords"
    if node_key == "extract_paper_data":
        n = len(update.get("paper_extracts") or [])
        skipped = update.get("skipped_papers") or []
        n_full = sum(1 for e in update.get("paper_extracts") or [] if e.get("extraction_confidence") == "full_text")
        return (f"Successfully extracted structured data from **{n}** papers ({n_full} full text, {n - n_full} abstract only)"
                + (f" · **{len(skipped)}** skipped: " + "; ".join(f"{(p.get('title') or p['id'])[:60]} ({p['reason']})"
                                                         for p in skipped) if skipped else ""))
    if node_key == "compare_papers":
        trend = update.get("trend_summary") or ""
        preview = trend[:280] + ("…" if len(trend) > 280 else "")
        return f"Trend summary: {preview}"
    if node_key == "analyze_gaps":
        gaps = update.get("gap_candidates") or []
        if not gaps:
            return "No potential research gaps survived verification"
        return f"**{evidence.ANSWER_TITLE}** (from the analysed papers only):  \n" + "\n".join(f"- {g['claim']} _( {g['confidence_level']}, {g['confidence_score']}/100, "
                         f"verifier read {len(g.get('evidence_read_ids', []))}/{len(g.get('supporting_paper_ids', []))} sources )_"
                         for g in gaps)
    return ""


def _apply_paper_progress_line(card: dict, line: str):
    """Parse one raw log line from extract_paper_data's per-paper loop into card['papers']."""
    m = _EXTRACTING_RE.search(line)
    if m:
        i, total, method, title = m.groups()
        card["papers"][int(i)] = {"total": int(total), "title": title, "status": "running", "seconds": None, "method": method}
        return
    m = _DONE_RE.search(line)
    if m:
        i, total, secs, title = m.groups()
        card["papers"][int(i)] = {"total": int(total), "title": title, "status": "done", "seconds": float(secs)}
        return
    m = _FAILED_RE.search(line)
    if m:
        i, total, paper_id, secs, _err = m.groups()
        card["papers"][int(i)] = {"total": int(total), "title": paper_id, "status": "failed", "seconds": float(secs)}
        return
    m = _SKIPPED_RE.search(line)
    if m:
        i, total, title = m.groups()
        card["papers"][int(i)] = {"total": int(total), "title": title, "status": "skipped", "seconds": None}
        return


_PAPER_ICON = {"running": "⏳", "done": "✅", "failed": "❌", "skipped": "⏭️"}


def _paper_checklist_text(papers: dict) -> str:
    lines = []
    for i in sorted(papers):
        p = papers[i]
        icon = _PAPER_ICON.get(p["status"], "•")
        title = (p.get("title") or "")[:70]
        secs = f" ({p['seconds']:.0f}s)" if p.get("seconds") is not None else " — working..."
        lines.append(f"{icon} [{i}/{p['total']}] {title}{secs}")
    return "\n".join(lines) if lines else "starting..."


def _status_label(card: dict) -> str:
    label = {"pending": "pending", "running": "running", "done": "done", "error": "error"}[card["status"]]
    icon = {"pending": "⏸️", "running": "⏳", "done": "✅", "error": "⚠️"}[card["status"]]
    elapsed = ""
    if card["started_at"] is not None:
        end = card["finished_at"] if card["finished_at"] is not None else time.monotonic()
        elapsed = f" · {end - card['started_at']:.0f}s"
    attempt = f" · attempt {card['attempt']}" if card["attempt"] > 1 else ""
    return f"{icon} {label}{elapsed}{attempt}"


def _render_card(placeholder, node_key: str, card: dict):
    icon, title = NODE_META[node_key]
    with placeholder.container(border=True):
        st.markdown(f"**{icon} {title}**")
        st.caption(_status_label(card))
        if card["input_summary"]:
            st.caption(f"Input: {card['input_summary']}")
        if card["status"] == "running":
            if node_key == "extract_paper_data" and card["papers"]:
                with st.container(height=PAPER_BOX_HEIGHT, border=True):
                    st.code(_paper_checklist_text(card["papers"]), language=None)
            elif card["sub_status"]:
                st.caption(f"⏳ {card['sub_status']}")
        if card["status"] == "done":
            if node_key == "extract_paper_data" and card["papers"]:
                with st.container(height=PAPER_BOX_HEIGHT, border=True):
                    st.code(_paper_checklist_text(card["papers"]), language=None)
            if card["output_summary"]:
                st.markdown(card["output_summary"])
        if card["status"] == "error" and card["error"]:
            st.error(card["error"])


# Two pages: the agent run (everything below) and the Dashboard over all saved runs.
page = st.sidebar.radio("Page", ["🔬 Run agent", "📊 Dashboard"], horizontal=True, key="page")
if page == "📊 Dashboard":
    import dashboard_page
    dashboard_page.render()
    st.stop()

if "final_state" not in st.session_state:
    st.session_state.final_state = None
if "log_lines" not in st.session_state:
    st.session_state.log_lines = []

# ---------------------------------------------------------------------------
# Sidebar: runtime settings. These mutate the shared CONFIG object directly —
# every node reads CONFIG fresh on each call, so this takes effect immediately,
# no restart needed. See config.py / llm_client.py.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    backend = st.radio("LLM backend", ["cloud", "local"], index=0 if CONFIG.llm_backend == "cloud" else 1,
                        help="Cloud = Anthropic API (needs a key). Local = Ollama running on this machine.")
    CONFIG.llm_backend = backend

    if backend == "cloud":
        key = st.text_input("Anthropic API key", value=CONFIG.anthropic_api_key, type="password",
                             help="Or set ANTHROPIC_API_KEY in .env to avoid re-entering this.")
        CONFIG.anthropic_api_key = key or CONFIG.anthropic_api_key
    else:
        CONFIG.ollama_host = st.text_input("Ollama host", value=CONFIG.ollama_host)
        st.caption(f"Models: {CONFIG.local_model_fast} (fast) / {CONFIG.local_model_strong} (strong) "
                   f"— change in .env. Make sure `ollama pull` was run for both.")
        CONFIG.ollama_think = st.checkbox(
            "Enable model 'thinking' mode", value=CONFIG.ollama_think,
            help="Off by default for speed — reasoning models like Qwen3 spend most of their time on an "
                 "internal reasoning trace this project's tasks don't need. Turn on only if you have fast "
                 "hardware and want to try it for the 'strong' tier's reasoning quality.",
        )

    st.caption("Tracing: " + ("Langfuse ON (" + (os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "cloud") + ")"
                              if os.getenv("LANGFUSE_PUBLIC_KEY") else "off — set LANGFUSE_* in .env to enable"))
    CONFIG.rag_enabled = st.checkbox("Use RAG knowledge base", value=CONFIG.rag_enabled,
                                     help="Chunk + embed selected papers into a persistent store (rag_store/), reuse across runs, "
                                          "and give the verifier search_paper_chunks / search_corpus tools.")

    st.divider()
    CONFIG.max_deep_analysis_papers = st.slider(
        "Papers to deep-analyze", min_value=3, max_value=40, value=min(CONFIG.max_deep_analysis_papers, 40),
        help="Each paper adds roughly 45 s on a local 14B model (PDF + extraction + indexing). "
             f"Gaps can only reach High confidence with >= {CONFIG.high_confidence_min_papers} papers.",
    )
    source_options = ["semantic_scholar", "arxiv", "openalex"]
    selected_sources = st.multiselect("Data sources", source_options, default=list(CONFIG.enabled_sources))
    CONFIG.enabled_sources = tuple(selected_sources) if selected_sources else tuple(source_options)

    CONFIG.fetch_full_text = st.checkbox(
        "Fetch full-text PDFs", value=CONFIG.fetch_full_text,
        help="Downloads each selected paper's open-access PDF and converts it to Markdown for extraction, "
             "instead of using only the abstract — more accurate, but roughly doubles Node 5's time per "
             "paper. Results are cached (pdf_cache/), so re-running the same topic is fast the 2nd time.",
    )

# ---------------------------------------------------------------------------
# Main input
# ---------------------------------------------------------------------------
st.title("AI Research Assistant Agent")
st.caption("Find, compare, and identify research gaps across academic papers.")

# A form: typing the topic and clicking Run once is enough. (With a bare text_input + button, the
# first click only committed the text box and a second click was needed to start the run.)
with st.form("run_form", border=False):
    topic = st.text_input("Research topic", placeholder="e.g. few-shot named entity recognition in low-resource languages")
    c1, c2, c3 = st.columns([1, 1, 2])
    year_from = c1.number_input("Year from", min_value=1990, max_value=2030, value=2020, step=1)
    year_to = c2.number_input("Year to", min_value=1990, max_value=2030, value=2026, step=1)
    c3.markdown("<div style='height: 1.75rem'></div>", unsafe_allow_html=True)
    run_clicked = c3.form_submit_button("Run agent", type="primary", use_container_width=True)
missing_keys = [name for name, key in (("SEMANTIC_SCHOLAR_API_KEY", CONFIG.semantic_scholar_api_key),
                                       ("OPENALEX_API_KEY", CONFIG.openalex_api_key)) if not key]
if missing_keys:
    st.caption("⚠️ No " + " / ".join(missing_keys) + " in .env - those sources may be rate limited, which narrows "
               "the search (and caps gap confidence below High). Both keys are free; see .env.example.")

# ---------------------------------------------------------------------------
# Run in a background thread + live per-node step cards.
#
# Why a thread: LangGraph's graph.stream() only yields once a node finishes, so without a thread
# the UI would only update once per node (fine for "which step are we on", useless for seeing
# progress *inside* a slow node, e.g. paper 7 of 20 during extraction). Running the graph in a
# background thread and piping (a) its logger output and (b) each node's completion event
# through two queues lets the main Streamlit thread redraw the cards every ~150ms regardless of
# what the agent is doing.
#
# Thread safety: only the main thread calls any st.* function. The background thread only calls
# logging.info(...) and queue.put(...) (both thread-safe) — it never touches Streamlit.
# ---------------------------------------------------------------------------
if run_clicked:
    if not topic.strip():
        st.error("Enter a research topic first.")
    elif backend == "cloud" and not CONFIG.anthropic_api_key:
        st.error("Enter an Anthropic API key in the sidebar, or switch to the local backend.")
    else:
        metrics.reset()
        run_started = time.time()
        agent_graph = build_graph()
        initial_state = {
            "research_topic": topic,
            "year_from": int(year_from),
            "year_to": int(year_to),
            "max_papers": CONFIG.max_deep_analysis_papers,
            "prior_queries_tried": [],
            "retry_count": 0,
        }

        # Always capture everything the agent itself logs (no level picker needed) — the only
        # filtering is muting third-party wire-level noise (see logging_utils) so both the raw
        # log and the cards' sub-status lines stay readable.
        log_queue: "queue.Queue[str]" = queue.Queue()
        event_queue: "queue.Queue[tuple]" = queue.Queue()
        handler = QueueLogHandler(log_queue)
        handler.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        prev_level = root_logger.level
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)
        quiet_noisy_third_party_loggers()

        run_result = {"state": None, "error": None}

        tracing.init()
        ui_session_id = st.session_state.setdefault("ui_session_id", "streamlit-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        def _run_agent():
            try:
                final = dict(initial_state)
                with tracing.run_trace("streamlit", session_id=ui_session_id) as trace:
                    for step in agent_graph.stream(initial_state, {"recursion_limit": 60}):
                        for node_name, node_update in step.items():
                            final.update(node_update)
                            event_queue.put(("node_done", node_name, node_update, dict(final)))
                    final.setdefault("run_metrics", metrics.snapshot())
                    trace.finish(final)
                if trace.trace_id:
                    final["trace_id"] = trace.trace_id
                    if trace.trace_url:
                        final["trace_url"] = trace.trace_url
                tracing.flush()
                run_result["state"] = final
            except Exception as e:  # surfaced in the main thread below, not raised here
                run_result["error"] = e
            finally:
                event_queue.put(("run_finished",))

        worker = threading.Thread(target=_run_agent, daemon=True)
        worker.start()

        st.session_state.log_lines = []

        # --- card state setup ---
        cards = {key: _new_card_state() for key in NODE_ORDER}
        acc_state = dict(initial_state)
        cards["sanitize_input"]["status"] = "running"
        cards["sanitize_input"]["started_at"] = time.monotonic()
        cards["sanitize_input"]["input_summary"] = _summarize_input("sanitize_input", acc_state)

        st.caption("Each card below is one step of the pipeline — see architecture doc §2.2.")
        placeholders = {key: st.empty() for key in NODE_ORDER}
        for key in NODE_ORDER:
            _render_card(placeholders[key], key, cards[key])

        # Full raw log, live — kept visible under the cards for the whole run (not just after),
        # per request: this is the "print everything" box, always on screen, not tucked behind a
        # click. The cards above are the readable summary; this is the unfiltered detail under it.
        st.divider()
        st.subheader("Full run log")
        log_box_placeholder = st.empty()

        def _draw_log_box():
            text = "\n".join(st.session_state.log_lines) if st.session_state.log_lines else "waiting for the agent to start..."
            with log_box_placeholder.container(height=LOG_BOX_HEIGHT, border=True):
                st.code(text, language=None)

        _draw_log_box()

        run_finished = False
        while not run_finished:
            redraw = set()
            log_updated = False

            # Drain raw log lines: feed the currently-running card's sub-status line, and (for
            # the extraction node) parse per-paper progress out of them.
            while True:
                try:
                    line = log_queue.get_nowait()
                except queue.Empty:
                    break
                st.session_state.log_lines.append(line)
                log_updated = True
                running_key = next((k for k in NODE_ORDER if cards[k]["status"] == "running"), None)
                if running_key:
                    msg = _LOG_PREFIX_RE.sub("", line, count=1)
                    cards[running_key]["sub_status"] = msg.strip()[:160]
                    if running_key == "extract_paper_data":
                        _apply_paper_progress_line(cards[running_key], line)
                    redraw.add(running_key)

            # Drain node-completion events.
            while True:
                try:
                    event = event_queue.get_nowait()
                except queue.Empty:
                    break
                if event[0] == "run_finished":
                    run_finished = True
                    continue
                _, node_name, node_update, acc_state = event
                if node_name not in cards:
                    continue
                card = cards[node_name]
                card["status"] = "done"
                card["finished_at"] = time.monotonic()
                if card["started_at"] is None:
                    card["started_at"] = card["finished_at"]
                card["output_summary"] = _summarize_output(node_name, node_update, acc_state)
                card["sub_status"] = ""
                redraw.add(node_name)

                # Figure out what runs next (mirrors graph.py's edges, including the retry loop).
                if node_name == "check_sufficiency":
                    if node_update.get("sufficient") or node_update.get("retry_count", 0) >= CONFIG.max_retry_loops:
                        next_key = "extract_paper_data"
                    else:
                        next_key = "generate_keywords"
                        attempt = card["attempt"] + 1
                        for reset_key in RETRY_RESET_KEYS:
                            cards[reset_key] = _new_card_state()
                            cards[reset_key]["attempt"] = attempt
                            redraw.add(reset_key)
                elif node_name == "privacy_guard":
                    next_key = None
                elif node_name == "sanitize_input" and (node_update.get("input_decision") or {}).get("action") == "REJECT":
                    next_key = None
                else:
                    next_key = NEXT_NODE.get(node_name)

                if next_key:
                    cards[next_key]["status"] = "running"
                    cards[next_key]["started_at"] = time.monotonic()
                    cards[next_key]["input_summary"] = _summarize_input(next_key, acc_state)
                    redraw.add(next_key)

            if redraw:
                for key in redraw:
                    _render_card(placeholders[key], key, cards[key])
            if log_updated:
                _draw_log_box()
            time.sleep(0.15)

        # final drain — a few log lines can land after "run_finished"
        while True:
            try:
                st.session_state.log_lines.append(log_queue.get_nowait())
            except queue.Empty:
                break
        _draw_log_box()

        root_logger.removeHandler(handler)
        root_logger.setLevel(prev_level)

        if run_result["error"]:
            running_key = next((k for k in NODE_ORDER if cards[k]["status"] == "running"), None)
            if running_key:
                cards[running_key]["status"] = "error"
                cards[running_key]["finished_at"] = time.monotonic()
                cards[running_key]["error"] = str(run_result["error"])
                _render_card(placeholders[running_key], running_key, cards[running_key])
            st.error(f"Agent run failed: {run_result['error']}")
        else:
            final_state = run_result["state"]
            final_state.setdefault("run_metrics", metrics.snapshot())
            final_state["run_info"] = run_info(run_started, CONFIG.max_deep_analysis_papers, "streamlit")
            st.session_state.final_state = final_state

            os.makedirs(CONFIG.output_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(os.path.join(CONFIG.output_dir, f"{ts}_state.json"), "w", encoding="utf-8") as f:
                json.dump(outbound.enforce(final_state, "state.json"), f, indent=2, default=str, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Full raw log, persisted — the run above already showed this live; this keeps it visible (not
# collapsed) after the run completes or after any unrelated rerun (e.g. moving a sidebar
# slider), so "print everything" stays on screen rather than requiring an extra click.
# ---------------------------------------------------------------------------
if st.session_state.log_lines and not run_clicked:
    st.divider()
    st.subheader("Full run log (last run)")
    with st.container(height=LOG_BOX_HEIGHT, border=True):
        st.code("\n".join(st.session_state.log_lines), language=None)

if st.session_state.log_lines:
    st.download_button("Download run log (.txt)", data="\n".join(st.session_state.log_lines),
                        file_name="agent_run_log.txt", mime="text/plain")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
state = st.session_state.final_state
if state and (state.get("input_decision") or {}).get("action") == "REJECT":
    d = state["input_decision"]
    st.error(f"🛡️ Input rejected by the decision model (rule {d.get('rule')}): {d.get('message')}")
elif state:
    d = state.get("input_decision") or {}
    if d.get("action") == "PROCEED_REDACTED":
        st.warning(f"🛡️ {d.get('message')} Topic used: “{state.get('research_topic')}” (rule {d.get('rule')})")

    if state.get("trace_url"):
        st.info(f"🔭 Langfuse trace of this run: {state['trace_url']}")
    st.subheader("🔎 Search scope (ขอบเขตการค้น)")
    st.caption("How wide the search behind the answer was - a gap is only as convincing as this search.")
    st.markdown("\n".join(evidence.search_scope_lines(evidence.search_scope(state))))

    papers = state.get("selected_papers", [])
    st.subheader(f"Papers analyzed ({len(papers)} of {len(state.get('candidate_papers', []))} found)")
    for p in papers:
        # author names are never shown or stored - privacy.scrub_paper keeps only the count
        link = links.markdown_link({"title": p.get("title"), "year": p.get("year"), "url": p.get("url") or links.paper_url(p)})
        st.markdown(f"- **{link}** — {p.get('n_authors', '?')} author(s) "
                    f"· `{p.get('id')}` · similarity {p.get('similarity_score', 0):.2f}")

    st.subheader("Comparison table")
    st.markdown(state.get("comparison_table", "_(none)_"))

    st.subheader("Trends across papers")
    st.write(state.get("trend_summary", "_(none)_"))

    # ---- FINAL ANSWER ----
    st.header(f"✅ Final answer — {evidence.ANSWER_TITLE}")
    st.warning(evidence.LIMITATION_NOTE)
    st.caption("Each gap is checked by the verifier agent, which reads the cited papers through the "
               "get_paper_text tool. The context it read is in the tool-call panel below (agent = verifier_agent).")
    gaps = state.get("gap_candidates", [])
    if not gaps:
        st.info("No potential research gaps survived verification — try a broader topic or more papers.")
    for i, gap in enumerate(gaps, 1):
        with st.container(border=True):
            color = CONFIDENCE_COLOR.get(gap.get("confidence_level"), "gray")
            st.markdown(f"#### {i}. {gap['claim']}")
            st.markdown(f":{color}[**{gap['confidence_level']} confidence** — {gap['confidence_score']}/100] · "
                        f"verifier read **{len(gap.get('evidence_read_ids', []))}/{len(gap.get('supporting_paper_ids', []))}** "
                        f"cited sources via tool")
            if gap.get("confidence_caps"):
                st.caption("Capped below High: " + "; ".join(gap["confidence_caps"]))
            st.write(gap.get("rationale", ""))
            st.markdown("**Supporting papers** (links built from paper metadata, not written by the AI):  \n"
                        + "  \n".join(f"• {links.markdown_link(sp)}" for sp in gap.get("supporting_papers", [])))
            st.caption(f"Verifier notes: {gap.get('verifier_notes', '')}")
            if gap.get("verifier_decision_reason"):
                f = gap.get("verifier_facts") or {}
                st.caption(f"Verdict rule: {gap['verifier_decision_reason']} · facts: relevant={f.get('cited_papers_relevant')}, "
                           f"counterexample={f.get('counterexample_found')}, overstated={f.get('overstated')}")
                if f.get("counterexample_quote"):
                    st.caption(f"Counterexample quote ({'verified in its text' if f.get('counterexample_verified') else 'not found in its text - ignored'}): "
                               f"“{f['counterexample_quote'][:300]}”")
            if gap.get("counterexample_searches"):
                st.caption("🔎 Counterexample searches across the corpus (search_corpus): "
                           + "; ".join(f"“{q}”" for q in gap["counterexample_searches"]))
            if gap.get("dropped_ids"):
                st.caption(f"⚠️ Dropped cited ids not among analyzed papers: {', '.join(gap['dropped_ids'])}")

    # ---- TOOL CALLS: context returned to the agents ----
    trace = state.get("tool_trace") or []
    st.header(f"🔧 Tool calls — context returned from tools ({len(trace)})")
    st.caption("Every row is one tool call the LLM issued: tool name + arguments it chose, and the context "
               "the harness returned to it. agent = harness_fallback means the model did not call tools itself.")
    st.dataframe([{
        "#": i, "agent": t.get("agent"), "gap": str(t.get("gap_index", "")), "tool": t.get("tool_name"),
        "arguments": json.dumps(t.get("arguments", {}), ensure_ascii=False),
        "ok": t.get("ok", True), "items": t.get("n_items", ""), "context chars": t.get("result_chars", 0),
        "notes": "; ".join((t.get("policy_notes") or []) + ([t["error"]] if t.get("error") else [])),
    } for i, t in enumerate(trace, 1)], width="stretch", hide_index=True)
    for i, t in enumerate(trace, 1):
        with st.expander(f"#{i} {t.get('agent')} → {t.get('tool_name')}({json.dumps(t.get('arguments', {}), ensure_ascii=False)[:90]})"):
            st.markdown("**Arguments (from the LLM)**")
            st.json(t.get("arguments", {}))
            st.markdown("**Context returned to the LLM**")
            try:
                st.json(json.loads(t.get("result_preview", "")))
            except (ValueError, TypeError):
                st.code(t.get("result_preview", ""), language=None)

    # ---- PRIVACY DECISIONS ----
    plog = state.get("privacy_log") or []
    st.header(f"🛡️ Privacy decisions — decision model ({len(plog)})")
    st.caption("Detectors (regex / paper metadata / LLM) extract candidates; decision_model/sensitive_info.dmn.json "
               "decides. Original strings are never stored — only a masked form.")
    st.dataframe(plog, width="stretch", hide_index=True)

    kb = state.get("knowledge_base")
    if kb:
        m = state.get("run_metrics") or {}
        st.header("📚 Knowledge base (RAG, persists across runs)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Papers stored", kb.get("papers", 0))
        c2.metric("Chunks", kb.get("chunks", 0))
        c3.metric("New this run", int(m.get("rag_papers_indexed", 0)))
        c4.metric("Reused this run", int(m.get("rag_papers_reused", 0) + m.get("rag_full_text_reused", 0)))
        st.caption(f"Extractions reused from the knowledge base this run: {int(m.get('extract_cache_hits', 0))} · "
                   f"embedding: {kb.get('embedding')} · store: {kb.get('path')}")

    with st.expander("Run metrics (contracts, tool calls, tokens)"):
        st.json(state.get("run_metrics") or {})

    st.divider()
    report_md = render_report(state)
    st.download_button("Download full report (.md)", data=report_md,
                        file_name="research_gap_report.md", mime="text/markdown")
