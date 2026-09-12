# AI Research Assistant Agent

Companion code for `AI_Research_Assistant_Agent_Architecture.docx` (see that document for full
design rationale — §10 there covers the same install/run/use/test ground as this README, in
more depth, alongside the full architecture writeup).

Given a research topic, this agent searches academic sources, filters to the relevant papers,
extracts structured facts from each (optionally reading the full PDF, not just the abstract),
compares them, and proposes evidence-grounded Research Gap candidates with a confidence score.

## Status

The source connectors (`sources/semantic_scholar.py`, `sources/arxiv.py`, `sources/openalex.py`,
`sources/pdf_extract.py`), the embedding ranker, the hybrid cloud/local LLM client, and the
Streamlit UI are real, working code — this has been run end-to-end on real topics, on both the
cloud and local backends. The LangGraph nodes (`nodes.py`) are wired correctly with real prompts
and state flow, but the LLM-facing prompts will keep improving as you see more real outputs —
that tuning is the team's ongoing work (see "What to build next" at the bottom, and the docx §6).

## Table of contents

1. [Architecture, in brief](#architecture-in-brief)
2. [Node input/output reference](#node-inputoutput-reference)
3. [Fact-checking](#fact-checking)
4. [Prerequisites](#prerequisites)
5. [Installation, step by step](#installation-step-by-step)
6. [Configuration reference](#configuration-reference)
7. [Choosing a model backend](#choosing-a-model-backend)
8. [Running the agent](#running-the-agent)
9. [How to use the web UI](#how-to-use-the-web-ui)
10. [Full-text PDF extraction](#full-text-pdf-extraction)
11. [Testing and verifying your setup](#testing-and-verifying-your-setup)
12. [Troubleshooting](#troubleshooting)
13. [Project layout](#project-layout)
14. [Run the MCP server standalone](#run-the-mcp-server-standalone)
15. [What to build next](#what-to-build-next)

## Architecture, in brief

One LangGraph agent, seven nodes in sequence, with a single conditional retry loop. Full
rationale for every design choice is in the docx (§1–§9); this is just the map:

| # | Node (`nodes.py`) | What it does |
|---|---|---|
| 1 | `generate_keywords` | Turns the topic into 3–5 varied search queries (LLM, "fast" tier) |
| 2 | `search_sources` | Queries every enabled source in parallel-ish (semantic_scholar / arxiv / openalex), dedupes across queries and sources |
| 3 | `filter_and_rank` | Ranks candidates by embedding cosine-similarity to the topic; keeps the top N above a threshold |
| 4 | `check_sufficiency` | No LLM call — if too few papers passed the threshold, loops back to node 1 with feedback (capped at `MAX_RETRY_LOOPS`) |
| 5 | `extract_paper_data` | Per paper: fetches the open-access PDF and converts it to Markdown if available (else uses the abstract), then extracts method/dataset/findings/limitations/metrics (LLM, "fast" tier) |
| 6 | `compare_papers` | Builds a comparison table across all extracted papers, then synthesizes cross-paper trends (LLM, "strong" tier) |
| 7 | `analyze_gaps` | Proposes 2–4 candidate Research Gaps grounded in the table, then a separate verifier pass fact-checks each one and assigns a confidence score (LLM, "strong" tier, twice) |

The "fast" vs "strong" model tiers, the hybrid cloud/local backend, the confidence-scoring
formula, and the evaluation methodology are all covered in depth in the docx — this README is
deliberately just the practical "how do I run this" companion to it.

## Node input/output reference

Exact state keys in and out of each node — see `state.py` for the full `AgentState` schema, and
`prompts.py` for every prompt referenced below. "Reads" only lists keys that node actually uses,
not the whole state (every node receives the whole state object, per LangGraph's standard
pattern, but nodes only touch what they need).

**1. `generate_keywords`**
- Reads: `research_topic` (str); `prior_queries_tried` (list[str], only non-empty on a retry)
- Calls: 1 LLM call, `KEYWORD_GEN_PROMPT` (+ `RETRY_CONTEXT_TEMPLATE` appended when retrying), tier `fast`, JSON mode
- Writes: `search_queries` (list[str], 3–5 items); `prior_queries_tried` (extended)
- Example: `research_topic="gold trading"` → `search_queries=["gold trading market dynamics", "gold futures pricing", "central bank gold reserves", ...]`

**2. `search_sources`**
- Reads: `search_queries` (list[str]); `year_from`/`year_to`; `CONFIG.enabled_sources`
- Calls: no LLM — pure tool orchestration (`sources.search_all`, one call per query per enabled source)
- Writes: `candidate_papers` (list of `Paper` dicts — `id`, `source`, `title`, `authors`, `year`, `abstract`, `doi`, `oa_pdf_url`, `citation_count` — deduped by DOI/title, capped at `MAX_CANDIDATE_PAPERS`)

**3. `filter_and_rank`**
- Reads: `candidate_papers`; `research_topic`
- Calls: no LLM — local embedding model (`BAAI/bge-small-en-v1.5`) cosine similarity
- Writes: `ranked_papers` (all candidates + a `similarity_score` field, sorted descending); `selected_papers` (top `MAX_DEEP_ANALYSIS_PAPERS` of those scoring above `SIMILARITY_THRESHOLD`)

**4. `check_sufficiency`**
- Reads: `selected_papers`; `retry_count`
- Calls: no LLM — a single rule: `len(selected_papers) >= MIN_PAPERS_BEFORE_RETRY`
- Writes: `sufficient` (bool); `retry_count` (incremented only when insufficient)
- The conditional edge (`route_after_sufficiency`) then routes back to node 1 if insufficient and under `MAX_RETRY_LOOPS`, otherwise forward to node 5 regardless — see [Architecture, in brief](#architecture-in-brief)

**5. `extract_paper_data`**
- Reads: `selected_papers` (each paper's `abstract`, `oa_pdf_url`, and `full_text` if already fetched on an earlier attempt)
- Side effect: may fetch + cache a PDF via `pdf_extract.fetch_and_convert` and write the result onto `paper["full_text"]` in place — see [Full-text PDF extraction](#full-text-pdf-extraction)
- Calls: 1 LLM call per paper, `EXTRACT_PROMPT`, tier `fast`, JSON mode — text truncated to 12,000 chars if full-text was fetched, 4,000 chars if only the abstract is available
- Writes: `paper_extracts` (list of `PaperExtract` — `paper_id`, `method`, `dataset`, `key_findings`, `limitations_stated`, `metrics`, `extraction_confidence` = `"full_text"` or `"abstract_only"`, `supporting_quote`, `quote_verified`)
- Also runs `_quote_verified()` (`nodes.py`, no LLM call) on each extraction's `supporting_quote` against the exact text the model was shown — see [Fact-checking](#fact-checking)

**6. `compare_papers`**
- Reads: `paper_extracts`; `selected_papers` (for each paper's year); `research_topic`
- Calls: 1 LLM call, `COMPARE_PROMPT`, tier `strong`, plain text (not JSON) — built from the structured `comparison_table`, never from raw paper text, which is what keeps this step grounded
- Writes: `comparison_table` (str, a Markdown table, now with a "Quote verified" column — see [Fact-checking](#fact-checking)); `trend_summary` (str, 3–5 paragraphs)

**7. `analyze_gaps`**
- Reads: `comparison_table`; `trend_summary`; `paper_extracts`; `selected_papers`; `research_topic`
- Calls: 2 LLM calls per candidate gap, both tier `strong`, both JSON mode:
  1. `GAP_ANALYSIS_PROMPT` (once) → raw candidates: `claim`, `supporting_paper_ids`, `rationale`
  2. `VERIFIER_PROMPT` (once per candidate with ≥ `MIN_SUPPORTING_PAPERS_FOR_GAP` supporting papers) → `supported` (bool), `notes`, `unsupported_aspects` — see [Fact-checking](#fact-checking) just below
- Writes: `gap_candidates` (list of `GapCandidate` — `claim`, `supporting_paper_ids`, `rationale`, `confidence_score` 0–100, `confidence_level` High/Medium/Low, `verifier_notes`)

## Fact-checking

Yes — there's already a fact-checking pass, and it's mandatory, not optional: every proposed
Research Gap goes through it before it's ever shown to the user. It happens inside node 7
(`analyze_gaps`), as a second, separate LLM call per candidate:

1. **Proposer** (`GAP_ANALYSIS_PROMPT`) drafts 2–4 candidate gaps from the comparison table and
   trend summary. Its output schema *requires* each claim to cite specific `supporting_paper_ids`
   — a claim with no citations is structurally impossible to express in the schema, so this can't
   be skipped by the model.
2. **Structural filter**: any candidate citing fewer than `MIN_SUPPORTING_PAPERS_FOR_GAP` (default
   2) papers is rejected outright, before the verifier even runs — a rule, not a judgment call.
3. **Verifier** (`VERIFIER_PROMPT`, a separate LLM call — deliberately not the same call that
   proposed the gap, so it isn't just agreeing with itself) is instructed to be skeptical and
   checks three things against the cited papers' `paper_extracts`: does the evidence actually
   support the claim (or is it overstated), is the evidence recent or mostly old, and are there
   any counterexamples in the cited data that contradict the claim. It returns `supported: true/false`
   plus `notes` explaining what it checked — that verdict directly feeds the confidence score
   (`nodes.analyze_gaps`, the weighted formula covered in the docx §7.3), and the notes are shown
   to the user alongside each gap in both the web UI and the Markdown report.

**The limitation this used to have, and how it's now closed:** the node 7 verifier above checks
each gap claim against `paper_extracts` — the LLM's *own* structured summary of each paper from
node 5 — not against the original paper text. That makes it very good at catching *overclaiming*
(a gap claim that goes further than even the extracted summary supports), but on its own it could
never catch a *hallucinated extraction* (if node 5 already misread or invented something from the
source, the verifier had no way to know, since it was checking against that same wrong summary).

To close that gap, node 5 now also does **extraction-quote grounding**: `EXTRACT_PROMPT` requires
the model to return a `supporting_quote` — a short (under 40 words) excerpt copied *verbatim* from
the paper text, backing up `key_findings` — alongside the other extracted fields. Right after the
LLM call returns, `nodes._quote_verified()` checks that quote against the exact text the model was
shown, with **plain string matching, zero extra LLM calls, and zero added latency or cost**:

```python
def _quote_verified(quote: str, source_text: str) -> bool:
    quote = (quote or "").strip()
    if not quote:
        return False
    return _normalize_whitespace(quote) in _normalize_whitespace(source_text)
```

("Whitespace-normalized" only — runs of spaces/newlines collapse to one space, since PDF-to-Markdown
conversion reflows line breaks unpredictably — but the match is still case-sensitive and exact
otherwise, since "verbatim" is the entire point: a paraphrase doesn't count as verified.) The
result is stored as `quote_verified` (bool) on every `PaperExtract`, logged live in the run log
(`✅ Quote verified` / `⚠️ Quote NOT found verbatim — possible hallucination`), and shown as a
"Quote verified" column in the comparison table itself — so it's visible in the web UI, the
downloaded run log, the comparison table, and the final Markdown report, not buried in `state.json`.

This is a genuinely different, complementary check from the node 7 verifier, not a duplicate of
it: node 7 catches an LLM *reasoning too far* from correct extracted data; extraction-quote
grounding catches an LLM *inventing* the extracted data in the first place. A gap can now fail
either check independently — a claim can be well-reasoned from an extraction whose quote doesn't
verify, or perfectly verified but still overclaimed at the gap-analysis step — and both signals
are visible to the user rather than collapsed into one score.

**What's still not done, if you want to go further:** node 7's verifier itself still only ever
sees `paper_extracts`, never the original source text. **Source-grounded verification** — giving
the verifier the actual abstract/full text for each cited paper, not just its summary — would let
it catch reasoning drift introduced *during* gap proposal against the real source, the same way
extraction-quote grounding now catches drift introduced during extraction. It's a heavier lift
(more tokens per verification call, since it's reading source text instead of a short structured
summary) and wasn't implemented here, in part because `full_text` isn't reliably available for
every paper (only open-access PDFs get fetched — see [Full-text PDF extraction](#full-text-pdf-extraction)),
so for abstract-only papers it would mostly re-check the abstract against itself. Flagging it as
the next step if the team wants to strengthen this further before a demo.

## Prerequisites

- **Python 3.11 or newer.** Check with `python --version` (Windows: `python --version` in
  PowerShell). If missing, install from [python.org](https://www.python.org/downloads/) —
  on Windows, tick **"Add python.exe to PATH"** during setup, or the commands below won't find it.
- **This project folder** — `ai_research_gap_agent/`, with all the files described in
  [Project layout](#project-layout) below.
- **Either** an Anthropic API key (cloud backend — needs an internet connection and a paid/free
  key from [console.anthropic.com](https://console.anthropic.com)) **or** ~5GB of free disk
  space for a local model via Ollama (local backend — free per call, runs on your own hardware,
  no key needed). You can set up both and switch between them any time — see
  [Choosing a model backend](#choosing-a-model-backend).
- No GPU is required for either path. A GPU only speeds up the local backend if you have one;
  the cloud backend doesn't touch your hardware at all beyond the network call.

## Installation, step by step

Open a terminal **inside the `ai_research_gap_agent/` folder** and run these in order:

| # | Step | Windows (PowerShell) | macOS / Linux |
|---|---|---|---|
| 1 | Create a virtual environment | `python -m venv .venv` | `python3 -m venv .venv` |
| 2 | Activate it | `.venv\Scripts\activate` | `source .venv/bin/activate` |
| 3 | Install dependencies | `pip install -r requirements.txt` | `pip install -r requirements.txt` |
| 4 | Create your config file | `copy .env.example .env` | `cp .env.example .env` |
| 5 | Fill in `.env` | open it in Notepad / VS Code and edit | same |

A few notes on what to expect:

- **Step 2** — you should see `(.venv)` appear at the start of your terminal prompt once it's
  active. Every command after this point (steps 3+, and every time you come back to run the
  agent later) needs to happen with the venv active — if you close the terminal, just re-run
  step 2 (not step 1) before doing anything else.
- If PowerShell refuses step 2 with a message about "running scripts is disabled on this
  system", see [Troubleshooting](#troubleshooting) below — it's a one-line fix.
- **Step 3** installs LangGraph, the Anthropic client, `httpx`, `sentence-transformers` (the
  local embedding model used for paper ranking), `pymupdf4llm` (PDF→Markdown conversion), and
  Streamlit (the web UI). It also downloads a small (~130MB) embedding model the *first time the
  ranking step actually runs* (not during `pip install`) — expect a one-time pause the first time
  you run the agent, not a hang.
- **Step 5** — you do *not* need every key filled in. `arxiv` and `openalex` work with no key at
  all; `semantic_scholar` works unauthenticated at a lower rate limit. Add source API keys later
  if you start hitting rate limits. You do need **either** `ANTHROPIC_API_KEY` filled in **or**
  Ollama installed locally (next section) before the agent can actually run — see
  [Choosing a model backend](#choosing-a-model-backend).

Once step 5 is done, jump to [Testing and verifying your setup](#testing-and-verifying-your-setup) to
confirm everything works before your first real run.

## Configuration reference

Everything in `.env` (copy from `.env.example`, see step 4 above). Every setting has a sane
default — you only need to touch the ones relevant to your setup.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `cloud` | `cloud` (Anthropic API) or `local` (Ollama) — see [Choosing a model backend](#choosing-a-model-backend) |
| `ANTHROPIC_API_KEY` | *(empty)* | Required if `LLM_BACKEND=cloud` |
| `CLOUD_MODEL_FAST` / `CLOUD_MODEL_STRONG` | `claude-haiku-4-5` / `claude-sonnet-4-5` | Cloud models for the two tiers (see [Architecture](#architecture-in-brief)) |
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama is listening — only change this if Ollama runs on a different machine/port |
| `LOCAL_MODEL_FAST` / `LOCAL_MODEL_STRONG` | `qwen3:8b` / `qwen3:14b` | Local models for the two tiers — must be `ollama pull`ed first |
| `OLLAMA_READ_TIMEOUT` | `120` | Max seconds between streamed chunks from Ollama (not total generation time) — raise only if you still see timeouts on very slow hardware |
| `OLLAMA_THINK` | `false` | Whether to let reasoning models (Qwen3 etc.) generate an internal "thinking" trace before answering — off by default, since it's pure overhead for this project's tasks and can dramatically slow local runs |
| `OLLAMA_MAX_CTX` | `8192` | Ceiling on the local model's context window — raise only if you have RAM to spare; higher = slower |
| `FETCH_FULL_TEXT` | `true` | Fetch and convert each paper's open-access PDF instead of using only the abstract — see [Full-text PDF extraction](#full-text-pdf-extraction) |
| `PDF_CACHE_DIR` | `pdf_cache` | Where converted PDF text is cached |
| `PDF_MAX_PAGES` | `40` | Per-PDF page cap |
| `PDF_MAX_DOWNLOAD_MB` | `25` | Per-PDF size cap |
| `SEMANTIC_SCHOLAR_API_KEY` | *(empty)* | Optional — raises Semantic Scholar's rate limit |
| `OPENALEX_API_KEY` / `OPENALEX_MAILTO` | *(empty)* | Optional — `OPENALEX_MAILTO` puts you in OpenAlex's more-reliable "polite pool," recommended even though it's not strictly required |
| `UNPAYWALL_EMAIL` | *(empty)* | Reserved for a future Unpaywall connector — see "What to build next" |
| `MAX_CANDIDATE_PAPERS` | `80` | Broad search pool size (metadata only, cheap) |
| `MAX_DEEP_ANALYSIS_PAPERS` | `20` | How many papers get full extraction (the expensive tier) — overridable per-run via `--max-papers` or the web UI's slider |
| `MIN_PAPERS_BEFORE_RETRY` | `5` | Node 4's threshold for triggering a broader-search retry |
| `MAX_RETRY_LOOPS` | `3` | Hard cap on retry loops, so a bad topic can't loop forever |

## Choosing a model backend

### Option A — Cloud (fastest to get running)

Open `.env` and set:

```
LLM_BACKEND=cloud
ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at [console.anthropic.com](https://console.anthropic.com). That's it — no local model
to download, no extra install step.

### Option B — Local, via Ollama (no API key, runs on your own machine)

Free per call, but only as fast and as capable as your hardware.

1. Download Ollama for your OS from [ollama.com/download](https://ollama.com/download) and run
   the installer. On Windows this needs no admin rights — it installs to your user profile and
   starts a background service automatically (look for its icon in the system tray).
2. Confirm it installed: `ollama --version`
3. Pull a model (sizing guidance below): `ollama pull qwen3:4b` — downloads once (a few GB) and
   caches locally.
4. Sanity-check it directly, outside this project: `ollama run qwen3:4b "In one sentence, what
   is a research gap?"` — this also shows you the response speed to expect from the agent itself.
5. In `.env`, set:
   ```
   LLM_BACKEND=local
   LOCAL_MODEL_FAST=qwen3:4b
   LOCAL_MODEL_STRONG=qwen3:4b
   ```
   (`OLLAMA_HOST` already defaults to `http://localhost:11434`, which matches where Ollama
   listens — no change needed unless Ollama runs on a different machine.)

**Which model to pull, sized to your hardware** (quantized automatically by Ollama — "how big a
model" mostly means "how much RAM and how fast a CPU/GPU you have"):

| Model (Ollama tag) | Size (Q4) | Recommendation |
|---|---|---|
| `qwen3:4b` | ~2.5 GB | Default choice for a 4-core CPU laptop with no dedicated GPU (e.g. 16GB RAM, integrated graphics) — best speed/quality balance |
| `qwen3:8b` | ~5 GB | Noticeably slower on weak CPUs but higher quality — worth trying once; fall back to 4b if it feels sluggish |
| `qwen3:14b` or larger | 7 GB+ | Only if you have a discrete GPU with 8GB+ VRAM, or don't mind slow (multi-minute) responses — technically fits in 16GB RAM but too slow for an interactive demo on CPU alone |

**Hybrid tip:** the "fast" tier (keyword generation, per-paper extraction) is called many times
per run; the "strong" tier (comparison synthesis, gap analysis, verification) is called once or
twice per run but needs better reasoning. On a weak laptop, running the fast tier locally
(`LOCAL_MODEL_FAST=qwen3:4b`, free, called most often) while keeping `LLM_BACKEND=cloud` for the
one or two calls that most need good reasoning is a reasonable middle ground — though the
current `llm_client.py` switches backends per-run, not per-tier, so splitting them would be a
small follow-on code change if the team wants it (see the docx §10.6 for more on this).

## Running the agent

### Command line

```bash
python main.py --topic "few-shot named entity recognition in low-resource languages" \
                --year-from 2021 --max-papers 20 --verbose
```

Flags (all except `--topic` are optional):

| Flag | Meaning | Default |
|---|---|---|
| `--topic` | Research topic / question (**required**) | — |
| `--year-from` | Only consider papers published in/after this year | no limit |
| `--year-to` | Only consider papers published in/before this year | no limit |
| `--max-papers` | How many papers get full extraction (the expensive tier) | `MAX_DEEP_ANALYSIS_PAPERS` from `.env` |
| `--verbose` | Print progress for every pipeline stage as it runs, not just the final result | off |

Output: a Markdown report at `output/<timestamp>_report.md`, plus a full JSON state dump at
`output/<timestamp>_state.json` (useful for debugging and for the evaluation worksheet — see
[Testing](#testing-and-verifying-your-setup)).

### Web UI

```bash
streamlit run app.py
```

Opens the same pipeline in a browser tab, with a topic box, live per-step progress, and results
rendered as readable cards instead of raw Markdown. See the next section for how to use it.

## How to use the web UI

1. **Sidebar — settings** (all take effect immediately, no restart needed):
   - **LLM backend**: switch between cloud and local without touching `.env`.
   - **Anthropic API key** (cloud) or **Ollama host** + **"Enable model 'thinking' mode"**
     (local) — the thinking toggle is off by default for speed; see `OLLAMA_THINK` above for why.
   - **Papers to deep-analyze**: the same budget as `--max-papers` on the CLI.
   - **Data sources**: which of semantic_scholar / arxiv / openalex to query.
   - **Fetch full-text PDFs**: on by default — see [Full-text PDF extraction](#full-text-pdf-extraction).
2. **Enter a topic**, optionally narrow the year range, and click **Run agent**.
3. **Watch the step cards** — one per pipeline node (see [Architecture](#architecture-in-brief)).
   Each card shows a status badge, what went into that step, and — once it finishes — a
   readable summary of what came out (queries generated, papers found, the actual gap claims
   with confidence scores, etc.). The per-paper extraction step additionally shows a small live
   checklist (one line per paper) since that step is naturally the longest-running one.
4. **The full run log stays visible below the cards**, live, for the whole run — this is the
   unfiltered detail (every LLM call, every search query, every retry) behind the cards above
   it. It's there by design, not hidden behind a click, so you can always see exactly what the
   agent is doing; there's also a download button for it.
5. **Once the run finishes**, scroll down for: the list of papers actually analyzed (with
   similarity scores), the cross-paper comparison table, the trend synthesis, and the candidate
   Research Gaps — each with a color-coded confidence level (green/orange/red for
   High/Medium/Low), the rationale, which papers support it, and the verifier's notes. A
   **"Download full report (.md)"** button at the bottom exports everything as one Markdown file
   — the same content the CLI writes to `output/`.
6. A run also writes its state to `output/<timestamp>_state.json` in the background, same as the
   CLI — useful if you want to inspect the raw data later, e.g. for the evaluation worksheet.

## Full-text PDF extraction

By default, Node 5 (`nodes.extract_paper_data`) downloads each selected paper's open-access PDF
(when one is available — the `oa_pdf_url` field the search sources return) and converts it to
Markdown before extraction, instead of using only the abstract. This gets extraction closer to
what a human reading the paper would pick up (real methods/results/limitations sections, not
just the summary), and Markdown specifically — not raw PDF text — matters for quality: naive PDF
text extraction interleaves repeated page headers/footers and multi-column reading-order glitches
into the middle of sentences, which both wastes the model's attention and can make it misread the
paper's content.

This roughly doubles Node 5's time per paper (a PDF download + parse on top of the LLM call), so
it can be turned off for a faster, original-behavior run:
- Web UI: uncheck **"Fetch full-text PDFs"** in the sidebar.
- CLI/env: set `FETCH_FULL_TEXT=false` in `.env`.

Downloaded/converted text is cached in `pdf_cache/` (one file per paper id), so re-running the
same topic — or resuming after a crash mid-run — doesn't re-download and re-convert the same
PDFs. Delete that folder any time to force a fresh fetch. A paper with no open-access PDF, or
whose PDF download fails for any reason (404, paywall page, oversized file), silently falls back
to its abstract, same as before this feature existed — this never aborts a run.

## Testing and verifying your setup

There's no automated test suite in this scaffold (a course-scope call — see "What to build
next" if you want to add one), so "testing" here means two different things: **confirming your
install works**, and **judging whether the agent's actual output is good**.

### 1. Quick smoke test (do this right after installing)

From `ai_research_gap_agent/`, with the virtual environment active:

```bash
python main.py --topic "few-shot named entity recognition" --max-papers 10 --verbose
```

A working install prints progress through all 7 pipeline stages (query generation → search →
ranking → sufficiency check → extraction → comparison → gap analysis) and ends with a `Done.`
line naming the two output files. If it fails, the `--verbose` output plus
[Troubleshooting](#troubleshooting) below should point at the cause — the error is almost always
either "no LLM backend configured" (§[Choosing a model backend](#choosing-a-model-backend)) or a
missing dependency (re-run `pip install -r requirements.txt` with the venv active).

For the UI version of the same check, run `streamlit run app.py`, enter any topic, click **Run
agent**, and confirm the step cards progress through all 7 nodes to a "done" state with results
rendered at the bottom.

### 2. Judging output quality

This is the harder, more important kind of testing — a run "succeeding" (no crash) doesn't mean
the *gap it proposed is actually good*. Use `evaluation/eval_plan.md` (a fill-in-the-blanks
worksheet — full rationale in the docx §8) to check, across a handful of test topics:

- **Retrieval quality** — did search actually find papers you already know are relevant (hand-
  curate a small "gold list" per topic first)?
- **Extraction accuracy** — read 5–10 papers yourself and compare against what `paper_extracts`
  in the saved `output/*_state.json` says the method/dataset/findings were.
- **Gap quality** — rate each proposed gap 1–5 on novelty, evidence-groundedness, and
  actionability.
- **Confidence calibration** — do gaps scored "High" actually hold up better under your own
  manual review than ones scored "Low"? If not, the scoring formula in `nodes.analyze_gaps`
  needs retuning (there's a `TODO` marking exactly where).

Run this against at least 2–3 different topics — a system that only works on one cherry-picked
topic isn't validated yet.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| PowerShell: `running scripts is disabled on this system` (step 2 of install) | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then retry `.venv\Scripts\activate` |
| `ModuleNotFoundError: No module named '...'` | Either the venv isn't active (re-run step 2) or dependencies weren't installed (re-run step 3) |
| `ANTHROPIC_API_KEY is not set` | You're on `LLM_BACKEND=cloud` with no key in `.env` — fill it in, or switch to `LLM_BACKEND=local` |
| Connection refused to `localhost:11434` | Ollama isn't running — open the Ollama app, or check its system-tray icon; confirm with `ollama --version` |
| Local responses are very slow (multi-minute per call) | Expected on CPU-only hardware with a large model — try a smaller model (`qwen3:4b`), confirm `OLLAMA_THINK=false`, and see the hardware-sizing table above |
| A run finds 0 or very few papers | Topic is too narrow, or the year range excludes everything relevant — try a broader topic or widen/remove `--year-from`/`--year-to` |
| `[source] FAILED ... 429` in the log for one source | That source rate-limited you — this is expected and handled gracefully (the run continues with the other sources); add that source's API key in `.env` if it happens often |
| Extraction confidence is always `abstract_only`, never `full_text` | Either `FETCH_FULL_TEXT=false`, or these particular papers have no open-access PDF — check the log for "No usable PDF" lines; delete `pdf_cache/` to force a retry if you changed a source's rate limits or added a key |
| `streamlit run app.py` says the port is already in use | Another instance is still running — close it, or run `streamlit run app.py --server.port 8502` |
| First run pauses for a while with no visible progress before the first search | Expected once — `sentence-transformers` downloads its ~130MB embedding model the first time ranking actually runs |

## Project layout

```
config.py            # all tunables: max_papers, thresholds, model tiers, source toggles
llm_client.py         # hybrid cloud/local LLM wrapper (single call()/call_json() used everywhere)
embeddings.py          # local embedding + cosine-similarity ranking
state.py               # the shared state schema (TypedDict) that flows through the graph
prompts.py              # every LLM prompt template, kept in one place for easy iteration
nodes.py                 # the 7 graph nodes (see "Architecture, in brief" above)
graph.py                  # wires nodes.py into a LangGraph StateGraph with the retry loop
logging_utils.py          # shared logging setup used by both the CLI (--verbose) and the web UI
sources/
  semantic_scholar.py      # search_papers / get_paper / get_full_text
  arxiv.py                  # same interface, arXiv backend
  openalex.py                # same interface, OpenAlex backend
  pdf_extract.py              # shared PDF -> Markdown full-text fetcher, used by Node 5
app.py                        # Streamlit web UI
mcp_server.py                  # exposes sources/* as MCP tools
main.py                         # CLI entrypoint
evaluation/eval_plan.md          # the evaluation methodology worksheet (see docx §8)
```

## Run the MCP server standalone

`mcp_server.py` exposes the same search tools over MCP, so you can also point Claude Desktop
or Claude Code at this project for manual literature scanning, independent of the agent graph:

```bash
python mcp_server.py
```

Then add it to your MCP client config (see the MCP docs for your client) pointing at this
command. Useful for debugging a connector by hand before trusting the agent to call it.

## What to build next

Team split — see architecture doc §6.

1. **Member A** — tune `config.SIMILARITY_THRESHOLD` and the Node 3 ranking logic against real
   searches; test the Node 4 retry loop actually broadens results on a deliberately narrow topic.
   (The full-text PDF connector is done — see `sources/pdf_extract.py` and
   [Full-text PDF extraction](#full-text-pdf-extraction) above — but it only covers open-access
   PDFs; wiring in Unpaywall's API to resolve more DOIs to OA copies would raise the full-text
   hit rate further.)
2. **Member B** — iterate the extraction schema/prompt in `prompts.py` (`EXTRACT_PROMPT`) against
   papers you've read yourself; tighten `nodes.compare_papers` if the trend narrative drifts
   from the table.
3. **Member C** — iterate `GAP_ANALYSIS_PROMPT` and `VERIFIER_PROMPT`; the confidence scoring
   formula in `nodes.analyze_gaps` is a working first pass, not a validated one — run the
   calibration check in [Testing](#testing-and-verifying-your-setup) and retune the weights (marked
   with a `TODO` in the code) before treating the scores as trustworthy.
