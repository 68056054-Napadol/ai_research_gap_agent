# AI Research Assistant Agent

Companion code for `AI_Research_Assistant_Agent_Architecture.docx` (see that document for full
design rationale — §10 there covers the same install/run/use/test ground as this README, in
more depth, alongside the full architecture writeup).

Given a research topic, this agent searches academic sources, filters to the relevant papers,
extracts structured facts from each (optionally reading the full PDF, not just the abstract),
compares them, and proposes **Potential Research Gaps** with the evidence behind each one, an evidence
level (Strong / Moderate / Limited) and the limits of the search that produced them. It does not
claim to *discover* gaps: everything is proposed from the papers this run found and analysed, which
is not a systematic literature review, so every gap should be checked by a person.

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
   - [Evidence levels](#evidence-levels-strong--moderate--limited)
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
15. [Tool calling (the agent harness)](#tool-calling-the-agent-harness)
16. [Privacy: LLM extraction + decision model](#privacy-llm-extraction--decision-model)
17. [Contracts and deterministic sampling](#contracts-and-deterministic-sampling)
18. [Evaluation scripts and offline tests](#evaluation-scripts-and-offline-tests)
19. [RAG knowledge base (KM)](#rag-knowledge-base-km)
20. [Paper links in the answer](#paper-links-in-the-answer)
21. [Tracing with Langfuse + outbound contract](#tracing-with-langfuse--outbound-contract)
22. [What to build next](#what-to-build-next)

## Architecture, in brief

One LangGraph graph, nine nodes, two conditional edges, and two **tool-calling agents** inside it
(node 2 and node 7 — the LLM itself chooses tool name + arguments, see
[Tool calling](#tool-calling-the-agent-harness)). Full rationale for the original seven nodes is in
the docx (§1–§9); this is just the map:

| # | Node (`nodes.py`) | What it does |
|---|---|---|
| 0 | `sanitize_input` | Privacy check on the typed topic: regex + LLM entity extraction, then the **decision model** decides REDACT/KEEP per entity and PROCEED / PROCEED_REDACTED / REJECT for the input (rejected → END) |
| 1 | `generate_keywords` | Turns the topic into 3–5 varied search queries (LLM, "fast" tier) |
| 2 | `search_sources` | **Search agent**: the LLM calls the `search_papers` tool (query, source, year range, limit) as many times as it decides, sees each result, adapts; candidates accumulate across retry loops |
| 3 | `filter_and_rank` | Ranks candidates by embedding cosine-similarity to the topic; keeps the top N above a threshold |
| 4 | `check_sufficiency` | No LLM call — if too few papers passed the threshold, loops back to node 1 with feedback (capped at `MAX_RETRY_LOOPS`) |
| 5 | `extract_paper_data` | Per paper: fetches the open-access PDF and converts it to Markdown if available (else uses the abstract), then extracts problem/method/dataset/findings/limitations/metrics (LLM, "fast" tier) |
| 6 | `compare_papers` | Builds a comparison table across all extracted papers, then synthesizes cross-paper trends (LLM, "strong" tier) |
| 7 | `analyze_gaps` | Proposes 2–4 Potential Research Gaps grounded in the table, drops cited ids that don't exist, then a **verifier agent** reads each cited paper's source text through the `get_paper_text` tool and returns a contract-validated verdict → confidence score |
| 8 | `privacy_guard` | Privacy check on the **final answer** (trend summary + gaps): regex + author list + LLM extraction → decision model → redaction |

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
- Reads: `search_queries` (list[str], as suggestions); `year_from`/`year_to`; `CONFIG.enabled_sources`; existing `candidate_papers` (on a retry)
- Calls: **tool-calling loop** (`llm_client.run_tool_loop`, tier `fast`, `SEARCH_AGENT_PROMPT`) — the LLM issues `search_papers(query, source, year_from, year_to, limit)` calls; each is validated (`contracts.SearchPapersArgs`), policy-checked, executed and its result returned to the LLM as context. Every returned paper goes through `privacy.scrub_paper` (author names dropped, names in title/abstract redacted)
- Writes: `candidate_papers` (list of `Paper` dicts — `id`, `source`, `title`, `n_authors`, `year`, `abstract`, `doi`, `oa_pdf_url`, `citation_count` — deduped, accumulated across retries, capped at `MAX_CANDIDATE_PAPERS`); `search_summary`; `search_used_fallback`; appends to `tool_trace` and `privacy_log`

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
- Writes: `paper_extracts` (list of `PaperExtract` — `paper_id`, `problem`, `method`, `dataset`, `key_findings`, `limitations_stated`, `metrics`, `extraction_confidence` = `"full_text"` or `"abstract_only"`, `supporting_quote`, `quote_verified`)
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
- Writes: `gap_candidates` (list of `GapCandidate` — `claim`, `supporting_paper_ids`, `rationale`, `evidence_level` Strong/Moderate/Limited with its counts in `evidence` (see [Evidence levels](#evidence-levels-strong--moderate--limited)), legacy `confidence_score` 0–100, `verifier_notes`)

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

**Now done — source-grounded verification:** the verifier is a tool-calling agent. It is given
the extracted summaries *and* the `get_paper_text` tool, is told to read every cited paper's
source text before deciding, and its confidence score only gets full verifier points if it really
read at least half of the cited sources (`evidence_coverage`, shown next to every gap). Cited ids
that are not among the analyzed papers are dropped before verification (`dropped_ids`) — they
used to count as evidence. The original note, kept for context: node 7's verifier used to only
see `paper_extracts`, never the original source text. **Source-grounded verification** — giving
the verifier the actual abstract/full text for each cited paper, not just its summary — would let
it catch reasoning drift introduced *during* gap proposal against the real source, the same way
extraction-quote grounding now catches drift introduced during extraction. It's a heavier lift
(more tokens per verification call, since it's reading source text instead of a short structured
summary) and wasn't implemented here, in part because `full_text` isn't reliably available for
every paper (only open-access PDFs get fetched — see [Full-text PDF extraction](#full-text-pdf-extraction)),
so for abstract-only papers it would mostly re-check the abstract against itself. Flagging it as
the next step if the team wants to strengthen this further before a demo.

## Evidence levels (Strong / Moderate / Limited)

Every Potential Research Gap gets an **evidence level** decided by code from facts anyone can
count in `output/*_state.json` — never from a number the LLM gives itself. The criteria were
written down (here and in `config.py`) before looking at results; change them only in
`config.py` / `.env`, and say so when you report results.

The counts shown under every gap (report, web UI and Dashboard — `evidence.gap_evidence`):

| Count | Where it comes from |
|---|---|
| supporting papers | `supporting_paper_ids` that really are among the analysed papers (made-up ids are dropped first) |
| …of which full text | supporting papers whose extraction read the full text (`paper_extracts[].extraction_confidence == "full_text"`) |
| contradicting papers | a counterexample the verifier found **and** whose quote was found verbatim in that paper's text (`counterexample_verified`) |
| evidence years | min–max `year` of the supporting papers |
| verifier read the sources | `evidence_coverage` = cited papers the verifier actually read through its tools ÷ cited papers |

The rule (`evidence.evidence_level`, first match wins):

| Level | Thai | Condition (defaults) |
|---|---|---|
| **Limited** | หลักฐานจำกัด | verifier did not support the claim, **or** ≥ 1 contradicting paper, **or** < `EVIDENCE_MODERATE_MIN_SUPPORTING` (2) supporting papers, **or** verifier read < `EVIDENCE_MODERATE_MIN_COVERAGE` (50 %) of them |
| **Strong** | หลักฐานมาก | not Limited **and** ≥ `EVIDENCE_STRONG_MIN_SUPPORTING` (3) supporting papers **and** ≥ `EVIDENCE_STRONG_MIN_FULL_TEXT` (2) of them read in full text **and** verifier read ≥ `EVIDENCE_STRONG_MIN_COVERAGE` (100 %) **and** newest supporting paper ≤ `EVIDENCE_STRONG_MAX_AGE_YEARS` (3) years old **and** no cap below |
| **Moderate** | หลักฐานปานกลาง | everything else |

The ceilings from `nodes._confidence_caps` are kept: when fewer than `HIGH_CONFIDENCE_MIN_PAPERS`
(8) papers were analysed, when fewer than `HIGH_CONFIDENCE_MIN_SOURCES` (2) sources answered, or
when the verifier named a counterexample it could not quote, a gap cannot be Strong. Each gap
also lists the criteria that kept it from the next level up (`evidence_level_reasons`).

The old 0–100 score (evidence 40 + recency 30 + verifier 30) is still computed and stored in
`confidence_score` so new runs can be compared with old ones, but it is not shown as the answer.
`confidence_level` now carries the same value as `evidence_level`. For runs saved before evidence
levels existed (High/Medium/Low only), the Dashboard rebuilds the counts from what those runs
stored and applies the same rule.

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
| `SEMANTIC_SCHOLAR_API_KEY` | *(empty)* | Free key ([request](https://www.semanticscholar.org/product/api#api-key)) — 1 request/s of your own; without it every call shares one public pool and is often refused (429) |
| `OPENALEX_API_KEY` | *(empty)* | Free key ([get one](https://openalex.org/settings/api)) — **required by OpenAlex since Feb 2026**; without it the daily allowance is tiny |
| `OPENALEX_MAILTO` | *(empty)* | Optional, ignored by OpenAlex since keys replaced the polite pool. Placeholder values such as `you@example.com` are dropped |
| `SOURCE_MAX_RETRIES` / `SOURCE_RETRY_MAX_WAIT` / `SOURCE_DISABLE_AFTER_FAILURES` | `3` / `10` / `3` | Retries on 429/5xx (Retry-After honoured), and how many failed calls in a row switch a source off for the rest of the run |
| `RANKING_EMBEDDING` | *(= `RAG_EMBEDDING`)* | Embedding Node 3 ranks candidates with |
| `SIMILARITY_THRESHOLD` | *(per model)* | Override the relevance cut-off; defaults per model in `embeddings.RANKING_THRESHOLDS` |
| `HIGH_CONFIDENCE_MIN_PAPERS` / `HIGH_CONFIDENCE_MIN_SOURCES` | `8` / `2` | A gap can only be **High** when the run analysed this many papers from this many sources that answered |
| `PRIVACY_LLM_ON_PAPERS` | `always` | `always` / `auto` / `never` — see [Privacy](#privacy-llm-extraction--decision-model) |
| `UNPAYWALL_EMAIL` | *(empty)* | Reserved for a future Unpaywall connector — see "What to build next" |
| `MAX_CANDIDATE_PAPERS` | `80` | Broad search pool size (metadata only, cheap) |
| `MAX_DEEP_ANALYSIS_PAPERS` | `8` | How many papers get full extraction (~45 s each on a local 14B model) (the expensive tier) — overridable per-run via `--max-papers` or the web UI's slider |
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

Run `python -m pytest -q tests` first: 23 offline tests (fictional fake papers + a scripted LLM,
no network or API key) covering tool calling, contracts, the decision model and privacy. Beyond
that, "testing" means two things: **confirming your install works**, and **judging whether the
agent's actual output is good**.

### 1. Quick smoke test (do this right after installing)

From `ai_research_gap_agent/`, with the virtual environment active:

```bash
python main.py --topic "few-shot named entity recognition" --max-papers 10 --verbose
```

A working install prints progress through all pipeline stages (input privacy check → query generation →
search agent → ranking → sufficiency check → extraction → comparison → gap analysis → privacy guard) and ends with a `Done.`
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
nodes.py                 # the 9 graph nodes (see "Architecture, in brief" above)
graph.py                  # wires nodes.py into a LangGraph StateGraph (input gate + retry loop)
tools.py                  # tool definitions + executors (search_papers, get_paper_text, search_paper_chunks, search_corpus)
rag_store.py              # persistent RAG knowledge base: section chunking, SQLite+numpy store, extraction cache
links.py                  # paper links from DOI/arXiv/OpenAlex/S2 metadata (never from the LLM)
tracing.py                # optional Langfuse tracing (no-op without keys)
outbound.py               # outbound contract: no PII in traces, saved state or reports
contracts.py              # pydantic contracts for every LLM output and tool argument
privacy.py                # regex / metadata / LLM detectors + redaction
decision_model/           # sensitive_info.dmn.json decision tables + evaluator
metrics.py                # per-run counters (contracts, tool calls, tokens)
tests/                    # offline tests with fictional fake papers + scripted LLM
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
evaluation/consistency_test.py   # same question N times -> similarity of answers
evaluation/compare_models.py     # open-source (Ollama) vs commercial (Claude) on the same topics
```

## Run the MCP server standalone

`mcp_server.py` exposes the same search tools over MCP, so you can also point Claude Desktop
or Claude Code at this project for manual literature scanning, independent of the agent graph:

```bash
python mcp_server.py
```

Then add it to your MCP client config (see the MCP docs for your client) pointing at this
command. Useful for debugging a connector by hand before trusting the agent to call it.

## Tool calling (the agent harness)

`llm_client.run_tool_loop()` is a native tool-use loop for both backends — Anthropic Messages API
`tools=[...]` with `tool_use`/`tool_result` blocks, and Ollama `/api/chat` `tools=[...]` with
`message.tool_calls`. Tools are defined in `tools.py`; their JSON schemas are generated from the
argument contracts in `contracts.py`.

| Agent | Tool | Arguments the LLM chooses | Context returned to the LLM |
|---|---|---|---|
| search agent (node 2) | `search_papers` | `query`, `source` (semantic_scholar / arxiv / openalex / all), `year_from`, `year_to`, `limit` | ids, titles, years, citation counts, has_pdf, how many new papers, calls left |
| verifier agent (node 7) | `get_paper_text` | `paper_id`, `max_chars` | the paper's opening text (full-text excerpt or abstract), already privacy-redacted |
| verifier agent (node 7) | `search_paper_chunks` | `paper_id`, `query`, `k` | the k passages of that paper most relevant to the query, with their section (RAG) |
| verifier agent (node 7) | `search_corpus` | `query`, `k`, `scope` (this_run / knowledge_base) | passages across all analyzed papers (or everything stored) — used to look for counterexamples to "nobody has done X" claims |

The harness, not the LLM, enforces policy on every call: arguments must pass the contract (a bad
call returns an error the LLM can read and fix), only sources enabled in the sidebar are allowed,
years are clamped to the user's range, and there is a call budget (`MAX_SEARCH_TOOL_CALLS`).
If a model never calls a tool (some small local models), the harness runs the suggested queries
itself and labels those entries `harness_fallback` — so the trace never pretends the LLM chose them.

**Where to point during the demo** (web UI, after a run):
- **🔧 Tool calls — context returned from tools**: one row per call (agent, tool name, arguments);
  expand a row to see the exact JSON context that went back to the LLM.
- **✅ Final answer — candidate research gaps**: the system's answer; each gap shows how many of its
  cited sources the verifier read via the tool.
- The live log shows the same thing as `TOOL CALL search_papers({...})` / `TOOL RESULT ...` lines.

## Privacy: LLM extraction + decision model

The LLM only **extracts** candidate sensitive entities (`prompts.SENSITIVE_EXTRACT_PROMPT`,
validated by `contracts.SensitiveEntitiesOutput`): text, type (PERSON, EMAIL, PHONE, ID_NUMBER,
ADDRESS, ORGANIZATION), role (author, cited_author, subject_person, eponym, affiliation…) and
confidence. Two non-LLM detectors run alongside it: regex (emails, phones, 13-digit IDs,
"Surname et al." citations) and metadata (the paper's own author list from the source API, kept
in memory only). The **decision model** (`decision_model/sensitive_info.dmn.json`, evaluated by
`decision_model/__init__.py`, hit policy FIRST) then **decides**:

- table `entity_action` — per entity, REDACT (with a placeholder like `[AUTHOR]`) or KEEP. E.g.
  rule R2 keeps an *eponym* ("Kalman" in "Kalman filter" is vocabulary, not a person), rule R5 uses
  a lower confidence bar for what the user typed than rule R6 does for paper text.
- table `input_action` — for the topic: PROCEED, PROCEED_REDACTED (person removed, research part
  kept) or REJECT (nothing but a person lookup left, e.g. "งานวิจัยของ ดร. …").

It runs on three channels: the user's topic (node 0), every title/abstract/PDF text before any LLM
or the UI sees it (search tool + node 5), and the final answer (node 8). Author names are never
stored in state, reports, or `output/*.json` — papers keep only `n_authors`. Every decision is
logged with the rule that fired and a masked form of the text (`J*** D**`), shown in the UI under
**🛡️ Privacy decisions**. Try it live: `few-shot NER papers by Dr. <any name> for Thai`.

Paper headers also get a deterministic **affiliation detector** (`privacy.affiliation_detect`):
header lines naming an institution, and short lines ending in a country, are redacted without an
LLM. `PRIVACY_LLM_ON_PAPERS=auto` would skip the header LLM call when these detectors plus the
author list seem to cover everything — but `scripts/validate_header_privacy.py` on 12 real PDFs
showed it skipping only 4/12 headers (57 s saved) while leaving 16 items the LLM would have
redacted (2 author names among them). So the default stays `always`; `auto` is there for runs
where speed matters more than a few leaked citation names.

## Contracts and deterministic sampling

- `contracts.py`: one pydantic model per LLM output (keywords, extraction, gap proposal, verifier,
  sensitive entities) and per tool's arguments. `llm_client.call_json(..., schema=...)` re-asks
  the model with the exact validation error when JSON is valid but breaks the contract.
- `LLM_TEMPERATURE=0` and `LLM_SEED=42` (both backends) — the same question gives the same
  substance. Measured by `evaluation/consistency_test.py`.
- `metrics.py` counts contract violations, retries, tool calls, invalid tool arguments and tokens
  per run (UI: *Run metrics* expander; `run_metrics` in the state JSON).

## Evaluation scripts and offline tests

```bash
python -m pytest -q tests                      # 23 offline tests, no network/API key (fictional data)
python evaluation/consistency_test.py --runs 5 --max-papers 6          # same question x5
python evaluation/compare_models.py --runs 2 --max-papers 5            # Claude vs Ollama (qwen3)
```

Both scripts write Markdown + JSON to `evaluation/results/`. `--offline` runs them on the fake
data in `tests/fakes.py` — only to check the script works, the numbers are not an evaluation.

## RAG knowledge base (KM)

Two tiers, so the expensive work is only done where it pays off:

| Tier | Which papers | What happens |
|---|---|---|
| 1 — screening | every candidate the search agent found (30–80) | title + abstract embedding, ranked in node 3 (unchanged) |
| 2 — knowledge base | only the papers selected for deep analysis | full text (redacted) is split by section (abstract / method / experiments / results / limitations / conclusion; references are never indexed), embedded and stored in `rag_store/store.sqlite` |

What the knowledge base changes:
- **Extraction (node 5)** no longer reads only the first 12,000 characters (mostly intro + related
  work). It reads the paper's opening plus the chunks retrieved for *method*, *dataset*, *results*
  and *limitations* — so the results and limitations sections near the end actually get read.
- **Verifier (node 7)** gets `search_paper_chunks` and `search_corpus` (see the tool table above)
  and is told to search for counterexamples before accepting an "X is unexplored" claim; the
  queries it ran are shown under each gap.
- **It persists (this is the KM part):** a paper indexed once is never downloaded, converted,
  redacted or embedded again; an extraction with the same text + prompt + model is reused instead
  of calling the LLM. `search_corpus(scope="knowledge_base")` searches everything earlier runs
  stored. The UI's *📚 Knowledge base* panel shows papers stored / new / reused per run.

Storage is SQLite + numpy (exact cosine search; milliseconds for thousands of chunks) — no vector-DB
dependency. Only privacy-redacted text is stored, and `pdf_cache/` is overwritten with the
redacted text too. Both folders are git-ignored: never commit them (real paper text).
`RAG_EMBEDDING=ollama:bge-m3` switches to a multilingual embedding model (Thai included) running
in Ollama; changing the model re-embeds papers lazily. Delete `rag_store/` to start from empty —
worth doing once in a demo, so the examiner sees a cold run as well as a warm one.

## Paper links in the answer

The LLM only ever cites `paper_id`s. After verification, `links.py` builds each supporting paper's
link from its metadata — `https://doi.org/<doi>` when a DOI exists, otherwise the arXiv / OpenAlex /
Semantic Scholar page — and drops any id that is not among the analyzed papers. An LLM cannot put a
link into the answer, so it cannot invent one (a live run already produced a mistyped OpenAlex id,
which was dropped).

## Tracing with Langfuse + outbound contract

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL` in `.env` (self-hosted
Langfuse, e.g. `http://localhost:3000`, or Langfuse Cloud) and every run becomes one trace:

| Observation | Records |
|---|---|
| `research_gap_agent` (agent) | redacted topic, final answer (gaps + links), run metrics |
| `node:<name>` (chain) | each graph node's input keys and output summary |
| `llm:fast` / `llm:strong` (generation) | system prompt + user prompt, output, token usage, model, temperature; **reasoning** when `OLLAMA_THINK=true` |
| `llm:tool_turn:<agent>` (generation) | system prompt, conversation so far, tool list → assistant text + tool calls (name, arguments) |
| `tool:<name>` (tool) | arguments the LLM chose → the exact context returned to it |

Nothing is traced when the keys are missing (tracing.py is a no-op), so the agent never depends
on Langfuse being up.

**Outbound contract** (`outbound.py`): nothing that leaves the process — trace, `output/*_state.json`,
report — may contain personal data. Three layers: pre-tool (placeholders stripped from tool
arguments), post-tool (search results scrubbed before the LLM sees them) and this boundary check,
which Langfuse runs as its `mask` hook on every input/output/metadata and `save_outputs` runs on
every file. The two spans that must look at raw personal data (the typed topic, PDF headers being
scanned for names) record `[withheld: …]` instead of their content. `tests/test_pipeline.py`
exports a full run's spans to memory and asserts no name, e-mail or raw topic appears anywhere.

## Choosing models (measured on this project)

Hardware: RTX 5070 Ti 16 GB, Ollama, temperature 0. Raw results: `evaluation/results/`.

**Embeddings** (`evaluation/compare_embeddings.py`): 13 real papers, 577 chunks, 49 questions per
language; a question counts as a hit if a chunk from the right section is in the top 3.

| Model | hit@3 EN | hit@3 TH | TH–EN agreement |
|---|---|---|---|
| **ollama:embeddinggemma:300m** | **0.88** | **0.76** | 0.59 |
| sentence_transformers:BAAI/bge-small-en-v1.5 | 0.65 | 0.39 | 0.16 |
| ollama:bge-m3 | 0.55 | 0.59 | 0.63 |
| ollama:qwen3-embedding:0.6b | 0.51 | 0.49 | 0.60 |

→ `RAG_EMBEDDING=ollama:embeddinggemma:300m`. Each model is used with the query/document format
its model card prescribes (`embeddings.EMBEDDING_FORMATS`) - in the evaluation and the pipeline alike.

**LLMs** (`evaluation/compare_models.py`): 2 topics × 2 runs each, max 5 papers, same harness.

| | qwen3:14b | gpt-oss:20b | qwen3.5:9b |
|---|---|---|---|
| completed / failed runs | 4 / 0 | 3 / 1 (Ollama 500) | 4 / 0 |
| seconds per run (mean) | **377** | 564 | 452 |
| tokens (all runs) | **225k** | 239k | 886k |
| quote verified (extraction faithful to source) | 0.80 | 0.47 | **0.90** |
| verifier read the cited sources | **1.00** | 0.92 | 0.83 |
| contract pass on first try | 0.98 | 0.97 | 0.96 |
| tool arguments valid | 0.83 | **1.00** | **1.00** |
| same answer when asked twice | 1.00 | 1.00 | 1.00 |

→ default stays **qwen3:14b**: fastest, no failed runs, faithful extraction, cheapest in tokens.
gpt-oss:20b proposes the most specific gaps but paraphrases its "verbatim" quotes (0.47) and needs a
model profile (it cannot disable thinking and returns empty content under `format=json` - see
`llm_client.OLLAMA_PROFILES`). qwen3.5:9b extracts most faithfully but its verifier makes ~25 tool
calls per run (4× the tokens). Caveats: small sample; "gaps supported" is judged by each model's own
verifier, so it is not comparable across models and is left out of the table.

## Quality fixes from a live app run

A real run in the web UI ("automatic short answer grading with large language models") showed five
problems. What changed, and what was measured:

1. **Paper sources silently down.** Every Semantic Scholar and OpenAlex call got HTTP 429, so the
   corpus was arXiv-only and nothing said so. `sources/net.py` now throttles per source, retries
   429/5xx with backoff (Retry-After honoured), switches a source off after 3 failed calls in a row,
   and records per-source health — shown on the search card, in the report, and used in (3).
   Placeholder settings (`you@example.com`) are no longer sent. OpenAlex now requires a free API key
   (Feb 2026); the UI warns when keys are missing.
2. **Relevance cut-off never cut.** `evaluation/calibrate_ranking.py` on 136 hand-labelled
   candidates (2 topics, `evaluation/data/`): ranking order was already good for every model
   (AUC ≈ 0.99; the top 5 and top 8 were all on topic), but the 0.35 cut-off sat below the lowest
   score bge-small ever gave (0.53), so 66/66 candidates "passed". Node 3 now ranks with the RAG
   embedding (embeddinggemma, with its query/document formats) and a per-model cut-off at 95 %
   on-topic recall (0.50 → 1/57 off-topic kept). arXiv queries now AND the topic words and filter
   years server-side (the old `all:` query ORed them and pulled in genome and quantum-ML papers).
3. **Overconfident gaps.** A counterexample now counts only if the verifier copies a sentence
   from that paper's text verbatim (checked against what the tools returned); a gap lost to a
   title that "may be relevant" no longer happens. Absence claims are capped below High when the
   run analysed < 8 papers or < 2 sources answered (`confidence_caps`, shown per gap). In the
   re-run, a counterexample was accepted on a verified sentence ("…spanning various languages…").
4. **Slow privacy pass.** Measured instead of assumed — see [Privacy](#privacy-llm-extraction--decision-model):
   skipping the header LLM call leaked names, so it stays on; the deterministic affiliation
   detector was added on top.
5. **UI.** One click runs (form), default 8 papers, and the trend summary names papers by short
   title (`Grade Guard (2025)`) instead of ids or the internal "Quote verified" flag.

### Thinking mode (`OLLAMA_THINK`, `OLLAMA_THINK_PURPOSES`) — measured

Same topic ("large language models for automated code review"), 8 papers, all three sources,
extraction cache off, qwen3:14b, one run each. "Hybrid" = `OLLAMA_THINK=false`,
`OLLAMA_THINK_PURPOSES=propose-gaps,synthesize-trends` (thinking only where gaps are written).

| | thinking off | hybrid | thinking everywhere |
|---|---|---|---|
| run time | 7.0 min | **6.5 min** | 12.5 min |
| output tokens | **4.5k** | 5.2k | 18.7k |
| same 8 papers as "off" | — | 8/8 | 1/8 |
| extraction quotes found in the paper | 4/8 | 4/8 | 4/8 |
| verifier read every cited paper | yes (12 tool calls) | yes (9) | **no** — one gap read nothing |
| gaps: supported / refuted / not supported / unquoted counterexample | 3 / 0 / 0 / 1 | 1 / 1 / 1 / 0 | 3 / 0 / 0 / 0 |
| gap wording | "there is a lack of X" | specific ("CodeReviewQA's three-step decomposition…") | concrete proposals |

- **Off** is reproducible: two runs hours apart gave the same 4 gaps word for word (temperature 0,
  seed 42). Its gaps are generic and pass verification easily.
- **Hybrid** costs nothing extra in time, keeps the verifier thorough, and writes bolder, more
  specific gaps — and the verifier then rejected two of three (one refuted by a quoted sentence,
  one whose papers did not concern it). Fewer gaps survive, but the survivor is a real one.
- **Thinking everywhere** is twice as slow and made the verifier skip its tools.

One run per setting is a direction, not a verdict, so the default stays **off** and hybrid is one
line in `.env`. The thinking-everywhere run is why a verifier that reads none of the cited papers
can no longer mark a gap as supported (`nodes.analyze_gaps`), and with thinking on for a step the
reply budget in the context window grows by 6k tokens so the reasoning cannot push the paper text out.
A caveat the hybrid run exposed: the counterexample check proves the quote is in the paper, not
that it refutes the gap ("the chosen datasets encompass the entire code review process" was used
against a gap about multi-modal datasets) — the Dashboard shows the quote so a person can judge.

## Dashboard

Sidebar → **Page: 📊 Dashboard** reads every saved run in `output/` (`dashboard.py` for the data,
`dashboard_page.py` for the page) and shows:

- **KPIs** — runs, topics, papers analysed, gaps, supported / refuted, High-confidence count, mean score.
- **Gaps by category** and **confidence by verdict** charts. The gap proposer now gives every gap one
  of 8 categories (`contracts.GAP_CATEGORIES`); runs saved before that get one from keyword rules
  (marked "inferred").
- **All gaps** table with filters (topic, category, verdict, confidence) and a CSV download.
- **Gaps refuted by a paper** — the refuting paper with its link (built from metadata) and the
  verbatim sentence the verifier quoted from it.
- **Gaps that recur across different topics** — near-identical gaps proposed for unrelated topics
  (topic words removed before comparing), usually a stock answer such as "non-English settings are
  under-explored" rather than something these papers motivate.
- **Runs** — duration, model, candidates vs analysed papers, which sources answered / failed,
  tokens, privacy redactions, Langfuse trace link.
- **Papers** — analysed papers by year and by source, and the papers cited most by gaps.

Duplicate papers: the same title is only treated as the same paper when the authors match too
(order-free, initials ignored), or — when one side has no author list — when the years agree. The
author comparison uses one-way hashes (`sources.author_keys`) kept after the names are dropped.

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
