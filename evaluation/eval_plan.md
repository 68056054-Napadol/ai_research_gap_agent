# Evaluation Plan

Full rationale in `AI_Research_Assistant_Agent_Architecture.docx` §8. This file is the
checklist/worksheet version — fill it in as you run each check.

## 1. Retrieval quality (component-level)

Pick 3-5 test topics. For each, hand-curate a small "gold" list of papers you already know are
relevant (from coursework, or papers cited in your own project brief).

| Topic | Gold papers found by system | Gold papers missed | Irrelevant papers returned | Precision | Recall |
|---|---|---|---|---|---|
| | | | | | |

## 2. Extraction accuracy (component-level)

Pick 5-10 papers from a test run. Read them yourself, then compare against `paper_extracts` in
the saved `output/*_state.json`.

| Paper | Method match? | Dataset match? | Findings match? | Notes |
|---|---|---|---|---|
| | | | | |

## 3. Gap quality (end-to-end)

For each test topic's `gap_candidates`, rate 1-5 on each dimension (team or instructor rating):

| Topic | Gap claim | Novelty (1-5) | Evidence-groundedness (1-5) | Actionability (1-5) |
|---|---|---|---|---|
| | | | | |

## 4. Confidence calibration (end-to-end)

Do gaps scored "High" actually hold up better under your manual review (§3 above) than gaps
scored "Low"? If not, revisit the weights in `nodes.analyze_gaps` (see the TODO comment there).

| Confidence level | Avg. quality rating from §3 | Count |
|---|---|---|
| High | | |
| Medium | | |
| Low | | |

## 5. Cost & latency (operational)

Run with `--verbose` and note timing; log token usage if your LLM client library exposes it
(Anthropic's response includes `usage.input_tokens` / `usage.output_tokens` — consider adding
a running total to `llm_client.call()` once you're tracking this for real).

| Topic | Papers analyzed | Wall-clock time | Approx. LLM calls | Approx. cost (cloud only) |
|---|---|---|---|---|
| | | | | |

This table is also your evidence for the paper-count budget decision in the architecture
doc §7.4 — bring it to your instructor if asked to justify why the cap is set where it is.
