# Evaluation Plan

Full rationale in `AI_Research_Assistant_Agent_Architecture.docx` §8. This file is the
checklist/worksheet: what is measured, by which script, and **which part a person has to do**.

Teacher feedback this plan follows:

- the system proposes **Potential Research Gaps**, it does not "discover" gaps — so evaluation asks
  whether a proposed gap is real *in the papers found*, and how wide that search was;
- evidence levels must rest on countable facts (README "Evidence levels"), so they are checked
  against human judgment, not against the system's own numbers;
- every quality score is a comparison with **answers people wrote** — no script writes those
  answers. Scripts only build empty templates and compare once the group has filled them in.

Legend — **Auto**: a script measures it, no human input. **Human**: a person must read papers /
gaps and write the reference answer or rating. **Auto + Human**: the script needs a human-written
reference to compare with.

| # | Dimension | Script | Who | Human input needed |
|---|---|---|---|---|
| 1 | Retrieval & ranking | `calibrate_ranking.py` | Auto + Human | on/off-topic labels per candidate (`data/ranking_labels.json`, already done for 2 topics — add more topics) |
| 2 | Search scope & query adaptation | `eval_query_adaptation.py` | Auto (+ Human check) | read the new queries per round: sensible broadening or drift? |
| 3 | Extraction accuracy | `eval_extraction.py` | Auto + Human | fill in `data/extraction_gold_TEMPLATE.csv` (problem, method, dataset, findings) |
| 4 | Citation correctness | `eval_citations.py` | Auto (+ Human check) | check the rows the report lists as mismatches / "by title" |
| 5 | Gap quality vs human answers | — (worksheet below) | Human | reference gaps per topic + rating per proposed gap |
| 6 | Evidence-level calibration | — (worksheet below, uses §5 ratings) | Human | §5 ratings |
| 7 | Reliability, privacy & cost | `consistency_test.py`, `compare_models.py`, test suite | Auto | none |

Result files go to `evaluation/results/` (`.md` is committed, `.json` is git-ignored).

---

## 1. Retrieval & ranking — Auto + Human

Hand-label each candidate of a real run as on-topic / off-topic by title + abstract (labels in
`evaluation/data/ranking_labels.json` point at the run's state file; `ranking_set.json` is built
from it once and committed). Then:

```
python evaluation/calibrate_ranking.py --models "gemma=ollama:embeddinggemma:300m" "bge=sentence_transformers:BAAI/bge-small-en-v1.5"
```

Reports ROC AUC, the cut-off at 95 % recall, off-topic papers that pass it, and **precision@5 /
@8 / @10** per topic. Current data: 138 candidates, 2 topics
(`results/calibrate_ranking_20261119_132034.md`, before p@10 was added — re-run to get p@10).

**Human TODO:** label 1–3 more topics (ideally one per group member) so p@k is not based on 2
topics only. Optional recall check against a gold list:

| Topic | Gold papers (known relevant, chosen by a person) | Found by system | Missed | Recall |
|---|---|---|---|---|
| | | | | |

## 2. Search scope & query adaptation — Auto (+ Human check)

Every report now has a *Search scope* section (sources asked / answered / failed, all queries,
years, candidate → cut-off → selected → analysed counts, full text vs abstract, skipped papers).
`eval_query_adaptation.py` runs deliberately narrow topics and measures from `search_rounds`
whether the retry loop fired, whether later rounds used new queries, and whether the paper pool
grew.

```
python evaluation/eval_query_adaptation.py --offline        # smoke test only (fake papers)
python evaluation/eval_query_adaptation.py --max-papers 8   # real run, ~10 min per topic locally
```

**Human TODO:** for each topic, judge the new queries:

| Topic | Retry? | New queries sensible (yes / partly / drift)? | Papers increased? | Notes |
|---|---|---|---|---|
| | | | | |

## 3. Extraction accuracy — Auto + Human

1. Make a template for a run (paper ids + titles, answers empty):
   `python evaluation/eval_extraction.py --make-template output/<run>_state.json`
   (or copy `data/extraction_gold_TEMPLATE.csv`).
2. **Human TODO:** each annotator reads the papers *themselves* (full text if open access) and
   fills in `problem`, `method`, `dataset`, `findings` and their name in `annotator`. Do not look
   at the system's extraction first. At least 2 annotators on ≥ 3 shared papers gives
   inter-annotator agreement — the ceiling the system can be expected to reach.
3. `python evaluation/eval_extraction.py --gold evaluation/data/extraction_gold_<run>.csv`
   → token F1 and match rate per field, split by full text vs abstract-only extraction.

Token F1 is a proxy. Read the low-scoring rows and note real errors:

| Paper | Field | System said | Human said | Real error? |
|---|---|---|---|---|
| | | | | |

## 4. Citation correctness — Auto (+ Human check)

`python evaluation/eval_citations.py` checks the title / year / DOI of every paper a gap cites
(supporting and refuting) against OpenAlex and Crossref and reports match rates.

**Human TODO:** open each paper the report lists under "mismatches to check" and decide whether
it is a real error (wrong paper / wrong year) or a lookup artefact (preprint vs published year,
title redacted by the privacy scrub, a same-title paper found by title search).

## 5. Gap quality vs human answers — Human

The comparison with *answers people made*:

1. **Before** looking at the system's output, for each test topic a group member reads the
   analysed papers (listed in the report) and writes 2–4 gaps they see in *those* papers —
   `data/gap_reference_TEMPLATE.csv` (topic, reference_gap_id, reference_gap,
   supporting_papers, annotator).
2. Then rate every proposed gap of the run — `data/gap_rating_TEMPLATE.csv`: is it a real gap in
   these papers (yes / partly / no), does the cited evidence support it (1–5), novelty (1–5),
   actionability (1–5), do you agree with its evidence level, and which reference gap it matches
   (or none).

Summary to fill in:

| Topic | Proposed gaps | Real (yes / partly) | Matched a reference gap | Reference gaps the system missed | Mean evidence-support (1–5) |
|---|---|---|---|---|---|
| | | | | | |

## 6. Evidence-level calibration — Human (uses §5)

Do gaps the system marks **Strong** hold up better under the §5 ratings than **Moderate** and
**Limited** ones? The levels come from the pre-written criteria in `config.py` / README
"Evidence levels"; if the table below shows no difference, change the criteria in `config.py`
and say so in the report (do not tune them on the same topics you then report).

| Evidence level | Count | Share rated "real" in §5 | Mean evidence-support (1–5) | Share where raters agreed with the level |
|---|---|---|---|---|
| Strong | | | | |
| Moderate | | | | |
| Limited | | | | |

## 7. Reliability, privacy & cost — Auto

- Consistency: `python evaluation/consistency_test.py` (same topic 5×, temperature 0) —
  `results/consistency_20261118_171247.md`: gap similarity 1.0, selected-paper Jaccard 1.0.
- Model comparison (time, contract pass rate, quote-verified rate): `evaluation/compare_models.py`.
- Privacy & secrets: `python -m pytest -q tests` (no author name / e-mail / API key in state,
  report, trace or run log; decision-table rules).
- Cost & latency per run: `run_info.seconds` and `run_metrics` (llm_calls, input/output tokens)
  in each `output/*_state.json`, also on the Dashboard's Runs table. `MAX_RUN_TOKENS` caps a run.

| Topic | Papers analysed | Wall-clock time | LLM calls | Tokens (in / out) |
|---|---|---|---|---|
| | | | | |

This table is also the evidence for the paper-count budget (architecture doc §7.4).
