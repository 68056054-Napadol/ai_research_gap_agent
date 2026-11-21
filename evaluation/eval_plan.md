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
| 8 | Grounding: no invented papers / DOIs / results | `eval_grounding.py` | Auto + Human | label the sentences in the audit sheet (supported / partly / no) |

Order of work: §0 (run the test set once) → §1–§8 on those saved runs → fill the targets table below.

Result files go to `evaluation/results/` (`.md` is committed, `.json` is git-ignored).

---

## 0. Test set and frozen runs — do this first

**Topics** (`data/test_topics.json`, chosen before any result was seen):

| id | split | topic | years | conditions | what it tests | owner / 2nd annotator |
|---|---|---|---|---|---|---|
| D1 | dev | few-shot named entity recognition in low-resource languages | 2021–2026 | – | ranking labels exist; tune here | A / B |
| D2 | dev | automatic short answer grading with large language models | 2022–2026 | – | ranking labels exist; tune here | B / C |
| T1 | test | LLM hallucination detection | any | – | broad area → **narrow** rule; extraction; gaps | A / C |
| T2 | test | Thai named entity recognition | any | – | mostly abstract-only extraction; Thai qualifier kept | B / A |
| T3 | test | retrieval-augmented generation evaluation | 2024–2026 | – | metrics extraction; numbers in the answer | C / B |
| T4 | test | prompt injection defense for LLM agents | 2024– | – | full-text extraction; citation check | A / B |
| T5 | test | Thai legal document summarization | any | – | **too few → broaden**; nothing passed on 21 Nov → must stay honest | anyone |
| T6 | test | clinical named entity recognition with transformer models | 2020–2026 | include `clinical\|medical\|biomedical`, exclude `survey, review`, focus EHR / clinical notes | **user conditions** | C / A |

Why these: T1–T5 are the topics the group already ran on 21 Nov 2026 (`results/eval_summary_2026-11-21.md`),
so the prepared extraction-gold and gap-rating sheets carry over; T6 is the only one that exercises the
user conditions; D1–D2 already have ranking labels and are the only topics prompts/thresholds may be
tuned on. The query-adaptation script keeps its own very narrow topics (Lao legal NER etc.) for the
stop conditions. The 21 Nov runs used the old settings (3 sources, 80 candidates, 8 papers) — re-run
T1–T6 with the frozen MVP settings before annotating; keep the old runs only as a before/after comparison.

**Frozen runs:** `python evaluation/run_test_set.py` (real backend, settings in the file's
`frozen_settings`; the script warns when `.env` differs). It saves
`output/testset_<id>_<time>_state.json` + report. **Every** human task below uses these files; do
not re-run a topic between annotators (a new run can pick different papers). If a run fails or is
re-run, write down why in the table at the end.

**Blind order:** for §3 and §5, the annotator reads the papers and writes their own answer
**before** opening the system's report for that topic.

**Workload** (3 members, ~2 h each): per owned topic — ranking labels (~30 candidates, 20 min), extraction
gold for 4 papers (60 min), 2–4 reference gaps (20 min), rate the proposed gaps (15 min), grounding
audit sheet (~15 sentences, 15 min). Second annotator: extraction gold for 2 of the 4 papers + gap
ratings, for agreement.

### Targets (written before results — change only with a reason recorded here)

| # | Measure | Target | Result |
|---|---|---|---|
| 1 | precision@10 of the ranking, per test topic | ≥ 0.70 | |
| 2 | narrow rule fires on T3 and p@10 after narrowing ≥ before; broaden rule fires on T4 and the pool grows | yes / yes | |
| 3 | extraction judged correct by the annotator (method, dataset, findings), per field | ≥ 0.80 | |
| 3 | inter-annotator agreement on the same fields (token F1 between the 2 annotators) | report (ceiling) | |
| 4 | cited papers `verified` or `title_only` in the run's own citation check | ≥ 0.95, every mismatch explained | |
| 5 | proposed gaps rated real (yes / partly) | ≥ 0.60 | |
| 5 | mean "evidence supports claim" (1–5) | ≥ 3.5 | |
| 6 | share rated real: Strong ≥ Moderate ≥ Limited | ordering holds | |
| 7 | runs within `MAX_RUN_TOKENS` / `MAX_RUN_SECONDS`; time and tokens per topic reported | all | |
| 8 | invented references in the answer (DOIs / ids / "X et al." outside the analysed set) | 0 | |
| 8 | sentences rated "no" (not supported) in the grounding audit | ≤ 0.10 | |
| 8 | invented numbers confirmed by the audit | 0 | |

### Rating rubric (§5 and §8)

- **Real gap in these papers** — yes: the analysed papers show the thing is missing and none of them does it;
  partly: true but already partly done by one paper, or only for a narrower setting; no: one of the
  papers already does it, or the papers do not concern it.
- **Evidence supports claim (1–5)** — 5: every cited paper clearly backs its part; 3: the claim is
  plausible but one cited paper is only loosely related; 1: the cited papers do not support it.
- **Novelty (1–5)** — 5: not an obvious next step; 1: restates a limitation the papers already list as future work.
- **Actionability (1–5)** — 5: a student could start a project from it; 1: too vague to act on.
- **Grounding sentence** — yes: every fact in the sentence is in the cited papers' text; partly: the
  gist is there but a detail (number, qualifier, name) is not; no: a fact is not in the papers.

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
grew. Each round now records its `mode` (initial / broaden / narrow) and `verdict` (too_few /
too_broad / sufficient); T3 should show a narrow round, T4 broaden rounds (§0 targets).

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

Note before starting: in the 32 runs up to 21 Nov 2026 no gap reached Strong (0 / 98), partly
because "Strong" needed ≥ 2 answering sources, which the single-source MVP can never meet — that
default is now 1 (`HIGH_CONFIDENCE_MIN_SOURCES`). If the test set still gives no Strong gap, the table
below can only compare Moderate with Limited; say so rather than loosening the criteria on the test topics.

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
  in each `output/*_state.json`, also on the Dashboard's Runs table and in every report's *Search
  scope → Limits* line. `MAX_RUN_TOKENS` (default 250k) and `MAX_RUN_SECONDS` (default 1800) cap a
  run; `run_test_set.py` prints time and tokens per topic.

| Topic | Papers analysed | Wall-clock time | LLM calls | Tokens (in / out) |
|---|---|---|---|---|
| | | | | |

This table is also the evidence for the paper-count budget (architecture doc §7.4).

## 8. Grounding: no invented papers, DOIs or results — Auto + Human

What the run already guarantees: papers come only from the search API, cited ids not among the
analysed papers are dropped (`dropped_ids`, = `gap_ids_hallucinated` in `compare_models.py`), and every
cited paper's title / year / DOI is checked against Crossref/OpenAlex before the report
(`citation_checks`, §4). What can still be invented is in the **free text** the user reads (trend
summary, gap claims, rationales, verifier notes). `eval_grounding.py` checks it against the analysed
papers' title, abstract, full text, extraction and the comparison table:

| Check | Flags | Meaning |
|---|---|---|
| A. references | DOI / arXiv id / source id not of an analysed paper; "Name et al." | refers to a paper outside the set — **must be 0** |
| B. years | "(2019)" not the year of any analysed paper | possibly a paper outside the set |
| C. numbers | % and decimals not in the corpus | possibly an invented result |
| D. names | acronyms / CamelCase / names with digits not in the corpus | possibly an invented dataset or method |

```
python evaluation/eval_grounding.py --states output/testset_*_state.json            # automatic flags
python evaluation/eval_grounding.py --states output/testset_*_state.json --make-audit   # sheets for people
python evaluation/eval_grounding.py --audit evaluation/data/grounding_audit_*.csv     # after labelling
```

**Human TODO:** each audit sheet holds every flagged sentence plus 10 random unflagged ones. Open the
cited papers (the `cited_papers` column) and label each sentence yes / partly / no with the rubric in
§0, noting what is not supported. The score reports the unsupported rate, the rate among flagged vs
unflagged sentences (do the flags find the problems?) and how many unsupported sentences the flags
missed — if that number is high, the automatic check is not enough on its own and the report must say so.

| Topic | Sentences audited | Flagged | Rated "no" | "no" among unflagged | Invented references | Invented numbers |
|---|---|---|---|---|---|---|
| | | | | | | |

## Runs used

| Topic | state file | backend / models | settings differing from frozen | notes (re-runs, failures) |
|---|---|---|---|---|
| | | | | |
