"""
Extraction accuracy against answers written by PEOPLE (teacher feedback: evaluation must compare
with a human reference, not with the system's own opinion).

Group members fill in a copy of evaluation/data/extraction_gold_TEMPLATE.csv after reading each
paper themselves (one row per paper per annotator):

    paper_id,title,problem,method,dataset,findings,annotator

This script never writes answers. It only compares what people wrote with `paper_extracts` in the
saved runs (output/*_state.json; the newest run that analysed a paper wins) and reports, per field
(problem / method / dataset / findings):

  - token F1 between the system's text and the human's (SQuAD-style: lower-case, punctuation and
    articles removed). A proxy for "says the same thing" - read the low-scoring rows yourself.
  - match rate: share of papers with F1 >= --match-f1 (default 0.5), and "not stated" agreement
    (both sides say the paper does not state it)
  - inter-annotator F1 when two or more people annotated the same paper (the ceiling the system
    can be expected to reach)

    # optional: a template pre-filled with the paper ids + titles of one run (answers left empty)
    python evaluation/eval_extraction.py --make-template output/20261120_011606_state.json
    # after the group has filled it in
    python evaluation/eval_extraction.py --gold evaluation/data/extraction_gold.csv
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS_DIR, ROOT, save  # noqa: E402

FIELDS = {"problem": "problem", "method": "method", "dataset": "dataset", "findings": "key_findings"}
TEMPLATE_HEADER = ["paper_id", "title", "problem", "method", "dataset", "findings", "annotator"]
_NOT_STATED = {"", "not stated", "none", "n/a", "na", "-", "ไม่ระบุ"}


def normalize(text: str) -> list:
    text = (text or "").lower()
    text = "".join(" " if ch in string.punctuation else ch for ch in text)
    return [w for w in text.split() if w not in ("a", "an", "the")]


def is_not_stated(text: str) -> bool:
    return (text or "").strip().lower().rstrip(".") in _NOT_STATED


def token_f1(pred: str, gold: str) -> float:
    if is_not_stated(pred) and is_not_stated(gold):
        return 1.0
    if is_not_stated(pred) or is_not_stated(gold):
        return 0.0
    p, g = normalize(pred), normalize(gold)
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if not same:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def load_gold(paths: list) -> list:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
                if r.get("paper_id") and any(r.get(k) for k in FIELDS):
                    r["annotator"] = r.get("annotator") or os.path.basename(path)
                    rows.append(r)
    return rows


def load_extracts(state_glob: str) -> dict:
    """{paper_id: (run_id, extract)} - newest run first, so a later re-extraction wins."""
    out = {}
    for path in sorted(glob.glob(state_glob), reverse=True):
        try:
            state = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        run_id = os.path.basename(path).replace("_state.json", "")
        for ex in state.get("paper_extracts") or []:
            out.setdefault(ex.get("paper_id"), (run_id, ex))
    return out


def evaluate(gold: list, extracts: dict, match_f1: float = 0.5) -> dict:
    per_paper, missing = [], []
    for g in gold:
        hit = extracts.get(g["paper_id"])
        if not hit:
            missing.append(g["paper_id"])
            continue
        run_id, ex = hit
        row = {"paper_id": g["paper_id"], "annotator": g["annotator"], "run_id": run_id,
               "extraction_confidence": ex.get("extraction_confidence"), "quote_verified": ex.get("quote_verified")}
        for gold_key, sys_key in FIELDS.items():
            if not g.get(gold_key):
                continue          # the annotator left this field empty - not scored
            row[f"{gold_key}_f1"] = round(token_f1(ex.get(sys_key, ""), g[gold_key]), 3)
            row[f"{gold_key}_system"] = ex.get(sys_key, "")
            row[f"{gold_key}_human"] = g[gold_key]
        per_paper.append(row)

    fields = {}
    for k in FIELDS:
        scores = [r[f"{k}_f1"] for r in per_paper if f"{k}_f1" in r]
        if scores:
            fields[k] = {"n": len(scores), "mean_f1": round(sum(scores) / len(scores), 3),
                         "match_rate": round(sum(1 for s in scores if s >= match_f1) / len(scores), 3)}
    by_mode = {}
    for mode in ("full_text", "abstract_only"):
        rows = [r for r in per_paper if r["extraction_confidence"] == mode]
        scores = [r[f"{k}_f1"] for r in rows for k in FIELDS if f"{k}_f1" in r]
        if scores:
            by_mode[mode] = {"n_papers": len(rows), "mean_f1": round(sum(scores) / len(scores), 3)}

    # inter-annotator agreement: same paper, two different people
    by_paper = collections.defaultdict(list)
    for g in gold:
        by_paper[g["paper_id"]].append(g)
    iaa = {}
    for k in FIELDS:
        scores = []
        for rows in by_paper.values():
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    if rows[i].get(k) and rows[j].get(k) and rows[i]["annotator"] != rows[j]["annotator"]:
                        scores.append(token_f1(rows[i][k], rows[j][k]))
        if scores:
            iaa[k] = {"pairs": len(scores), "mean_f1": round(sum(scores) / len(scores), 3)}
    return {"match_f1": match_f1, "n_gold_rows": len(gold), "n_scored": len(per_paper),
            "missing_in_runs": missing, "fields": fields, "by_extraction_mode": by_mode,
            "inter_annotator": iaa, "per_paper": per_paper}


def to_markdown(res: dict, gold_paths: list) -> str:
    lines = ["# Extraction accuracy vs human answers", "",
             f"Gold: {', '.join(gold_paths)} · {res['n_gold_rows']} annotated rows, {res['n_scored']} matched to a "
             f"saved extraction, {len(res['missing_in_runs'])} not found in any run.", "",
             f"Token F1 (SQuAD-style) between the system's extraction and the human answer; match = F1 >= "
             f"{res['match_f1']}. Both sides 'not stated' counts as a match. F1 is a proxy - read the rows below.", "",
             "| field | papers | mean F1 | match rate | inter-annotator F1 |", "|---|---|---|---|---|"]
    for k, v in res["fields"].items():
        ia = res["inter_annotator"].get(k)
        lines.append(f"| {k} | {v['n']} | {v['mean_f1']} | {v['match_rate']} | "
                     + (f"{ia['mean_f1']} ({ia['pairs']} pairs)" if ia else "-") + " |")
    if res["by_extraction_mode"]:
        lines += ["", "| extracted from | papers | mean F1 (all fields) |", "|---|---|---|"]
        lines += [f"| {m} | {v['n_papers']} | {v['mean_f1']} |" for m, v in res["by_extraction_mode"].items()]
    lines += ["", "## Per paper", "", "| paper | annotator | " + " | ".join(f"{k} F1" for k in FIELDS) + " |",
              "|---|---|" + "---|" * len(FIELDS)]
    for r in res["per_paper"]:
        lines.append(f"| `{r['paper_id']}` | {r['annotator']} | "
                     + " | ".join(str(r.get(f"{k}_f1", "-")) for k in FIELDS) + " |")
    if res["missing_in_runs"]:
        lines += ["", "Not found in any saved run: " + ", ".join(f"`{p}`" for p in res["missing_in_runs"])]
    return "\n".join(lines) + "\n"


def make_template(state_path: str, out_path: str) -> str:
    state = json.load(open(state_path, encoding="utf-8"))
    papers = {p["id"]: p for p in state.get("selected_papers") or []}
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(TEMPLATE_HEADER)
        for ex in state.get("paper_extracts") or []:
            w.writerow([ex["paper_id"], papers.get(ex["paper_id"], {}).get("title", ""), "", "", "", "", ""])
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", nargs="+", help="filled-in copies of extraction_gold_TEMPLATE.csv")
    ap.add_argument("--states", default=os.path.join(ROOT, "output", "*_state.json"))
    ap.add_argument("--match-f1", type=float, default=0.5)
    ap.add_argument("--make-template", metavar="STATE_JSON",
                    help="write evaluation/data/extraction_gold_<run>.csv with this run's paper ids + titles, answers empty")
    args = ap.parse_args()
    if args.make_template:
        run_id = os.path.basename(args.make_template).replace("_state.json", "")
        out = make_template(args.make_template, os.path.join(ROOT, "evaluation", "data", f"extraction_gold_{run_id}.csv"))
        print("Template (answers empty - to be filled in by people):", out)
        return
    if not args.gold:
        ap.error("--gold is required (or --make-template)")
    gold = load_gold(args.gold)
    if not gold:
        sys.exit("No filled-in rows in the gold file(s) yet - the template has to be completed by the group first.")
    res = evaluate(gold, load_extracts(args.states), args.match_f1)
    md = to_markdown(res, args.gold)
    print(md)
    print("Saved:", *save("eval_extraction", res, md), f"(in {RESULTS_DIR})")


if __name__ == "__main__":
    main()
