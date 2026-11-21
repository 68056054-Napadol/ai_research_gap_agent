"""
Grounding of the final answer: does the text a person reads (trend summary, gap claims, rationales,
verifier notes) mention papers, DOIs, numbers or dataset/method names that are NOT in the papers the
run analysed? (Teacher feedback, evaluation item 6: "การไม่สร้าง Paper, DOI หรือผลการศึกษาที่ไม่มีอยู่จริง".)

What the run already guarantees, so this script does not re-measure it:
  - every paper in the answer came from a search API (ids are dropped when not among the analysed
    papers - `dropped_ids`, measured as gap_ids_hallucinated by compare_models.py);
  - each cited paper's title / year / DOI is checked against Crossref/OpenAlex (citations.py,
    measured by eval_citations.py).

What can still be invented is in the FREE TEXT. Four automatic checks against the "corpus" = title,
abstract, full text (when the run fetched it) and extraction of every analysed paper, plus the
comparison table:

  A. references   DOIs (10.xxxx/...), arXiv ids, source ids (openalex:W..., arxiv:...) that are not
                  those of an analysed paper, and "<Name> et al." citations (author names are never
                  stored, so such a citation cannot come from the analysed set)          -> must be 0
  B. years        "(2019)"-style years that are not the year of any analysed paper    -> a paper outside the set?
  C. numbers      percentages and decimals ("4.2 F1", "93%") not found in the corpus  -> invented results?
  D. names        dataset / model / method names (acronyms, CamelCase, names with digits such
                  as XLM-R, CoNLL-2003, MasakhaNER) not found in the corpus            -> invented datasets/methods?

Automatic flags are candidates, not verdicts (a number can be rounded, a name abbreviated). The
human part: `--make-audit` writes a CSV of sentences - every flagged one plus a random sample of the
others - for a person to mark supported / partly / no against the cited papers; `--audit` turns the
filled CSV into the unsupported-sentence rate. Protocol: evaluation/eval_plan.md §8.

    python evaluation/eval_grounding.py                                   # every output/*_state.json
    python evaluation/eval_grounding.py --states output/X_state.json --make-audit
    python evaluation/eval_grounding.py --audit evaluation/data/grounding_audit_X.csv
"""
import argparse
import csv
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, save  # noqa: E402

DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s,;)\]]+", re.I)
ARXIV_RE = re.compile(r"(?<![\w.])(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?\b", re.I)
SOURCE_ID_RE = re.compile(r"\b(?:openalex|semantic_scholar|arxiv):[\w.]+")
ET_AL_RE = re.compile(r"\b([A-Z][\w'’-]+)\s+et\s+al\.?", re.U)
YEAR_RE = re.compile(r"\((?:[^()]*?\b)?((?:19|20)\d{2})\)")
NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s?(%|percent|points?|pp\b|F1|BLEU|accuracy)?", re.I)
NAME_RE = re.compile(r"\b(?:[A-Z][a-z]+[A-Z]\w*|[A-Z]{2,}[\w-]*|\w*[A-Za-z]-?\d+[\w-]*|[A-Z][\w]*-[A-Z0-9][\w-]*)\b")

# generic words that look like names but say nothing about a specific dataset/method
GENERIC_NAMES = {"NLP", "AI", "ML", "LLM", "LLMS", "NER", "MT", "QA", "API", "PDF", "JSON", "URL", "DOI", "ID", "II",
                 "III", "IV", "USA", "UK", "EU", "GPU", "CPU", "RAG", "SOTA", "TH", "EN", "OA", "BM25"}


def sentences(text: str) -> list:
    text = " ".join((text or "").split())
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", text) if len(s.split()) >= 4]


def _norm(t: str) -> str:
    return " ".join((t or "").lower().split())


def corpus_of(state: dict) -> dict:
    papers = state.get("selected_papers") or []
    parts = [state.get("comparison_table") or ""]
    for p in papers:
        parts += [p.get("title") or "", p.get("abstract") or "", p.get("full_text") or ""]
    for e in state.get("paper_extracts") or []:
        parts += [str(e.get(k) or "") for k in ("problem", "method", "dataset", "key_findings", "limitations_stated",
                                                "metrics", "supporting_quote")]
    ids = {p.get("id") for p in papers if p.get("id")}
    for g in state.get("gap_candidates") or []:
        cp = g.get("counterexample_paper") or {}
        if cp.get("id"):
            ids.add(cp["id"])
    return {
        "text": _norm("\n".join(parts)),
        "ids": ids,
        "dois": {p["doi"].lower() for p in papers if p.get("doi")},
        "arxiv": {p["id"].split(":", 1)[1].split("v")[0] for p in papers if str(p.get("id", "")).startswith("arxiv:")},
        "years": {p.get("year") for p in papers if isinstance(p.get("year"), int)},
    }


def answer_fields(state: dict) -> list:
    """(field, text) the user reads."""
    out = [("trend_summary", state.get("trend_summary") or "")]
    for i, g in enumerate(state.get("gap_candidates") or [], 1):
        for k in ("claim", "rationale", "verifier_notes"):
            out.append((f"gap{i}.{k}", g.get(k) or ""))
    return [(f, t) for f, t in out if t.strip()]


def check_sentence(s: str, corpus: dict) -> list:
    """Automatic flags for one sentence: [(check, item)]."""
    flags = []
    for doi in DOI_RE.findall(s):
        if doi.lower().rstrip(".") not in corpus["dois"]:
            flags.append(("reference", f"DOI {doi}"))
    for aid in ARXIV_RE.findall(s):
        if aid not in corpus["arxiv"]:
            flags.append(("reference", f"arXiv {aid}"))
    for sid in SOURCE_ID_RE.findall(s):
        if sid.rstrip(".") not in corpus["ids"]:
            flags.append(("reference", f"id {sid}"))
    for name in ET_AL_RE.findall(s):
        flags.append(("reference", f"{name} et al."))
    s = SOURCE_ID_RE.sub(" ", ARXIV_RE.sub(" ", DOI_RE.sub(" ", s)))   # ids are not numbers or names
    for y in YEAR_RE.findall(s):
        if int(y) not in corpus["years"]:
            flags.append(("year", y))
    for num, unit in NUMBER_RE.findall(s):
        if not unit and "." not in num:
            continue                     # plain integers ("3 papers", "two") are counts, not results
        if 1900 <= float(num) <= 2100 and not unit:
            continue
        if num not in corpus["text"]:
            flags.append(("number", num + ("%" if unit == "%" else f" {unit}" if unit else "")))
    for name in NAME_RE.findall(s):
        if name.upper() in GENERIC_NAMES or len(name) < 3 or re.fullmatch(r"(?:19|20)\d{2}", name):
            continue
        if _norm(name) not in corpus["text"]:
            flags.append(("name", name))
    return flags


def evaluate_state(state: dict, run_id: str) -> dict:
    corpus = corpus_of(state)
    rows = []
    for field, text in answer_fields(state):
        for s in sentences(text):
            rows.append({"run_id": run_id, "field": field, "sentence": s, "flags": check_sentence(s, corpus)})
    counts = {k: 0 for k in ("reference", "year", "number", "name")}
    for r in rows:
        for check, _ in r["flags"]:
            counts[check] += 1
    numbers_total = sum(1 for r in rows for num, unit in NUMBER_RE.findall(r["sentence"]) if unit or "." in num)
    return {"run_id": run_id, "topic": state.get("research_topic"), "n_sentences": len(rows),
            "n_flagged_sentences": sum(1 for r in rows if r["flags"]), "flag_counts": counts,
            "n_result_numbers": numbers_total,
            "ungrounded_number_rate": round(counts["number"] / numbers_total, 3) if numbers_total else None,
            "rows": rows}


def to_markdown(results: list) -> str:
    lines = ["# Grounding of the final answer (free text vs the analysed papers)", "",
             "Flags are candidates for a person to check, not verdicts. `reference` must be 0: the answer may only "
             "refer to analysed papers.", "",
             "| run | topic | sentences | flagged | references | years | numbers (ungrounded / total) | names |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        c = r["flag_counts"]
        lines.append(f"| {r['run_id']} | {(r['topic'] or '')[:50]} | {r['n_sentences']} | {r['n_flagged_sentences']} "
                     f"| {c['reference']} | {c['year']} | {c['number']} / {r['n_result_numbers']} | {c['name']} |")
    for r in results:
        flagged = [x for x in r["rows"] if x["flags"]]
        lines += ["", f"## {r['run_id']}: flagged sentences ({len(flagged)})", ""]
        for x in flagged[:40]:
            items = "; ".join(f"{k}: {v}" for k, v in x["flags"])
            lines.append(f"- `{x['field']}` — {x['sentence'][:300]}  \n  _flags: {items}_")
        if not flagged:
            lines.append("None.")
    return "\n".join(lines) + "\n"


AUDIT_COLUMNS = ["run_id", "field", "sentence", "auto_flags", "cited_papers",
                 "supported (yes/partly/no)", "what is not supported", "annotator"]


def make_audit(result: dict, state: dict, path: str, n_random: int = 10, seed: int = 42) -> int:
    """Every flagged sentence + n_random unflagged ones, in answer order."""
    flagged = [r for r in result["rows"] if r["flags"]]
    other = [r for r in result["rows"] if not r["flags"]]
    random.Random(seed).shuffle(other)
    chosen = {id(r) for r in flagged + other[:n_random]}
    gaps = state.get("gap_candidates") or []
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(AUDIT_COLUMNS)
        for r in result["rows"]:
            if id(r) not in chosen:
                continue
            m = re.match(r"gap(\d+)\.", r["field"])
            cited = ", ".join(gaps[int(m.group(1)) - 1].get("supporting_paper_ids") or []) if m else "all analysed papers"
            w.writerow([r["run_id"], r["field"], r["sentence"], "; ".join(f"{k}: {v}" for k, v in r["flags"]),
                        cited, "", "", ""])
    return len(chosen)


def score_audit(path: str) -> dict:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    labels = [(r.get("supported (yes/partly/no)") or "").strip().lower() for r in rows]
    done = [lab for lab in labels if lab in ("yes", "partly", "no")]
    flagged = [(r.get("supported (yes/partly/no)") or "").strip().lower() for r in rows if r.get("auto_flags")]
    flagged = [lab for lab in flagged if lab in ("yes", "partly", "no")]
    unflagged = [(r.get("supported (yes/partly/no)") or "").strip().lower() for r in rows if not r.get("auto_flags")]
    unflagged = [lab for lab in unflagged if lab in ("yes", "partly", "no")]
    rate = lambda xs: round(sum(1 for x in xs if x == "no") / len(xs), 3) if xs else None   # noqa: E731
    return {"n_rows": len(rows), "n_labelled": len(done), "unsupported_rate": rate(done),
            "partly_rate": round(sum(1 for x in done if x == "partly") / len(done), 3) if done else None,
            "unsupported_rate_flagged": rate(flagged), "unsupported_rate_unflagged": rate(unflagged),
            # how well the automatic flags find the problems a person finds
            "flag_precision": rate(flagged), "n_unsupported_missed_by_flags": sum(1 for x in unflagged if x == "no")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", nargs="+", default=None, help="state files (default: output/*_state.json)")
    ap.add_argument("--make-audit", action="store_true", help="write evaluation/data/grounding_audit_<run>.csv per state")
    ap.add_argument("--random", type=int, default=10, help="unflagged sentences added to each audit sheet")
    ap.add_argument("--audit", nargs="+", default=None, help="filled audit CSV(s) to score")
    args = ap.parse_args()

    if args.audit:
        res = {os.path.basename(p): score_audit(p) for p in args.audit}
        md = ["# Grounding audit (human labels)", "", "| sheet | labelled | unsupported | partly | unsupported among "
              "flagged | among unflagged | missed by flags |", "|---|---|---|---|---|---|---|"]
        for name, r in res.items():
            md.append(f"| {name} | {r['n_labelled']}/{r['n_rows']} | {r['unsupported_rate']} | {r['partly_rate']} | "
                      f"{r['unsupported_rate_flagged']} | {r['unsupported_rate_unflagged']} | {r['n_unsupported_missed_by_flags']} |")
        md = "\n".join(md) + "\n"
        print(md)
        print("Saved:", *save("eval_grounding_audit", res, md))
        return

    states = args.states or sorted(glob.glob(os.path.join(ROOT, "output", "*_state.json")))
    results = []
    for path in states:
        try:
            st = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not st.get("gap_candidates") and not st.get("trend_summary"):
            continue
        run_id = os.path.basename(path).replace("_state.json", "")
        res = evaluate_state(st, run_id)
        results.append(res)
        if args.make_audit:
            out = os.path.join(ROOT, "evaluation", "data", f"grounding_audit_{run_id}.csv")
            n = make_audit(res, st, out, n_random=args.random)
            print(f"Audit sheet ({n} sentences): {out}")
    if not results:
        sys.exit("No runs with an answer found in the given state files.")
    md = to_markdown(results)
    print(md)
    print("Saved:", *save("eval_grounding", {"runs": [{k: v for k, v in r.items()} for r in results]}, md))


if __name__ == "__main__":
    main()
