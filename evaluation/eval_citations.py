"""
Citation accuracy: are the papers a gap cites real, and are their title / year / DOI right?

For every paper cited by a gap (supporting papers + the paper that refutes a gap) in the saved
runs, the metadata the agent stored (title, year, DOI) is checked against an independent record:

  1. OpenAlex - by DOI, else by OpenAlex id, else by title search (needs OPENALEX_API_KEY since
     Feb 2026, read from .env like the agent does)
  2. Crossref - by DOI, else by bibliographic title search (no key; set UNPAYWALL_EMAIL or
     OPENALEX_MAILTO for Crossref's polite pool). Skipped with --no-crossref or when unreachable.

Reported per check: title match (normalised text similarity >= --title-sim, default 0.9), year
match (exact, and within 1 year - preprints are often a year earlier than the published version),
DOI match (case-insensitive; only when both sides have one), and "not found".

Caveats the report repeats: titles in state.json went through the privacy scrub, so a title that
contained a person's name was redacted and will not match; a title-search hit can be a different
paper with the same title (rows marked "by title" - check them by hand).

    python evaluation/eval_citations.py                       # every output/*_state.json
    python evaluation/eval_citations.py --states output/20261120_011606_state.json --no-crossref
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, save  # noqa: E402

sys.path.insert(0, ROOT)
# The lookup / comparison functions are shared with the agent's own pre-report check (citations.py,
# Node 7b verify_citations), so the evaluation measures exactly what the run checked.
from citations import (CROSSREF, OPENALEX, _crossref_record, _openalex_record, compare, lookup_crossref,  # noqa: E402,F401
                       lookup_openalex, norm_doi, norm_title, title_similarity)


def cited_papers(states: list) -> list:
    """Unique (run, paper) pairs cited by gaps, with the metadata the agent stored."""
    out, seen = [], set()
    for path in states:
        try:
            st = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        run_id = os.path.basename(path).replace("_state.json", "")
        papers = {p["id"]: p for p in (st.get("selected_papers") or []) if p.get("id")}
        for g in st.get("gap_candidates") or []:
            refs = [(pid, "supporting") for pid in g.get("supporting_paper_ids") or []]
            if g.get("counterexample_paper"):
                refs.append((g["counterexample_paper"]["id"], "refuting"))
            for pid, role in refs:
                if (run_id, pid) in seen:
                    continue
                seen.add((run_id, pid))
                p = papers.get(pid) or g.get("counterexample_paper") or {"id": pid}
                out.append({"run_id": run_id, "id": pid, "role": role, "title": p.get("title") or "",
                            "year": p.get("year"), "doi": p.get("doi")})
    return out


def rates(rows: list, key: str) -> dict:
    checks = [r[key] for r in rows if r.get(key) is not None]
    found = [r for r in rows if r.get("found")]
    if key == "found":
        return {"n": len(rows), "rate": round(len(found) / len(rows), 3) if rows else None}
    return {"n": len(checks), "rate": round(sum(1 for c in checks if c) / len(checks), 3) if checks else None}


def evaluate(papers: list, lookups: dict, title_sim: float = 0.9, pause: float = 0.0) -> dict:
    """lookups: {"openalex": fn(paper) -> (record, how), "crossref": ...}."""
    per_source = {}
    for name, fn in lookups.items():
        rows = []
        for p in papers:
            rec, how = fn(p)
            rows.append({**p, "how": how, **compare(p, rec, title_sim)})
            if pause:
                time.sleep(pause)
        per_source[name] = {"rows": rows, "summary": {k: rates(rows, k) for k in
                                                       ("found", "title_match", "year_match", "year_within_1", "doi_match")}}
    return {"n_cited_papers": len(papers), "title_sim": title_sim, "sources": per_source}


def to_markdown(res: dict) -> str:
    lines = ["# Citation accuracy (papers cited by gaps vs OpenAlex / Crossref)", "",
             f"{res['n_cited_papers']} cited papers (unique per run). Title match = normalised similarity >= "
             f"{res['title_sim']}. Rates are over the papers where both sides have the field.", "",
             "Caveats: titles in state.json passed the privacy scrub (a redacted name will not match); a "
             "'by title' hit may be another paper with the same title - check those rows by hand.", "",
             "| source | found | title match | year exact | year ±1 | DOI match |", "|---|---|---|---|---|---|"]

    def cell(s):
        return f"{s['rate']} (n={s['n']})" if s["rate"] is not None else f"- (n={s['n']})"
    for name, src in res["sources"].items():
        s = src["summary"]
        lines.append(f"| {name} | {cell(s['found'])} | {cell(s['title_match'])} | {cell(s['year_match'])} "
                     f"| {cell(s['year_within_1'])} | {cell(s['doi_match'])} |")
    for name, src in res["sources"].items():
        bad = [r for r in src["rows"] if not r.get("found") or r.get("title_match") is False
               or r.get("year_match") is False or r.get("doi_match") is False]
        lines += ["", f"## {name}: mismatches to check ({len(bad)})", ""]
        if not bad:
            lines.append("None.")
            continue
        lines += ["| run | paper | stored title / year / DOI | found | remote title / year / DOI |", "|---|---|---|---|---|"]
        for r in bad:
            esc = lambda t: str(t or "").replace("|", "/")[:120]   # noqa: E731
            lines.append(f"| {r['run_id']} | `{r['id']}` ({r['role']}) | {esc(r['title'])} / {r['year']} / {r['doi'] or '-'} "
                         f"| {r['how']} | " + (f"{esc(r.get('remote_title'))} / {r.get('remote_year')} / {r.get('remote_doi') or '-'}"
                                               if r.get("found") else "-") + " |")
    return "\n".join(lines) + "\n"


def http_getter(timeout: float = 20.0):
    import httpx

    def get(url, params):
        try:
            r = httpx.get(url, params=params, timeout=timeout, follow_redirects=True,
                          headers={"User-Agent": "research-gap-agent-eval (course project)"})
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None
    return get


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", nargs="+", default=None, help="state files (default: output/*_state.json)")
    ap.add_argument("--title-sim", type=float, default=0.9)
    ap.add_argument("--no-crossref", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="check at most N cited papers (0 = all)")
    args = ap.parse_args()
    from config import CONFIG
    states = args.states or sorted(glob.glob(os.path.join(ROOT, "output", "*_state.json")))
    papers = cited_papers(states)
    if args.limit:
        papers = papers[: args.limit]
    if not papers:
        sys.exit("No gap citations found in the given state files.")
    if not CONFIG.openalex_api_key:
        print("Note: no OPENALEX_API_KEY in .env - OpenAlex may refuse or rate limit these lookups.")
    get = http_getter()
    lookups = {"openalex": lambda p: lookup_openalex(p, get, CONFIG.openalex_api_key)}
    if not args.no_crossref:
        mailto = CONFIG.unpaywall_email or CONFIG.openalex_mailto
        lookups["crossref"] = lambda p: lookup_crossref(p, get, mailto)
    res = evaluate(papers, lookups, args.title_sim, pause=0.2)
    md = to_markdown(res)
    print(md)
    print("Saved:", *save("eval_citations", res, md))


if __name__ == "__main__":
    main()
