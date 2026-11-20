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
import difflib
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, save  # noqa: E402

OPENALEX = "https://api.openalex.org/works"
CROSSREF = "https://api.crossref.org/works"


def norm_title(t: str) -> str:
    t = re.sub(r"<[^>]+>", " ", t or "")            # Crossref titles can carry <i>..</i>
    return " ".join(re.sub(r"[^\w\s]", " ", t.lower()).split())


def title_similarity(a: str, b: str) -> float:
    a, b = norm_title(a), norm_title(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def norm_doi(d: str) -> str:
    d = (d or "").strip().lower()
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", d)


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


def _openalex_record(w: dict) -> dict:
    return {"title": w.get("display_name") or w.get("title") or "", "year": w.get("publication_year"),
            "doi": norm_doi(w.get("doi"))}


def lookup_openalex(paper: dict, get, api_key: str = "") -> tuple:
    """(record or None, how). `get(url, params)` returns parsed JSON or None."""
    params = {"api_key": api_key} if api_key else {}
    if paper.get("doi"):
        w = get(f"{OPENALEX}/doi:{norm_doi(paper['doi'])}", params)
        if w and w.get("id"):
            return _openalex_record(w), "by DOI"
    if paper["id"].startswith("openalex:"):
        w = get(f"{OPENALEX}/{paper['id'].split(':', 1)[1]}", params)
        if w and w.get("id"):
            return _openalex_record(w), "by OpenAlex id"
    if paper.get("title"):
        data = get(OPENALEX, {**params, "search": paper["title"][:300], "per-page": 1})
        hits = (data or {}).get("results") or []
        if hits:
            return _openalex_record(hits[0]), "by title"
    return None, "not found"


def _crossref_record(m: dict) -> dict:
    year = None
    for k in ("published-print", "published-online", "issued", "created"):
        parts = ((m.get(k) or {}).get("date-parts") or [[None]])[0]
        if parts and parts[0]:
            year = parts[0]
            break
    return {"title": ((m.get("title") or [""])[0]), "year": year, "doi": norm_doi(m.get("DOI"))}


def lookup_crossref(paper: dict, get, mailto: str = "") -> tuple:
    params = {"mailto": mailto} if mailto else {}
    if paper.get("doi"):
        data = get(f"{CROSSREF}/{norm_doi(paper['doi'])}", params)
        if data and data.get("message"):
            return _crossref_record(data["message"]), "by DOI"
    if paper.get("title"):
        data = get(CROSSREF, {**params, "query.bibliographic": paper["title"][:300], "rows": 1})
        items = ((data or {}).get("message") or {}).get("items") or []
        if items:
            return _crossref_record(items[0]), "by title"
    return None, "not found"


def compare(paper: dict, rec: dict | None, title_sim: float) -> dict:
    if rec is None:
        return {"found": False}
    sim = title_similarity(paper.get("title"), rec.get("title"))
    y1, y2 = paper.get("year"), rec.get("year")
    d1, d2 = norm_doi(paper.get("doi")), rec.get("doi") or ""
    return {"found": True, "title_sim": round(sim, 3), "title_match": sim >= title_sim,
            "year_match": (y1 == y2) if (y1 and y2) else None,
            "year_within_1": (abs(int(y1) - int(y2)) <= 1) if (y1 and y2) else None,
            "doi_match": (d1 == d2) if (d1 and d2) else None,
            "remote_title": rec.get("title"), "remote_year": y2, "remote_doi": d2 or None}


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
