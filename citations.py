"""
Citation check BEFORE the report (teacher feedback: "ตรวจสอบแหล่งอ้างอิงก่อนสร้างรายงาน").

Every paper a Potential Research Gap cites (supporting papers + a refuting paper) is looked up in an
independent bibliographic record and its title / year / DOI compared with what the agent stored:

  Crossref  by DOI, else by bibliographic title search   (no key; mailto from UNPAYWALL_EMAIL / OPENALEX_MAILTO)
  OpenAlex  by DOI, else OpenAlex id, else title search   (only for papers that did NOT come from
                                                           OpenAlex - checking OpenAlex against itself
                                                           proves nothing; needs OPENALEX_API_KEY)

Status per paper (code decides, no LLM):
  verified    a record was found by DOI or id, the title matches (normalised similarity >=
              CITATION_TITLE_SIM), the year matches within 1 (preprint vs published) and the DOI,
              when both sides have one, is the same
  mismatch    a record was found but the title, year or DOI disagrees - the report shows what differs
  title_only  found only by a title search and it matches - plausible, but a same-title paper is
              possible, so it is not counted as verified
  not_found   no record in any checked service
  unchecked   no lookup possible (VERIFY_CITATIONS=false, network error, service switched off)

Effect on the answer: a gap whose supporting papers are not all `verified` or `title_only` cannot be
Strong (README "Evidence levels"); the status of every cited paper is printed in the report.

evaluation/eval_citations.py uses the same lookup functions to measure citation accuracy over many runs.
"""
import difflib
import logging
import re

logger = logging.getLogger(__name__)

OPENALEX = "https://api.openalex.org/works"
CROSSREF = "https://api.crossref.org/works"

VERIFIED, MISMATCH, TITLE_ONLY, NOT_FOUND, UNCHECKED = "verified", "mismatch", "title_only", "not_found", "unchecked"
OK_STATUSES = (VERIFIED, TITLE_ONLY)
STATUS_TH = {VERIFIED: "ตรวจแล้วตรงกัน", MISMATCH: "ข้อมูลไม่ตรง", TITLE_ONLY: "พบจากชื่อเรื่องเท่านั้น",
             NOT_FOUND: "ไม่พบในฐานข้อมูลอ้างอิง", UNCHECKED: "ยังไม่ได้ตรวจ"}
STATUS_MARK = {VERIFIED: "✅", MISMATCH: "⚠️", TITLE_ONLY: "🔎", NOT_FOUND: "❌", UNCHECKED: "⏸️"}


# ---------------------------------------------------------------------------
# Normalisation + comparison
# ---------------------------------------------------------------------------
def norm_title(t: str) -> str:
    t = re.sub(r"<[^>]+>", " ", t or "")            # Crossref titles can carry <i>..</i>
    t = re.sub(r"\[[A-Z_]+\]", " ", t)               # privacy placeholders such as [PERSON]
    return " ".join(re.sub(r"[^\w\s]", " ", t.lower()).split())


def title_similarity(a: str, b: str) -> float:
    a, b = norm_title(a), norm_title(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def norm_doi(d: str) -> str:
    d = (d or "").strip().lower()
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", d)


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
    if str(paper.get("id", "")).startswith("openalex:"):
        w = get(f"{OPENALEX}/{paper['id'].split(':', 1)[1]}", params)
        if w and w.get("id"):
            return _openalex_record(w), "by OpenAlex id"
    if paper.get("title"):
        data = get(OPENALEX, {**params, "search": norm_title(paper["title"])[:300], "per-page": 1})
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
        data = get(CROSSREF, {**params, "query.bibliographic": norm_title(paper["title"])[:300], "rows": 1})
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
            "local_doi": d1 or None,
            "remote_title": rec.get("title"), "remote_year": y2, "remote_doi": d2 or None}


# Preprint DOIs (arXiv's DataCite prefix): the same paper is often found under its published DOI.
PREPRINT_DOI_PREFIXES = ("10.48550/",)


def status_of(cmp: dict, how: str) -> tuple:
    """(status, problems) for one lookup result."""
    if not cmp.get("found"):
        return NOT_FOUND, []
    problems = []
    if not cmp["title_match"]:
        problems.append(f"title similarity {cmp['title_sim']:.2f}")
    if cmp["year_within_1"] is False:
        problems.append(f"year {cmp['remote_year']}")
    published_as = None
    if cmp["doi_match"] is False:
        if (cmp.get("local_doi") or "").startswith(PREPRINT_DOI_PREFIXES) and cmp["title_match"] \
                and cmp["year_within_1"] is not False:
            published_as = cmp["remote_doi"]      # arXiv preprint of a published paper - same work
        else:
            problems.append(f"DOI {cmp['remote_doi']}")
    if problems:
        return MISMATCH, problems
    if published_as:
        return VERIFIED, [f"preprint; published as {published_as}"]
    return (TITLE_ONLY if how == "by title" else VERIFIED), []


_RANK = {VERIFIED: 0, TITLE_ONLY: 1, MISMATCH: 2, NOT_FOUND: 3, UNCHECKED: 4}


def check_paper(paper: dict, lookups: dict, title_sim: float) -> dict:
    """Runs every lookup; the best status wins (one service may lack a record the other has)."""
    results = []
    for name, fn in lookups.items():
        try:
            rec, how = fn(paper)
        except Exception as e:                       # network trouble must not break the run
            logger.info("    citation lookup %s failed for %s: %s", name, paper.get("id"), type(e).__name__)
            results.append({"service": name, "status": UNCHECKED, "how": f"error: {type(e).__name__}", "problems": []})
            continue
        cmp = compare(paper, rec, title_sim)
        status, problems = status_of(cmp, how)
        results.append({"service": name, "status": status, "how": how, "problems": problems,
                        "title_sim": cmp.get("title_sim"), "remote_year": cmp.get("remote_year"),
                        "remote_doi": cmp.get("remote_doi")})
    if not results:
        return {"status": UNCHECKED, "checks": [], "summary": "no lookup service available"}
    best = min(results, key=lambda r: _RANK[r["status"]])
    if best["status"] == NOT_FOUND and any(r["status"] == UNCHECKED for r in results):
        best = next(r for r in results if r["status"] == UNCHECKED)   # absence not shown, just not checked
    summary = f"{best['service']} {best['how']}" + (f" - {', '.join(best['problems'])}" if best["problems"] else "")
    if best["status"] == VERIFIED:
        best["problems"] = []                         # a note (published version), not a problem
    return {"status": best["status"], "checks": results, "summary": summary}


def cited_paper_ids(gaps: list) -> list:
    ids = []
    for g in gaps or []:
        for pid in g.get("supporting_paper_ids") or []:
            if pid not in ids:
                ids.append(pid)
        cp = g.get("counterexample_paper") or {}
        if cp.get("id") and cp["id"] not in ids:
            ids.append(cp["id"])
    return ids


def lookups_for(paper: dict, get, cfg) -> dict:
    out = {"crossref": lambda p: lookup_crossref(p, get, cfg.unpaywall_email or cfg.openalex_mailto)}
    if paper.get("source") != "openalex" and not str(paper.get("id", "")).startswith("openalex:"):
        out["openalex"] = lambda p: lookup_openalex(p, get, cfg.openalex_api_key)
    return out


def http_getter(timeout: float = 15.0, use_net: bool = True):
    """`get(url, params) -> dict | None`. With use_net the shared throttled/retrying client
    (sources/net.py) is used under the labels citation_crossref / citation_openalex, so a service that
    keeps failing is switched off for the rest of the run instead of slowing every lookup."""
    import httpx

    def get(url, params):
        label = "citation_crossref" if "crossref" in url else "citation_openalex"
        headers = {"User-Agent": "research-gap-agent (course project; citation check)"}
        try:
            if use_net:
                from sources import net
                r = net.get(label, url, params=params, headers=headers, timeout=timeout, follow_redirects=True)
            else:
                r = httpx.get(url, params=params, timeout=timeout, follow_redirects=True, headers=headers)
        except Exception as e:
            raise ConnectionError(str(e)) from e
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            raise ConnectionError(f"HTTP {r.status_code}")
        return r.json()
    return get


def gap_citation_summary(gap: dict, checks: dict) -> dict:
    """Counts for one gap: how many of its supporting papers checked out."""
    ids = gap.get("supporting_paper_ids") or []
    statuses = [(checks.get(pid) or {}).get("status", UNCHECKED) for pid in ids]
    return {"n_cited": len(ids), "n_ok": sum(1 for s in statuses if s in OK_STATUSES),
            "n_verified": statuses.count(VERIFIED),
            "problems": [pid for pid, s in zip(ids, statuses) if s not in OK_STATUSES]}


def cap_reason(summary: dict) -> str:
    """Why this gap cannot be Strong, or "" when every supporting citation checked out."""
    if summary["n_cited"] and summary["n_ok"] < summary["n_cited"]:
        return (f"{summary['n_cited'] - summary['n_ok']} of {summary['n_cited']} supporting citations did not "
                f"check out against Crossref/OpenAlex")
    return ""


def status_line(check: dict | None) -> str:
    check = check or {"status": UNCHECKED, "summary": "not checked"}
    s = check["status"]
    return f"{STATUS_MARK[s]} citation {s} ({STATUS_TH[s]}): {check.get('summary', '')}"
