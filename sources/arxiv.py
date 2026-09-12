"""
arXiv API — https://info.arxiv.org/help/api/user-manual.html
No key needed, but arXiv enforces a hard rate limit of ~1 request / 3 seconds — this module
does not implement its own throttling (single-call use here is fine); add a delay if you start
calling it in a tight loop across many queries.
Essential source for CS/NLP topics; has no citation-count field.
"""
import time
import httpx
import xml.etree.ElementTree as ET

BASE = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}

_last_call_ts = 0.0


def _throttle():
    global _last_call_ts
    elapsed = time.time() - _last_call_ts
    if elapsed < 3.0:
        time.sleep(3.0 - elapsed)
    _last_call_ts = time.time()


def _to_paper(entry) -> dict:
    arxiv_id = entry.findtext("atom:id", default="", namespaces=NS).rsplit("/", 1)[-1]
    published = entry.findtext("atom:published", default="", namespaces=NS)
    year = int(published[:4]) if published[:4].isdigit() else None
    authors = [a.findtext("atom:name", default="", namespaces=NS) for a in entry.findall("atom:author", NS)]
    pdf_url = None
    for link in entry.findall("atom:link", NS):
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            pdf_url = link.get("href")
    return {
        "id": f"arxiv:{arxiv_id}",
        "source": "arxiv",
        "title": (entry.findtext("atom:title", default="", namespaces=NS) or "").strip().replace("\n", " "),
        "authors": authors,
        "year": year,
        "abstract": (entry.findtext("atom:summary", default="", namespaces=NS) or "").strip().replace("\n", " "),
        "doi": entry.findtext("{http://arxiv.org/schemas/atom}doi", default=None, namespaces=NS),
        "oa_pdf_url": pdf_url,
        "citation_count": None,  # arXiv doesn't provide this
    }


def search_papers(query: str, year_from=None, year_to=None, limit: int = 30) -> list[dict]:
    _throttle()
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": min(limit, 100),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    resp = httpx.get(BASE, params=params, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    papers = [_to_paper(e) for e in root.findall("atom:entry", NS)]
    if year_from or year_to:
        papers = [p for p in papers if p["year"] is None or (
            (year_from is None or p["year"] >= year_from) and (year_to is None or p["year"] <= year_to)
        )]
    return papers


def get_paper(paper_id: str) -> dict | None:
    raw_id = paper_id.split(":", 1)[-1]
    _throttle()
    resp = httpx.get(BASE, params={"id_list": raw_id}, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    entries = root.findall("atom:entry", NS)
    return _to_paper(entries[0]) if entries else None


def get_full_text(paper_id: str) -> str | None:
    # arXiv PDFs are always open access. nodes.py's extract_paper_data calls pdf_extract directly
    # with the oa_pdf_url it already has (from the search result), which is why this function
    # looks the paper back up first — it exists for standalone/API callers that only have an id.
    from . import pdf_extract
    paper = get_paper(paper_id)
    if not paper:
        return None
    return pdf_extract.fetch_and_convert(paper.get("oa_pdf_url"), paper_id)
