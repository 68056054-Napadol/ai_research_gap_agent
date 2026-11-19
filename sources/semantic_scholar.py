"""
Semantic Scholar Graph API — https://api.semanticscholar.org/api-docs/graph
No key required for light use; SEMANTIC_SCHOLAR_API_KEY (if set) raises the rate limit.
Best all-around source: abstracts, citation counts, and an AI-generated TLDR field.
"""
from config import CONFIG
from . import net

BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,abstract,year,authors,externalIds,citationCount,openAccessPdf,tldr"


def _headers():
    h = {}
    if CONFIG.semantic_scholar_api_key:
        h["x-api-key"] = CONFIG.semantic_scholar_api_key
    return h


def _to_paper(raw: dict) -> dict:
    ext = raw.get("externalIds") or {}
    oa = raw.get("openAccessPdf") or {}
    return {
        "id": f"semantic_scholar:{raw.get('paperId')}",
        "source": "semantic_scholar",
        "title": raw.get("title") or "",
        "authors": [a.get("name") for a in (raw.get("authors") or [])],
        "year": raw.get("year"),
        "abstract": raw.get("abstract") or (raw.get("tldr") or {}).get("text") or "",
        "doi": ext.get("DOI"),
        "oa_pdf_url": oa.get("url"),
        "citation_count": raw.get("citationCount"),
    }


def search_papers(query: str, year_from=None, year_to=None, limit: int = 30) -> list[dict]:
    params = {"query": query, "fields": FIELDS, "limit": min(limit, 100)}
    if year_from or year_to:
        params["year"] = f"{year_from or ''}-{year_to or ''}"
    resp = net.get("semantic_scholar", f"{BASE}/paper/search", params=params, headers=_headers())
    resp.raise_for_status()
    return [_to_paper(r) for r in resp.json().get("data", [])]


def get_paper(paper_id: str) -> dict | None:
    raw_id = paper_id.split(":", 1)[-1]
    resp = net.get("semantic_scholar", f"{BASE}/paper/{raw_id}", params={"fields": FIELDS}, headers=_headers())
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return _to_paper(resp.json())


def get_full_text(paper_id: str) -> str | None:
    # Semantic Scholar doesn't host full text itself, only an oa_pdf_url when one exists.
    # nodes.py's extract_paper_data calls pdf_extract directly with the oa_pdf_url it already has
    # (from the search result); this function exists for standalone/API callers that only have an
    # id, so it looks the paper back up first.
    from . import pdf_extract
    paper = get_paper(paper_id)
    if not paper:
        return None
    return pdf_extract.fetch_and_convert(paper.get("oa_pdf_url"), paper_id)
