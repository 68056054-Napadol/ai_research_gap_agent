"""
OpenAlex API — https://docs.openalex.org
Free, CC0-licensed, no key strictly required, but supplying `mailto` puts you in the "polite
pool" with much better reliability. ~240M works; good broad-coverage fallback/cross-check
against Semantic Scholar, and includes open-access status directly.
"""
from config import CONFIG
from . import net

BASE = "https://api.openalex.org/works"


def _params(extra: dict) -> dict:
    # Since Feb 2026 OpenAlex wants an api_key on every call (free); mailto is ignored by the API
    # but harmless, so a real address is still sent. Placeholders are dropped in config._secret.
    p = dict(extra)
    if CONFIG.openalex_mailto:
        p["mailto"] = CONFIG.openalex_mailto
    if CONFIG.openalex_api_key:
        p["api_key"] = CONFIG.openalex_api_key
    return p


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """OpenAlex stores abstracts as an inverted index ({word: [positions]}) to satisfy
    publisher copyright terms — this rebuilds the plain-text abstract from it."""
    if not inverted_index:
        return ""
    positions = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def _to_paper(raw: dict) -> dict:
    oa = raw.get("open_access") or {}
    primary_loc = raw.get("primary_location") or {}
    return {
        "id": f"openalex:{raw.get('id', '').rsplit('/', 1)[-1]}",
        "source": "openalex",
        "title": raw.get("title") or raw.get("display_name") or "",
        "authors": [a.get("author", {}).get("display_name") for a in (raw.get("authorships") or [])],
        "year": raw.get("publication_year"),
        "abstract": _reconstruct_abstract(raw.get("abstract_inverted_index")),
        "doi": (raw.get("doi") or "").replace("https://doi.org/", "") or None,
        "oa_pdf_url": oa.get("oa_url") or primary_loc.get("pdf_url"),
        "citation_count": raw.get("cited_by_count"),
    }


def search_papers(query: str, year_from=None, year_to=None, limit: int = 30) -> list[dict]:
    filters = [f"default.search:{query}"]
    if year_from:
        filters.append(f"from_publication_date:{year_from}-01-01")
    if year_to:
        filters.append(f"to_publication_date:{year_to}-12-31")
    resp = net.get("openalex", BASE, params=_params({
        "search": query,
        "filter": ",".join(filters[1:]) if len(filters) > 1 else None,
        "per_page": min(limit, 100),
    }))
    resp.raise_for_status()
    return [_to_paper(r) for r in resp.json().get("results", [])]


def get_paper(paper_id: str) -> dict | None:
    raw_id = paper_id.split(":", 1)[-1]
    resp = net.get("openalex", f"{BASE}/{raw_id}", params=_params({}))
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return _to_paper(resp.json())


def get_full_text(paper_id: str) -> str | None:
    # OpenAlex gives an OA URL, not full text itself. nodes.py's extract_paper_data calls
    # pdf_extract directly with the oa_pdf_url it already has (from the search result); this
    # function exists for standalone/API callers that only have an id, so it looks the paper back
    # up first.
    from . import pdf_extract
    paper = get_paper(paper_id)
    if not paper:
        return None
    return pdf_extract.fetch_and_convert(paper.get("oa_pdf_url"), paper_id)
