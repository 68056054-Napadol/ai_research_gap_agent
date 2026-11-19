"""
Every module in this package exposes the same three functions (architecture doc §3.2), so the
search node can call any source interchangeably and add new ones without touching graph code:

    search_papers(query: str, year_from: int|None, year_to: int|None, limit: int) -> list[dict]
    get_paper(paper_id: str) -> dict | None
    get_full_text(paper_id: str) -> str | None

Every returned paper dict has at least: id, source, title, authors, year, abstract, doi,
oa_pdf_url, citation_count — matching state.Paper. A source that lacks a field (e.g. arXiv has
no citation_count) just sets it to None rather than omitting the key, so downstream code can
rely on the shape.

pdf_extract.py (in this package) is the shared PDF-to-Markdown full-text fetcher used by Node 5
(nodes.extract_paper_data) — it's called with the oa_pdf_url already present on a paper dict
rather than through each source's get_full_text(paper_id), to avoid a redundant paper lookup.
Each source's get_full_text still works standalone (it looks the paper up first) for API/CLI
callers that only have an id.
"""
import logging
import time

from . import semantic_scholar, arxiv, openalex

logger = logging.getLogger(__name__)

REGISTRY = {
    "semantic_scholar": semantic_scholar,
    "arxiv": arxiv,
    "openalex": openalex,
}


def search_all(query: str, sources: list[str], year_from=None, year_to=None, limit_per_source: int = 30) -> list[dict]:
    """
    Calls search_papers on every enabled source and returns the combined, deduplicated list.
    A source that errors (rate limit, network) is skipped with a warning, not a crash — this is
    the "graceful degradation" behavior described in architecture doc §7.1.
    """
    results = []
    for name in sources:
        module = REGISTRY.get(name)
        if module is None:
            logger.warning("Unknown source %r, skipping", name)
            continue
        start = time.monotonic()
        try:
            papers = module.search_papers(query, year_from=year_from, year_to=year_to, limit=limit_per_source)
            logger.info("    [%s] %d results in %.1fs", name, len(papers), time.monotonic() - start)
            results.extend(papers)
        except Exception as e:
            logger.warning("    [%s] FAILED after %.1fs for query %r: %s",
                            name, time.monotonic() - start, query, e)
    return _dedupe(results)


def _title_key(title: str) -> str:
    return "".join(ch for ch in (title or "").lower() if ch.isalnum())


def _dedupe(papers: list[dict]) -> list[dict]:
    """One entry per paper: a paper is a duplicate when its DOI OR its normalized title was seen.
    (Before, a title was only checked when the paper had no DOI, so the same paper from arXiv - no
    DOI - and OpenAlex - with a DOI - was analysed twice in a live run.) Fields the kept entry is
    missing (abstract, PDF link, DOI, citations) are filled from the duplicate."""
    by_doi, by_title, out = {}, {}, []
    for p in papers:
        doi = (p.get("doi") or "").strip().lower()
        tkey = _title_key(p.get("title"))
        kept = (by_doi.get(doi) if doi else None) or (by_title.get(tkey) if tkey else None)
        if kept is not None:
            for field in ("abstract", "oa_pdf_url", "doi", "citation_count", "year"):
                if not kept.get(field) and p.get(field):
                    kept[field] = p[field]
            if doi:
                by_doi.setdefault(doi, kept)
            continue
        if doi:
            by_doi[doi] = p
        if tkey:
            by_title[tkey] = p
        out.append(p)
    return out
