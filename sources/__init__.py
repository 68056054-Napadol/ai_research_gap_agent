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


def _dedupe(papers: list[dict]) -> list[dict]:
    """Dedup by DOI first (most reliable), then by lowercased/normalized title."""
    seen_doi, seen_title, out = set(), set(), []
    for p in papers:
        doi = (p.get("doi") or "").strip().lower()
        title_key = "".join((p.get("title") or "").lower().split())
        if doi and doi in seen_doi:
            continue
        if not doi and title_key in seen_title:
            continue
        if doi:
            seen_doi.add(doi)
        seen_title.add(title_key)
        out.append(p)
    return out
