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


def author_keys(authors) -> list:
    """Order-free fingerprint of an author list: one short one-way hash per name token of 3+
    letters (accents folded). Survives "Wei Wang" vs "Wang Wei" and "J. Smith" vs "John Smith"
    (initials are ignored). Kept on the paper after privacy.scrub_paper drops the names, so
    duplicates can still be told apart - the hashes are not shown anywhere."""
    import hashlib
    import unicodedata
    tokens = set()
    for name in authors or []:
        if not isinstance(name, str):
            continue
        folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
        for tok in "".join(ch if ch.isalpha() else " " for ch in folded).split():
            if len(tok) >= 3:
                tokens.add(hashlib.sha1(tok.encode()).hexdigest()[:10])
    return sorted(tokens)


def _keys_of(p: dict) -> set:
    return set(p.get("author_keys") or author_keys(p.get("authors")))


AUTHOR_OVERLAP_SAME = 0.5   # share of the shorter author fingerprint that must match


def same_paper(a: dict, b: dict) -> tuple:
    """(is_duplicate, reason). DOI first; otherwise the same title is only the same paper when the
    authors match too - or, if either side has no author list, when the years agree (+-1)."""
    da, db = (a.get("doi") or "").strip().lower(), (b.get("doi") or "").strip().lower()
    if da and da == db:
        return True, "same DOI"
    ta, tb = _title_key(a.get("title")), _title_key(b.get("title"))
    if not ta or ta != tb:
        return False, "different title"
    ka, kb = _keys_of(a), _keys_of(b)
    if ka and kb:
        overlap = len(ka & kb) / min(len(ka), len(kb))
        return (overlap >= AUTHOR_OVERLAP_SAME,
                f"same title, author overlap {overlap:.0%}")
    ya, yb = a.get("year"), b.get("year")
    if isinstance(ya, int) and isinstance(yb, int) and abs(ya - yb) > 1:
        return False, "same title, no author list, years differ"
    return True, "same title, no author list to compare, same year"


def _dedupe(papers: list[dict]) -> list[dict]:
    """One entry per paper (see same_paper). Before, a title was only compared when a paper had
    no DOI, so the same paper from arXiv (no DOI) and OpenAlex (DOI) - or a preprint and its
    published version (two DOIs) - was analysed twice or three times in a live run. Fields the
    kept entry is missing (abstract, PDF link, DOI, citations) are filled from the duplicate."""
    out, by_title = [], {}
    for p in papers:
        tkey = _title_key(p.get("title"))
        doi = (p.get("doi") or "").strip().lower()
        match = None
        for kept in by_title.get(tkey, []) + ([k for k in out if doi and (k.get("doi") or "").strip().lower() == doi]):
            dup, _ = same_paper(kept, p)
            if dup:
                match = kept
                break
        if match is not None:
            for field in ("abstract", "oa_pdf_url", "doi", "citation_count", "year"):
                if not match.get(field) and p.get(field):
                    match[field] = p[field]
            if not match.get("author_keys") and not match.get("authors") and (p.get("author_keys") or p.get("authors")):
                match["author_keys"] = sorted(_keys_of(p))
            match["duplicates_merged"] = match.get("duplicates_merged", 0) + 1
            continue
        out.append(p)
        if tkey:
            by_title.setdefault(tkey, []).append(p)
    return out
