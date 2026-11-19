"""
arXiv API — https://info.arxiv.org/help/api/user-manual.html
No key needed; arXiv asks for at most 1 request / 3 seconds (enforced in sources/net.py).
Essential source for CS/NLP topics; has no citation-count field.

Query building: `all:automatic short answer grading` is NOT a phrase search - arXiv applies the
`all:` prefix to the first word only and ORs the rest by relevance, so a live run pulled in papers
on genome interpretation and quantum machine learning for an automatic-grading topic. Content words
are now ANDed (`all:automatic AND all:short AND ...`), the year range is applied server-side via
submittedDate, and only when the strict query finds too little is it relaxed.
"""
import re
import xml.etree.ElementTree as ET

from . import net

BASE = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}

_STOPWORDS = {"a", "an", "and", "are", "as", "at", "based", "by", "for", "from", "in", "into", "is", "of", "on",
              "or", "the", "to", "using", "via", "with", "without", "towards", "toward", "approach", "approaches",
              "method", "methods", "study", "studies", "techniques", "technique", "paper", "papers", "research"}
# Words so common in CS abstracts that requiring them adds nothing; the relaxed query drops them.
_GENERIC = {"large", "language", "languages", "model", "models", "learning", "machine", "deep", "neural", "ai",
            "system", "systems", "data", "evaluation", "automated", "automatic", "new", "novel", "improving",
            "efficient", "framework", "task", "tasks"}
MIN_STRICT_RESULTS = 5   # fewer than this from the AND query -> retry with the relaxed query


def content_terms(query: str) -> list:
    """Words that carry the topic: lowercased, stopwords and 1-letter tokens dropped, hyphenated
    words kept whole (arXiv tokenises "few-shot" as a phrase)."""
    words = re.findall(r"[\w][\w-]*", query.lower())
    return list(dict.fromkeys(w for w in words if w not in _STOPWORDS and len(w) > 1))


def _term(t: str) -> str:
    return f'all:"{t.replace("-", " ")}"' if "-" in t else f"all:{t}"


def build_query(query: str, year_from=None, year_to=None, strict: bool = True) -> str:
    terms = content_terms(query) or [query.strip()]
    if strict:
        core = " AND ".join(_term(t) for t in terms)
    else:
        # relaxed: only the topic-specific words must appear (generic CS words dropped)
        must = [t for t in terms if t not in _GENERIC] or sorted(terms, key=len, reverse=True)[:3]
        core = " AND ".join(_term(t) for t in must)
    if len(terms) > 1:
        core = f"({core})"
    if year_from or year_to:
        lo = f"{year_from or 1990}01010000"
        hi = f"{year_to or 2100}12312359"
        core += f" AND submittedDate:[{lo} TO {hi}]"
    return core


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


def _fetch(search_query: str, limit: int) -> list[dict]:
    resp = net.get("arxiv", BASE, params={"search_query": search_query, "start": 0, "max_results": min(limit, 100),
                                          "sortBy": "relevance", "sortOrder": "descending"},
                   follow_redirects=True)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    return [_to_paper(e) for e in root.findall("atom:entry", NS)]


def search_papers(query: str, year_from=None, year_to=None, limit: int = 30) -> list[dict]:
    papers = _fetch(build_query(query, year_from, year_to, strict=True), limit)
    if len(papers) < MIN_STRICT_RESULTS and len(content_terms(query)) > 3:
        seen = {p["id"] for p in papers}
        papers += [p for p in _fetch(build_query(query, year_from, year_to, strict=False), limit) if p["id"] not in seen]
    if year_from or year_to:   # submittedDate is the first version; keep the guard for the published year
        papers = [p for p in papers if p["year"] is None or (
            (year_from is None or p["year"] >= year_from) and (year_to is None or p["year"] <= year_to)
        )]
    return papers


def get_paper(paper_id: str) -> dict | None:
    raw_id = paper_id.split(":", 1)[-1]
    resp = net.get("arxiv", BASE, params={"id_list": raw_id}, follow_redirects=True)
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
