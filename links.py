"""
Paper links, built by CODE from source metadata - never written by the LLM.

LLMs invent plausible-looking URLs and ids (a live run already produced a mistyped OpenAlex id),
so the model only ever cites `paper_id`s; after verification, nodes.analyze_gaps maps each id to
a real paper dict and this module turns that dict into a link:

    DOI present          -> https://doi.org/<doi>              (preferred: the canonical landing page)
    arxiv:<id>           -> https://arxiv.org/abs/<id>
    openalex:<W...>      -> https://openalex.org/<W...>
    semantic_scholar:<id>-> https://www.semanticscholar.org/paper/<id>
"""
from urllib.parse import quote

_BY_SOURCE = {
    "arxiv": "https://arxiv.org/abs/{}",
    "openalex": "https://openalex.org/{}",
    "semantic_scholar": "https://www.semanticscholar.org/paper/{}",
}


def paper_url(paper: dict) -> str | None:
    doi = (paper.get("doi") or "").strip()
    if doi:
        doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        return "https://doi.org/" + quote(doi, safe="/:;()._-")
    pid = paper.get("id") or ""
    source, _, raw = pid.partition(":")
    template = _BY_SOURCE.get(source)
    return template.format(quote(raw, safe="._-")) if template and raw else None


def supporting_papers(paper_ids: list, papers_by_id: dict) -> list:
    """[{id, title, year, url}] for ids that exist among the analyzed papers - anything else is
    dropped, so a link can only ever point at a paper the agent really retrieved."""
    out = []
    for pid in paper_ids:
        p = papers_by_id.get(pid)
        if p is None:
            continue
        out.append({"id": pid, "title": p.get("title") or pid, "year": p.get("year"),
                    "url": p.get("url") or paper_url(p)})
    return out


def markdown_link(sp: dict) -> str:
    title = str(sp["title"]).replace("[", "(").replace("]", ")")
    label = f"{title} ({sp.get('year') or 'n.d.'})"
    return f"[{label}]({sp['url']})" if sp.get("url") else label
