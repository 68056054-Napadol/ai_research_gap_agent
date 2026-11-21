"""
Source quality filter (Node 3, before ranking), deterministic - README "Decision rules" rule 7b.

Why: a live OpenAlex run on 21 Nov 2026 selected 8 Zenodo uploads from 2026 out of 10 papers - near
identical titles, the same abstract, 0 citations. They are not reviewed papers, they crowd out real
ones, and a gap "supported" by eight copies of one text is not supported by eight papers.

  1. DOI prefix   a candidate whose DOI starts with CONFIG.exclude_doi_prefixes (default Zenodo 10.5281,
                  figshare 10.6084 - data/upload repositories) is dropped
  2. same abstract  of candidates whose abstracts are the same (first 300 normalised characters) only the
                  most cited one is kept

Every dropped candidate is returned with its reason; the report lists them in Search scope.
"""
import re


def _abstract_key(p: dict) -> str:
    a = re.sub(r"\W+", " ", (p.get("abstract") or "").lower()).strip()
    return a[:300] if len(a) >= 100 else ""          # short / missing abstracts are not compared


def apply(papers: list, cfg=None) -> tuple:
    """(kept, dropped [{id, title, reason}])."""
    if cfg is None:
        from config import CONFIG as cfg
    kept, dropped = [], []
    for p in papers:
        doi = (p.get("doi") or "").lower()
        bad = next((pre for pre in cfg.exclude_doi_prefixes if doi.startswith(pre + "/") or doi == pre), None)
        if bad:
            dropped.append({"id": p.get("id"), "title": p.get("title"), "reason": f"DOI prefix {bad} (data/upload repository)"})
        else:
            kept.append(p)
    if not cfg.drop_duplicate_abstracts:
        return kept, dropped
    groups = {}
    for p in kept:
        k = _abstract_key(p)
        if k:
            groups.setdefault(k, []).append(p)
    losers = set()
    for group in groups.values():
        if len(group) > 1:
            best = max(group, key=lambda p: (p.get("citation_count") or 0, -len(p.get("title") or "")))
            for p in group:
                if p is not best:
                    losers.add(id(p))
                    dropped.append({"id": p.get("id"), "title": p.get("title"),
                                    "reason": f"same abstract as {best.get('id')}"})
    return [p for p in kept if id(p) not in losers], dropped
