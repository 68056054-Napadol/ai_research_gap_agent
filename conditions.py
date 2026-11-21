"""
User conditions for a search (README "User conditions") - MVP step 1: "รับหัวข้อ ช่วงปี และเงื่อนไขจากผู้ใช้".

Besides the topic and the year range, the user can give:

  focus    free text, e.g. "Thai and Vietnamese only, transformer-based methods". It is passed to the
           planning (Node 1), search (Node 2) and gap (Node 7) prompts. It goes through the same
           privacy check as the topic (Node 0) before any prompt sees it.
  include  terms a paper's title or abstract MUST contain. Comma-separated; every term is required
           (AND); "a|b" inside one term means either is enough (OR). "NER, Thai|Vietnamese" = the
           paper mentions NER and (Thai or Vietnamese).
  exclude  terms that drop a paper when its title or abstract contains any of them, e.g. "survey, review".

include / exclude are applied by code (Node 3, before ranking), never by the LLM, and every paper they
drop is listed in the report's Search scope with the reason - so a reader can see what was filtered.
Matching is case-insensitive and on whole words for Latin-script terms ("NER" does not match
"inner"); Thai and other scripts without spaces match as substrings.
"""
import re

MAX_TERMS = 10
MAX_FOCUS_CHARS = 300


def parse_terms(raw) -> list:
    """'NER, Thai|Vietnamese' -> [['ner'], ['thai', 'vietnamese']]. Accepts a string or a list."""
    if raw is None:
        return []
    parts = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")
    out = []
    for part in parts:
        alts = part if isinstance(part, (list, tuple)) else str(part).split("|")
        alts = [" ".join(a.split()).lower() for a in alts if a and str(a).strip()]
        alts = [a for a in alts if len(a) >= 2]
        if alts:
            out.append(list(dict.fromkeys(alts)))
    return out[:MAX_TERMS]


def normalize(focus: str = "", include=None, exclude=None) -> dict:
    """The conditions as they are stored in state (`search_conditions`)."""
    # in an exclude list "|" means nothing special - every listed term excludes
    excl = [a for alts in parse_terms(exclude) for a in alts]
    return {
        "focus": " ".join((focus or "").split())[:MAX_FOCUS_CHARS],
        "include": parse_terms(include),
        "exclude": list(dict.fromkeys(excl))[:MAX_TERMS],
    }


def is_empty(cond: dict | None) -> bool:
    return not cond or not (cond.get("focus") or cond.get("include") or cond.get("exclude"))


def _pattern(term: str) -> re.Pattern:
    if re.fullmatch(r"[\x00-\x7f]+", term):            # Latin/ASCII: whole words only
        return re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", re.I)
    return re.compile(re.escape(term), re.I)            # Thai etc.: no spaces between words


def _contains(text: str, term: str) -> bool:
    return bool(_pattern(term).search(text or ""))


def check_paper(paper: dict, cond: dict | None) -> tuple:
    """(keep, reason). Title + abstract only - what every source returns."""
    if is_empty(cond):
        return True, ""
    text = f"{paper.get('title') or ''}\n{paper.get('abstract') or ''}"
    for term in cond.get("exclude") or []:
        if _contains(text, term):
            return False, f"contains excluded term “{term}”"
    for alts in cond.get("include") or []:
        if not any(_contains(text, a) for a in alts):
            return False, "missing required term “" + " | ".join(alts) + "”"
    return True, ""


def apply(papers: list, cond: dict | None) -> tuple:
    """(kept papers, [{id, title, reason}] of the dropped ones)."""
    kept, dropped = [], []
    for p in papers:
        ok, why = check_paper(p, cond)
        if ok:
            kept.append(p)
        else:
            dropped.append({"id": p.get("id"), "title": p.get("title"), "reason": why})
    return kept, dropped


def describe(cond: dict | None) -> list:
    """Human-readable lines (prompts, report, UI)."""
    if is_empty(cond):
        return []
    lines = []
    if cond.get("focus"):
        lines.append(f"Focus: {cond['focus']}")
    if cond.get("include"):
        lines.append("Papers must mention: " + "; ".join(" or ".join(a) for a in cond["include"]))
    if cond.get("exclude"):
        lines.append("Papers must NOT mention: " + ", ".join(cond["exclude"]))
    return lines


def prompt_block(cond: dict | None) -> str:
    """Text for the {conditions} slot of a prompt; empty string when there are none."""
    import prompts
    lines = describe(cond)
    if not lines:
        return ""
    return prompts.CONDITIONS_TEMPLATE.format(lines="\n".join(f"- {line}" for line in lines))
