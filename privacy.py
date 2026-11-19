"""
Sensitive-information handling = LLM extraction + decision model.

    text --> 3 detectors (EXTRACT candidates)            --> decision model (DECIDE)          --> redacted text
             1. regex     : emails, phones, ID numbers,       decision_model/sensitive_info      + decision log
                            "Surname et al." citations          .dmn.json, table entity_action
             2. metadata  : the paper's own author list       (hit policy FIRST, every decision
                            from the source API                 records which rule fired)
             3. LLM       : contracts.SensitiveEntitiesOutput
                            (names, roles like author vs
                            eponym "Kalman filter", affiliations)

Applied on three channels (the `channel` input of the decision table):
  - user_input   : the research topic, before anything else runs (Node 0 sanitize_input);
                   a second table (input_action) decides PROCEED / PROCEED_REDACTED / REJECT
  - paper_text   : every abstract / title / PDF full text, before an LLM or the UI sees it
                   (search tool + Node 5)
  - final_answer : the trend summary + gap claims/rationales/verifier notes (Node 8 privacy_guard)

Paper author lists are used only as an in-memory gazetteer for this run (never stored in the
agent state, report, or output JSON) and then dropped from every paper dict.
The decision log stores a masked form ("J*** D**"), never the original string.
"""
import logging
import re
import threading

import decision_model
import llm_client
import metrics
import prompts
import tracing
from config import CONFIG
from contracts import SensitiveEntitiesOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Run-scoped gazetteer of author names (memory only)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_gazetteer: set = set()
_gazetteer_re = None


def reset_run() -> None:
    global _gazetteer_re
    with _lock:
        _gazetteer.clear()
        _gazetteer_re = None


def _name_variants(name: str) -> set:
    name = " ".join((name or "").split())
    parts = [p for p in re.split(r"\s+", name) if p]
    out = set()
    if len(name) >= 4:
        out.add(name)
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        out.add(f"{first} {last}")
        out.add(f"{first[0]}. {last}")
        out.add(f"{first[0]}.{last}")
    return {v for v in out if len(v) >= 4}


def register_authors(authors) -> None:
    global _gazetteer_re
    new = set()
    for a in authors or []:
        if isinstance(a, str):
            new |= _name_variants(a)
    with _lock:
        if not new - _gazetteer:
            return
        _gazetteer.update(new)
        _gazetteer_re = None


def _gazetteer_regex():
    global _gazetteer_re
    with _lock:
        if _gazetteer_re is None and _gazetteer:
            alts = sorted(_gazetteer, key=len, reverse=True)
            _gazetteer_re = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(a) for a in alts) + r")(?!\w)")
        return _gazetteer_re


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# grouped addresses as printed in paper headers: {alice, bob}@lab.example.com
_EMAIL_GROUP_RE = re.compile(r"\{[^{}\n]{1,400}\}\s*@[\w-]+(?:\.[\w-]+)+")
# phones start with "+", "0" or "(" (international / Thai local format) - this keeps ISBNs, grant
# numbers, years and page ranges out
_PHONE_RE = re.compile(r"(?<![\w.,])(?:\+\d{1,3}[\s-]?\d{1,3}|0\d{1,2}|\(0\d{1,2}\))(?:[\s-]?\d{3,4}){2}(?![\w.])")
_THAI_ID_RE = re.compile(r"(?<!\d)\d[\s-]?\d{4}[\s-]?\d{5}[\s-]?\d{2}[\s-]?\d(?!\d)")
# "Smith et al." / "Smith and Jones (2020)" / "Smith & Jones" in citations
_CITATION_RE = re.compile(
    r"\b([A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?)(?:\s+(?:and|&)\s+([A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?))?\s+et\s+al\.?"
    r"|\b([A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?)\s+(?:and|&)\s+([A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?)\s+\(\s*(?:19|20)\d{2}"
)


def regex_detect(text: str) -> list:
    ents = []
    for m in _EMAIL_GROUP_RE.finditer(text):
        ents.append({"text": m.group(0), "type": "EMAIL", "role": "contact", "detected_by": "regex", "confidence": 1.0})
    for m in _EMAIL_RE.finditer(text):
        ents.append({"text": m.group(0), "type": "EMAIL", "role": "contact", "detected_by": "regex", "confidence": 1.0})
    for m in _THAI_ID_RE.finditer(text):
        ents.append({"text": m.group(0), "type": "ID_NUMBER", "role": "other", "detected_by": "regex", "confidence": 0.9})
    for m in _PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        # avoid years, arXiv ids, page ranges: require phone-like length AND a separator
        if 9 <= len(digits) <= 13 and re.search(r"[\s()+-]", m.group(0).strip()):
            ents.append({"text": m.group(0).strip(), "type": "PHONE", "role": "contact", "detected_by": "regex", "confidence": 0.8})
    for m in _CITATION_RE.finditer(text):
        for g in m.groups():
            if g:
                ents.append({"text": g, "type": "PERSON", "role": "cited_author", "detected_by": "regex", "confidence": 0.8})
    return ents


def metadata_detect(text: str) -> list:
    rx = _gazetteer_regex()
    if rx is None:
        return []
    return [{"text": m.group(0), "type": "PERSON", "role": "author", "detected_by": "metadata", "confidence": 1.0}
            for m in rx.finditer(text)]


def llm_detect(text: str, channel: str) -> list:
    """LLM entity extraction (contract-validated). Entities whose text does not literally occur
    in the input are dropped - the LLM can only point at text, not invent it."""
    if not text.strip():
        return []
    try:
        # this call necessarily sees the raw text - its trace records a marker, not the content
        with tracing.withheld(f"{channel} text inspected for personal data before redaction"):
            out = llm_client.call_json(prompts.SENSITIVE_EXTRACT_PROMPT.format(channel=channel, text=text),
                                       tier="fast", schema=SensitiveEntitiesOutput, purpose="detect-personal-data")
    except llm_client.LLMError as e:
        logger.warning("Privacy LLM extraction failed (regex/metadata detectors still apply): %s", e)
        metrics.inc("privacy_llm_failures")
        return []
    ents = []
    for e in out["entities"]:
        if e["text"] in text:
            ents.append({**e, "detected_by": "llm"})
        else:
            metrics.inc("privacy_llm_entities_not_in_text")
    return ents


# ---------------------------------------------------------------------------
# Decision + redaction
# ---------------------------------------------------------------------------
def mask(s: str) -> str:
    return " ".join(w[0] + "*" * (len(w) - 1) for w in s.split())


def _replace(text: str, needle: str, label: str) -> tuple:
    if not needle:
        return text, 0
    if re.match(r"\w", needle) and re.search(r"[A-Za-z]", needle):
        # letter boundaries only: PDF headers glue superscript markers to names/affiliations
        # ("1AWS AI Labs", "Smith2"), which a \w boundary would refuse to match
        # (no "[" / "]" exclusion: PDF Markdown wraps e-mails as [a@b.c](mailto:...) - a live run
        # leaked one through that gap. Placeholders like [AUTHOR] are safe anyway: an entity's own
        # text never equals a placeholder's upper-case word.)
        pattern = re.compile(r"(?<![A-Za-z])" + re.escape(needle) + r"(?![A-Za-z])")
    else:  # Thai / symbols: no word boundaries in Thai script
        pattern = re.compile(re.escape(needle))
    return pattern.subn(label, text)


def decide_and_redact(text: str, entities: list, channel: str, where: str, count_metrics: bool = True) -> tuple:
    """Run every entity through the entity_action decision table, apply REDACT decisions.
    Returns (redacted_text, decision_records)."""
    records, seen = [], set()
    # longest first so "Jane Doe" is replaced before "Doe"
    for ent in sorted(entities, key=lambda e: len(e["text"]), reverse=True):
        key = (ent["text"], ent["type"])
        if key in seen:
            continue
        seen.add(key)
        decision = decision_model.evaluate("entity_action", {
            "type": ent["type"], "role": ent.get("role", "other"), "detected_by": ent["detected_by"],
            "confidence": ent.get("confidence", 0.8), "channel": channel,
        })
        n = 0
        if decision["action"] == "REDACT":
            text, n = _replace(text, ent["text"], decision["label"])
        records.append({
            "channel": channel, "where": where, "type": ent["type"], "role": ent.get("role", "other"),
            "detected_by": ent["detected_by"], "confidence": round(float(ent.get("confidence", 0.8)), 2),
            "rule": decision["rule"], "action": decision["action"], "label": decision["label"],
            "occurrences": n, "masked": mask(ent["text"]),
        })
        if count_metrics:
            metrics.inc(f"privacy_{decision['action'].lower()}")
    return text, records


def sanitize(text: str, channel: str, where: str, use_llm: bool) -> tuple:
    if not text:
        return text, []
    ents = regex_detect(text) + metadata_detect(text)
    if use_llm:
        ents += llm_detect(text, channel)
    return decide_and_redact(text, ents, channel, where)


# ---------------------------------------------------------------------------
# Outbound contract helpers (no LLM): used by tracing.mask and save_outputs
# ---------------------------------------------------------------------------
def find_pii(text: str) -> list:
    """Deterministic PII check for text leaving the system: regex detectors + this run's author
    list. (No LLM call - this runs on every trace attribute and every saved file.)"""
    if not text:
        return []
    return [e for e in regex_detect(text) + metadata_detect(text) if e["type"] != "PERSON" or e["detected_by"] == "metadata"
            or e["role"] == "cited_author"]


def mask_text(text: str) -> str:
    """Redact whatever find_pii finds, through the same decision table (channel=final_answer)."""
    ents = find_pii(text)
    if not ents:
        return text
    out, _ = decide_and_redact(text, ents, "final_answer", "outbound", count_metrics=False)
    return out


def mask_any(obj):
    """Recursively apply mask_text to every string in a JSON-like structure."""
    if isinstance(obj, str):
        return mask_text(obj)
    if isinstance(obj, dict):
        return {k: mask_any(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [mask_any(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Channel helpers used by nodes.py / tools.py
# ---------------------------------------------------------------------------
_STOPWORDS_EN = {"paper", "papers", "research", "work", "works", "study", "studies", "about", "find", "search",
                 "the", "and", "for", "with", "from", "his", "her", "their", "published", "written", "author",
                 "authors", "prof", "professor", "dr", "phd", "all", "related", "topic", "topics", "by", "of"}
_STOPWORDS_TH = ["งานวิจัย", "ผลงาน", "บทความ", "เกี่ยวกับ", "ของ", "ค้นหา", "หา", "ช่วย", "ทั้งหมด",
                 "ดร.", "ดร", "อาจารย์", "ศาสตราจารย์", "รองศาสตราจารย์", "ผศ.", "รศ.", "ศ.", "คุณ", "นาย", "นาง", "นางสาว", "ที่", "เขียน", "โดย"]


def content_words_left(text: str) -> int:
    t = re.sub(r"\[[A-Z_]+\]", " ", text)
    for w in sorted(_STOPWORDS_TH, key=len, reverse=True):
        t = t.replace(w, " ")
    n = 0
    for w in re.findall(r"[^\W\d_][^\W\d_-]*(?:-[^\W\d_]+)*", t):   # unicode letter runs, any script
        if w.isascii():
            n += len(w) >= 3 and w.lower() not in _STOPWORDS_EN
        else:
            n += len(w) >= 2          # Thai/CJK/... run (Thai stopwords were removed above)
    return n


_PLACEHOLDER_RE = re.compile(r"(?:\b(?:by|from|of|authored by|written by)\s+|(?:ของ|โดย)\s*)?\[[A-Z_]+\]", re.I)


def strip_placeholders(text: str) -> str:
    """'few-shot NER papers by [PERSON] for Thai' -> 'few-shot NER papers for Thai'. Used for
    anything sent to a search API or a planning prompt: a placeholder is not a search term."""
    out = _PLACEHOLDER_RE.sub(" ", text or "")
    return " ".join(out.split())


def sanitize_topic(topic: str) -> dict:
    redacted, records = sanitize(topic, channel="user_input", where="research_topic", use_llm=True)
    redacted_records = [r for r in records if r["action"] == "REDACT" and r["occurrences"]]
    decision = decision_model.evaluate("input_action", {
        "n_person_redacted": sum(1 for r in redacted_records if r["type"] == "PERSON"),
        "n_contact_redacted": sum(1 for r in redacted_records if r["type"] != "PERSON"),
        "content_words_left": content_words_left(redacted),
    })
    return {"topic": " ".join(redacted.split()), "decision": decision, "records": records}


def scrub_paper(paper: dict) -> list:
    """Register the paper's authors in the gazetteer, drop them from the dict, redact title/abstract.
    Mutates `paper`; returns decision records."""
    authors = paper.pop("authors", None) or []
    register_authors(authors)
    paper["n_authors"] = len(authors)
    if authors and not paper.get("author_keys"):
        from sources import author_keys
        paper["author_keys"] = author_keys(authors)   # one-way hashes, for duplicate detection only
    records = []
    for fld in ("title", "abstract"):
        if paper.get(fld):
            paper[fld], recs = sanitize(paper[fld], channel="paper_text", where=f"{paper.get('id')}:{fld}", use_llm=False)
            records += recs
    return records


# ---------------------------------------------------------------------------
# Paper headers: deterministic affiliation/address lines + "does this header still need the LLM?"
# ---------------------------------------------------------------------------
_ABSTRACT_HEADING_RE = re.compile(r"(?im)^[\W_]{0,8}abstract\b")
_AFFIL_RE = re.compile(
    r"(Universit|Institut|Laborator|\bLabs?\b|Department|\bDept\b|College|School of|Faculty|Academy|"
    r"Research (?:Center|Centre|Institute|Lab)|Cent(?:er|re) for|\bInc\b|\bLtd\b|Corporation|\bCorp\b|GmbH|"
    r"\bLLC\b|Hospital|Foundation|Polytechnic|\bIIT\b|Google|Microsoft|\bMeta\b|DeepMind|Amazon|\bIBM\b|"
    r"NVIDIA|Alibaba|Tencent|Baidu|Huawei|ByteDance|Samsung|OpenAI|"
    r"มหาวิทยาลัย|สถาบัน|คณะ|ภาควิชา|ศูนย์)", re.I)
_COUNTRIES = {
    "usa", "u.s.a.", "united states", "united states of america", "uk", "united kingdom", "england", "canada",
    "china", "p.r. china", "pr china", "japan", "korea", "south korea", "republic of korea", "taiwan", "hong kong",
    "singapore", "india", "pakistan", "bangladesh", "thailand", "vietnam", "viet nam", "indonesia", "malaysia",
    "philippines", "australia", "new zealand", "germany", "france", "italy", "spain", "portugal", "netherlands",
    "the netherlands", "belgium", "switzerland", "austria", "sweden", "norway", "denmark", "finland", "poland",
    "czech republic", "czechia", "greece", "turkey", "israel", "iran", "saudi arabia", "uae",
    "united arab emirates", "qatar", "egypt", "nigeria", "kenya", "south africa", "brazil", "mexico", "argentina",
    "chile", "colombia", "russia", "ukraine", "ireland", "scotland", "hungary", "romania", "slovakia", "slovenia",
    "estonia", "latvia", "lithuania", "ethiopia", "morocco", "tunisia", "nepal", "sri lanka", "ไทย", "ประเทศไทย"}
_NAME_SEGMENT_RE = re.compile(r"^[A-Z][a-zA-Z'\u2019-]+(?:\s+(?:[A-Z]\.|[A-Z][a-zA-Z'\u2019-]+)){1,3}$")
_NON_LATIN_LETTER_RE = re.compile(r"[^\W\d_a-zA-Z\u00C0-\u024F]")
_MARKUP_RE = re.compile(r"\[(?:AUTHOR|AFFILIATION|CONTACT|PERSON|CITED_AUTHOR)\]|[*_#>`|\[\]()^{}]|[\d\u2020\u2021\u00a7\u00b6\u2217]+")


def header_region(text: str) -> str:
    """The part of a paper before its Abstract heading (title, authors, affiliations, emails),
    capped at privacy_header_chars."""
    head = text[:CONFIG.privacy_header_chars]
    m = _ABSTRACT_HEADING_RE.search(head)
    return head[:m.start()] if m else head


def _header_lines(head: str, title: str = "") -> list:
    title_key = re.sub(r"\W+", "", (title or "").lower())[:40]
    out = []
    for i, line in enumerate(head.splitlines()):
        line = line.strip()
        if not line:
            continue
        key = re.sub(r"\W+", "", line.lower())
        if line.startswith("#") or (title_key and key and (key.startswith(title_key) or title_key.startswith(key[:40]))):
            continue   # the title is not personal data
        out.append(line)
    return out


def affiliation_detect(head: str, title: str = "") -> list:
    """Whole header lines that name an institution, and short lines that end in a country
    (postal addresses). Deterministic stand-in for what the LLM mostly found in paper headers."""
    ents = []
    for line in _header_lines(head, title):
        bare = line.strip("*_ ").rstrip(".")
        if len(bare) > 240:
            continue
        if _AFFIL_RE.search(bare):
            ents.append({"text": bare, "type": "ORGANIZATION", "role": "affiliation", "detected_by": "regex", "confidence": 0.9})
        elif len(bare) <= 90 and bare.split(",")[-1].strip().lower().strip(".") in _COUNTRIES:
            ents.append({"text": bare, "type": "ADDRESS", "role": "affiliation", "detected_by": "regex", "confidence": 0.85})
    return ents


def header_needs_llm(head_redacted: str, has_author_metadata: bool, title: str = "") -> str:
    """'' when regex/metadata/affiliation detectors can vouch for this (already redacted) header,
    else the reason the LLM pass is still needed."""
    if not has_author_metadata:
        return "source gave no author list"
    if _NON_LATIN_LETTER_RE.search(_MARKUP_RE.sub(" ", head_redacted)):
        return "non-Latin script in the header"
    for line in _header_lines(head_redacted, title):
        for seg in re.split(r",|;|\band\b|&", _MARKUP_RE.sub(" ", line)):
            seg = " ".join(seg.split())
            if seg and seg.lower() not in _COUNTRIES and _NAME_SEGMENT_RE.match(seg):
                return "a name-like line survived the deterministic detectors"
    return ""


def scrub_full_text(text: str, paper_id: str, has_author_metadata: bool = True, title: str = "") -> tuple:
    """Full text: regex + gazetteer over everything; affiliation/address lines in the header; the
    LLM extractor over the header only when needed (CONFIG.privacy_llm_on_papers)."""
    header_len = CONFIG.privacy_header_chars
    head, body = text[:header_len], text[header_len:]
    region = header_region(head)
    det = regex_detect(head) + metadata_detect(head) + affiliation_detect(region, title)
    mode = CONFIG.privacy_llm_on_papers
    llm_ents, reason = [], ""
    if mode == "always":
        reason = "PRIVACY_LLM_ON_PAPERS=always"
    elif mode == "auto":
        if "[AUTHOR]" in region and "[AFFILIATION]" in region:
            reason = ""          # header already scrubbed (cached PDF from an earlier run)
        else:
            probe, _ = decide_and_redact(region, det, "paper_text", f"{paper_id}:probe", count_metrics=False)
            reason = header_needs_llm(probe, has_author_metadata, title)
    if reason:
        llm_ents = llm_detect(head, "paper_text")
        metrics.inc("privacy_header_llm_calls")
        logger.info("    [privacy] header LLM check for %s: %s", paper_id, reason)
    elif mode != "never":
        metrics.inc("privacy_header_llm_skipped")
    head, rec_h = decide_and_redact(head, det + llm_ents, "paper_text", f"{paper_id}:full_text_header")
    # names the LLM found in the header (authors) are also redacted wherever they recur in the body
    body_ents = regex_detect(body) + metadata_detect(body) + [e for e in llm_ents if e["text"] in body]
    body, rec_b = decide_and_redact(body, body_ents, "paper_text", f"{paper_id}:full_text_body")
    return head + body, rec_h + rec_b
