"""
Downloads an open-access PDF and converts it to Markdown for Node 5 (extract_paper_data in
nodes.py). Converting to Markdown instead of just dumping raw extracted text matters for
quality, not just looks: naive PDF text extraction interleaves repeated page headers/footers
and reading-order glitches from multi-column layouts into the middle of sentences, which wastes
the LLM's attention and hurts extraction accuracy. pymupdf4llm's heuristic (font-size-based
headers, paragraph grouping, column-aware reading order) produces much cleaner input for the
extraction prompt, for a fraction of the token cost of raw text.

Pinned to pymupdf4llm==0.0.17 on purpose (see requirements.txt) — the 1.x line pulls in an ONNX
layout-detection model plus a Tesseract OCR fallback by default, which is unnecessary weight and
CPU time for the digitally-native academic PDFs this project deals with, on hardware with no
GPU. 0.0.17 is the older, dependency-light heuristic converter (PyMuPDF only).

Results are cached to disk (config.pdf_cache_dir), keyed by paper id, so re-running the same
topic — or resuming after a crash mid-run — doesn't re-download and re-convert the same PDFs.
A paper whose PDF turned out to be unusable (404, encrypted, too large, not actually a PDF) is
also cached as "nothing found" so it isn't retried every single run.
"""
import logging
import os
import re

import httpx

from config import CONFIG

logger = logging.getLogger(__name__)


def _cache_path(paper_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", paper_id)
    return os.path.join(CONFIG.pdf_cache_dir, f"{safe}.md")


def fetch_and_convert(oa_pdf_url: str | None, paper_id: str) -> str | None:
    """
    Returns Markdown text extracted from the paper's open-access PDF, or None if no usable PDF
    was available. Never raises — every failure mode (no URL, 404, timeout, password-protected
    or corrupt PDF, oversized file, non-PDF response) degrades to None so the caller falls back
    to the abstract, matching this project's "graceful degradation" principle (architecture doc
    §7.1: a single bad source/paper should never abort the whole run).
    """
    if not oa_pdf_url:
        return None

    os.makedirs(CONFIG.pdf_cache_dir, exist_ok=True)
    cache_file = _cache_path(paper_id)
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info("    PDF full text: cache hit (%d chars)", len(text))
        return text or None  # an empty cached file means "tried before, found nothing usable"

    tmp_pdf = cache_file + ".download.pdf"
    text = ""
    try:
        _download_pdf(oa_pdf_url, tmp_pdf)
        text = _convert_to_markdown(tmp_pdf)
    except Exception as e:
        logger.warning("    PDF full text unavailable for %s (%s): %s", paper_id, oa_pdf_url, e)
    finally:
        if os.path.exists(tmp_pdf):
            os.remove(tmp_pdf)

    # Cache the outcome either way — including "nothing usable" — so a bad PDF isn't
    # re-downloaded and re-attempted on every future run of the same topic.
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(text)
    return text or None


def _download_pdf(url: str, dest_path: str) -> None:
    """
    Downloads to dest_path, then verifies the file actually starts with a %PDF- magic byte
    signature before returning. Trusting the URL suffix or the Content-Type header isn't enough:
    open-access links occasionally resolve to an HTML paywall/"access denied" page instead of the
    real PDF (some publishers block bots, or the OA copy moved), and without this check that page
    would silently get "converted" and fed to the LLM as if it were the paper's real text.
    """
    max_bytes = CONFIG.pdf_max_download_mb * 1024 * 1024
    timeout = httpx.Timeout(connect=15.0, read=30.0, write=15.0, pool=15.0)
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
        resp.raise_for_status()
        written = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_bytes():
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError(f"exceeds the {CONFIG.pdf_max_download_mb}MB cap, aborting download")
                f.write(chunk)

    with open(dest_path, "rb") as f:
        head = f.read(5)
    if head != b"%PDF-":
        content_type = resp.headers.get("content-type", "?")
        raise ValueError(f"response isn't actually a PDF (content-type={content_type!r}, "
                          f"first bytes={head!r}) — likely a paywall/redirect page")


def _convert_to_markdown(pdf_path: str) -> str:
    import pymupdf
    import pymupdf4llm

    doc = pymupdf.open(pdf_path)
    try:
        if doc.needs_pass:
            raise ValueError("PDF is password-protected")
        n_pages = min(len(doc), CONFIG.pdf_max_pages)
        # Capping pages (default 40 — see config.py) keeps a single huge PDF from stalling a
        # run; academic papers are almost always well under that anyway.
        md = pymupdf4llm.to_markdown(doc, pages=list(range(n_pages)), show_progress=False)
        return md.strip()
    finally:
        doc.close()
