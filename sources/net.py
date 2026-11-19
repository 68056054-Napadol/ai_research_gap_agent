"""
Shared HTTP layer for the paper sources: per-source throttling, retry with backoff on 429/5xx
(honouring Retry-After), and a run-scoped circuit breaker plus health record.

Why: a live run sent 10 search calls to Semantic Scholar and OpenAlex and every one came back
429, so all candidates came from arXiv alone - and nothing in the answer said so. Now:
  - each source waits its documented minimum interval between calls,
  - a 429/5xx is retried a few times with backoff instead of failing at once,
  - after a source keeps failing it is switched off for the rest of the run (no more wasted
    seconds), and
  - source_health() reports which sources really answered, so the UI, the report and the gap
    confidence (nodes.analyze_gaps) can say "this corpus came from arXiv only".
"""
import logging
import threading
import time

import httpx

from config import CONFIG

logger = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}

# Minimum seconds between two calls to the same source. Semantic Scholar: 1 request/second with a
# key; the keyless pool is shared by everyone, so go slower. OpenAlex: 10/s with a key.
def _min_interval(source: str) -> float:
    if source == "semantic_scholar":
        return 1.1 if CONFIG.semantic_scholar_api_key else 3.0
    if source == "openalex":
        return 0.15
    if source == "arxiv":
        return 3.0
    return 0.0


_lock = threading.Lock()
_last_call: dict = {}
_health: dict = {}


def reset_run() -> None:
    with _lock:
        _health.clear()


def _h(source: str) -> dict:
    return _health.setdefault(source, {"ok": 0, "failed": 0, "retries": 0, "rate_limited": 0,
                                       "consecutive_failures": 0, "disabled": False, "last_error": ""})


class SourceUnavailable(RuntimeError):
    """Raised without a network call once a source has been switched off for this run."""


def source_health() -> dict:
    """{source: {ok, failed, retries, rate_limited, disabled, last_error}} for this run."""
    with _lock:
        return {k: dict(v) for k, v in _health.items()}


def answered_sources() -> list:
    """Sources that returned at least one successful response in this run."""
    return sorted(s for s, h in source_health().items() if h["ok"] > 0)


def _throttle(source: str) -> None:
    gap = _min_interval(source)
    if gap <= 0:
        return
    with _lock:
        wait = _last_call.get(source, 0.0) + gap - time.monotonic()
        _last_call[source] = time.monotonic() + max(wait, 0.0)
    if wait > 0:
        time.sleep(wait)


_BACKOFF_BASE = {"arxiv": 5.0}   # arXiv answers 429 without Retry-After and wants a real pause


def _retry_after(resp: httpx.Response, attempt: int, source: str = "") -> float:
    header = resp.headers.get("Retry-After", "")
    try:
        wait = float(header)
    except ValueError:
        wait = _BACKOFF_BASE.get(source, 2.0) * (2 ** attempt)   # 2s, 4s, 8s ... (arXiv 5s, 10s ...)
    return min(max(wait, 0.5), CONFIG.source_retry_max_wait)


def get(source: str, url: str, params: dict | None = None, headers: dict | None = None,
        timeout: float = 30, follow_redirects: bool = False) -> httpx.Response:
    """GET with throttling + retries. Returns the final response (raise_for_status is the caller's
    job, so a 404 can still mean "not found"). Raises SourceUnavailable if the source was switched
    off earlier in this run, or httpx errors after the last retry."""
    with _lock:
        h = _h(source)
        if h["disabled"]:
            raise SourceUnavailable(f"{source} skipped: {h['last_error']} (switched off for this run)")
    params = {k: v for k, v in (params or {}).items() if v is not None}
    attempts = CONFIG.source_max_retries + 1
    for attempt in range(attempts):
        _throttle(source)
        try:
            resp = httpx.get(url, params=params, headers=headers, timeout=timeout, follow_redirects=follow_redirects)
        except httpx.TransportError as e:
            if attempt + 1 < attempts:
                with _lock:
                    _h(source)["retries"] += 1
                time.sleep(min(2.0 * (2 ** attempt), CONFIG.source_retry_max_wait))
                continue
            _record_failure(source, f"network error: {type(e).__name__}")
            raise
        if resp.status_code in RETRY_STATUS and attempt + 1 < attempts:
            wait = _retry_after(resp, attempt, source)
            with _lock:
                _h(source)["retries"] += 1
            logger.info("    [%s] HTTP %d - retrying in %.1fs (attempt %d/%d)",
                        source, resp.status_code, wait, attempt + 2, attempts)
            time.sleep(wait)
            continue
        if resp.status_code in RETRY_STATUS:
            _record_failure(source, _explain(source, resp.status_code), rate_limited=resp.status_code == 429)
        elif resp.status_code < 400 or resp.status_code == 404:
            with _lock:
                _h(source)["ok"] += 1
                _h(source)["consecutive_failures"] = 0
        elif resp.status_code in (401, 403):
            key = {"semantic_scholar": "SEMANTIC_SCHOLAR_API_KEY", "openalex": "OPENALEX_API_KEY"}.get(source)
            _record_failure(source, f"key rejected (HTTP {resp.status_code})"
                            + (f" - check {key} in .env" if key else ""))
        else:
            _record_failure(source, f"HTTP {resp.status_code}")
        return resp
    raise AssertionError("unreachable")


def _explain(source: str, status: int) -> str:
    if status != 429:
        return f"HTTP {status} after retries"
    if source == "semantic_scholar" and not CONFIG.semantic_scholar_api_key:
        return "rate limited (429) - no SEMANTIC_SCHOLAR_API_KEY, so it shares the public pool"
    if source == "openalex" and not CONFIG.openalex_api_key:
        return "rate limited (429) - OpenAlex needs a free OPENALEX_API_KEY since Feb 2026"
    return "rate limited (429) after retries"


def _record_failure(source: str, reason: str, rate_limited: bool = False) -> None:
    with _lock:
        h = _h(source)
        h["failed"] += 1
        h["consecutive_failures"] += 1
        h["last_error"] = reason
        if rate_limited:
            h["rate_limited"] += 1
        # circuit breaker: N failed calls in a row (each already retried) switch the source off for
        # the rest of the run - a live run spent ~80 s retrying Semantic Scholar after it had stopped
        # answering, because one early call had succeeded
        if h["consecutive_failures"] >= CONFIG.source_disable_after_failures and not h["disabled"]:
            h["disabled"] = True
            logger.warning("    [%s] switched off for the rest of this run: %s", source, reason)
