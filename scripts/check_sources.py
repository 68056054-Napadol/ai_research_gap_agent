"""
One search per paper source, to check API keys and rate limits before a long run.
Prints status, result count and (for OpenAlex) the remaining daily budget - never the keys.

    python scripts/check_sources.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG  # noqa: E402
from sources import REGISTRY, net  # noqa: E402


def main():
    print(f"SEMANTIC_SCHOLAR_API_KEY: {'set' if CONFIG.semantic_scholar_api_key else 'not set'}")
    print(f"OPENALEX_API_KEY:         {'set' if CONFIG.openalex_api_key else 'not set'}")
    for name, module in REGISTRY.items():
        start = time.monotonic()
        try:
            papers = module.search_papers("short answer grading", 2023, 2026, limit=5)
            print(f"{name:17s} OK   {len(papers)} results in {time.monotonic() - start:.1f}s")
        except Exception as e:
            h = net.source_health().get(name, {})
            print(f"{name:17s} FAIL {type(e).__name__}: {h.get('last_error') or e}"[:200])
    if CONFIG.openalex_api_key:
        import httpx
        try:
            r = httpx.get("https://api.openalex.org/rate-limit", params={"api_key": CONFIG.openalex_api_key}, timeout=20)
            rl = r.json().get("rate_limit", {}) if r.status_code == 200 else {}
            print(f"openalex budget   HTTP {r.status_code}; remaining today: "
                  f"{rl.get('daily_remaining_usd', '?')} of {rl.get('daily_budget_usd', '?')} USD")
        except Exception as e:
            print(f"openalex budget   could not check ({type(e).__name__})")


if __name__ == "__main__":
    main()
