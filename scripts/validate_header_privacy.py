"""
Check PRIVACY_LLM_ON_PAPERS=auto against the old always-LLM behaviour on real paper headers.

For each paper: download its PDF (into a temporary folder that is deleted at the end - the raw
headers contain real names and are never kept), then
  1. run privacy.scrub_full_text in auto mode (does it call the LLM? why?),
  2. run the LLM entity extractor on the raw header as the reference (what the old mode found),
  3. count reference entities the decision model would REDACT that are still present in the
     auto-mode output = what auto mode would have leaked.
Only counts, types and masked forms are printed. Papers: OpenAlex candidates with an open-access
PDF from earlier runs (arXiv's API was rate-limiting this machine when this was written).

    python scripts/validate_header_privacy.py --n 12
"""
import argparse
import collections
import datetime
import json
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import decision_model  # noqa: E402
import privacy  # noqa: E402
from config import CONFIG  # noqa: E402
from sources import openalex, pdf_extract  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    # Papers: OpenAlex candidates with an open-access PDF from earlier runs' state files (OpenAlex
    # gives the author list back on lookup; the state files themselves never keep authors).
    import glob
    cands = {}
    for f in sorted(glob.glob("output/*_state.json")):
        for p in json.load(open(f, encoding="utf-8")).get("candidate_papers") or []:
            if p["id"].startswith("openalex:") and p.get("oa_pdf_url"):
                cands[p["id"]] = p
    ids = sorted(cands)
    random.Random(args.seed).shuffle(ids)
    metas = []
    for pid in ids[: args.n * 3]:
        try:
            m = openalex.get_paper(pid)
        except Exception as e:
            print(f"{pid}: lookup failed ({type(e).__name__})")
            continue
        if m and m.get("oa_pdf_url"):
            metas.append(m)

    rows, missed_all = [], collections.Counter()
    with tempfile.TemporaryDirectory() as tmp:
        CONFIG.pdf_cache_dir = tmp
        for meta in metas:
            if len(rows) >= args.n:
                break
            pid = meta["id"]
            if not meta.get("oa_pdf_url"):
                continue
            try:
                text = pdf_extract.fetch_and_convert(meta["oa_pdf_url"], pid)
            except Exception as e:   # one bad PDF must not end the check
                print(f"{pid}: PDF failed ({type(e).__name__})")
                continue
            if not text:
                continue
            authors, title = meta.get("authors") or [], meta.get("title", "")

            # 1. auto mode
            calls = []
            real_llm = privacy.llm_detect

            def spy(t, ch):
                calls.append(1)
                return real_llm(t, ch)
            privacy.llm_detect = spy
            CONFIG.privacy_llm_on_papers = "auto"
            privacy.reset_run()
            privacy.register_authors(authors)
            t0 = time.monotonic()
            out_auto, _ = privacy.scrub_full_text(text, pid, has_author_metadata=bool(authors), title=title)
            t_auto = time.monotonic() - t0
            privacy.llm_detect = real_llm

            # 2. reference: the LLM on the raw header (old behaviour)
            head = text[:CONFIG.privacy_header_chars]
            t0 = time.monotonic()
            ref = privacy.llm_detect(head, "paper_text")
            t_llm = time.monotonic() - t0

            # 3. what the reference would redact that auto mode left in place
            auto_head = out_auto[:CONFIG.privacy_header_chars + 200]
            missed = []
            for e in ref:
                d = decision_model.evaluate("entity_action", {"type": e["type"], "role": e.get("role", "other"),
                                                              "detected_by": "llm", "confidence": e.get("confidence", 0.8),
                                                              "channel": "paper_text"})
                if d["action"] == "REDACT" and e["text"] in auto_head:
                    missed.append(f"{e['type']}/{e.get('role')} {privacy.mask(e['text'])}")
                    missed_all[f"{e['type']}/{e.get('role')}"] += 1
            rows.append({"paper": pid, "authors_from_source": len(authors), "auto_called_llm": bool(calls),
                         "auto_seconds": round(t_auto, 1), "llm_seconds": round(t_llm, 1),
                         "reference_entities": len(ref), "missed_by_auto": missed})
            print(f"{pid}: auto LLM={'yes' if calls else 'no '} {t_auto:5.1f}s | reference {len(ref)} ents "
                  f"{t_llm:4.1f}s | missed {len(missed)} {missed}")
            privacy.reset_run()

    skipped = [r for r in rows if not r["auto_called_llm"]]
    saved = sum(r["llm_seconds"] for r in skipped)
    summary = {
        "papers": len(rows), "auto_skipped_llm": len(skipped),
        "seconds_saved": round(saved, 1),
        "missed_entities_total": sum(len(r["missed_by_auto"]) for r in rows),
        "missed_in_skipped_papers": sum(len(r["missed_by_auto"]) for r in skipped),
        "missed_by_type": dict(missed_all),
    }
    print(json.dumps(summary, indent=1))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("evaluation/results", exist_ok=True)
    path = f"evaluation/results/validate_header_privacy_{stamp}.json"
    json.dump({"summary": summary, "rows": rows}, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("Saved:", path)


if __name__ == "__main__":
    main()
