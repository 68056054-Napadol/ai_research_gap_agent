"""
Re-apply the (fixed) deterministic redaction to data earlier runs left on disk, IN PLACE - nothing
is deleted. Run after a privacy-rule fix (e.g. the Markdown-link e-mail leak):

    python scripts/rescrub_local_data.py            # dry run: only counts
    python scripts/rescrub_local_data.py --apply    # rewrite files

Covers pdf_cache/*.md, output/*.json + *.md, and the chunk text in every rag_store*/**/store.sqlite.
Uses regex detectors only (no LLM): e-mails, phones, ID numbers, "X et al." citations.
"""
import argparse
import glob
import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import outbound  # noqa: E402
import privacy  # noqa: E402


def scrub_text(text: str) -> tuple:
    n = len(outbound.violations(text))
    return (outbound.enforce(text, "rescrub") if n else text), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    privacy.reset_run()
    total = 0
    for path in sorted(glob.glob("pdf_cache/*.md") + glob.glob("output/*.md")):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        new, n = scrub_text(text)
        if n:
            total += n
            print(f"{path}: {n} item(s)")
            if args.apply:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new)
    for path in sorted(glob.glob("output/*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        n = len(outbound.violations(data))
        if n:
            total += n
            print(f"{path}: {n} item(s)")
            if args.apply:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(outbound.enforce(data, "rescrub"), f, indent=2, ensure_ascii=False, default=str)
    for db_path in sorted(glob.glob("rag_store*/**/store.sqlite", recursive=True) + glob.glob("rag_store*/store.sqlite")):
        db = sqlite3.connect(db_path)
        rows = db.execute("SELECT paper_id, idx, text FROM chunks").fetchall()
        changed = 0
        for pid, idx, text in rows:
            new, n = scrub_text(text)
            if n:
                changed += n
                if args.apply:
                    db.execute("UPDATE chunks SET text=? WHERE paper_id=? AND idx=?", (new, pid, idx))
        if args.apply:
            db.execute("DELETE FROM extracts")   # cached extractions may quote the old text
            db.commit()
        db.close()
        if changed:
            total += changed
            print(f"{db_path}: {changed} item(s) in chunk text")
    print(f"{'Redacted' if args.apply else 'Would redact'} {total} item(s) in total")


if __name__ == "__main__":
    main()
