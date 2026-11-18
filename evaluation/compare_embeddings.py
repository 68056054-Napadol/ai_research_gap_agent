"""
Compare embedding models on THIS project's retrieval job (the RAG knowledge base), not on a
generic benchmark.

    python evaluation/compare_embeddings.py
    python evaluation/compare_embeddings.py --models "sentence_transformers:BAAI/bge-small-en-v1.5" ollama:bge-m3

Task: for every paper in pdf_cache/ (real converted PDFs, already downloaded by earlier runs),
chunk it exactly like rag_store.py does, embed the chunks, then ask questions whose answer lives
in a known section - "what method do they propose?" should retrieve a `method` chunk, "what are
the limitations?" a `limitations` chunk. Section labels come from the paper's own headings, so no
manual labelling is needed (they are weak labels: a method can also be described in the intro).

Asked twice: in English, and in Thai (users may type Thai topics/questions; the papers are English).

Metrics per model:
  hit@3 (EN / TH)   share of questions where a chunk of the right section is in the top 3
  MRR (EN / TH)     1/rank of the first right-section chunk, averaged
  TH-EN agreement   overlap of the top-3 chunks for the Thai vs English version of a question
  chunks/sec        embedding throughput on this machine
Nothing from the papers' text is written to the results - only numbers.
"""
import argparse
import glob
import logging
import os
import time

import common  # sets sys.path
import numpy as np

import embeddings  # noqa: E402
import rag_store  # noqa: E402

DEFAULT_MODELS = [
    "sentence_transformers:BAAI/bge-small-en-v1.5",   # current default (English-only, 384-d)
    "ollama:bge-m3",                                  # multilingual, 1024-d
    "ollama:qwen3-embedding:0.6b",                    # multilingual, instruction-aware
    "ollama:embeddinggemma:300m",                     # multilingual, small
]

# (section, English question, Thai question)
QUESTIONS = [
    ("method", "What method or model architecture do the authors propose?",
     "ผู้เขียนเสนอวิธีการหรือสถาปัตยกรรมโมเดลแบบใด"),
    ("experiments", "Which datasets and experimental setup are used for evaluation?",
     "งานนี้ใช้ชุดข้อมูลและการตั้งค่าการทดลองอะไรในการประเมินผล"),
    ("results", "What are the main results compared with the baselines?",
     "ผลลัพธ์หลักเมื่อเทียบกับ baseline เป็นอย่างไร"),
    ("limitations", "What limitations or future work do the authors mention?",
     "ผู้เขียนกล่าวถึงข้อจำกัดหรืองานในอนาคตอะไรบ้าง"),
    ("related_work", "What prior work does this paper build on?",
     "งานนี้ต่อยอดจากงานวิจัยก่อนหน้าใดบ้าง"),
    ("conclusion", "What do the authors conclude?",
     "ผู้เขียนสรุปผลการศึกษาว่าอย่างไร"),
]

def fmt(spec: str):
    """Same per-model query/document formats the RAG pipeline uses (embeddings.EMBEDDING_FORMATS)."""
    q = embeddings.format_texts(spec, ["{q}"], "query")[0]
    d = embeddings.format_texts(spec, ["{d}"], "document")[0]
    return q, d


def embed(spec: str, texts: list) -> np.ndarray:
    backend, _, model = spec.partition(":")
    return embeddings._embed_for_rag(backend, model, texts)


def load_corpus(min_chars: int) -> list:
    papers = []
    for path in sorted(glob.glob(os.path.join(common.ROOT, "pdf_cache", "*.md"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if len(text) < min_chars:
            continue
        chunks = rag_store.chunk_text(text, "full_text")
        sections = {s for s, _ in chunks}
        papers.append({"id": os.path.basename(path)[:-3], "chunks": chunks, "sections": sections})
    return papers


def evaluate(spec: str, papers: list, k: int = 3) -> dict:
    qf, df = fmt(spec)
    all_chunks = [(p_i, s, c) for p_i, p in enumerate(papers) for s, c in p["chunks"]]
    start = time.monotonic()
    mat = np.vstack([embed(spec, [df.format(d=c) for _, _, c in all_chunks[i:i + 64]])
                     for i in range(0, len(all_chunks), 64)])
    secs = time.monotonic() - start
    q_texts = [qf.format(q=en) for _, en, _ in QUESTIONS] + [qf.format(q=th) for _, _, th in QUESTIONS]
    qv = embed(spec, q_texts)
    res = {"en_hits": [], "th_hits": [], "en_rr": [], "th_rr": [], "agree": []}
    owner = np.array([p_i for p_i, _, _ in all_chunks])
    for p_i, p in enumerate(papers):
        idx = np.where(owner == p_i)[0]
        for q_i, (section, _, _) in enumerate(QUESTIONS):
            if section not in p["sections"]:
                continue
            tops = {}
            for lang, row in (("en", q_i), ("th", q_i + len(QUESTIONS))):
                order = idx[np.argsort(-(mat[idx] @ qv[row]))]
                labels = [all_chunks[j][1] for j in order]
                res[f"{lang}_hits"].append(section in labels[:k])
                rank = labels.index(section) + 1 if section in labels else None
                res[f"{lang}_rr"].append(1 / rank if rank else 0.0)
                tops[lang] = set(order[:k].tolist())
            res["agree"].append(len(tops["en"] & tops["th"]) / k)
    return {
        "model": spec, "dims": int(mat.shape[1]), "n_chunks": len(all_chunks), "n_questions": len(res["en_hits"]),
        "hit@3_en": round(float(np.mean(res["en_hits"])), 3), "mrr_en": round(float(np.mean(res["en_rr"])), 3),
        "hit@3_th": round(float(np.mean(res["th_hits"])), 3), "mrr_th": round(float(np.mean(res["th_rr"])), 3),
        "th_en_agreement@3": round(float(np.mean(res["agree"])), 3),
        "chunks_per_sec": round(len(all_chunks) / secs, 1) if secs else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--min-chars", type=int, default=5000, help="skip cache entries shorter than this (failed PDFs)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    papers = load_corpus(args.min_chars)
    if not papers:
        raise SystemExit("pdf_cache/ has no converted papers yet - run the agent once with FETCH_FULL_TEXT=true")
    print(f"{len(papers)} papers, {sum(len(p['chunks']) for p in papers)} chunks")
    rows = []
    for spec in args.models:
        try:
            r = evaluate(spec, papers)
        except Exception as e:
            print(f"{spec}: FAILED ({type(e).__name__}: {e})")
            continue
        rows.append(r)
        print(r)

    cols = ["hit@3_en", "mrr_en", "hit@3_th", "mrr_th", "th_en_agreement@3", "chunks_per_sec", "dims"]
    md = ["# Embedding comparison - section retrieval inside real papers", "",
          f"- corpus: {len(papers)} papers from pdf_cache/, {rows[0]['n_chunks'] if rows else 0} chunks, "
          f"{rows[0]['n_questions'] if rows else 0} questions per language", "- chance level for hit@3 depends on how many "
          "chunks a paper has (~30-40), i.e. roughly 0.1-0.25 for a random ranking", "",
          "| Model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    md += [f"| {r['model']} | " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    jp, mp = common.save("compare_embeddings", {"rows": rows, "questions": QUESTIONS}, "\n".join(md))
    print("\n".join(md))
    print(f"\nSaved: {mp}")


if __name__ == "__main__":
    main()
