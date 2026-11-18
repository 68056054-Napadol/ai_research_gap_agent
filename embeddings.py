"""
Local embedding + cosine-similarity ranking (architecture doc §4.3, §7.2 stage 1).
Runs on CPU, never leaves the machine, and needs no API key — this is what keeps the search
node from having to pay an LLM call just to rank a hundred abstracts against the topic.
"""
import logging
import time

import numpy as np
from functools import lru_cache

from config import CONFIG

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _model():
    # Imported lazily so `python main.py --help` doesn't pay the model-load cost.
    # Logged because the first call downloads/loads the model and can take 10-30s — without
    # this line it just looks like the agent is stuck.
    logger.info("Loading embedding model %s (first call only, may take a while)...", CONFIG.embedding_model_name)
    start = time.monotonic()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(CONFIG.embedding_model_name)
    logger.info("Embedding model ready (%.1fs)", time.monotonic() - start)
    return model


def embed(texts: list[str]) -> np.ndarray:
    """Returns an (N, D) array of L2-normalized embeddings."""
    import tracing
    with tracing.observe("embed-texts", "embedding", model=CONFIG.embedding_model_name,
                         input={"n_texts": len(texts)}, metadata={"purpose": "rank candidates"} if len(texts) > 1 else None) as obs:
        vectors = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
        obs.update(output={"n_vectors": len(vectors)})
    return np.asarray(vectors)


def _normalize(vectors) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n == 0, 1, n)


@lru_cache(maxsize=4)
def _st_model(name: str):
    from sentence_transformers import SentenceTransformer
    logger.info("Loading RAG embedding model %s ...", name)
    return SentenceTransformer(name)


# Query/document formatting each model's card prescribes. Using a model without its format
# throws away quality (evaluation/compare_embeddings.py measures with these same formats).
EMBEDDING_FORMATS = {
    "bge-small-en": ("Represent this sentence for searching relevant passages: {t}", "{t}"),
    "qwen3-embedding": ("Instruct: Given a question about a research paper, retrieve the passage that answers it\nQuery: {t}", "{t}"),
    "embeddinggemma": ("task: search result | query: {t}", "title: none | text: {t}"),
}


def format_texts(spec: str, texts: list[str], kind: str) -> list[str]:
    """kind = 'query' or 'document'."""
    for key, (q, d) in EMBEDDING_FORMATS.items():
        if key in spec:
            template = q if kind == "query" else d
            return [template.format(t=t) for t in texts]
    return list(texts)


def embed_for_rag(texts: list[str], kind: str = "document") -> np.ndarray:
    """Embeddings for the RAG knowledge base (CONFIG.rag_embedding = "<backend>:<model>").
    Kept separate from embed() so switching the RAG model (e.g. to multilingual ollama:bge-m3)
    doesn't change Node 3's ranking - and with it the consistency-test baseline."""
    backend, _, model = CONFIG.rag_embedding.partition(":")
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    texts = format_texts(CONFIG.rag_embedding, texts, kind)
    import tracing
    if kind == "query":           # one per search - recorded by the retriever observation, not separately
        return _embed_for_rag(backend, model, texts)
    with tracing.observe("embed-texts", "embedding", model=CONFIG.rag_embedding, input={"n_texts": len(texts)},
                         metadata={"purpose": "index paper chunks"}) as obs:
        out = _embed_for_rag(backend, model, texts)
        obs.update(output={"n_vectors": len(out)})
    return out


def _embed_for_rag(backend: str, model: str, texts: list[str]) -> np.ndarray:
    if backend == "ollama":
        import httpx
        out = []
        for i in range(0, len(texts), 32):
            resp = httpx.post(f"{CONFIG.ollama_host}/api/embed", json={"model": model, "input": texts[i:i + 32]},
                              timeout=120)
            resp.raise_for_status()
            out += resp.json()["embeddings"]
        return _normalize(out)
    if backend == "sentence_transformers":
        if model == CONFIG.embedding_model_name:
            return _normalize(_model().encode(texts, normalize_embeddings=True, show_progress_bar=False))
        return _normalize(_st_model(model).encode(texts, show_progress_bar=False))
    raise ValueError(f"Unknown RAG_EMBEDDING backend {backend!r} (use sentence_transformers:<model> or ollama:<model>)")


def rank_by_similarity(query: str, papers: list[dict]) -> list[dict]:
    """
    Ranks papers by cosine similarity of their abstract to the query, highest first.
    Each paper dict gets a 'similarity_score' field added. Papers with no abstract are
    ranked last with score 0.0 rather than dropped, so they're still visible for debugging.
    """
    if not papers:
        return []

    texts, indices = [], []
    for i, p in enumerate(papers):
        abstract = (p.get("abstract") or "").strip()
        if abstract:
            texts.append(f"{p.get('title', '')}. {abstract}")
            indices.append(i)

    scored = list(papers)  # shallow copy of refs; we mutate dicts in place below
    for p in scored:
        p.setdefault("similarity_score", 0.0)

    if texts:
        query_vec = embed([query])[0]
        paper_vecs = embed(texts)
        sims = paper_vecs @ query_vec  # both L2-normalized -> dot product == cosine similarity
        for idx, sim in zip(indices, sims):
            scored[idx]["similarity_score"] = float(sim)

    return sorted(scored, key=lambda p: p["similarity_score"], reverse=True)


def filter_by_threshold(ranked_papers: list[dict], threshold: float = None) -> list[dict]:
    threshold = CONFIG.similarity_threshold if threshold is None else threshold
    return [p for p in ranked_papers if p.get("similarity_score", 0.0) >= threshold]
