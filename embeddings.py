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
    vectors = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors)


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
