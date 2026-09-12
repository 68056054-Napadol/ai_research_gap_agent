"""
All tunables live here so the team can adjust behavior without hunting through node code.
Values are read from .env with sane fallbacks — see .env.example for what each one means.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass
class Config:
    # --- LLM backend ---
    llm_backend: str = os.getenv("LLM_BACKEND", "cloud")  # "cloud" or "local"

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    cloud_model_fast: str = os.getenv("CLOUD_MODEL_FAST", "claude-haiku-4-5")
    cloud_model_strong: str = os.getenv("CLOUD_MODEL_STRONG", "claude-sonnet-4-5")

    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    local_model_fast: str = os.getenv("LOCAL_MODEL_FAST", "qwen3:8b")
    local_model_strong: str = os.getenv("LOCAL_MODEL_STRONG", "qwen3:14b")
    # Max seconds allowed between successive streamed chunks from Ollama (NOT a cap on total
    # generation time — see llm_client._call_ollama). Raise this if you see timeouts on slow/
    # CPU-only hardware; it should rarely need to change since streaming already avoids the
    # "one long silent wait" failure mode a flat non-streaming timeout has.
    ollama_read_timeout: int = _int("OLLAMA_READ_TIMEOUT", 120)
    # Off by default: reasoning models (Qwen3 and similar) burn most of their time on an
    # internal <think> trace that this project's tasks don't benefit from — see the docstring
    # in llm_client._call_ollama. Set OLLAMA_THINK=true to re-enable if you have faster
    # hardware and want to try it for the "strong" tier's reasoning quality.
    ollama_think: bool = os.getenv("OLLAMA_THINK", "false").strip().lower() in ("1", "true", "yes")
    # Ceiling for llm_client's per-call num_ctx sizing (see _call_ollama). Ollama otherwise
    # defaults to a small context window (often 2048-4096 tokens) regardless of prompt size,
    # which would silently truncate a long full-text extraction prompt from the *model's* side
    # even though our own code sent it in full. A bigger ceiling lets longer prompts (full-text
    # papers) fit, but costs more RAM and CPU time per call — 8192 is a reasonable ceiling for
    # 16GB-RAM, no-GPU hardware; raise it if you have more RAM to spare.
    ollama_max_ctx: int = _int("OLLAMA_MAX_CTX", 8192)

    # --- Data source keys (all optional) ---
    semantic_scholar_api_key: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    openalex_api_key: str = os.getenv("OPENALEX_API_KEY", "")
    openalex_mailto: str = os.getenv("OPENALEX_MAILTO", "")
    unpaywall_email: str = os.getenv("UNPAYWALL_EMAIL", "")

    # --- Which sources are enabled for this run ---
    enabled_sources: tuple = ("semantic_scholar", "arxiv", "openalex")

    # --- Full-text PDF extraction (Node 5 — see sources/pdf_extract.py) ---
    # Off switch: full-text extraction roughly doubles Node 5's network+CPU work per paper, so
    # this can be turned off (sidebar checkbox, or FETCH_FULL_TEXT=false) to fall back to the
    # faster abstract-only behavior this project shipped with originally.
    fetch_full_text: bool = os.getenv("FETCH_FULL_TEXT", "true").strip().lower() in ("1", "true", "yes")
    pdf_cache_dir: str = os.getenv("PDF_CACHE_DIR", "pdf_cache")
    pdf_max_pages: int = _int("PDF_MAX_PAGES", 40)          # cap so one huge PDF can't stall a run
    pdf_max_download_mb: int = _int("PDF_MAX_DOWNLOAD_MB", 25)

    # --- Paper-count / cost-control budget (architecture doc §7.4) ---
    max_candidate_papers: int = _int("MAX_CANDIDATE_PAPERS", 80)      # broad search pool, metadata only
    max_deep_analysis_papers: int = _int("MAX_DEEP_ANALYSIS_PAPERS", 20)  # papers that get full extraction

    # --- Sufficiency-check retry loop (architecture doc §2.2, Node 4) ---
    min_papers_before_retry: int = _int("MIN_PAPERS_BEFORE_RETRY", 5)
    max_retry_loops: int = _int("MAX_RETRY_LOOPS", 3)

    # --- Relevance ranking (architecture doc §7.2) ---
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    similarity_threshold: float = _float("SIMILARITY_THRESHOLD", 0.35)
    llm_rerank_top_n: int = _int("LLM_RERANK_TOP_N", 40)  # only re-rank the top N by embedding score

    # --- Confidence scoring (architecture doc §7.3) ---
    min_supporting_papers_for_gap: int = 2

    # --- Output ---
    output_dir: str = "output"


CONFIG = Config()
