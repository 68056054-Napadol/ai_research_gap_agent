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

    # --- Sampling (Harness tuning: same question -> same substance) ---
    # temperature=0 makes both backends pick the most likely token every time, so asking the
    # same topic repeatedly gives (near-)identical answers. llm_seed additionally pins Ollama's
    # sampler. See evaluation/consistency_test.py for the 5-run check.
    llm_temperature: float = _float("LLM_TEMPERATURE", 0.0)
    llm_seed: int = _int("LLM_SEED", 42)

    # --- Tool-calling agents (search agent in Node 2, verifier agent in Node 7) ---
    max_tool_steps: int = _int("MAX_TOOL_STEPS", 8)          # LLM turns per agent before forcing an answer
    max_search_tool_calls: int = _int("MAX_SEARCH_TOOL_CALLS", 10)  # budget for search_papers calls per run
    verifier_text_chars: int = _int("VERIFIER_TEXT_CHARS", 3000)   # source text returned per get_paper_text call

    # --- Privacy (decision model + LLM sensitive-info extraction, see privacy.py) ---
    # Run the LLM entity extractor on each paper's header (where author names/affiliations/emails
    # live). Regex + metadata-name redaction always runs regardless; this adds 1 fast-tier call per
    # full-text paper.
    privacy_llm_on_papers: bool = os.getenv("PRIVACY_LLM_ON_PAPERS", "true").strip().lower() in ("1", "true", "yes")
    privacy_header_chars: int = _int("PRIVACY_HEADER_CHARS", 2500)

    # --- RAG knowledge base (rag_store.py) - persists across runs = KM ---
    rag_enabled: bool = os.getenv("RAG_ENABLED", "true").strip().lower() in ("1", "true", "yes")
    rag_store_dir: str = os.getenv("RAG_STORE_DIR", "rag_store")
    # "<backend>:<model>": sentence_transformers:<hf id>  or  ollama:<model> (e.g. ollama:bge-m3,
    # multilingual incl. Thai, runs on the GPU through Ollama). Changing it re-embeds papers lazily.
    rag_embedding: str = os.getenv("RAG_EMBEDDING", "sentence_transformers:BAAI/bge-small-en-v1.5")
    rag_chunk_chars: int = _int("RAG_CHUNK_CHARS", 1200)
    rag_chunk_overlap: int = _int("RAG_CHUNK_OVERLAP", 200)
    rag_reuse_extracts: bool = os.getenv("RAG_REUSE_EXTRACTS", "true").strip().lower() in ("1", "true", "yes")

    # --- Tracing (tracing.py): on automatically when LANGFUSE_PUBLIC_KEY/SECRET_KEY are set ---
    tracing_enabled: bool = os.getenv("TRACING_ENABLED", "true").strip().lower() in ("1", "true", "yes")

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
