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


def _optional_float(name: str):
    raw = os.getenv(name, "").split("#", 1)[0].strip()
    return float(raw) if raw else None


# Values copied from .env.example ("you@example.com", "your-key-here") are not real settings.
# Sending them is worse than sending nothing: a fake contact email or key looks configured
# while every request still lands in the anonymous, rate-limited pool.
_PLACEHOLDER_MARKERS = ("example.com", "example.org", "your-", "your_", "<", "xxx", "changeme")


def _privacy_mode(raw: str) -> str:
    v = raw.split("#", 1)[0].strip().lower()
    v = {"true": "always", "1": "always", "yes": "always", "false": "never", "0": "never", "no": "never"}.get(v, v)
    return v if v in ("auto", "always", "never") else "always"


KNOWN_SOURCES = ("openalex", "semantic_scholar", "arxiv")


def _sources(raw: str) -> tuple:
    names = [n.strip().lower() for n in raw.split("#", 1)[0].split(",") if n.strip()]
    names = [n for n in names if n in KNOWN_SOURCES]
    return tuple(dict.fromkeys(names)) or ("openalex",)


def _secret(name: str) -> str:
    raw = os.getenv(name, "").split(" #", 1)[0].strip().strip('"').strip("'")
    if not raw or raw.lower() in ("none", "null", "false") or any(m in raw.lower() for m in _PLACEHOLDER_MARKERS):
        return ""
    return raw


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
    # Thinking for chosen steps only (comma-separated LLM-call purposes, e.g.
    # "propose-gaps,synthesize-trends"; tool loops use their agent name: "search-agent",
    # "verifier-agent"). Overrides OLLAMA_THINK=false for those steps. See README "Thinking mode".
    ollama_think_purposes: frozenset = frozenset(
        p.strip() for p in os.getenv("OLLAMA_THINK_PURPOSES", "").split("#", 1)[0].split(",") if p.strip())
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
    # LLM entity extractor on each paper's header (where author names/affiliations/emails live).
    # Regex + metadata-name + affiliation-line redaction always runs. Modes:
    #   always - one fast-tier call per full-text paper (default)
    #   auto   - call the LLM only when the deterministic detectors can't vouch for the header.
    #            Measured on 12 real PDFs (scripts/validate_header_privacy.py): it skipped the LLM on
    #            4/12 (57 s saved) but those 4 kept 16 items the LLM would have redacted (2 author
    #            names, cited authors, affiliations) - so it is opt-in, not the default.
    #   never  - deterministic detectors only
    # Legacy true/false map to always/never.
    privacy_llm_on_papers: str = _privacy_mode(os.getenv("PRIVACY_LLM_ON_PAPERS", "always"))
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
    # Semantic Scholar: free key -> 1 request/s of your own instead of the shared public pool
    # (https://www.semanticscholar.org/product/api#api-key). OpenAlex: an API key is required since
    # Feb 2026 (free, https://openalex.org/settings/api); the old mailto "polite pool" is ignored.
    semantic_scholar_api_key: str = _secret("SEMANTIC_SCHOLAR_API_KEY")
    openalex_api_key: str = _secret("OPENALEX_API_KEY")
    openalex_mailto: str = _secret("OPENALEX_MAILTO")
    # Rate limits: retries with backoff on 429/5xx (Retry-After honoured, capped), and a source that
    # fails this many times without ever answering is switched off for the rest of the run.
    source_max_retries: int = _int("SOURCE_MAX_RETRIES", 3)
    source_retry_max_wait: float = _float("SOURCE_RETRY_MAX_WAIT", 10.0)
    source_disable_after_failures: int = _int("SOURCE_DISABLE_AFTER_FAILURES", 3)
    unpaywall_email: str = os.getenv("UNPAYWALL_EMAIL", "")

    # --- Which sources are enabled for this run ---
    # MVP scope (teacher feedback): ONE source by default, so a run has one data format, one rate
    # limit and no cross-source duplicates to reconcile. OpenAlex: abstracts, DOIs and open-access
    # links in one API (needs the free OPENALEX_API_KEY). Comma-separated list to widen it, e.g.
    # ENABLED_SOURCES=openalex,semantic_scholar,arxiv - say so when you report results.
    enabled_sources: tuple = _sources(os.getenv("ENABLED_SOURCES", "openalex"))

    # --- Full-text PDF extraction (Node 5 — see sources/pdf_extract.py) ---
    # Off switch: full-text extraction roughly doubles Node 5's network+CPU work per paper, so
    # this can be turned off (sidebar checkbox, or FETCH_FULL_TEXT=false) to fall back to the
    # faster abstract-only behavior this project shipped with originally.
    fetch_full_text: bool = os.getenv("FETCH_FULL_TEXT", "true").strip().lower() in ("1", "true", "yes")
    pdf_cache_dir: str = os.getenv("PDF_CACHE_DIR", "pdf_cache")
    pdf_max_pages: int = _int("PDF_MAX_PAGES", 40)          # cap so one huge PDF can't stall a run
    pdf_max_download_mb: int = _int("PDF_MAX_DOWNLOAD_MB", 25)

    # --- Paper-count / cost-control budget (architecture doc §7.4) ---
    # MVP scope (teacher feedback): ~20-30 candidates, of which 8-12 with enough abstract/metadata
    # are analysed. Enough for a demo, NOT a systematic review - the report says so.
    max_candidate_papers: int = _int("MAX_CANDIDATE_PAPERS", 30)      # candidate pool, metadata only
    min_candidate_papers: int = _int("MIN_CANDIDATE_PAPERS", 20)      # fewer -> "too few" -> broaden (Node 4)
    max_deep_analysis_papers: int = _int("MAX_DEEP_ANALYSIS_PAPERS", 10)  # papers that get full extraction
    mvp_min_papers: int = _int("MVP_MIN_PAPERS", 8)                   # UI slider range for papers analysed
    mvp_max_papers: int = _int("MVP_MAX_PAPERS", 12)
    # (10: ~8-12 min on a local 14B model. Each extra paper costs ~45 s - extraction + PDF + RAG.)

    # --- Sufficiency-check retry loop (architecture doc §2.2, Node 4) ---
    min_papers_before_retry: int = _int("MIN_PAPERS_BEFORE_RETRY", 8)
    max_retry_loops: int = _int("MAX_RETRY_LOOPS", 3)
    # "Too broad" rule (Node 4): the candidate pool is full (>= MAX_CANDIDATE_PAPERS) but fewer than
    # BROAD_MAX_PASS_RATE of the candidates pass the relevance cut-off -> the queries are too
    # general, so Node 1 is asked for MORE SPECIFIC queries (at most MAX_NARROW_LOOPS times).
    broad_max_pass_rate: float = _float("BROAD_MAX_PASS_RATE", 0.4)
    max_narrow_loops: int = _int("MAX_NARROW_LOOPS", 1)
    # Token budget for one run (input + output tokens, metrics.py). 0 = no limit. Once used up, the
    # optional, repeatable work stops: no further retry loop (Node 4) and no further verifier runs
    # (Node 7 - the remaining gaps are kept but marked unverified, so their evidence is Limited).
    # Default 250k: measured runs used ~45-60k tokens for 5 papers (evaluation/results/compare_models_*),
    # so 10-12 papers plus retries fit with room; a run that loops or stalls is cut off.
    max_run_tokens: int = _int("MAX_RUN_TOKENS", 250_000)
    # Wall-clock budget for one run in seconds (0 = off); checked at the same points as the tokens.
    max_run_seconds: int = _int("MAX_RUN_SECONDS", 1800)

    # --- Relevance ranking (architecture doc §7.2) ---
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    # Node 3 ranks candidates with this embedding ("<backend>:<model>", same format as
    # RAG_EMBEDDING). Default: the RAG model, so ranking and retrieval agree and use the model
    # that won evaluation/compare_embeddings.py.
    ranking_embedding: str = os.getenv("RANKING_EMBEDDING", "").split("#", 1)[0].strip() or os.getenv(
        "RAG_EMBEDDING", "sentence_transformers:BAAI/bge-small-en-v1.5").split("#", 1)[0].strip()
    # Cosine scores are not comparable across models, so the cut-off is per model
    # (embeddings.RANKING_THRESHOLDS, calibrated by evaluation/calibrate_ranking.py on hand-labelled
    # candidates). SIMILARITY_THRESHOLD overrides it.
    similarity_threshold: float | None = _optional_float("SIMILARITY_THRESHOLD")
    llm_rerank_top_n: int = _int("LLM_RERANK_TOP_N", 40)  # only re-rank the top N by embedding score

    # --- Confidence scoring (architecture doc §7.3) ---
    min_supporting_papers_for_gap: int = 2
    # A gap is mostly an absence claim ("nobody has done X"), and absence is only as convincing as
    # the search behind it. "High" needs at least this many analysed papers and this many paper
    # sources that actually answered; otherwise the score is capped just below High.
    high_confidence_min_papers: int = _int("HIGH_CONFIDENCE_MIN_PAPERS", 8)
    # 1 with the single-source MVP: the report already names the one source, and requiring 2 would
    # make Strong impossible by construction. Raise to 2 when you enable several sources.
    high_confidence_min_sources: int = _int("HIGH_CONFIDENCE_MIN_SOURCES", 1)

    # --- Evidence level of a potential gap (README "Evidence levels") ---
    # Written down before looking at results, and only COUNTABLE facts: how many analysed papers
    # support the gap, how many of those were read in full text, whether a paper contradicts it
    # (a counterexample whose quote was verified in that paper's text), how recent the evidence is,
    # and whether the verifier actually read the cited papers. No number the LLM gives itself.
    # Limited  = verifier did not support it, OR >= 1 contradicting paper, OR fewer than
    #            EVIDENCE_MODERATE_MIN_SUPPORTING supporting papers, OR verifier read less than
    #            EVIDENCE_MODERATE_MIN_COVERAGE of them
    # Strong   = not Limited AND every EVIDENCE_STRONG_* threshold met AND no confidence cap
    #            (nodes._confidence_caps: narrow search / unquoted counterexample)
    # Moderate = everything else
    evidence_strong_min_supporting: int = _int("EVIDENCE_STRONG_MIN_SUPPORTING", 3)
    evidence_strong_min_full_text: int = _int("EVIDENCE_STRONG_MIN_FULL_TEXT", 2)
    evidence_strong_min_coverage: float = _float("EVIDENCE_STRONG_MIN_COVERAGE", 1.0)
    evidence_strong_max_age_years: int = _int("EVIDENCE_STRONG_MAX_AGE_YEARS", 3)   # age of the NEWEST supporting paper
    evidence_moderate_min_supporting: int = _int("EVIDENCE_MODERATE_MIN_SUPPORTING", 2)
    evidence_moderate_min_coverage: float = _float("EVIDENCE_MODERATE_MIN_COVERAGE", 0.5)

    # --- Citation check before the report (citations.py, Node 8 verify_citations) ---
    # Every paper a gap cites is looked up in Crossref (by DOI, else title) and, when it did not come
    # from OpenAlex, in OpenAlex; title / year / DOI must agree. A gap citing a paper that does not
    # check out cannot be Strong. VERIFY_CITATIONS=false skips the network lookups (offline runs).
    verify_citations: bool = os.getenv("VERIFY_CITATIONS", "true").strip().lower() in ("1", "true", "yes")
    citation_title_sim: float = _float("CITATION_TITLE_SIM", 0.9)
    citation_timeout: float = _float("CITATION_TIMEOUT", 15.0)

    # --- Output ---
    output_dir: str = "output"


CONFIG = Config()
