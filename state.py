"""
The shared state object that flows through every node in the graph.
See architecture doc §2.1 for the field-by-field "who sets it / who reads it" table.

Kept as plain TypedDicts (not a heavier ORM/pydantic model) because LangGraph's StateGraph
natively merges TypedDict updates returned by each node — this is the standard pattern.
"""
from typing import TypedDict, Optional


class Paper(TypedDict, total=False):
    id: str                # source-qualified id, e.g. "semantic_scholar:abcd1234"
    source: str
    title: str
    n_authors: int         # author NAMES are never stored (privacy.scrub_paper) - only the count
    year: Optional[int]
    abstract: str
    doi: Optional[str]
    oa_pdf_url: Optional[str]
    citation_count: Optional[int]
    url: Optional[str]                    # built by links.paper_url from DOI/source id, never by the LLM
    similarity_score: Optional[float]     # set by Node 3
    full_text: Optional[str]              # set by Node 5, if OA copy was found


class PaperExtract(TypedDict, total=False):
    paper_id: str
    problem: str                # research problem / question the paper addresses
    method: str
    dataset: str
    key_findings: str
    limitations_stated: str
    metrics: str
    extraction_confidence: str  # "abstract_only" | "full_text"
    supporting_quote: str       # verbatim quote the LLM claims backs key_findings — set by Node 5
    quote_verified: bool        # True iff supporting_quote is an exact (whitespace-normalized)
                                 # substring of the text actually fed to the extraction prompt —
                                 # a deterministic, no-LLM check that the extraction isn't
                                 # hallucinated (architecture doc §7.3 / README "Fact-checking")


class GapCandidate(TypedDict, total=False):
    claim: str
    supporting_paper_ids: list
    supporting_papers: list    # [{id, title, year, url}] - links.supporting_papers, only real analyzed papers
    rationale: str
    evidence_level: str        # "Strong" | "Moderate" | "Limited" - evidence.evidence_level, from countable facts
    evidence: dict             # evidence.gap_evidence: n_supporting, n_supporting_full_text, n_contradicting,
                               # year_min/year_max, evidence_coverage, verifier_supported
    evidence_level_reasons: list  # which pre-written criterion kept the gap from the next level up
    confidence_level: str      # same value as evidence_level (runs before Nov 2026: "Low" | "Medium" | "High")
    confidence_score: float    # legacy 0-100 score (nodes.analyze_gaps) - kept for comparison, not shown
    verifier_notes: str
    unsupported_aspects: str
    evidence_read_ids: list    # cited papers the verifier actually read via get_paper_text
    evidence_coverage: float   # len(evidence_read_ids) / len(supporting_paper_ids)
    counterexample_searches: list  # search_corpus queries the verifier ran before accepting the claim
    dropped_ids: list          # cited ids that do not exist among the analyzed papers (hallucinated)


class AgentState(TypedDict, total=False):
    # --- input (research_topic is the REDACTED topic - the raw input is never stored) ---
    research_topic: str
    year_from: Optional[int]
    year_to: Optional[int]
    max_papers: int

    # --- Node 0: sanitize_input (decision model) ---
    input_decision: dict        # {action: PROCEED|PROCEED_REDACTED|REJECT, message, rule, why}

    # --- Node 1 ---
    search_queries: list        # list[str]
    prior_queries_tried: list   # accumulates across retry loops

    # --- Node 2: search agent (LLM tool calling) ---
    candidate_papers: list      # list[Paper], accumulated across retry loops
    search_summary: str         # the search agent's own final message
    search_used_fallback: bool  # True if the model made no tool calls and the harness searched for it
    source_health: dict         # per source: ok/failed/retries/rate_limited/disabled/last_error (sources/net.py)
    enabled_sources: list       # sources the search agent was allowed to use this run (sidebar / CONFIG)

    # --- Node 3 ---
    ranked_papers: list         # list[Paper], sorted by similarity_score desc
    selected_papers: list       # list[Paper], top max_deep_analysis_papers

    # --- Node 4 ---
    retry_count: int
    sufficient: bool

    # --- Node 5 ---
    paper_extracts: list        # list[PaperExtract]
    knowledge_base: dict        # rag_store stats after this run (papers, chunks, cached extracts)
    skipped_papers: list        # [{id, title, reason}] selected papers that produced no extract
                                 # (no abstract/full text, or the extraction call failed)

    # --- Node 6 ---
    comparison_table: str       # markdown table, built from paper_extracts
    trend_summary: str
    paper_labels: dict          # {paper_id: "Short title (year)"} - how the summary names papers

    # --- Node 7 ---
    gap_candidates: list        # list[GapCandidate]

    # --- Node 8: privacy_guard on the final answer ---
    final_answer_redactions: int

    # --- cross-cutting, accumulated by several nodes ---
    tool_trace: list            # every tool call: agent, tool_name, arguments, returned context
    privacy_log: list           # every decision-model decision (masked, never the original text)
    run_metrics: dict           # metrics.snapshot() at the end of the run
    run_info: dict              # main.run_info: started_at, seconds, models, sources (for the Dashboard)

    # --- final ---
    report_markdown: str
