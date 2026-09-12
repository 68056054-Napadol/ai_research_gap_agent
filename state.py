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
    authors: list
    year: Optional[int]
    abstract: str
    doi: Optional[str]
    oa_pdf_url: Optional[str]
    citation_count: Optional[int]
    similarity_score: Optional[float]     # set by Node 3
    full_text: Optional[str]              # set by Node 5, if OA copy was found


class PaperExtract(TypedDict, total=False):
    paper_id: str
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
    rationale: str
    confidence_level: str      # "Low" | "Medium" | "High"
    confidence_score: float    # 0-100, see nodes.analyze_gaps
    verifier_notes: str


class AgentState(TypedDict, total=False):
    # --- input ---
    research_topic: str
    year_from: Optional[int]
    year_to: Optional[int]
    max_papers: int

    # --- Node 1 ---
    search_queries: list        # list[str]
    prior_queries_tried: list   # accumulates across retry loops

    # --- Node 2 ---
    candidate_papers: list      # list[Paper]

    # --- Node 3 ---
    ranked_papers: list         # list[Paper], sorted by similarity_score desc
    selected_papers: list       # list[Paper], top max_deep_analysis_papers

    # --- Node 4 ---
    retry_count: int
    sufficient: bool

    # --- Node 5 ---
    paper_extracts: list        # list[PaperExtract]

    # --- Node 6 ---
    comparison_table: str       # markdown table, built from paper_extracts
    trend_summary: str

    # --- Node 7 ---
    gap_candidates: list        # list[GapCandidate]

    # --- final ---
    report_markdown: str
