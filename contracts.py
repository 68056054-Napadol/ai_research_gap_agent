"""
Contracts: the exact shape every LLM output and every tool argument must have.

Two kinds of contract live here:

1. **LLM output contracts** - one pydantic model per prompt in prompts.py. llm_client.call_json()
   validates the model's JSON against the contract; a response that parses as JSON but breaks
   the contract (missing field, wrong type, empty query list...) is treated exactly like invalid
   JSON: the model is re-asked with the validation error appended, up to max_attempts.
   Before this file existed, nodes did `result.get("queries", ...)` and silently carried on with
   whatever came back.

2. **Tool argument contracts** - one model per tool in tools.py. When the LLM issues a tool call,
   its arguments are validated *before* the tool runs. A bad argument (unknown source, limit=500,
   malformed paper_id) is returned to the LLM as a tool error so it can correct itself, instead of
   crashing the run or hitting an external API with garbage.

Every contract violation is counted in metrics.py, which is what evaluation/compare_models.py
reports as "contract pass rate".
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

SourceName = Literal["semantic_scholar", "arxiv", "openalex"]


# ---------------------------------------------------------------------------
# LLM output contracts
# ---------------------------------------------------------------------------
class KeywordOutput(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=8)
    key_concepts: list[str] = Field(default_factory=list)

    @field_validator("queries")
    @classmethod
    def _non_empty_queries(cls, v: list[str]) -> list[str]:
        v = [q.strip() for q in v if isinstance(q, str) and q.strip()]
        if not v:
            raise ValueError("queries must contain at least one non-empty string")
        return v


class PaperExtractOutput(BaseModel):
    method: str
    dataset: str = "not stated"
    key_findings: str
    limitations_stated: str = "not stated"
    metrics: str = "not stated"
    supporting_quote: str = ""

    @field_validator("method", "key_findings")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be blank (use 'not stated' if the text has no answer)")
        return v.strip()


class GapItem(BaseModel):
    claim: str = Field(min_length=10)
    supporting_paper_ids: list[str]
    rationale: str


class GapProposalOutput(BaseModel):
    gap_candidates: list[GapItem] = Field(max_length=6)


class VerifierOutput(BaseModel):
    """The verifier REPORTS facts; nodes.verifier_decision() DECIDES 'supported' from them.
    (A live run showed qwen3 reading "no paper does X" as evidence AGAINST the claim "X is missing"
    - the opposite of what it means. Facts are easier for the model to get right than the verdict.)"""
    cited_papers_relevant: bool       # cited papers really concern the claim's topic AND its qualifiers
    counterexample_found: bool        # a retrieved passage shows a paper already doing what the claim says is missing
    counterexample_paper_id: Optional[str] = None
    # the sentence from that paper's passage that shows it already does the thing - must be copied
    # verbatim; the harness checks it against the text the tools returned for that paper
    counterexample_quote: Optional[str] = None
    overstated: bool                  # claim goes further than the source text allows
    notes: str
    unsupported_aspects: str = "none"
    supported: Optional[bool] = None  # model's own opinion - logged for comparison, not used


EntityType = Literal["PERSON", "EMAIL", "PHONE", "ID_NUMBER", "ADDRESS", "ORGANIZATION", "OTHER"]
EntityRole = Literal["author", "cited_author", "subject_person", "eponym", "affiliation", "contact", "other"]


class SensitiveEntity(BaseModel):
    text: str = Field(min_length=1)
    type: EntityType
    role: EntityRole = "other"
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)


class SensitiveEntitiesOutput(BaseModel):
    entities: list[SensitiveEntity] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool argument contracts (validated before a tool executes)
# ---------------------------------------------------------------------------
class SearchPapersArgs(BaseModel):
    query: str = Field(min_length=2, max_length=200)
    source: Literal["semantic_scholar", "arxiv", "openalex", "all"] = "all"
    year_from: Optional[int] = Field(default=None, ge=1900, le=2100)
    year_to: Optional[int] = Field(default=None, ge=1900, le=2100)
    limit: int = Field(default=20, ge=1, le=50)

    @model_validator(mode="after")
    def _year_order(self):
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("year_from must be <= year_to")
        return self


class GetPaperTextArgs(BaseModel):
    # No max_chars: how much source text goes back into the context is the harness's decision
    # (CONFIG.verifier_text_chars), not the model's. When it was exposed, qwen3:14b asked for
    # 10000 > le=8000 in 18 of 183 live tool calls - each one a wasted model turn, for a value
    # the harness then capped at 3000 anyway.
    paper_id: str = Field(pattern=r"^(semantic_scholar|arxiv|openalex):.+")


class SearchPaperChunksArgs(BaseModel):
    paper_id: str = Field(pattern=r"^(semantic_scholar|arxiv|openalex):.+")
    query: str = Field(min_length=3, max_length=300)
    k: int = Field(default=3, ge=1, le=8)


class SearchCorpusArgs(BaseModel):
    query: str = Field(min_length=3, max_length=300)
    k: int = Field(default=5, ge=1, le=10)
    scope: Literal["this_run", "knowledge_base"] = "this_run"
