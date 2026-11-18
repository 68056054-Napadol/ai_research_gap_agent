"""
The agent's tools: what the LLM can call, with what arguments, and what context comes back.

Each tool has
  - a name + description + JSON schema (generated from its pydantic contract in contracts.py) -
    this is exactly what is sent to the LLM in the `tools=[...]` parameter;
  - an executor that validates the LLM's arguments against the contract, enforces run policy
    (enabled sources, the user's year range, call budget), runs the real connector in sources/,
    and returns a compact JSON string - the "context returned from the tool".

Two toolsets are used by the graph:
  SEARCH_TOOLS   -> Node 2 search agent   : search_papers
  VERIFIER_TOOLS -> Node 7 verifier agent : get_paper_text
"""
import json
import logging

from pydantic import ValidationError

import metrics
import links
import privacy
from config import CONFIG
import rag_store
from contracts import GetPaperTextArgs, SearchCorpusArgs, SearchPaperChunksArgs, SearchPapersArgs
from sources import REGISTRY, _dedupe

logger = logging.getLogger(__name__)


def _spec(name: str, description: str, args_model) -> dict:
    schema = args_model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return {"name": name, "description": description, "input_schema": schema}


SEARCH_PAPERS_SPEC = _spec(
    "search_papers",
    "Search an academic database for papers matching a query. Returns up to `limit` papers with id, "
    "title, year, citation count and whether an open-access PDF exists. source='all' searches every "
    "enabled source. Use specific technical phrases as queries.",
    SearchPapersArgs,
)
GET_PAPER_TEXT_SPEC = _spec(
    "get_paper_text",
    "Return the source text of one paper that was selected for analysis: an excerpt of its full text "
    "if an open-access PDF was read, otherwise its abstract. Use it to check claims against the "
    "original paper rather than a summary.",
    GetPaperTextArgs,
)
SEARCH_PAPER_CHUNKS_SPEC = _spec(
    "search_paper_chunks",
    "Semantic search INSIDE one analyzed paper (RAG knowledge base). Returns the k passages of that "
    "paper most relevant to `query`, each labelled with its section (method, experiments, results, "
    "limitations...). Use it to check a specific part of a claim against the paper's own text.",
    SearchPaperChunksArgs,
)
SEARCH_CORPUS_SPEC = _spec(
    "search_corpus",
    "Semantic search ACROSS papers in the RAG knowledge base. scope='this_run' searches every paper "
    "analyzed in this run; scope='knowledge_base' also includes papers stored by earlier runs. Use it "
    "to look for counterexamples before accepting a claim that something is missing or unexplored.",
    SearchCorpusArgs,
)
SEARCH_TOOLS = [SEARCH_PAPERS_SPEC]
VERIFIER_TOOLS = [GET_PAPER_TEXT_SPEC, SEARCH_PAPER_CHUNKS_SPEC, SEARCH_CORPUS_SPEC]
VERIFIER_TOOLS_NO_RAG = [GET_PAPER_TEXT_SPEC]


def _err(msg: str, **extra) -> tuple:
    return json.dumps({"error": msg}, ensure_ascii=False), {"ok": False, "error": msg, **extra}


def _validate(args_model, arguments: dict):
    try:
        return args_model.model_validate(arguments), None
    except ValidationError as e:
        metrics.inc("tool_args_invalid")
        msg = "; ".join(f"{'.'.join(str(p) for p in er['loc']) or 'args'}: {er['msg']}" for er in e.errors()[:5])
        return None, f"invalid arguments ({msg}) - fix them and call again"


class SearchToolContext:
    """Run-scoped state for the search agent: policy limits + every paper any call returned."""

    def __init__(self, year_from=None, year_to=None, enabled_sources=None, budget=None):
        self.year_from = year_from
        self.year_to = year_to
        self.enabled_sources = list(enabled_sources or CONFIG.enabled_sources)
        self.budget = budget or CONFIG.max_search_tool_calls
        self.calls_made = 0
        self.papers: list = []
        self.privacy_records: list = []

    def execute(self, name: str, arguments: dict) -> tuple:
        if name != "search_papers":
            return _err(f"unknown tool {name!r}; available: search_papers")
        args, problem = _validate(SearchPapersArgs, arguments)
        if problem:
            return _err(problem, contract_error=True)
        if self.calls_made >= self.budget:
            return _err(f"search budget of {self.budget} calls is used up - give your final summary now")

        # --- policy: only enabled sources, never outside the user's year range ---
        notes = []
        if args.source == "all":
            sources = self.enabled_sources
        elif args.source not in self.enabled_sources:
            return _err(f"source {args.source!r} is disabled for this run; enabled: {self.enabled_sources} or 'all'")
        else:
            sources = [args.source]
        year_from, year_to = args.year_from, args.year_to
        if self.year_from and (year_from is None or year_from < self.year_from):
            notes.append(f"year_from clamped to user's {self.year_from}")
            year_from = self.year_from
        if self.year_to and (year_to is None or year_to > self.year_to):
            notes.append(f"year_to clamped to user's {self.year_to}")
            year_to = self.year_to

        query = privacy.strip_placeholders(args.query)
        if query != args.query:
            notes.append("redaction placeholder removed from query")
        if len(query) < 2:
            return _err("query is empty after removing redaction placeholders - search for the research topic itself")

        self.calls_made += 1
        found, failures = [], []
        for src in sources:
            try:
                found += REGISTRY[src].search_papers(query, year_from=year_from, year_to=year_to, limit=args.limit)
            except Exception as e:  # graceful degradation: one source down != tool failure
                failures.append(f"{src}: {type(e).__name__}")
                logger.warning("    [%s] FAILED for query %r: %s", src, query, e)
        found = _dedupe(found)
        for p in found:
            self.privacy_records += privacy.scrub_paper(p)   # drop authors, redact names in title/abstract
            p["url"] = links.paper_url(p)                      # link from metadata, never from the LLM
        before = len(self.papers)
        self.papers = _dedupe(self.papers + found)
        new = len(self.papers) - before

        # Compact context for the LLM: enough to judge relevance, small enough to keep the loop cheap.
        context = {
            "query": query, "sources": sources, "year_range": [year_from, year_to],
            "n_results": len(found), "n_new_unique": new, "total_unique_so_far": len(self.papers),
            "calls_left": self.budget - self.calls_made,
            "papers": [{"id": p["id"], "title": p.get("title", "")[:160], "year": p.get("year"),
                        "citations": p.get("citation_count"), "has_pdf": bool(p.get("oa_pdf_url"))}
                       for p in found[:10]],
        }
        if failures:
            context["source_errors"] = failures
        if notes:
            context["policy_notes"] = notes
        return json.dumps(context, ensure_ascii=False), {
            "ok": True, "n_items": len(found), "n_new_unique": new, "policy_notes": notes,
            "source_errors": failures, "effective_args": {"sources": sources, "year_from": year_from, "year_to": year_to},
        }


class VerifierToolContext:
    """Verifier's read access to the analyzed papers: plain text (get_paper_text) and, when the RAG
    knowledge base is on, passage search inside one paper or across the corpus. All text here is
    already privacy-redacted."""

    def __init__(self, selected_papers: list, store=None):
        self.by_id = {p["id"]: p for p in selected_papers}
        self.store = store
        self.read_ids: set = set()          # cited papers whose own text the verifier looked at
        self.corpus_searches: list = []

    def execute(self, name: str, arguments: dict) -> tuple:
        if name == "get_paper_text":
            return self._get_paper_text(arguments)
        if name == "search_paper_chunks" and self.store is not None:
            return self._search_paper_chunks(arguments)
        if name == "search_corpus" and self.store is not None:
            return self._search_corpus(arguments)
        available = "get_paper_text" + (", search_paper_chunks, search_corpus" if self.store is not None else "")
        return _err(f"unknown tool {name!r}; available: {available}")

    def _unknown(self, paper_id: str):
        return _err(f"paper {paper_id!r} is not among the analyzed papers (it may be a made-up id)", unknown_id=True)

    def _get_paper_text(self, arguments: dict) -> tuple:
        args, problem = _validate(GetPaperTextArgs, arguments)
        if problem:
            return _err(problem, contract_error=True)
        paper = self.by_id.get(args.paper_id)
        if paper is None:
            return self._unknown(args.paper_id)
        max_chars = min(args.max_chars, CONFIG.verifier_text_chars)
        if paper.get("full_text"):
            kind, text = "full_text_excerpt", paper["full_text"][:max_chars]
        else:
            kind, text = "abstract", (paper.get("abstract") or "")[:max_chars]
        self.read_ids.add(args.paper_id)
        ctx = {"paper_id": args.paper_id, "title": paper.get("title"), "year": paper.get("year"),
               "text_kind": kind, "text": text}
        return json.dumps(ctx, ensure_ascii=False), {"ok": True, "n_items": 1, "text_kind": kind}

    @staticmethod
    def _passages(hits: list, max_chars: int = 900) -> list:
        return [{"paper_id": h["paper_id"], "title": h["title"], "year": h["year"], "section": h["section"],
                 "score": h["score"], "text": h["text"][:max_chars]} for h in hits]

    def _search_paper_chunks(self, arguments: dict) -> tuple:
        args, problem = _validate(SearchPaperChunksArgs, arguments)
        if problem:
            return _err(problem, contract_error=True)
        if args.paper_id not in self.by_id:
            return self._unknown(args.paper_id)
        hits = self.store.search(args.query, paper_ids=[args.paper_id], k=args.k)
        self.read_ids.add(args.paper_id)
        ctx = {"paper_id": args.paper_id, "query": args.query, "passages": self._passages(hits)}
        return json.dumps(ctx, ensure_ascii=False), {"ok": True, "n_items": len(hits),
                                                     "sections": sorted({h["section"] for h in hits})}

    def _search_corpus(self, arguments: dict) -> tuple:
        args, problem = _validate(SearchCorpusArgs, arguments)
        if problem:
            return _err(problem, contract_error=True)
        ids = None if args.scope == "knowledge_base" else list(self.by_id)
        hits = self.store.search(args.query, paper_ids=ids, k=args.k)
        self.corpus_searches.append(args.query)
        ctx = {"query": args.query, "scope": args.scope, "n_papers_searched": "all stored" if ids is None else len(ids),
               "passages": self._passages(hits, 600)}
        return json.dumps(ctx, ensure_ascii=False), {"ok": True, "n_items": len(hits), "scope": args.scope,
                                                     "paper_ids_hit": sorted({h["paper_id"] for h in hits})}
