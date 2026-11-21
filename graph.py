"""
Wires nodes.py into a LangGraph StateGraph: the 7 analysis nodes of architecture doc §2.2, plus a
privacy node at each end, with two conditional edges.

    sanitize_input --reject--> END                       (decision model says the topic was a person lookup)
          | proceed
    generate_keywords -> search_sources* -> filter_and_rank -> check_sufficiency
          ^                                                          |
          +------------------------ retry --------------------------+
                                                                     | extract
    check_sufficiency --narrow--> generate_keywords       (too broad: more specific queries)
    extract_paper_data -> compare_papers -> analyze_gaps* -> verify_citations -> privacy_guard -> END

    * = LLM tool-calling agents (search_papers / get_paper_text), see tools.py
"""
from langgraph.graph import StateGraph, END

import nodes
import tracing
from state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("sanitize_input", tracing.wrap_node("sanitize_input", nodes.sanitize_input))

    graph.add_node("generate_keywords", tracing.wrap_node("generate_keywords", nodes.generate_keywords))
    graph.add_node("search_sources", tracing.wrap_node("search_sources", nodes.search_sources))
    graph.add_node("filter_and_rank", tracing.wrap_node("filter_and_rank", nodes.filter_and_rank))
    graph.add_node("check_sufficiency", tracing.wrap_node("check_sufficiency", nodes.check_sufficiency))
    graph.add_node("extract_paper_data", tracing.wrap_node("extract_paper_data", nodes.extract_paper_data))
    graph.add_node("compare_papers", tracing.wrap_node("compare_papers", nodes.compare_papers))
    graph.add_node("analyze_gaps", tracing.wrap_node("analyze_gaps", nodes.analyze_gaps))
    graph.add_node("verify_citations", tracing.wrap_node("verify_citations", nodes.verify_citations))
    graph.add_node("privacy_guard", tracing.wrap_node("privacy_guard", nodes.privacy_guard))

    graph.set_entry_point("sanitize_input")
    graph.add_conditional_edges("sanitize_input", nodes.route_after_input,
                                {"proceed": "generate_keywords", "reject": END})
    graph.add_edge("generate_keywords", "search_sources")
    graph.add_edge("search_sources", "filter_and_rank")
    graph.add_edge("filter_and_rank", "check_sufficiency")

    # The adaptive branch (architecture doc §1.1, §2.2 Node 4): loop back to keyword generation
    # with feedback - broader queries if too few relevant papers were found (retry, capped by
    # MAX_RETRY_LOOPS), more specific ones if the results were too broad (narrow, capped by
    # MAX_NARROW_LOOPS); MAX_RUN_TOKENS / MAX_RUN_SECONDS stop both. See nodes.route_after_sufficiency.
    graph.add_conditional_edges(
        "check_sufficiency",
        nodes.route_after_sufficiency,
        {"retry": "generate_keywords", "narrow": "generate_keywords", "extract": "extract_paper_data"},
    )

    graph.add_edge("extract_paper_data", "compare_papers")
    graph.add_edge("compare_papers", "analyze_gaps")
    graph.add_edge("analyze_gaps", "verify_citations")    # check every cited paper before the report
    graph.add_edge("verify_citations", "privacy_guard")
    graph.add_edge("privacy_guard", END)

    return graph.compile()
