"""
Wires nodes.py into a LangGraph StateGraph matching architecture doc Figure 1 / §2.2 exactly:
the 7 nodes in sequence, with one conditional edge (the sufficiency-check retry loop).

    generate_keywords -> search_sources -> filter_and_rank -> check_sufficiency
                                                                       |
                                            retry (loop to generate_keywords) <-+
                                                                       |
                                                                   extract (proceed)
                                                                       |
                                    extract_paper_data -> compare_papers -> analyze_gaps -> END
"""
from langgraph.graph import StateGraph, END

import nodes
from state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("generate_keywords", nodes.generate_keywords)
    graph.add_node("search_sources", nodes.search_sources)
    graph.add_node("filter_and_rank", nodes.filter_and_rank)
    graph.add_node("check_sufficiency", nodes.check_sufficiency)
    graph.add_node("extract_paper_data", nodes.extract_paper_data)
    graph.add_node("compare_papers", nodes.compare_papers)
    graph.add_node("analyze_gaps", nodes.analyze_gaps)

    graph.set_entry_point("generate_keywords")
    graph.add_edge("generate_keywords", "search_sources")
    graph.add_edge("search_sources", "filter_and_rank")
    graph.add_edge("filter_and_rank", "check_sufficiency")

    # The one adaptive branch in the whole system (architecture doc §1.1, §2.2 Node 4):
    # loop back to keyword generation with feedback if too few relevant papers were found,
    # capped by config.MAX_RETRY_LOOPS inside nodes.route_after_sufficiency.
    graph.add_conditional_edges(
        "check_sufficiency",
        nodes.route_after_sufficiency,
        {"retry": "generate_keywords", "extract": "extract_paper_data"},
    )

    graph.add_edge("extract_paper_data", "compare_papers")
    graph.add_edge("compare_papers", "analyze_gaps")
    graph.add_edge("analyze_gaps", END)

    return graph.compile()
