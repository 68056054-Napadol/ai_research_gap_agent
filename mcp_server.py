"""
Exposes the source connectors (sources/*) as MCP tools (architecture doc §3), so you can point
Claude Desktop / Claude Code / any MCP client at this project for manual literature scanning —
independent of the LangGraph agent. Handy for debugging a connector by hand: call search_papers
yourself and see exactly what a source returns before trusting the agent to interpret it.

Run standalone:  python mcp_server.py
Then add it to your MCP client's config as a stdio server pointing at this command.
"""
from mcp.server.fastmcp import FastMCP

from sources import REGISTRY, search_all

mcp = FastMCP("research-gap-sources")


@mcp.tool()
def search_papers(query: str, source: str = "all", year_from: int | None = None,
                   year_to: int | None = None, limit: int = 20) -> list[dict]:
    """Search academic papers. source: one of 'semantic_scholar', 'arxiv', 'openalex', or 'all'."""
    if source == "all":
        return search_all(query, sources=list(REGISTRY.keys()), year_from=year_from, year_to=year_to,
                           limit_per_source=limit)
    module = REGISTRY.get(source)
    if module is None:
        raise ValueError(f"Unknown source {source!r}. Choose from: {list(REGISTRY.keys())} or 'all'.")
    return module.search_papers(query, year_from=year_from, year_to=year_to, limit=limit)


@mcp.tool()
def get_paper(paper_id: str) -> dict | None:
    """Get full metadata for one paper. paper_id must be prefixed, e.g. 'arxiv:2401.12345'."""
    source = paper_id.split(":", 1)[0]
    module = REGISTRY.get(source)
    if module is None:
        raise ValueError(f"Unrecognized paper_id prefix {source!r}. Expected one of: {list(REGISTRY.keys())}")
    return module.get_paper(paper_id)


if __name__ == "__main__":
    mcp.run()
