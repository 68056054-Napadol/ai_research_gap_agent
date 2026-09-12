"""
Every LLM prompt in one place — this is deliberately the file the team will iterate on most
(architecture doc README appendix: "Member B/C iterate the prompts here"). Keeping prompts out
of nodes.py means you can tune wording without touching control flow.
"""

KEYWORD_GEN_PROMPT = """You are helping a researcher find related academic papers on a topic.

Research topic: {topic}
{retry_context}

Generate 3-5 different search queries that would find papers related to this topic. Papers on
the same topic often use different terminology, so vary phrasing, use synonyms, and include
both broad and specific queries. Return ONLY a JSON object, no other text:

{{"queries": ["query 1", "query 2", "query 3"], "key_concepts": ["concept 1", "concept 2"]}}
"""

RETRY_CONTEXT_TEMPLATE = """
Previous search attempt(s) returned too few relevant papers. Queries already tried (avoid
repeating these, go broader or use adjacent terminology instead):
{prior_queries}
"""

EXTRACT_PROMPT = """Extract structured information from this paper. Base your answer ONLY on
the text provided — if a field isn't mentioned, say "not stated" rather than guessing.

Title: {title}
Year: {year}
Text: {text}

Return ONLY a JSON object, no other text:

{{
  "method": "the approach/technique used, 1-2 sentences",
  "dataset": "dataset(s) used, or 'not stated'",
  "key_findings": "main result(s), 1-2 sentences",
  "limitations_stated": "limitations the authors themselves mention, or 'not stated'",
  "metrics": "evaluation metrics reported, or 'not stated'",
  "supporting_quote": "a short quote (under 40 words) copied EXACTLY, word-for-word, character-for-character from the Text above, that best backs up key_findings — copy-paste it, do not paraphrase or summarize it"
}}
"""

COMPARE_PROMPT = """You are comparing multiple research papers on: {topic}

Here is a structured comparison table extracted from each paper:
{comparison_table}

Write a short synthesis (3-5 paragraphs) covering:
1. Common approaches/methods across these papers, and where they diverge.
2. Common datasets used, and any notable gaps in dataset coverage.
3. Trends in results/findings over time (if the papers span multiple years).
4. Any explicit disagreements or contradictions between papers.

Ground every claim in the table above — do not introduce information not present in it.
"""

GAP_ANALYSIS_PROMPT = """You are identifying research gaps from a literature comparison, to
help a researcher choose a new research topic to pursue.

Topic: {topic}
Comparison table:
{comparison_table}

Trend synthesis:
{trend_summary}

Propose 2-4 candidate Research Gaps: specific, actionable directions that are NOT already
covered by the papers above, but are motivated by patterns in them (e.g. a dataset never
combined with a method, a limitation multiple papers share, an unexplored population/domain).

Every gap MUST cite at least 2 supporting paper_ids from the table as evidence for why it's a
real gap (not just a guess) — a gap with only 0-1 supporting papers should not be proposed.

Return ONLY a JSON object, no other text:

{{"gap_candidates": [
  {{"claim": "...", "supporting_paper_ids": ["id1", "id2"], "rationale": "why this is a gap, referencing the evidence"}}
]}}
"""

VERIFIER_PROMPT = """You are fact-checking a proposed research gap against the actual
extracted paper data below. Be skeptical — your job is to catch overclaiming.

Proposed gap: {claim}
Rationale given: {rationale}
Cited papers and their extracted data:
{cited_paper_data}

Check: (1) Do the cited papers actually support this claim, or is it overstated? (2) Is the
evidence recent, or mostly old papers? (3) Are there any obvious counterexamples in the cited
data that contradict the claim?

Return ONLY a JSON object, no other text:

{{
  "supported": true/false,
  "notes": "1-2 sentences on what you checked and any concerns",
  "unsupported_aspects": "specific part of the claim not backed by the cited data, or 'none'"
}}
"""
