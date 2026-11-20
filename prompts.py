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
Refer to papers by their short name in the "Paper" column, e.g. "Grade Guard (2025)" - never by an
id. Findings marked "(not confirmed in the text)" could not be matched to the paper itself: rely on
them less, but do not discuss this marker, how the table was built, or any checking process.
"""

GAP_ANALYSIS_PROMPT = """You are proposing POTENTIAL research gaps from a literature comparison, to
help a researcher choose a new research topic to pursue. You have seen only the papers in the table
below - a small sample found by an automated search, not a systematic literature review - so you
cannot know that something has never been done. Word every claim as a potential gap in THIS set of
papers (e.g. "X appears under-explored in the analysed papers"), never as a discovered fact
(avoid "no research exists", "has never been studied").

Topic: {topic}
Comparison table:
{comparison_table}

Trend synthesis:
{trend_summary}

Propose 2-4 Potential Research Gaps: specific, actionable directions that are NOT already
covered by the papers above, but are motivated by patterns in them (e.g. a dataset never
combined with a method, a limitation multiple papers share, an unexplored population/domain).

Every gap MUST cite at least 2 supporting paper_ids from the table as evidence for why it's a
real gap (not just a guess) — a gap with only 0-1 supporting papers should not be proposed.

Give each gap ONE category, exactly one of: "Language & population", "Data & benchmarks",
"Method combination", "Evaluation & metrics", "Application domain & setting",
"Efficiency & deployment", "Reliability, fairness & explainability", "Other".

Return ONLY a JSON object, no other text:

{{"gap_candidates": [
  {{"claim": "...", "supporting_paper_ids": ["id1", "id2"], "rationale": "why this is a gap, referencing the evidence",
    "category": "one of the categories above"}}
]}}
"""

VERIFIER_SYSTEM = """You are a skeptical fact-checker for proposed research gaps. You can read the
papers' own text with tools: get_paper_text (opening of a paper), and - when available -
search_paper_chunks (passages inside one paper) and search_corpus (passages across all analyzed
papers, to find counterexamples). Always check the source text with the tools before deciding -
do not rely on the extracted summaries alone. Base your verdict ONLY on what the tools return."""

VERIFIER_PROMPT = """Fact-check this proposed research gap.

Proposed gap: {claim}
Rationale given: {rationale}
Cited paper ids: {cited_ids}
Extracted summaries of the cited papers (may contain extraction errors - verify against source):
{cited_paper_data}

Steps:
1. For each cited paper id, read its source text: call search_paper_chunks with a query about the
   specific part of the claim that paper is supposed to support (or get_paper_text if
   search_paper_chunks is not available).
2. If the claim says something is missing, lacking, unexplored or never combined, call
   search_corpus with a query describing exactly that thing, to look for a paper that already
   does it. A passage that does it is a counterexample - then the claim is not supported.
3. Report three FACTS (you do not decide the verdict - code does):
   - cited_papers_relevant: do the cited papers really concern the claim's topic AND every
     qualifier in it (language, domain, data type, setting)? They should motivate the gap -
     e.g. show the surrounding work or state the limitation.
   - counterexample_found: did step 2 return a passage showing a paper that ALREADY DOES what the
     claim says is missing? IMPORTANT: for a claim like "X has not been explored", finding NO
     paper that does X is NOT a counterexample - it is consistent with the claim. Only a passage
     that actually does X counts. A title that merely sounds related is NOT enough. If true, give
     that passage's paper_id AND copy, word for word, the sentence from that passage that shows the
     paper does X (counterexample_quote). No exact sentence -> counterexample_found is false.
   - overstated: does the claim go further than the source text allows (e.g. "never" when the
     papers only say "rarely", or a qualifier none of the papers mention)?
4. Then answer with ONLY a JSON object, no other text:

{{
  "cited_papers_relevant": true or false,
  "counterexample_found": true or false,
  "counterexample_paper_id": "paper id of the counterexample, or null",
  "counterexample_quote": "verbatim sentence from that paper's passage showing it does X, or null",
  "overstated": true or false,
  "notes": "1-2 sentences on what you checked in the source text, including the counterexample search",
  "unsupported_aspects": "specific part of the claim not backed by the source text, or 'none'",
  "supported": "your own overall opinion, true or false (recorded but not used)"
}}
"""

SEARCH_AGENT_SYSTEM = """You are a literature-search agent. You find academic papers by calling the
search_papers tool. You decide which queries to run, on which source, with which year range and
limit. Look at what each search returns before deciding the next call: if a query returns few or
off-topic results, rephrase it or try another source; if results are good, move on to the next
aspect of the topic. Stop when the topic is covered from several angles or the budget is used.
Never invent papers - only the tool results count."""

SEARCH_AGENT_PROMPT = """Research topic: {topic}
Year range requested by the user: {year_from} to {year_to}
Sources you may use: {sources}
Suggested queries from the planning step (use, refine or replace them):
{queries}
Budget: at most {budget} search_papers calls.

Use limit 15-25 per call so the ranking step has enough candidates. After the first results,
check whether the papers actually match EVERY part of the topic (e.g. the language/domain/setting,
not just the general method); if not, make follow-up calls aimed at the missing part.

Search now. When you are done, reply with a 2-3 sentence summary of what you searched and what
you found (no JSON needed)."""

SENSITIVE_EXTRACT_PROMPT = """Extract every piece of personal or identifying information from the text
below. You only EXTRACT - a separate decision model decides what to do with each item.

Channel: {channel}  (user_input = a topic typed by a user; paper_text = text of an academic paper;
final_answer = text the system is about to show the user)

For each item return:
- text: copied EXACTLY as it appears in the text (same spelling, same script)
- type: PERSON | EMAIL | PHONE | ID_NUMBER | ADDRESS | ORGANIZATION | OTHER
- role: author (writes/wrote the paper, byline) | cited_author (named as the author of a cited work)
        | subject_person (the person the text is about or asks about) | eponym (a person's name that
        is part of a method/theorem/algorithm/dataset name, e.g. "Kalman" in "Kalman filter",
        "Bayes" in "naive Bayes") | affiliation (university/company an author belongs to)
        | contact | other
- confidence: 0.0-1.0

Include Thai and other non-Latin names. Do NOT include model names (BERT, GPT-4), dataset names,
or ordinary words. If there is nothing, return an empty list.

Text:
<<<
{text}
>>>

Return ONLY a JSON object:
{{"entities": [{{"text": "...", "type": "PERSON", "role": "author", "confidence": 0.9}}]}}
"""
