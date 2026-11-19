"""
Offline fakes: a scripted LLM (including real-shaped tool calls) and a fake paper source with
FICTIONAL papers and FICTIONAL author names. Lets the whole graph run with no network, no API
key and no embedding-model download - for tests and for rehearsing the demo flow.

Nothing here is real data: every title, abstract and name below is invented.
"""
import hashlib
import json
import re

import numpy as np

# what a verifier reports when the claim holds; note `supported` (the model's opinion) is
# deliberately False - code decides from the facts, exactly the live-run failure mode
VERDICT = {"cited_papers_relevant": True, "counterexample_found": False, "counterexample_paper_id": None,
           "overstated": False, "notes": "Read the cited source text; no paper in the corpus does this.",
           "unsupported_aspects": "none", "supported": False}

FAKE_AUTHORS = ["Alex Testauthor", "Brook Samplename", "Casey Fictivo"]

FAKE_PAPERS = [
    {"id": "arxiv:0000.00001", "title": "Prompt-based few-shot NER for low-resource languages",
     "abstract": "We study few-shot named entity recognition in low-resource languages using prompt-based "
                 "learning. Following Testauthor et al. (2021), we evaluate on a ten-language benchmark. "
                 "Contact: alex.testauthor@example.org. Results improve F1 by 4 points over fine-tuning.",
     "year": 2023},
    {"id": "arxiv:0000.00002", "title": "Cross-lingual transfer for few-shot NER",
     "abstract": "Cross-lingual transfer from high-resource languages improves few-shot named entity "
                 "recognition. We use a Kalman filter inspired smoothing of label distributions. "
                 "Performance degrades on languages with non-Latin scripts.",
     "year": 2022},
    {"id": "arxiv:0000.00003", "title": "Data augmentation for low-resource named entity recognition",
     "abstract": "Data augmentation with label-preserving token replacement helps named entity recognition "
                 "in low-resource languages. Samplename and Fictivo (2020) reported similar gains. "
                 "We do not evaluate on code-switched text.",
     "year": 2024},
    {"id": "arxiv:0000.00004", "title": "Few-shot NER with large language models in low-resource settings",
     "abstract": "Large language models prompted with a handful of examples perform few-shot named entity "
                 "recognition in low-resource languages, but results on Southeast Asian languages remain low.",
     "year": 2025},
    {"id": "arxiv:0000.00005", "title": "Meta-learning for few-shot named entity recognition",
     "abstract": "Meta-learning approaches to few-shot named entity recognition evaluated on low-resource "
                 "languages show strong results but require many source tasks.",
     "year": 2023},
    {"id": "arxiv:0000.00006", "title": "A survey of crop yield prediction",
     "abstract": "Unrelated agricultural paper about crop yield prediction from satellite images.",
     "year": 2021},
]


class FakeSource:
    """Stands in for sources/arxiv.py etc."""

    def __init__(self, name):
        self.name = name
        self.queries = []

    def search_papers(self, query, year_from=None, year_to=None, limit=30):
        self.queries.append(query)
        out = []
        for p in FAKE_PAPERS:
            if year_from and p["year"] < year_from:
                continue
            if year_to and p["year"] > year_to:
                continue
            out.append({**p, "source": self.name, "authors": list(FAKE_AUTHORS), "doi": None,
                        "oa_pdf_url": None, "citation_count": 3,
                        "id": p["id"] if self.name == "arxiv" else p["id"].replace("arxiv:", f"{self.name}:")})
        return out[:limit]

    def get_paper(self, paper_id):
        return None


def fake_embed(texts, kind="document"):
    """Deterministic bag-of-words hashing embedding, L2-normalized."""
    vecs = []
    for t in texts:
        v = np.zeros(256)
        for w in re.findall(r"[a-z]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        n = np.linalg.norm(v)
        vecs.append(v / n if n else v)
    return np.array(vecs)


class FakeLLM:
    """Answers each prompt type the way a cooperative model would. Set `hallucinate_id` to make
    the gap proposer cite a paper id that does not exist; `break_contract_once` to make the
    keyword step return contract-invalid JSON on its first try."""

    def __init__(self, hallucinate_id=True, break_contract_once=True, use_tools=True):
        self.hallucinate_id = hallucinate_id
        self.break_contract_once = break_contract_once
        self.use_tools = use_tools
        self.calls = []
        self.prompts = []

    # --- plain / JSON calls: patched over llm_client.call ---
    def call(self, prompt, system="", tier="fast", json_mode=False):
        self.calls.append(prompt[:60])
        self.prompts.append(prompt)
        if "Extract every piece of personal or identifying information" in prompt:
            text = prompt.split("<<<", 1)[1].rsplit(">>>", 1)[0]
            ents = []
            for name, role in [("Dr. Pat Realperson", "subject_person"), ("Pat Realperson", "subject_person"),
                               ("ดร.สมมติ ไม่มีจริง", "subject_person"), ("สมมติ ไม่มีจริง", "subject_person"),
                               ("Kalman", "eponym")]:
                if name in text:
                    ents.append({"text": name, "type": "PERSON", "role": role, "confidence": 0.9})
            return json.dumps({"entities": ents})
        if "Generate 3-5 different search queries" in prompt:
            if self.break_contract_once:
                self.break_contract_once = False
                return json.dumps({"queries": []})          # violates KeywordOutput (min_length=1)
            return json.dumps({"queries": ["few-shot NER low-resource", "cross-lingual NER transfer",
                                           "NER data augmentation"], "key_concepts": ["few-shot", "NER"]})
        if "Extract structured information from this paper" in prompt:
            text = prompt.split("Text:", 1)[1]
            quote = " ".join(text.split()[:12])
            return json.dumps({"method": "prompt-based / transfer method", "dataset": "multilingual NER benchmark",
                               "key_findings": "improves few-shot NER", "limitations_stated": "few languages",
                               "metrics": "F1", "supporting_quote": quote})
        if "Write a short synthesis" in prompt:
            return ("Most papers use prompting or transfer. As Testauthor et al. note, scripts matter. "
                    "Evaluation focuses on F1.")
        if "Propose 2-4 candidate Research Gaps" in prompt:
            ids = re.findall(r"\| ((?:arxiv|openalex|semantic_scholar):[0-9.]+) \|", prompt)
            gaps = [{"claim": "Few-shot NER for code-switched low-resource text is unexplored",
                     "supporting_paper_ids": ids[:2] + (["arxiv:9999.99999"] if self.hallucinate_id else []),
                     "rationale": "Both papers note missing code-switched evaluation; Brook Samplename suggests it."},
                    {"claim": "Only one fake paper supports this weak gap claim",
                     "supporting_paper_ids": ids[:1] + (["arxiv:9999.99998"] if self.hallucinate_id else []),
                     "rationale": "weak"}]
            return json.dumps({"gap_candidates": gaps})
        if "Fact-check this proposed research gap" in prompt:
            return json.dumps(VERDICT)
        raise AssertionError(f"FakeLLM got an unexpected prompt: {prompt[:200]}")

    # --- tool turns: patched over llm_client._tool_turn (cloud/Anthropic message format) ---
    def tool_turn(self, messages, tool_specs, system, tier, allow_tools):
        names = [t["name"] for t in tool_specs]
        n_tool_results = (sum(1 for m in messages if m["role"] == "user" and isinstance(m["content"], list))
                          + sum(1 for m in messages if m["role"] == "tool"))   # Anthropic or Ollama format
        first = messages[0]["content"]

        def reply(text, calls):
            content = ([{"type": "text", "text": text}] if text else []) + [
                {"type": "tool_use", "id": f"toolu_{len(messages)}_{i}", "name": n, "input": a}
                for i, (n, a) in enumerate(calls)]
            messages.append({"role": "assistant", "content": content})
            from llm_client import ToolCall
            return text, [ToolCall(id=c["id"], name=c["name"], arguments=c["input"]) for c in content if c["type"] == "tool_use"]

        if not self.use_tools or not allow_tools:
            if "search_papers" in names:
                return reply("Searched the main angles.", [])
            return reply(json.dumps(VERDICT), [])

        if "search_papers" in names:
            plan = [
                ("search_papers", {"query": "few-shot NER low-resource", "source": "arxiv", "year_from": 2015, "limit": 20}),
                ("search_papers", {"query": "x", "source": "arxiv"}),                        # invalid: too short
                ("search_papers", {"query": "cross-lingual NER transfer", "source": "all", "limit": 20}),
            ]
            if n_tool_results < len(plan):
                return reply("", [plan[n_tool_results]])
            return reply("I searched arXiv and all sources for few-shot and cross-lingual NER; found ~5 relevant papers.", [])

        # verifier
        ids = re.findall(r"Cited paper ids: (.*)", first)[0].split(", ")
        if "search_paper_chunks" in names:
            plan = [("search_paper_chunks", {"paper_id": pid, "query": "evaluation on code-switched text"}) for pid in ids]
            plan.append(("search_corpus", {"query": "few-shot NER evaluated on code-switched text", "scope": "this_run"}))
        else:
            plan = [("get_paper_text", {"paper_id": pid}) for pid in ids]
        if n_tool_results < len(plan):
            return reply("", [plan[n_tool_results]])
        return reply(json.dumps(VERDICT), [])
