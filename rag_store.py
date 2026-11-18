"""
RAG knowledge base - persists across runs (this is the project's KM: knowledge the agent gathered
once is kept, searchable and reused, instead of being thrown away after each run).

Storage: one SQLite file (rag_store/store.sqlite) + embeddings stored as float32 BLOBs. A paper
corpus of a few hundred papers is a few thousand chunks, so exact cosine search over a numpy
matrix takes milliseconds - no vector-DB dependency (Chroma/FAISS wheels are not available for
every Python version; this has none).

    papers   (paper_id PK, title, year, url, text_kind, n_chunks, embed_spec, indexed_at)
    chunks   (paper_id, idx, section, text, vec)          section = abstract/method/results/...
    extracts (paper_id, key, json, created_at)           Node 5 output cache, key = hash(prompt+model+input)

Only privacy-redacted text is ever stored (nodes.extract_paper_data scrubs before indexing).
The "references" section is never indexed: it is mostly names and adds noise to retrieval.

Tier policy (see README "RAG knowledge base"): candidates are ranked on title+abstract only
(Node 3); only the papers selected for deep analysis are chunked and embedded here.
"""
import datetime
import json
import logging
import os
import re
import sqlite3
import threading

import numpy as np

import embeddings
import metrics
from config import CONFIG

logger = logging.getLogger(__name__)

TEXT_KIND_RANK = {"abstract": 0, "full_text": 1}

_SECTION_RULES = [  # first match wins; checked against a lower-cased markdown heading
    ("references", ("reference", "bibliograph")),
    ("acknowledgements", ("acknowledg",)),
    ("abstract", ("abstract",)),
    ("limitations", ("limitation", "threats to validity", "ethic")),
    ("conclusion", ("conclusion", "future work", "summary")),
    ("related_work", ("related work", "background", "prior work", "literature")),
    ("results", ("result", "evaluation", "analysis", "ablation", "discussion", "findings")),
    ("experiments", ("experiment", "setup", "dataset", "data", "corpus", "implementation")),
    ("method", ("method", "approach", "model", "framework", "architecture", "proposed", "system")),
    ("introduction", ("introduction", "motivation")),
    ("appendix", ("appendix", "supplementary")),
]
_SKIP_SECTIONS = {"references", "acknowledgements"}
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$|^\s*\*\*(\d+(?:\.\d+)*\.?\s+[A-Z][^*]{2,80})\*\*\s*$")


def _section_of(heading: str, current: str) -> str:
    h = re.sub(r"[^a-z ]", " ", heading.lower())
    for label, keys in _SECTION_RULES:
        if any(k in h for k in keys):
            return label
    return current  # unknown sub-heading: stay in the enclosing section


def chunk_text(text: str, text_kind: str, size: int = None, overlap: int = None) -> list:
    """Section-aware chunking of pymupdf4llm Markdown. Returns [(section, chunk_text), ...]."""
    size = size or CONFIG.rag_chunk_chars
    overlap = overlap if overlap is not None else CONFIG.rag_chunk_overlap
    if text_kind == "abstract":
        sections = [("abstract", text)]
    else:
        sections, current, buf = [], "front", []
        for line in text.splitlines():
            m = _HEADING_RE.match(line)
            if m:
                if buf:
                    sections.append((current, "\n".join(buf)))
                buf = []
                current = _section_of(m.group(1) or m.group(2) or "", current)
                if current == "references":
                    break               # everything after the bibliography is appendix noise or names
            buf.append(line)
        if buf and current not in _SKIP_SECTIONS:
            sections.append((current, "\n".join(buf)))
    out = []
    for section, body in sections:
        if section in _SKIP_SECTIONS:
            continue
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        start = 0
        while start < len(body):
            piece = body[start:start + size]
            if len(piece.strip()) > 20 or not out:
                out.append((section, piece.strip()))
            if start + size >= len(body):
                break
            start += size - overlap
    return out


class RagStore:
    def __init__(self, directory: str = None):
        self.directory = directory or CONFIG.rag_store_dir
        os.makedirs(self.directory, exist_ok=True)
        self.path = os.path.join(self.directory, "store.sqlite")
        self._lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS papers (paper_id TEXT PRIMARY KEY, title TEXT, year INTEGER, url TEXT,
                text_kind TEXT, n_chunks INTEGER, embed_spec TEXT, indexed_at TEXT);
            CREATE TABLE IF NOT EXISTS chunks (paper_id TEXT, idx INTEGER, section TEXT, text TEXT, vec BLOB,
                PRIMARY KEY (paper_id, idx));
            CREATE TABLE IF NOT EXISTS extracts (paper_id TEXT, key TEXT, json TEXT, created_at TEXT,
                PRIMARY KEY (paper_id, key));
        """)
        self.db.commit()

    # ---------------- indexing ----------------
    def indexed_kind(self, paper_id: str):
        """text_kind already indexed for this paper with the CURRENT embedding model, or None."""
        row = self.db.execute("SELECT text_kind, embed_spec FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
        if row and row[1] == CONFIG.rag_embedding:
            return row[0]
        return None

    def index_paper(self, paper: dict, text: str, text_kind: str) -> int:
        """Chunk + embed + store (replacing an older/weaker version). Returns the number of chunks."""
        have = self.indexed_kind(paper["id"])
        if have is not None and TEXT_KIND_RANK[have] >= TEXT_KIND_RANK[text_kind]:
            metrics.inc("rag_papers_reused")
            return 0
        chunks = chunk_text(text, text_kind)
        if not chunks:
            return 0
        vecs = embeddings.embed_for_rag([f"{paper.get('title', '')}\n{c}" for _, c in chunks]).astype(np.float32)
        with self._lock:
            self.db.execute("DELETE FROM chunks WHERE paper_id=?", (paper["id"],))
            self.db.executemany("INSERT INTO chunks VALUES (?,?,?,?,?)",
                                [(paper["id"], i, s, c, v.tobytes()) for i, ((s, c), v) in enumerate(zip(chunks, vecs))])
            self.db.execute("INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?,?,?)",
                            (paper["id"], paper.get("title"), paper.get("year"), paper.get("url"), text_kind,
                             len(chunks), CONFIG.rag_embedding, datetime.datetime.now().isoformat(timespec="seconds")))
            self.db.commit()
        metrics.inc("rag_papers_indexed")
        metrics.inc("rag_chunks_indexed", len(chunks))
        logger.info("    [rag] indexed %s: %d chunks (%s)", paper["id"], len(chunks), text_kind)
        return len(chunks)

    def paper_text(self, paper_id: str) -> str:
        """The stored (redacted) text of a paper, chunks joined - lets a later run skip the PDF download."""
        rows = self.db.execute("SELECT section, text FROM chunks WHERE paper_id=? ORDER BY idx", (paper_id,)).fetchall()
        return "\n\n".join(t for _, t in rows)

    def paper_opening(self, paper_id: str, max_chars: int = 2500) -> str:
        """First chunks of a paper in document order (title block + abstract). Built from the STORED
        chunks so a cold run and a warm run produce byte-identical extraction input."""
        out, size = [], 0
        for (text,) in self.db.execute("SELECT text FROM chunks WHERE paper_id=? ORDER BY idx", (paper_id,)):
            out.append(text)
            size += len(text)
            if size >= max_chars:
                break
        return "\n\n".join(out)[:max_chars]

    # ---------------- retrieval ----------------
    def search(self, query: str, paper_ids=None, k: int = 5, sections=None) -> list:
        import tracing
        with tracing.observe("retrieve-chunks", "retriever", input={"query": query, "k": k},
                             metadata={"n_papers_filter": None if paper_ids is None else len(list(paper_ids)),
                                       "sections": sections, "index": self.path}) as obs:
            hits = self._search(query, paper_ids, k, sections)
            obs.update(output=[{"paper_id": h["paper_id"], "section": h["section"], "score": h["score"],
                                "text": h["text"][:300]} for h in hits])
        return hits

    def _search(self, query: str, paper_ids=None, k: int = 5, sections=None) -> list:
        sql = ("SELECT c.paper_id, c.idx, c.section, c.text, c.vec, p.title, p.year, p.url FROM chunks c "
               "JOIN papers p USING (paper_id) WHERE p.embed_spec=?")
        params = [CONFIG.rag_embedding]
        if paper_ids is not None:
            paper_ids = list(paper_ids)
            if not paper_ids:
                return []
            sql += f" AND c.paper_id IN ({','.join('?' * len(paper_ids))})"
            params += paper_ids
        if sections:
            sql += f" AND c.section IN ({','.join('?' * len(sections))})"
            params += list(sections)
        rows = self.db.execute(sql, params).fetchall()
        if not rows:
            return []
        mat = np.vstack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
        q = embeddings.embed_for_rag([query], kind="query")[0].astype(np.float32)
        scores = mat @ q
        order = np.argsort(-scores)[:k]
        metrics.inc("rag_searches")
        return [{"paper_id": rows[i][0], "chunk": rows[i][1], "section": rows[i][2], "text": rows[i][3],
                 "title": rows[i][5], "year": rows[i][6], "url": rows[i][7], "score": round(float(scores[i]), 3)}
                for i in order]

    # ---------------- extraction cache ----------------
    def get_extract(self, paper_id: str, key: str):
        row = self.db.execute("SELECT json FROM extracts WHERE paper_id=? AND key=?", (paper_id, key)).fetchone()
        return json.loads(row[0]) if row else None

    def put_extract(self, paper_id: str, key: str, data: dict) -> None:
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO extracts VALUES (?,?,?,?)",
                            (paper_id, key, json.dumps(data, ensure_ascii=False), datetime.datetime.now().isoformat(timespec="seconds")))
            self.db.commit()

    def stats(self) -> dict:
        n_p, n_full = self.db.execute("SELECT COUNT(*), SUM(text_kind='full_text') FROM papers").fetchone()
        n_c = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_e = self.db.execute("SELECT COUNT(*) FROM extracts").fetchone()[0]
        return {"papers": n_p or 0, "full_text_papers": n_full or 0, "chunks": n_c, "cached_extracts": n_e,
                "embedding": CONFIG.rag_embedding, "path": self.path}


_store = None
_store_lock = threading.Lock()


def get_store() -> RagStore:
    global _store
    with _store_lock:
        if _store is None or _store.directory != CONFIG.rag_store_dir:
            _store = RagStore()
        return _store
