"""
Langfuse tracing - follows the Langfuse skill's instrumentation baseline and
https://langfuse.com/docs/observability/best-practices (fetched when this was written):

  * one trace = one agent run; the ROOT observation carries the trace-level input (the redacted
    research request) and output (the final answer: gaps + paper links)
  * stable, verb-first names; the model is stored on the generation, never in the name
  * most specific observation type: `agent` for the search / verifier sub-agents, `generation`
    for every model call, `retriever` for every lookup tool and chunk search, `embedding` for
    embedding calls, `guardrail` for the two privacy steps, `chain` for the other graph nodes
  * tool calls are siblings of the generation that requested them, under their `agent`
  * generation I/O is role-labelled messages (OpenAI format); requested tool calls appear in the
    assistant message's `tool_calls` with `arguments` as a JSON string and the same id the tool
    observation carries in its metadata
  * model + token usage on every generation; reasoning captured when the model produces it
  * tags (set at trace creation - they are immutable), environment, version, session id
  * after-the-fact quality labels are scores on the trace (gaps supported, quote-verified rate...)
  * PII: Langfuse's `mask` hook runs the outbound contract (outbound.py) on every payload, and
    the two steps that must read raw personal data record "[withheld: ...]" instead

Off (every function is a no-op) unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.
LANGFUSE_BASE_URL selects the server (EU cloud default, https://us.cloud.langfuse.com, or self-hosted).
"""
import contextlib
import contextvars
import json
import logging
import os

from config import CONFIG

logger = logging.getLogger(__name__)

APP_VERSION = "research-gap-agent-2.0"
TRACE_NAME = "research-gap-agent-run"

_client = None
_withheld = contextvars.ContextVar("trace_withheld", default=None)
_MAX_STR = 20000

# graph node -> (observation name, observation type)
NODE_OBSERVATIONS = {
    "sanitize_input": ("check-input-privacy", "guardrail"),
    "generate_keywords": ("plan-search-queries", "chain"),
    "search_sources": ("search-literature", "chain"),
    "filter_and_rank": ("rank-papers", "chain"),
    "check_sufficiency": ("check-sufficiency", "chain"),
    "extract_paper_data": ("extract-paper-facts", "chain"),
    "compare_papers": ("compare-papers", "chain"),
    "analyze_gaps": ("propose-and-verify-gaps", "chain"),
    "privacy_guard": ("redact-final-answer", "guardrail"),
}
# every agent tool is a lookup -> retriever
TOOL_OBSERVATION_TYPE = {"search_papers": "retriever", "get_paper_text": "retriever",
                         "search_paper_chunks": "retriever", "search_corpus": "retriever"}


def hyphen(name: str) -> str:
    return name.replace("_", "-")


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------
def init(span_exporter=None, force: bool = False, public_key: str = None):
    """Create the Langfuse client once (after config.py has loaded .env). `span_exporter`,
    `force` and `public_key` are for tests (in-memory export)."""
    global _client
    if _client is not None:
        return _client
    if not CONFIG.tracing_enabled:
        return None
    have_keys = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
    if not (have_keys or force):
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning("LANGFUSE_* keys are set but the langfuse package is not installed - tracing off")
        return None
    import outbound
    if os.getenv("LANGFUSE_BASE_URL") and not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]
    kwargs = {"mask": lambda *, data, **_: outbound.enforce(data, "trace"),
              "environment": os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development"),
              "release": APP_VERSION}
    if span_exporter is not None:
        # Langfuse keeps one client per public_key - tests pass a fresh key so their exporter is used
        kwargs.update(span_exporter=span_exporter, public_key=public_key or os.getenv("LANGFUSE_PUBLIC_KEY") or "pk-lf-test",
                      secret_key=os.getenv("LANGFUSE_SECRET_KEY") or "sk-lf-test")
    _client = Langfuse(**kwargs)
    logger.info("Langfuse tracing on (%s, environment=%s)", os.getenv("LANGFUSE_BASE_URL") or "default host",
                kwargs["environment"])
    return _client


def reset() -> None:
    """Tests only."""
    global _client
    _client = None


def enabled() -> bool:
    return _client is not None


def flush() -> None:
    if _client is not None:
        try:
            _client.flush()
        except Exception as e:  # tracing must never break a run
            logger.warning("Langfuse flush failed: %s", e)


# ---------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------
def _safe(value):
    """JSON-friendly + size-capped copy for payloads."""
    if value is None:
        return None
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > _MAX_STR:
        return text[:_MAX_STR] + f"... [truncated, {len(text)} chars]"
    return json.loads(text)


class _Noop:
    def update(self, **kwargs):
        pass

    def score_trace(self, **kwargs):
        pass


class _Handle:
    def __init__(self, obs, reason):
        self._obs, self._reason = obs, reason

    def update(self, **kwargs):
        if self._reason:
            for k in ("input", "output"):
                if k in kwargs:
                    kwargs[k] = f"[withheld: {self._reason}]"
            kwargs.pop("metadata", None)
        for k in ("output", "metadata", "input"):
            if k in kwargs:
                kwargs[k] = _safe(kwargs[k])
        self._obs.update(**{k: v for k, v in kwargs.items() if v is not None or k == "output"})

    def score_trace(self, **kwargs):
        try:
            self._obs.score_trace(**kwargs)
        except Exception as e:
            logger.warning("Langfuse score failed: %s", e)


@contextlib.contextmanager
def observe(name: str, as_type: str = "span", input=None, metadata=None, model=None, model_parameters=None):
    if _client is None or _quiet.get():
        yield _Noop()
        return
    reason = _withheld.get()
    if reason:
        input, metadata = f"[withheld: {reason}]", None
    kwargs = {"name": name, "as_type": as_type, "input": _safe(input), "metadata": _safe(metadata)}
    if as_type in ("generation", "embedding"):
        kwargs.update(model=model, model_parameters=model_parameters)
    with _client.start_as_current_observation(**kwargs) as obs:
        yield _Handle(obs, reason)


_quiet = contextvars.ContextVar("trace_quiet", default=False)


@contextlib.contextmanager
def quiet():
    """Observations inside are not emitted - for internals a parent observation already records
    (a retriever tool's own chunk search). Best practice: filter out noisy/duplicate spans."""
    token = _quiet.set(True)
    try:
        yield
    finally:
        _quiet.reset(token)


@contextlib.contextmanager
def withheld(reason: str):
    """Everything traced inside this block records a marker instead of its input/output."""
    token = _withheld.set(reason)
    try:
        yield
    finally:
        _withheld.reset(token)


# ---------------------------------------------------------------------------
# one trace per agent run
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def run_trace(entrypoint: str, session_id: str = None, extra_tags=None):
    """Root of the trace. Yields a handle whose .finish(state) sets the trace-level input (the
    REDACTED request - the raw topic is only known before sanitize_input and is never recorded),
    the trace-level output (final answer) and quality scores. Tags are set here because Langfuse
    tags are immutable after creation."""
    if _client is None:
        yield _RunNoop()
        return
    from langfuse import propagate_attributes
    model = CONFIG.cloud_model_strong if CONFIG.llm_backend == "cloud" else CONFIG.local_model_strong
    tags = ["research-gap", f"entrypoint:{entrypoint}", f"backend:{CONFIG.llm_backend}",
            f"rag:{'on' if CONFIG.rag_enabled else 'off'}"] + list(extra_tags or [])
    meta = {"backend": CONFIG.llm_backend, "model_strong": model,
            "model_fast": CONFIG.cloud_model_fast if CONFIG.llm_backend == "cloud" else CONFIG.local_model_fast,
            "rag_embedding": CONFIG.rag_embedding, "temperature": str(CONFIG.llm_temperature)}
    with propagate_attributes(trace_name=TRACE_NAME, tags=tags, session_id=session_id, version=APP_VERSION,
                              metadata={k: str(v) for k, v in meta.items()}):
        with observe("run-research-gap-agent", "agent",
                     input="[withheld until the topic has been checked for personal data]", metadata=meta) as root:
            handle = _RunHandle(root)
            yield handle


class _RunNoop:
    trace_id = None
    trace_url = None

    def finish(self, state: dict):
        pass


class _RunHandle:
    def __init__(self, root):
        self.root = root
        self.trace_id = _client.get_current_trace_id()
        self._url = None

    @property
    def trace_url(self):
        """Resolved lazily - it costs one API call (project lookup) and must never break a run."""
        if self._url is None and self.trace_id:
            try:
                self._url = _client.get_trace_url(trace_id=self.trace_id)
            except Exception as e:
                logger.warning("Could not resolve the Langfuse trace URL (%s); trace id %s", type(e).__name__, self.trace_id)
                self._url = ""
        return self._url or None

    def finish(self, state: dict):
        decision = state.get("input_decision") or {}
        self.root.update(
            input={"research_topic": state.get("research_topic"), "year_from": state.get("year_from"),
                   "year_to": state.get("year_to"), "max_papers": state.get("max_papers"),
                   "input_decision": decision.get("action")},
            output={"final_answer": [{"claim": g["claim"], "confidence": g["confidence_level"],
                                      "verifier_supported": g.get("verifier_supported"),
                                      "supporting_papers": g.get("supporting_papers", [])}
                                     for g in state.get("gap_candidates") or []],
                    "input_rejected": decision.get("action") == "REJECT"},
            metadata={"run_metrics": state.get("run_metrics"), "knowledge_base": state.get("knowledge_base")},
        )
        gaps = state.get("gap_candidates") or []
        extracts = state.get("paper_extracts") or []
        if decision.get("action") != "REJECT":
            self.root.score_trace(name="gaps-supported", value=float(sum(1 for g in gaps if g.get("verifier_supported"))),
                                  comment=f"of {len(gaps)} proposed gaps that passed the >=2 papers rule")
            if extracts:
                self.root.score_trace(name="quote-verified-rate",
                                      value=round(sum(1 for e in extracts if e.get("quote_verified")) / len(extracts), 3))
            if gaps:
                self.root.score_trace(name="evidence-coverage",
                                      value=round(sum(g.get("evidence_coverage", 0) for g in gaps) / len(gaps), 3))
        self.root.score_trace(name="outbound-pii-violations",
                              value=float((state.get("run_metrics") or {}).get("outbound_pii_violations", 0)))


# ---------------------------------------------------------------------------
# graph nodes
# ---------------------------------------------------------------------------
def wrap_node(name: str, fn):
    obs_name, obs_type = NODE_OBSERVATIONS.get(name, (hyphen(name), "chain"))

    def summarize(d: dict) -> dict:
        out = {}
        for k, v in (d or {}).items():
            if k in ("privacy_log", "tool_trace"):
                out[k] = f"{len(v)} entries" if isinstance(v, list) else v
            elif isinstance(v, list):
                out[k] = f"list[{len(v)}]"
            elif isinstance(v, str) and len(v) > 400:
                out[k] = v[:400] + "..."
            else:
                out[k] = v
        return out

    def wrapped(state):
        if _client is None:
            return fn(state)
        if name == "sanitize_input":   # raw user text: never recorded
            with observe(obs_name, obs_type, input="[withheld: raw user input before redaction]") as h:
                update = fn(state)
                h.update(output={"research_topic": update.get("research_topic"),
                                 "input_decision": update.get("input_decision"),
                                 "entity_decisions": len(update.get("privacy_log") or [])})
                return update
        with observe(obs_name, obs_type, input={"research_topic": state.get("research_topic")},
                     metadata={"graph_node": name, "state_keys": sorted(state.keys())}) as h:
            update = fn(state)
            h.update(output=summarize(update))
            return update
    wrapped.__name__ = getattr(fn, "__name__", name)
    return wrapped


# ---------------------------------------------------------------------------
# message formatting (OpenAI format, as the best-practices page asks)
# ---------------------------------------------------------------------------
def to_openai_messages(system: str, messages: list) -> list:
    """Convert the backend-native conversation (Anthropic content blocks or Ollama messages) to
    role-labelled OpenAI-format messages so the Langfuse UI renders tool calls as cards."""
    out = [{"role": "system", "content": system}] if system else []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if isinstance(content, str):
            msg = {"role": role, "content": content}
            if role == "assistant" and m.get("tool_calls"):          # Ollama assistant turn
                msg["tool_calls"] = [{"id": tc.get("id") or f"call_{i}", "type": "function",
                                      "function": {"name": tc.get("function", {}).get("name"),
                                                   "arguments": json.dumps(tc.get("function", {}).get("arguments") or {},
                                                                           ensure_ascii=False)}}
                                     for i, tc in enumerate(m["tool_calls"])]
            if role == "tool":
                msg["name"] = m.get("tool_name")
            out.append(msg)
            continue
        blocks = content or []
        if role == "assistant":                                       # Anthropic assistant turn
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}}
                     for b in blocks if b.get("type") == "tool_use"]
            out.append({"role": "assistant", "content": text, **({"tool_calls": calls} if calls else {})})
        else:                                                          # Anthropic tool results
            for b in blocks:
                if b.get("type") == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": b.get("content")})
                elif b.get("type") == "text":
                    out.append({"role": role, "content": b.get("text")})
    return out


def assistant_message(text: str, calls: list, reasoning: str = "") -> dict:
    msg = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = [{"id": c.id, "type": "function",
                              "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                             for c in calls]
    if reasoning:
        msg["reasoning"] = reasoning
    return msg
