"""
Single entry point every node uses to call an LLM: call(prompt, tier="fast"|"strong", json_mode=False).

This is the one place that knows whether we're on cloud (Anthropic) or local (Ollama, via its
OpenAI-compatible /v1 endpoint) — see config.LLM_BACKEND. Nodes never import anthropic/openai
directly, so flipping the backend in .env never touches nodes.py.

Tiering (architecture doc §4.1): "fast" = cheap/small model for high-volume, low-reasoning
calls (keyword gen, per-paper extraction). "strong" = larger model for the few calls per run
that need real reasoning (comparison synthesis, gap analysis, verifier).
"""
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional, Type

from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential, before_sleep_log

import metrics
import tracing
from config import CONFIG

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


DEFAULT_SYSTEM = "You are a precise research-assistant AI. Follow output format instructions exactly."

# Reasoning text of the last local call (Ollama streams it under message["thinking"] when
# OLLAMA_THINK=true). Thread-local because the UI runs the agent in a background thread.
_last = threading.local()


def _take_reasoning() -> str:
    text = getattr(_last, "reasoning", "") or ""
    _last.reasoning = ""
    return text


def _count_anthropic_usage(resp) -> None:
    usage = getattr(resp, "usage", None)
    if usage is not None:
        metrics.inc("input_tokens", getattr(usage, "input_tokens", 0) or 0)
        metrics.inc("output_tokens", getattr(usage, "output_tokens", 0) or 0)


# Network/server errors are retried; LLMError (missing key, unknown backend) is not - retrying a
# missing API key three times with backoff just wastes ~30s before showing the same error.
_RETRY = dict(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20),
              retry=retry_if_not_exception_type(LLMError),
              before_sleep=before_sleep_log(logger, logging.WARNING), reraise=True)


@retry(**_RETRY)
def call(prompt: str, system: str = "", tier: str = "fast", json_mode: bool = False, purpose: str = "call-llm") -> str:
    """
    Returns the model's raw text response. If json_mode=True, the caller still gets a string —
    use call_json() below when you want it parsed, since LLMs occasionally wrap JSON in prose
    even when asked not to, and that repair step belongs in one place, not scattered in nodes.

    Every call is logged with backend/model/tier and elapsed time — this is the main source of
    detail in the UI's live log box and the CLI's --verbose output, since most of the agent's
    wall-clock time is spent here.
    """
    model = (
        (CONFIG.cloud_model_strong if tier == "strong" else CONFIG.cloud_model_fast)
        if CONFIG.llm_backend == "cloud" else
        (CONFIG.local_model_strong if tier == "strong" else CONFIG.local_model_fast)
    )
    start = time.monotonic()
    logger.info("LLM call -> backend=%s model=%s tier=%s prompt_len=%d",
                CONFIG.llm_backend, model, tier, len(prompt))
    metrics.inc("llm_calls")
    tok_in, tok_out = metrics.get("input_tokens"), metrics.get("output_tokens")
    with tracing.observe(purpose, "generation", model=model,
                         input=[{"role": "system", "content": system or DEFAULT_SYSTEM},
                                {"role": "user", "content": prompt}],
                         model_parameters={"temperature": CONFIG.llm_temperature, "json_mode": json_mode},
                         metadata={"tier": tier, "backend": CONFIG.llm_backend}) as obs:
        try:
            if CONFIG.llm_backend == "cloud":
                result = _call_anthropic(prompt, system, tier, json_mode)
            elif CONFIG.llm_backend == "local":
                result = _call_ollama(prompt, system, tier, json_mode)
            else:
                raise LLMError(f"Unknown LLM_BACKEND: {CONFIG.llm_backend!r} (expected 'cloud' or 'local')")
        except Exception as e:
            logger.warning("LLM call FAILED after %.1fs (backend=%s model=%s): %s",
                            time.monotonic() - start, CONFIG.llm_backend, model, e)
            obs.update(level="ERROR", status_message=str(e)[:500])
            raise
        reasoning = _take_reasoning()
        obs.update(output={"role": "assistant", "content": result, **({"reasoning": reasoning} if reasoning else {})},
                   usage_details={"input": int(metrics.get("input_tokens") - tok_in),
                                  "output": int(metrics.get("output_tokens") - tok_out)})
    logger.info("LLM call <- done in %.1fs, response_len=%d", time.monotonic() - start, len(result))
    return result


def _extract_json_text(raw: str) -> str:
    """Strip ```json fences / leading prose and return the outermost {...} span."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[len("json"):]
    text = text.strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
    return text


def parse_with_contract(raw: str, schema: Optional[Type[BaseModel]] = None) -> dict:
    """Parse a model response as JSON and (if a contract is given) validate it.
    Raises json.JSONDecodeError or pydantic.ValidationError."""
    data = json.loads(_extract_json_text(raw))
    if schema is not None:
        data = schema.model_validate(data).model_dump()
    return data


def call_json(prompt: str, system: str = "", tier: str = "fast", schema: Optional[Type[BaseModel]] = None,
              max_attempts: int = 3, purpose: str = "call-llm") -> dict:
    """
    Call the model and return its JSON answer as a dict, validated against `schema` (a contract
    from contracts.py) when one is given.

    json_mode=True (passed to call()) already asks the backend to constrain its own output to
    JSON where supported (Ollama's "format": "json", a "{" prefill for Anthropic). Two things can
    still go wrong, and both are retried the same way - by re-asking with the exact error
    appended, up to max_attempts:
      1. not JSON at all (small local models sometimes reason out loud in prose)
      2. valid JSON that breaks the contract (missing field, wrong type, empty list...)
    """
    metrics.inc("json_calls")
    last_error = None
    raw = ""
    current_prompt = prompt
    for attempt in range(1, max_attempts + 1):
        raw = call(current_prompt, system=system, tier=tier, json_mode=True,
                   purpose=purpose if attempt == 1 else f"{purpose}-retry")
        try:
            data = parse_with_contract(raw, schema)
            if attempt > 1:
                metrics.inc("json_calls_needed_retry")
            return data
        except json.JSONDecodeError as e:
            last_error = e
            metrics.inc("json_invalid")
            if not raw.strip():
                metrics.inc("json_empty_response")
            logger.warning("  Model did not return valid JSON (attempt %d/%d): %s. Raw response starts: %r",
                           attempt, max_attempts, e, raw[:200])
            fix = ("your previous response was not valid JSON. Respond with ONLY the JSON object - no "
                   "explanation, no markdown code fences. Start with '{' and end with '}'.")
        except ValidationError as e:
            last_error = e
            metrics.inc("contract_violations")
            short = "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()[:5])
            logger.warning("  Contract violation (%s, attempt %d/%d): %s",
                           schema.__name__ if schema else "?", attempt, max_attempts, short)
            fix = (f"your previous JSON broke the required output contract: {short}. Return the JSON "
                   f"object again with every required field present and correctly typed.")
        if attempt < max_attempts:
            current_prompt = f"{prompt}\n\nIMPORTANT: {fix}"
    metrics.inc("json_calls_needed_retry")
    metrics.inc("json_calls_failed")
    raise LLMError(f"Model did not return contract-valid JSON after {max_attempts} attempts: {last_error}\n"
                   f"Last raw response:\n{raw[:500]}")


def _call_anthropic(prompt: str, system: str, tier: str, json_mode: bool = False) -> str:
    import anthropic

    if not CONFIG.anthropic_api_key:
        raise LLMError("ANTHROPIC_API_KEY is not set (see .env.example)")

    client = anthropic.Anthropic(api_key=CONFIG.anthropic_api_key)
    model = CONFIG.cloud_model_strong if tier == "strong" else CONFIG.cloud_model_fast
    messages = [{"role": "user", "content": prompt}]
    if json_mode:
        # Anthropic's Messages API has no dedicated "JSON mode" flag (unlike Ollama's
        # format="json"). The standard workaround is to prefill the start of the assistant turn
        # with "{" — the model is then forced to continue directly into JSON rather than
        # prefacing it with any explanation, since it can't "un-write" what's already there.
        # The prefilled text isn't echoed back in the response, so we stitch it back on below.
        messages.append({"role": "assistant", "content": "{"})
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=CONFIG.llm_temperature,
        system=system or DEFAULT_SYSTEM,
        messages=messages,
    )
    _count_anthropic_usage(resp)
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return "{" + text if json_mode else text


def _call_ollama(prompt: str, system: str, tier: str, json_mode: bool = False) -> str:
    """
    Streams the response instead of waiting for it in one shot. This matters on CPU-only
    hardware: with stream=False, Ollama sends nothing at all until generation is completely
    finished, so a single fixed timeout has to cover the *entire* generation — on a slow
    machine that's an easy, confusing false timeout (this is exactly what happened: a 120s
    flat timeout died mid-generation even though the model was working fine, just slowly).

    Streaming instead applies the timeout per network read, not to the whole call — as long as
    Ollama keeps sending token chunks (even slowly), the call won't time out. We also log
    progress every few seconds so a slow run is visible in the log box instead of looking stuck.

    think=False (config.OLLAMA_THINK, default off) matters a lot for reasoning models like
    Qwen3: with thinking left on, the model spends most of its time generating an internal
    <think>...</think> trace before it ever writes the actual answer. Ollama streams that trace
    under message["thinking"], not message["content"] — which is exactly what made a real,
    working generation look "stuck at 0 characters" for minutes at a time in testing: the model
    was working the whole time, just producing tokens we weren't counting. For this project's
    tasks (structured extraction, keyword lists) the reasoning step buys nothing, so turning it
    off both fixes the misleading progress display AND is usually the single biggest speed win
    on slow hardware — skipping the reasoning phase can cut a call's time dramatically.

    json_mode=True adds "format": "json" to the request. This is Ollama's constrained-decoding
    JSON mode — it forces every generated token to stay inside valid JSON grammar, which is a
    much stronger guarantee than the prompt saying "return ONLY JSON": a small/quantized local
    model that thinks out loud in prose despite the prompt asking it not to (e.g. "We are
    generating 3-5 search queries... Key concepts might include...") physically cannot do that
    once format="json" is set, since prose tokens aren't valid JSON continuations. think=False
    already reduces how much a model rambles; this is the second, independent layer against it.
    """
    import httpx

    model = CONFIG.local_model_strong if tier == "strong" else CONFIG.local_model_fast
    # Ollama silently truncates from the *start* of the prompt if it exceeds the model's context
    # window, and that window defaults to something small (often 2048 tokens) regardless of what
    # we send — so a long full-text extraction prompt could lose its system message and part of
    # the paper text without any error at all, and even a short prompt can run out of room if the
    # model rambles in its reply. Sizing num_ctx off this specific prompt (a rough 4 chars/token
    # estimate, plus headroom for the reply) avoids that; CONFIG.ollama_max_ctx caps it so a huge
    # prompt can't blow up RAM/CPU time on weak hardware (see config.py). The 4096 floor (up from
    # 2048) gives extra headroom even for a short prompt's reply.
    approx_prompt_tokens = (len(prompt) + len(system)) // 4
    num_ctx = min(max(approx_prompt_tokens + _reply_reserve(model), 4096), CONFIG.ollama_max_ctx)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system or DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "think": ollama_profile(model)["think"],
        "options": _ollama_options(num_ctx),
    }
    if json_mode and ollama_profile(model)["json_format"]:
        payload["format"] = "json"
    logger.info("  Ollama context window: %d tokens (prompt ~%d tokens)%s",
                num_ctx, approx_prompt_tokens, ", format=json" if json_mode else "")

    # connect/write/pool stay short; `read` is the one that matters here — it's the max gap
    # allowed between successive chunks arriving, not a cap on total generation time.
    timeout = httpx.Timeout(connect=30.0, read=CONFIG.ollama_read_timeout, write=30.0, pool=30.0)

    chunks: list[str] = []
    thinking: list[str] = []
    thinking_chars = 0
    last_progress_log = time.monotonic()
    with httpx.stream("POST", f"{CONFIG.ollama_host}/api/chat", json=payload, timeout=timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            message = data.get("message", {})
            piece = message.get("content", "")
            if piece:
                chunks.append(piece)
            if message.get("thinking"):
                thinking.append(message["thinking"])
                thinking_chars += len(message["thinking"])
            now = time.monotonic()
            if now - last_progress_log >= 5.0:
                # Reporting both counters is what makes a still-thinking model legible instead
                # of looking hung — see the docstring above.
                logger.info("  ... still generating locally (%d answer chars, %d thinking chars so far, model=%s)",
                            sum(len(c) for c in chunks), thinking_chars, model)
                last_progress_log = now
            if data.get("done"):
                metrics.inc("input_tokens", data.get("prompt_eval_count") or 0)
                metrics.inc("output_tokens", data.get("eval_count") or 0)
                break
    _last.reasoning = "".join(thinking)
    return "".join(chunks)


# Per-model quirks of the Ollama API, found while comparing models on this project:
#   gpt-oss: thinking cannot be switched off (it takes "low"/"medium"/"high", not false), and
#            format="json" makes it return EMPTY content - so: think="low", no grammar-forced
#            JSON; call_json's parser + contract retry handle the output instead.
# Everything else: think=CONFIG.ollama_think, format="json" for JSON calls.
OLLAMA_PROFILES = [
    ("gpt-oss", {"think": "low", "json_format": False}),
]


def ollama_profile(model: str) -> dict:
    for key, profile in OLLAMA_PROFILES:
        if key in model:
            return profile
    return {"think": CONFIG.ollama_think, "json_format": True}


def _ollama_options(num_ctx: int) -> dict:
    return {"num_ctx": num_ctx, "temperature": CONFIG.llm_temperature, "seed": CONFIG.llm_seed}


def _reply_reserve(model: str = "") -> int:
    """Tokens kept free for the reply. With thinking on, the reasoning trace is generated inside
    the same window before the answer - 2048 would let a long trace push the prompt out."""
    think = ollama_profile(model)["think"] if model else CONFIG.ollama_think
    return 2048 + (6144 if think else 0)


def _ollama_num_ctx(messages: list, system: str = "", model: str = "") -> int:
    approx = (sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) + len(system)) // 4
    return min(max(approx + _reply_reserve(model), 4096), CONFIG.ollama_max_ctx)



# ===========================================================================
# Tool calling (the agent "harness")
#
# run_tool_loop() is a real tool-use loop: the LLM is given tool definitions (name, description,
# JSON-schema of arguments) and decides on its own which tool to call and with what arguments.
# Each turn:
#     LLM  --tool_use{name, arguments}-->  harness validates args (contracts.py) and runs the tool
#     LLM  <--tool_result{context}------  harness returns the tool's output as context
# until the LLM answers without asking for a tool (= the agent's final answer) or max_steps runs
# out. Every call is recorded in the returned trace (agent, tool name, arguments, returned
# context), which the UI shows so you can point at "this is the tool call" vs "this is the final
# answer".
#
# Both backends speak a native tool-calling protocol:
#   cloud -> Anthropic Messages API `tools=[...]`, `tool_use` / `tool_result` content blocks
#   local -> Ollama /api/chat `tools=[...]`, `message.tool_calls`, role="tool" replies
# ===========================================================================
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolLoopResult:
    final_text: str
    trace: list = field(default_factory=list)
    steps: int = 0
    hit_step_limit: bool = False


# executor signature: (tool_name, arguments) -> (result_text_for_llm, trace_entry_dict)
ToolExecutor = Callable[[str, dict], tuple]


def run_tool_loop(system: str, user_prompt: str, tool_specs: list, executor: ToolExecutor,
                  tier: str = "fast", max_steps: Optional[int] = None, agent_name: str = "agent",
                  trace_metadata: Optional[dict] = None) -> ToolLoopResult:
    """tool_specs: list of {"name", "description", "input_schema"} (JSON schema).
    Traced as one `agent` observation; inside it, each model turn is a generation and each tool
    call a retriever/tool observation - siblings, as the Langfuse best practices ask."""
    with tracing.observe(tracing.hyphen(agent_name), "agent", input=user_prompt,
                         metadata={"tools": [t["name"] for t in tool_specs], **(trace_metadata or {})}) as agent_obs:
        result = _run_tool_loop(system, user_prompt, tool_specs, executor, tier, max_steps, agent_name)
        agent_obs.update(output=result.final_text,
                         metadata={"steps": result.steps, "tool_calls": len(result.trace),
                                   "hit_step_limit": result.hit_step_limit, **(trace_metadata or {})})
    return result


def _run_tool_loop(system, user_prompt, tool_specs, executor, tier, max_steps, agent_name) -> ToolLoopResult:
    max_steps = max_steps or CONFIG.max_tool_steps
    messages: list = [{"role": "user", "content": user_prompt}]
    trace: list = []
    for step in range(1, max_steps + 1):
        text, calls = _traced_tool_turn(agent_name, step, messages, tool_specs, system, tier, allow_tools=True)
        if not calls:
            logger.info("[%s] final answer after %d step(s), %d tool call(s)", agent_name, step, len(trace))
            return ToolLoopResult(final_text=text, trace=trace, steps=step)
        results = []
        for c in calls:
            logger.info("[%s] TOOL CALL %s(%s)", agent_name, c.name, json.dumps(c.arguments, ensure_ascii=False)[:300])
            metrics.inc("tool_calls")
            start = time.monotonic()
            with tracing.observe(tracing.hyphen(c.name), tracing.TOOL_OBSERVATION_TYPE.get(c.name, "tool"),
                                 input=c.arguments,
                                 metadata={"tool_call_id": c.id, "agent": agent_name, "step": step}) as tobs:
                try:
                    with tracing.quiet():
                        result_text, entry = executor(c.name, c.arguments)
                except Exception as e:  # a crashing tool becomes an error the LLM can see, not a dead run
                    logger.warning("[%s] tool %s raised: %s", agent_name, c.name, e)
                    result_text, entry = json.dumps({"error": f"tool failed: {e}"}), {"ok": False, "error": str(e)}
                try:
                    tobs.update(output=json.loads(result_text))       # the context returned to the LLM
                except ValueError:
                    tobs.update(output=result_text)
            entry = {"agent": agent_name, "step": step, "tool_call_id": c.id, "tool_name": c.name,
                     "arguments": c.arguments, "elapsed_s": round(time.monotonic() - start, 2),
                     "result_preview": result_text[:1500], "result_chars": len(result_text), **entry}
            trace.append(entry)
            logger.info("[%s] TOOL RESULT %s -> %s (%d chars returned as context)", agent_name, c.name,
                        "ok" if entry.get("ok", True) else "ERROR", len(result_text))
            results.append(result_text)
        _append_tool_results(messages, calls, results)

    # Out of steps: force a final answer without further tool use.
    logger.warning("[%s] hit max_steps=%d - asking for a final answer without tools", agent_name, max_steps)
    messages.append({"role": "user", "content": "Tool budget exhausted. Give your final answer now, "
                                                "using only the tool results you already have."})
    text, _ = _traced_tool_turn(agent_name, max_steps + 1, messages, tool_specs, system, tier, allow_tools=False)
    return ToolLoopResult(final_text=text, trace=trace, steps=max_steps, hit_step_limit=True)


def _traced_tool_turn(agent_name, step, messages, tool_specs, system, tier, allow_tools):
    """_tool_turn wrapped in a generation observation: system prompt + conversation so far + tool
    list in, assistant text + tool calls (name, arguments) out."""
    model = (CONFIG.cloud_model_strong if tier == "strong" else CONFIG.cloud_model_fast) if CONFIG.llm_backend == "cloud" \
        else (CONFIG.local_model_strong if tier == "strong" else CONFIG.local_model_fast)
    tok_in, tok_out = metrics.get("input_tokens"), metrics.get("output_tokens")
    with tracing.observe(f"{tracing.hyphen(agent_name)}-turn", "generation", model=model,
                         input=tracing.to_openai_messages(system, messages),
                         model_parameters={"temperature": CONFIG.llm_temperature},
                         metadata={"agent": agent_name, "step": step, "backend": CONFIG.llm_backend,
                                   "tools_offered": [t["name"] for t in tool_specs] if allow_tools else []}) as obs:
        text, calls = _tool_turn(messages, tool_specs, system, tier, allow_tools)
        obs.update(output=tracing.assistant_message(text, calls, _take_reasoning()),
                   usage_details={"input": int(metrics.get("input_tokens") - tok_in),
                                  "output": int(metrics.get("output_tokens") - tok_out)})
    return text, calls


@retry(**_RETRY)
def _tool_turn(messages: list, tool_specs: list, system: str, tier: str, allow_tools: bool):
    """One LLM turn with tools. Appends the assistant message to `messages` in the backend's
    native format and returns (assistant_text, [ToolCall, ...])."""
    metrics.inc("llm_calls")
    start = time.monotonic()
    if CONFIG.llm_backend == "cloud":
        text, calls = _tool_turn_anthropic(messages, tool_specs, system, tier, allow_tools)
    elif CONFIG.llm_backend == "local":
        text, calls = _tool_turn_ollama(messages, tool_specs, system, tier, allow_tools)
    else:
        raise LLMError(f"Unknown LLM_BACKEND: {CONFIG.llm_backend!r}")
    logger.info("LLM tool-turn <- %.1fs, %d tool call(s), text_len=%d", time.monotonic() - start, len(calls), len(text))
    return text, calls


def _append_tool_results(messages: list, calls: list, results: list) -> None:
    if CONFIG.llm_backend == "cloud":
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": c.id, "content": r} for c, r in zip(calls, results)
        ]})
    else:
        for c, r in zip(calls, results):
            messages.append({"role": "tool", "content": r, "tool_name": c.name})


def _tool_turn_anthropic(messages, tool_specs, system, tier, allow_tools):
    import anthropic

    if not CONFIG.anthropic_api_key:
        raise LLMError("ANTHROPIC_API_KEY is not set (see .env.example)")
    client = anthropic.Anthropic(api_key=CONFIG.anthropic_api_key)
    model = CONFIG.cloud_model_strong if tier == "strong" else CONFIG.cloud_model_fast
    kwargs = dict(model=model, max_tokens=4096, temperature=CONFIG.llm_temperature,
                  system=system or DEFAULT_SYSTEM, messages=messages, tools=tool_specs)
    if not allow_tools:
        kwargs["tool_choice"] = {"type": "none"}
    resp = client.messages.create(**kwargs)
    _count_anthropic_usage(resp)
    # keep the assistant's content blocks verbatim - Anthropic requires the tool_use blocks to be
    # echoed back before the matching tool_result blocks
    content = [b.model_dump(exclude_none=True) for b in resp.content]
    messages.append({"role": "assistant", "content": content})
    text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
    calls = [ToolCall(id=b["id"], name=b["name"], arguments=b.get("input") or {})
             for b in content if b.get("type") == "tool_use"]
    return text, calls


def _tool_turn_ollama(messages, tool_specs, system, tier, allow_tools):
    import httpx

    model = CONFIG.local_model_strong if tier == "strong" else CONFIG.local_model_fast
    full = [{"role": "system", "content": system or DEFAULT_SYSTEM}] + messages
    payload = {"model": model, "messages": full, "stream": True, "think": ollama_profile(model)["think"],
               "options": _ollama_options(_ollama_num_ctx(full, model=model))}
    if allow_tools:
        payload["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in tool_specs]
    timeout = httpx.Timeout(connect=30.0, read=CONFIG.ollama_read_timeout, write=30.0, pool=30.0)
    chunks, raw_calls, thinking = [], [], []
    with httpx.stream("POST", f"{CONFIG.ollama_host}/api/chat", json=payload, timeout=timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            msg = data.get("message", {})
            if msg.get("content"):
                chunks.append(msg["content"])
            if msg.get("thinking"):
                thinking.append(msg["thinking"])
            raw_calls.extend(msg.get("tool_calls") or [])
            if data.get("done"):
                metrics.inc("input_tokens", data.get("prompt_eval_count") or 0)
                metrics.inc("output_tokens", data.get("eval_count") or 0)
                break
    text = "".join(chunks)
    _last.reasoning = "".join(thinking)
    calls = []
    for rc in raw_calls:
        fn = rc.get("function", {})
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_unparseable": args}
        calls.append(ToolCall(id=rc.get("id") or f"call_{uuid.uuid4().hex[:8]}", name=fn.get("name", ""), arguments=args))
    assistant = {"role": "assistant", "content": text}
    if raw_calls:
        assistant["tool_calls"] = raw_calls
    messages.append(assistant)
    return text, calls
