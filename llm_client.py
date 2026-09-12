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
import time
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from config import CONFIG

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20),
       before_sleep=before_sleep_log(logger, logging.WARNING))
def call(prompt: str, system: str = "", tier: str = "fast", json_mode: bool = False) -> str:
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
        raise
    logger.info("LLM call <- done in %.1fs, response_len=%d", time.monotonic() - start, len(result))
    return result


def call_json(prompt: str, system: str = "", tier: str = "fast", max_attempts: int = 3) -> dict:
    """
    Call the model and parse a JSON object out of the response, tolerating minor wrapping.

    json_mode=True (passed to call()) already asks the backend to constrain its own output to
    valid JSON where that's supported (Ollama's "format": "json", a JSON-prefill trick for
    Anthropic — see _call_ollama/_call_anthropic below), which prevents most of this failure
    mode outright. But small local models in particular can still ignore that and just reason
    out loud in prose (e.g. "We are generating 3-5 search queries... Key concepts might
    include...") instead of ever emitting JSON — this used to raise immediately with zero
    retries, crashing the whole run on the very first LLM call. Now it retries up to
    max_attempts times, appending an increasingly blunt reminder each time, before giving up.
    """
    last_error = None
    raw = ""
    current_prompt = prompt
    for attempt in range(1, max_attempts + 1):
        raw = call(current_prompt, system=system, tier=tier, json_mode=True)
        text = raw.strip()
        # Strip common ```json ... ``` fencing if the model added it despite instructions not to.
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[len("json"):]
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning("  Model did not return valid JSON (attempt %d/%d): %s. Raw response starts: %r",
                            attempt, max_attempts, e, raw[:200])
            if attempt < max_attempts:
                current_prompt = (
                    f"{prompt}\n\n"
                    f"IMPORTANT: your previous response was not valid JSON. Respond with ONLY the "
                    f"JSON object — no explanation, no reasoning, no markdown code fences, nothing "
                    f"before or after it. Start your response with '{{' and end it with '}}'."
                )
    raise LLMError(f"Model did not return valid JSON after {max_attempts} attempts: {last_error}\n"
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
        system=system or "You are a precise research-assistant AI. Follow output format instructions exactly.",
        messages=messages,
    )
    text = resp.content[0].text
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
    num_ctx = min(max(approx_prompt_tokens + 2048, 4096), CONFIG.ollama_max_ctx)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system or "You are a precise research-assistant AI. Follow output format instructions exactly."},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "think": CONFIG.ollama_think,
        "options": {"num_ctx": num_ctx},
    }
    if json_mode:
        payload["format"] = "json"
    logger.info("  Ollama context window: %d tokens (prompt ~%d tokens)%s",
                num_ctx, approx_prompt_tokens, ", format=json" if json_mode else "")

    # connect/write/pool stay short; `read` is the one that matters here — it's the max gap
    # allowed between successive chunks arriving, not a cap on total generation time.
    timeout = httpx.Timeout(connect=30.0, read=CONFIG.ollama_read_timeout, write=30.0, pool=30.0)

    chunks: list[str] = []
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
            thinking_chars += len(message.get("thinking") or "")
            now = time.monotonic()
            if now - last_progress_log >= 5.0:
                # Reporting both counters is what makes a still-thinking model legible instead
                # of looking hung — see the docstring above.
                logger.info("  ... still generating locally (%d answer chars, %d thinking chars so far, model=%s)",
                            sum(len(c) for c in chunks), thinking_chars, model)
                last_progress_log = now
            if data.get("done"):
                break
    return "".join(chunks)
