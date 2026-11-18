"""
Outbound contract: NOTHING that leaves the agent process may contain personal data.

"Leaves the process" = a Langfuse trace (tracing.py calls this as Langfuse's `mask` function on
every observation input/output/metadata), output/*_state.json and the Markdown report
(main.save_outputs), and what the UI renders from them.

This is the last of three layers - by the time data gets here it has normally already been
redacted at the source (pre-tool: placeholders stripped from tool arguments; post-tool: search
results scrubbed before the LLM sees them; Node 8 privacy_guard on the final answer). The check
here is deterministic (regex + this run's author list, no LLM) so it can run on every single
trace attribute; anything it still finds is redacted and counted (metric outbound_pii_violations).

Expect a few per run: raw LLM generations (e.g. a trend summary that writes "As <name> et al.
show...") are traced before Node 6/7/8 redact them - the model can produce a name nobody fed it,
which is exactly why the boundary check exists. The tests assert the RESULT: no name, e-mail or
raw topic in any exported span, saved state, or report.
"""
import logging

import metrics
import privacy

logger = logging.getLogger(__name__)


def violations(obj) -> list:
    """All PII findings in a JSON-like object (empty list = contract satisfied)."""
    found = []
    if isinstance(obj, str):
        found += privacy.find_pii(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            found += violations(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found += violations(v)
    return found


def enforce(obj, where: str):
    """Return obj with any PII redacted; count + log (masked) if anything had to be fixed."""
    found = violations(obj)
    if not found:
        return obj
    metrics.inc("outbound_pii_violations", len(found))
    logger.warning("Outbound contract: redacting %d PII item(s) in %s: %s", len(found), where,
                   ", ".join(sorted({f"{e['type']}:{privacy.mask(e['text'])}" for e in found}))[:300])
    cleaned = privacy.mask_any(obj)
    # verify the result - a contract that only *tries* to redact is not a contract
    left = violations(cleaned)
    if left:
        metrics.inc("outbound_pii_force_redacted", len(left))
        logger.warning("Outbound contract: %d item(s) survived masking in %s - force-redacting", len(left), where)
        cleaned = _force(cleaned, {e["text"] for e in left})
    return cleaned


def _force(obj, needles: set):
    if isinstance(obj, str):
        for n in sorted(needles, key=len, reverse=True):
            obj = obj.replace(n, "[REDACTED]")
        return obj
    if isinstance(obj, dict):
        return {k: _force(v, needles) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_force(v, needles) for v in obj]
    return obj
