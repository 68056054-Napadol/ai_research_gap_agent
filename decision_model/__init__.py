"""
Tiny evaluator for the DMN-style decision tables in sensitive_info.dmn.json.

A decision table = input columns + output columns + rule rows. Each rule's `when` maps an input
id to a unary test (a simplified DMN/FEEL syntax):

    "PERSON"                 equals
    "in [\"A\",\"B\"]"       one of
    ">= 0.6", "< 2", ...     numeric comparison
    (input id absent)        "-" in DMN: matches anything

Hit policy FIRST: rules are checked top to bottom, the first matching row wins. This is the
decision part of "LLM + decision model": the LLM never decides whether to redact - it only
extracts candidates; the table decides, deterministically and explainably (every decision
records which rule fired).
"""
import json
import os
from functools import lru_cache

_PATH = os.path.join(os.path.dirname(__file__), "sensitive_info.dmn.json")


@lru_cache(maxsize=1)
def load_model() -> dict:
    with open(_PATH, encoding="utf-8") as f:
        model = json.load(f)
    return {d["id"]: d for d in model["decisions"]}


def _test(expr, value) -> bool:
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return value == expr
    expr = str(expr).strip()
    if expr in ("", "-"):
        return True
    if expr.startswith("in "):
        return value in json.loads(expr[3:])
    for op in (">=", "<=", ">", "<"):
        if expr.startswith(op):
            try:
                v = float(value)
            except (TypeError, ValueError):
                return False
            bound = float(expr[len(op):])
            return {">=": v >= bound, "<=": v <= bound, ">": v > bound, "<": v < bound}[op]
    return value == expr


def evaluate(decision_id: str, inputs: dict) -> dict:
    """Returns the matching rule's outputs plus `rule` (id) and `why`."""
    table = load_model()[decision_id]
    if table.get("hitPolicy", "FIRST") != "FIRST":
        raise ValueError("only hit policy FIRST is implemented")
    for rule in table["rules"]:
        if all(_test(expr, inputs.get(col)) for col, expr in rule["when"].items()):
            return {**rule["then"], "rule": rule["id"], "why": rule.get("why", "")}
    raise ValueError(f"decision table {decision_id!r} has no matching rule (add a default row)")
