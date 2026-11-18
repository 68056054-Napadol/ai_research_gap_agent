"""
Per-run counters used by the evaluation scripts (consistency_test.py, compare_models.py) and
shown in the UI. Plain module-level dict + lock: the agent runs in one background thread at a
time, the lock only guards against the Streamlit thread reading mid-update.

Call reset() before a run and snapshot() after it.
"""
import threading

_lock = threading.Lock()
_counters: dict = {}


def reset() -> None:
    with _lock:
        _counters.clear()


def inc(name: str, amount: float = 1) -> None:
    with _lock:
        _counters[name] = _counters.get(name, 0) + amount


def get(name: str, default: float = 0) -> float:
    with _lock:
        return _counters.get(name, default)


def snapshot() -> dict:
    with _lock:
        snap = dict(_counters)
    # derived rates, so reports don't have to recompute them
    calls = snap.get("json_calls", 0)
    if calls:
        snap["contract_first_try_rate"] = round(1 - snap.get("json_calls_needed_retry", 0) / calls, 3)
        snap["contract_final_pass_rate"] = round(1 - snap.get("json_calls_failed", 0) / calls, 3)
    tool_calls = snap.get("tool_calls", 0)
    if tool_calls:
        snap["tool_args_valid_rate"] = round(1 - snap.get("tool_args_invalid", 0) / tool_calls, 3)
    return snap
