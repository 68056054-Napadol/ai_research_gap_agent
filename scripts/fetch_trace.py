"""
Fetch a finished trace back from Langfuse (the "run and self-audit" step of the Langfuse skill).

    python scripts/fetch_trace.py                 # trace of the newest output/*_state.json
    python scripts/fetch_trace.py <trace_id>

Writes output/trace_audit_<trace_id>.json (every observation: name, type, parent, model, usage,
input/output) and prints a compact tree + a baseline checklist. Uses the keys in .env.
"""
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import config  # noqa: E402,F401  (loads .env before Langfuse reads the environment)

if os.getenv("LANGFUSE_BASE_URL") and not os.getenv("LANGFUSE_HOST"):
    os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]

from langfuse import get_client  # noqa: E402


def latest_trace_id() -> str:
    for path in sorted(glob.glob("output/*_state.json"), reverse=True):
        with open(path, encoding="utf-8") as f:
            tid = json.load(f).get("trace_id")
        if tid:
            return tid
    raise SystemExit("no output/*_state.json with a trace_id - run main.py with LANGFUSE_* keys set first")


def as_dict(o):
    return o.dict() if hasattr(o, "dict") else (o.model_dump() if hasattr(o, "model_dump") else dict(o))


def main():
    trace_id = sys.argv[1] if len(sys.argv) > 1 else latest_trace_id()
    lf = get_client()
    print("auth_check:", lf.auth_check())
    def fetch_all():
        out, cursor = [], None
        while True:
            resp = lf.api.observations.get_many(trace_id=trace_id, limit=100, cursor=cursor,
                                                fields="core,basic,io,usage,metadata,model")
            out += [as_dict(o) for o in resp.data]
            meta = as_dict(resp.meta) if getattr(resp, "meta", None) is not None else {}
            cursor = meta.get("cursor") or meta.get("next_cursor") or meta.get("nextCursor")
            if not cursor or not resp.data:
                return out

    obs = []
    for attempt in range(12):                     # ingestion is asynchronous - wait for it
        obs = fetch_all()
        if obs and any(o.get("name") == "run-research-gap-agent" for o in obs):
            break
        time.sleep(5)
    scores = [as_dict(s) for s in lf.api.scores.get_many(trace_id=trace_id, limit=50).data]
    out = os.path.join("output", f"trace_audit_{trace_id}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"trace_id": trace_id, "observations": obs, "scores": scores}, f, indent=2, default=str, ensure_ascii=False)

    by_id = {o["id"]: o for o in obs}
    children = {}
    for o in obs:
        children.setdefault(o.get("parentObservationId") or o.get("parent_observation_id") or None, []).append(o)

    def show(parent, depth):
        for o in sorted(children.get(parent, []), key=lambda x: str(x.get("startTime") or x.get("start_time"))):
            model = o.get("providedModelName") or o.get("model") or ""
            usage = o.get("usageDetails") or o.get("usage_details") or {}
            print(f"{'  ' * depth}- {o.get('type')}: {o.get('name')}  {model} {usage if usage else ''}")
            if depth > 6:
                return
            show(o["id"], depth + 1)
    show(None, 0)
    print(f"\n{len(obs)} observations, {len(scores)} scores: " + ", ".join(f"{s.get('name')}={s.get('value')}" for s in scores))
    print("saved", out)
    _ = by_id


if __name__ == "__main__":
    main()
