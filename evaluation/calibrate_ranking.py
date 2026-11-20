"""
Calibrate Node 3's relevance cut-off per embedding model.

Data: candidates from real runs, hand-labelled on-topic / off-topic by title
(evaluation/data/ranking_labels.json -> evaluation/data/ranking_set.json, built once from the
run's state file; the set file is what gets committed, so the numbers can be re-checked).

For each model it reports
  - ROC AUC: how well the score separates on-topic from off-topic (1.0 = perfectly),
  - the cut-off that keeps >= 95% of the on-topic papers (recall first: a wrongly dropped paper
    is lost to the whole run, a wrongly kept one only costs a slot in the ranking), and how many
    off-topic papers that cut-off still lets through,
  - precision@5 / @8 / @10: of the top papers that would go to deep analysis, how many are on topic
    (@10 = share of the top 10 by score; a topic with fewer than 10 candidates uses them all).

    python evaluation/calibrate_ranking.py --models "old=sentence_transformers:BAAI/bge-small-en-v1.5|raw" \
        "bge=sentence_transformers:BAAI/bge-small-en-v1.5" "gemma=ollama:embeddinggemma:300m"

"|raw" = embed the bare topic with no query instruction (what Node 3 did before this change).
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABELS = os.path.join(os.path.dirname(__file__), "data", "ranking_labels.json")
SET = os.path.join(os.path.dirname(__file__), "data", "ranking_set.json")


def build_set() -> list:
    if os.path.exists(SET):
        return json.load(open(SET, encoding="utf-8"))
    spec = json.load(open(LABELS, encoding="utf-8"))
    out = []
    for s in spec["sets"]:
        state = json.load(open(s["state_file"], encoding="utf-8"))
        cands = state["candidate_papers"]
        rel, skip = set(s["relevant_idx"]), set(s["skip_idx"])
        items = [{"id": p["id"], "title": p.get("title", ""), "abstract": (p.get("abstract") or "")[:1500],
                  "label": int(i in rel)} for i, p in enumerate(cands) if i not in skip]
        out.append({"topic": s["topic"], "items": items})
    json.dump(out, open(SET, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


def auc(scores, labels) -> float:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def precision_at(labels, scores, k: int) -> float:
    """Share of on-topic papers among the top k by score."""
    order = np.argsort(-np.asarray(scores, dtype=float))[:k]
    return float(np.mean([labels[i] for i in order])) if len(order) else float("nan")


def score_set(spec: str, raw: bool, topic: str, items: list) -> np.ndarray:
    import embeddings
    from config import CONFIG
    CONFIG.ranking_embedding = spec
    texts = [f"{it['title']}. {it['abstract']}" if it["abstract"] else it["title"] for it in items]
    if raw:   # the old Node 3: no query instruction
        backend, _, model = spec.partition(":")
        q = embeddings._embed_for_rag(backend, model, [topic])[0]
        d = embeddings._embed_for_rag(backend, model, [t[:2000] for t in texts])
    else:
        q = embeddings.embed_for_ranking([topic], "query")[0]
        d = embeddings.embed_for_ranking([t[:2000] for t in texts], "document")
    return d @ q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="label=backend:model[|raw]")
    ap.add_argument("--recall", type=float, default=0.95)
    args = ap.parse_args()
    data = build_set()
    import tracing
    tracing.init()

    rows = {}
    for entry in args.models:
        label, _, rest = entry.partition("=")
        spec, _, flag = rest.partition("|")
        all_s, all_l, per_topic = [], [], []
        for t in data:
            labels = [it["label"] for it in t["items"]]
            with tracing.quiet():
                s = score_set(spec, flag == "raw", t["topic"], t["items"])
            order = np.argsort(-s)
            p5 = float(np.mean([labels[i] for i in order[:5]]))
            p8 = float(np.mean([labels[i] for i in order[:8]]))
            p10 = precision_at(labels, s, 10)
            per_topic.append({"topic": t["topic"], "auc": auc(list(s), labels), "p@5": p5, "p@8": p8, "p@10": p10,
                              "pos_min": float(min(x for x, l in zip(s, labels) if l)),
                              "neg_max": float(max(x for x, l in zip(s, labels) if not l))})
            all_s += list(map(float, s)); all_l += labels
        pos_sorted = sorted((x for x, l in zip(all_s, all_l) if l), reverse=True)
        k = max(int(np.ceil(args.recall * len(pos_sorted))) - 1, 0)
        cut = pos_sorted[k]
        leak = sum(1 for x, l in zip(all_s, all_l) if not l and x >= cut)
        n_neg = sum(1 for l in all_l if not l)
        rows[label] = {"spec": spec + (" (no query instruction)" if flag == "raw" else ""), "auc": auc(all_s, all_l),
                       "cut_at_recall": round(cut, 3), "off_topic_kept": f"{leak}/{n_neg}",
                       "score_range": [round(min(all_s), 3), round(max(all_s), 3)], "per_topic": per_topic}
        print(f"{label}: AUC {rows[label]['auc']:.3f}  cut@{args.recall:.0%} recall {cut:.3f}  "
              f"off-topic kept {leak}/{n_neg}  range {rows[label]['score_range']}")
        for pt in per_topic:
            print(f"    {pt['topic'][:50]:50s} AUC {pt['auc']:.3f} p@5 {pt['p@5']:.2f} p@8 {pt['p@8']:.2f} "
                  f"p@10 {pt['p@10']:.2f}")

    os.makedirs("evaluation/results", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    n_items = sum(len(t["items"]) for t in data)
    lines = [f"# Ranking calibration ({n_items} hand-labelled candidates, {len(data)} topics)", "",
             f"Cut-off = score that keeps {args.recall:.0%} of on-topic papers.", "",
             "| model | AUC | cut-off | off-topic kept | p@5 (per topic) | p@8 (per topic) | p@10 (per topic) | score range |",
             "|---|---|---|---|---|---|---|---|"]
    for label, r in rows.items():
        p5 = " / ".join(f"{pt['p@5']:.2f}" for pt in r["per_topic"])
        p8 = " / ".join(f"{pt['p@8']:.2f}" for pt in r["per_topic"])
        p10 = " / ".join(f"{pt['p@10']:.2f}" for pt in r["per_topic"])
        lines.append(f"| {label}: {r['spec']} | {r['auc']:.3f} | {r['cut_at_recall']} | {r['off_topic_kept']} "
                     f"| {p5} | {p8} | {p10} | {r['score_range'][0]}-{r['score_range'][1]} |")
    path = f"evaluation/results/calibrate_ranking_{stamp}.md"
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(rows, open(path[:-3] + ".json", "w", encoding="utf-8"), indent=1)
    print("Saved:", path)


if __name__ == "__main__":
    main()
