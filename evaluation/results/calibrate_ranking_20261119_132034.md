# Ranking calibration (138 hand-labelled candidates, 2 topics)

Cut-off = score that keeps 95% of on-topic papers.

| model | AUC | cut-off | off-topic kept | p@5 (per topic) | p@8 (per topic) | score range |
|---|---|---|---|---|---|---|
| old: sentence_transformers:BAAI/bge-small-en-v1.5 (no query instruction) | 0.994 | 0.772 | 3/57 | 1.00 / 1.00 | 1.00 / 1.00 | 0.531-0.939 |
| bge: sentence_transformers:BAAI/bge-small-en-v1.5 | 0.994 | 0.8 | 2/57 | 1.00 / 1.00 | 1.00 / 1.00 | 0.559-0.909 |
| gemma: ollama:embeddinggemma:300m | 0.990 | 0.529 | 1/57 | 1.00 / 1.00 | 1.00 / 1.00 | 0.176-0.73 |
