# Embedding comparison - section retrieval inside real papers

- corpus: 13 papers from pdf_cache/, 577 chunks, 49 questions per language
- chance level for hit@3 depends on how many chunks a paper has (~30-40), i.e. roughly 0.1-0.25 for a random ranking

| Model | hit@3_en | mrr_en | hit@3_th | mrr_th | th_en_agreement@3 | chunks_per_sec | dims |
|---|---|---|---|---|---|---|---|
| sentence_transformers:BAAI/bge-small-en-v1.5 | 0.653 | 0.524 | 0.388 | 0.378 | 0.163 | 16.6 | 384 |
| ollama:bge-m3 | 0.551 | 0.491 | 0.592 | 0.519 | 0.626 | 7.0 | 1024 |
| ollama:qwen3-embedding:0.6b | 0.51 | 0.374 | 0.49 | 0.409 | 0.599 | 7.8 | 1024 |
| ollama:embeddinggemma:300m | 0.878 | 0.793 | 0.755 | 0.745 | 0.585 | 8.2 | 768 |