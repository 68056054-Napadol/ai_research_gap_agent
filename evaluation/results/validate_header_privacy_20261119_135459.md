# Header privacy: PRIVACY_LLM_ON_PAPERS=auto vs always (12 real PDFs)

- auto skipped the LLM on 4/12 headers, saving 57.0 s
- items the always-LLM pass would redact but auto left in place: 17 (16 in the skipped papers)
- by type: ADDRESS/affiliation 1, PERSON/author 2, PERSON/cited_author 12, ORGANIZATION/affiliation 2
- verdict: default stays `always`; `auto` is opt-in

| paper | authors from source | auto called LLM | LLM s | reference entities | missed by auto |
|---|---|---|---|---|---|
| openalex:W4385572438 | 16 | yes | 11.1 | 21 | 0 |
| openalex:W4281606168 | 4 | yes | 6.6 | 9 | 1 |
| openalex:W7131793457 | 3 | yes | 8.3 | 11 | 0 |
| openalex:W4386228564 | 5 | yes | 7.5 | 12 | 0 |
| openalex:W4393145752 | 2 | no | 9.8 | 16 | 12 |
| openalex:W4385572195 | 6 | yes | 7.9 | 12 | 0 |
| openalex:W4416127073 | 3 | yes | 9.3 | 15 | 0 |
| openalex:W4224095956 | 5 | no | 11.6 | 14 | 1 |
| openalex:W3200787399 | 4 | no | 20.6 | 13 | 1 |
| openalex:W4408163650 | 3 | yes | 8.7 | 5 | 0 |
| openalex:W4384648067 | 15 | no | 15.0 | 21 | 2 |
| openalex:W4407186320 | 5 | yes | 6.9 | 10 | 0 |
