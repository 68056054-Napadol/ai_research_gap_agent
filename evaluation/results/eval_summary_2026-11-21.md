# สรุปผลการรัน evaluation — 2026-11-21 (เครื่อง ihc-wctbp9102, backend local)

โค้ด: branch `main` commit `ba9f30d` (git pull: Already up to date) · ไม่ได้แก้โค้ด · ไม่ได้ commit
วิธีรัน: งาน .cmd ผ่าน `_live_test/claude_runner.bat` บน Windows (.venv Python 3.15) รันทีละงาน · log อยู่ที่ `evaluation/results/logs/`

## 1. ผลขั้น 0

| รายการ | ผล |
|---|---|
| เครื่อง / path | ihc-wctbp9102 (Windows x64) · `D:\NLP_mini_project\ai_research_gap_agent` |
| Shell | VM แยกเรียก Ollama ไม่ได้ → รันบน Windows ผ่าน job runner |
| Hardware | AMD Ryzen 7 9800X3D (8C/16T) · RAM 32 GB · NVIDIA RTX 5070 Ti 16 GB |
| Ollama 0.40.16 | มี qwen3:14b, embeddinggemma:300m · **ไม่มี qwen3:4b และ qwen3:8b** (ไม่ได้ pull ตามที่สั่ง) |
| git status ก่อนเริ่ม | สะอาด (main = origin/main) |
| pytest | 102 passed (46 deprecation warnings จาก langsmith/backoff) |
| .gitignore | ครอบ output/, pdf_cache*/, rag_store/, .env, _live_test/, evaluation/results/*.json · **ไม่ครอบ evaluation/results/logs/** |
| Cloud | ข้าม (ตามที่สั่ง; ANTHROPIC_API_KEY ว่างอยู่แล้ว) |

**ค่า .env ที่ต่างจากที่ให้ไว้ในแผน** (ตกลงให้ใช้ตามปัจจุบัน): LOCAL_MODEL_FAST=qwen3:14b (แผน: qwen3:4b), LOCAL_MODEL_STRONG=qwen3:14b (แผน: qwen3:8b), RAG_EMBEDDING=ollama:embeddinggemma:300m (ตัวอย่าง: bge-small), LANGFUSE_BASE_URL ตั้งไว้ (tracing เปิด), มี SCOPUS_API_KEY เกินมา, RAG_REUSE_EXTRACTS=true ใน .env แต่ **T1–T5 รันด้วย env var `RAG_REUSE_EXTRACTS=false`** (ไม่แก้ .env) — ยืนยันแล้ว extract_cache_hits = 0 ทุกรอบ

## 2. ผลรวม 5 หัวข้อ (backend local, qwen3:14b ทั้ง fast/strong)

| T | หัวข้อ | run_id | เลือก | วิเคราะห์ (full/abs) | ข้าม | นาที | LLM calls | tokens in/out | retry | queries | gaps | ระดับหลักฐาน |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | LLM hallucination detection | 20261121_162445 | 8 | 7 (6/1) | 1 LLMError | 10.0 | 31 | 71,372 / 9,478 | 0 | 5 | 4 | Moderate 3, Limited 1 |
| T2 | Thai named entity recognition | 20261121_163122 | 8 | 7 (2/5) | 1 LLMError | 6.5 | 27 | 45,268 / 5,513 | 0 | 5 | 4 | Moderate 4 |
| T3 | RAG evaluation (2024–2026) | 20261121_163922 | 8 | 7 (4/3) | 1 LLMError | 8.0 | 27 | 49,834 / 7,005 | 0 | 5 | 3 | Moderate 3 |
| T4 | prompt injection defense (2024–) | 20261121_164942 | 8 | 8 (8/0) | 0 | 10.3 | 31 | 70,506 / 9,487 | 0 | 5 | 4 | Moderate 4 |
| T5 | Thai legal document summarization | 20261121_165459 | 0 | 0 | 0 | 5.2 | 13 | 20,699 / 2,100 | 3 | 15 | 0 | — |

ไฟล์เต็ม: `evaluation/results/runs_summary.csv` · quote verified: T1 7/7, T2 6/7, T3 6/7, T4 5/8 · verifier supported: T1 3/4, T2 4/4, T3 3/3, T4 4/4 · evidence coverage 1.0 ทุก gap · ไม่มี Strong

## 3. สาเหตุ extraction failed (LLMError)

**สิ่งที่เห็นบนเครื่องนี้ (qwen3:14b):** ล้ม 3 ครั้งจาก 32 paper (T1 [1/8], T2 [1/8], T3 [7/8]) — **ทั้ง 3 เป็น abstract-only** prompt สั้น (~600–700 tokens, num_ctx 4096) จึงไม่ใช่ context ยาวเกิน, ไม่ใช่ timeout, ไม่ใช่ JSON ผิดรูป
สาเหตุจริง: **โมเดลคืน JSON template ของ prompt กลับมาทั้งก้อน** แล้ว contract ปฏิเสธครบ 3 ครั้ง
- หลักฐาน log: `logs/T1_local.log` บรรทัด 50–66 — `Contract violation ... copied the instruction text from the JSON template` ครบ attempt 1/3–3/3 แล้ว `Last raw response: {"problem": "the research problem or question the paper addresses, 1 sentence, or 'not stated'", "method": "the approach/technique used, 1-2 sentences", ...}` · แบบเดียวกันที่ `T2_local.log:54`, `T3_local.log:137` และ 10 บรรทัดใน `eval_query_adaptation.log`
- หลักฐานโค้ด: `prompts.py:32-43` ใส่คำอธิบาย field เป็น "ค่า" ใน JSON skeleton → ด้วย `format=json` + temperature 0 โมเดลลอกได้ง่าย · `contracts.py:60-70` ตรวจเจอและ raise · `llm_client.py:174-204` retry ด้วย prompt เดิม + ข้อความ IMPORTANT ต่อท้าย (temperature 0, seed เดิม) → โมเดลลอกซ้ำ ใช้เวลาแค่ ~3.8 วินาทีต่อครั้ง · `nodes.py:352-358` จับ LLMError แล้วข้าม paper
- full text 20/20 ฉบับบนเครื่องนี้สกัดสำเร็จทั้งหมด

**รอบของเพื่อน (T1 วิเคราะห์ได้ 3/8, LLMError 4 ฉบับ ตอนอ่าน full text):** ทำซ้ำบนเครื่องนี้ไม่ได้ (ไม่มี qwen3:4b, และไม่มี log ของเครื่องเพื่อน) สาเหตุที่น่าจะเป็น เรียงตามน้ำหนักหลักฐาน:
1. อาการเดียวกัน (ลอก template) แต่บ่อยกว่ามากกับ qwen3:4b ที่เพื่อนตั้งเป็น fast tier
2. prompt ถูกตัดหัว: `llm_client.py:276-277` ประมาณ token ด้วย `len/4` แล้วตั้ง `num_ctx = min(max(est + 2048, 4096), 8192)` · full-text prompt ประมาณ ~2.7–3.3k tokens → num_ctx ~4.7–5.3k ถ้า token จริงมากกว่าที่ประมาณ (สมการ ตาราง โค้ด ภาษาอื่น) Ollama จะตัด **ส่วนต้น** ของ prompt ซึ่งเป็นคำสั่ง เหลือแต่เนื้อหา + template → โมเดลลอก template · หลักฐานว่าเกิดได้จริง: Ollama `server-3.log` (19 พ.ย.) มี `truncated = 1` 30 บรรทัด ที่ n_tokens 4095/6143/8703/12287 (ระบุไม่ได้ว่ามาจากโปรแกรมไหน; server.log ของวันนี้ว่าง จึงยืนยันรอบวันนี้ไม่ได้)
ขอจากเพื่อน: บรรทัด `Last raw response` ใน log และ `truncated = 1` ใน `%LOCALAPPDATA%\Ollama\server.log` ช่วงเวลาที่รัน

**วิธีแก้ที่เสนอ (ยังไม่แก้):**
1. เอาเป้าที่ลอกได้ออก: JSON skeleton ใช้ค่าว่าง `""` แล้วย้ายคำอธิบายไปเป็นรายการ field ด้านบน หรือใช้ Ollama structured outputs `format = PaperExtractOutput.model_json_schema()` แทน `"json"`
2. retry ให้ต่างจากครั้งแรก: เมื่อเจอ echo ให้ส่ง prompt แบบไม่มี template (หรือเปลี่ยน seed/temperature 0.2) เพราะ temperature 0 + prompt เกือบเดิม = คำตอบเดิม
3. ตรวจ truncation: อ่าน `prompt_eval_count` จาก Ollama เทียบ num_ctx ถ้าเต็มให้ log เตือนและเรียกใหม่ด้วย num_ctx ใหญ่ขึ้น (GPU 16 GB รับ 14b ที่ 12–16k ได้) หรือประมาณ token ด้วย `len/3`
4. ถ้าสกัด full text ล้ม ให้ลองด้วย abstract ก่อนข้าม paper

## 4. eval_citations / eval_query_adaptation

**eval_citations** (`eval_citations_20261121_165841.md`, 20 paper ที่ gap อ้าง จาก T1–T4):

| แหล่ง | found | title match | year exact | year ±1 | DOI match |
|---|---|---|---|---|---|
| OpenAlex | 1.0 | 1.0 | 0.85 | 1.0 | 0.938 (n=16) |
| Crossref | 1.0 | 0.65 | 0.50 | 0.90 | 0.75 (n=16) |

ไม่มี paper ปลอม (หาเจอครบ 20/20) · OpenAlex ไม่ตรง 3 แถว (ปี preprint vs ฉบับตีพิมพ์ / DOI arXiv vs DOI ฉบับตีพิมพ์) · Crossref ไม่ตรง 10 แถว ส่วนใหญ่เป็นแถว "by title" ที่ Crossref คืน paper คนละเรื่อง (เช่น "Master RAG Systems", "Automatic Extraction of Entities ... Legal Documents") = lookup artefact · **รอคนตรวจ 13 แถวนี้**

**eval_query_adaptation** (`eval_query_adaptation_20261121_172211.md`, ใช้เวลา 23.5 นาที): retry เกิด 1/3 หัวข้อ (Lao legal NER: 3 รอบ, คำค้นใหม่ทุกรอบ แต่ผ่าน cut-off 0 ทุกรอบ → ไม่มี paper), COBOL/Thai banks และ Khmer essay scoring ไม่ retry (ได้ 17 และ 16 paper ผ่าน cut-off) · **รอคนตัดสินว่าคำค้นใหม่ "สมเหตุผลหรือ drift"** (eval_plan §2)

## 5. ปัญหาอื่นที่เจอ

1. **Candidate pool ตันที่ 80 ทำให้ retry ไม่ได้ paper ใหม่** — `nodes.py:146` `_dedupe(old + new)[:80]` เก็บของเก่าก่อน ตัดของใหม่ทิ้ง · T5: รอบ 2–3 เจอ 172/181 paper แต่ pool ยัง 80 ชุดเดิม ผ่าน cut-off 0/80 ทั้ง 3 รอบ · Lao: 68→80 แล้วตัน · ผลคือ retry loop แทบไม่ช่วยหัวข้อแคบ เสนอ: จัดอันดับ union ก่อนตัด หรือแทนที่ด้วยผลรอบใหม่
2. T5 "Thai legal document summarization" ไม่ได้ gap เลย (ไม่มี paper ผ่าน cut-off 0.50 — ผลค้นเป็นงาน legal summarization ของอินเดีย/อังกฤษ) รายงานควรบอกชัดว่า "ไม่พบ paper ที่เกี่ยวข้องพอ"
3. Semantic Scholar 429 แทบทุกรอบ (ระบบ retry และไปต่อได้)
4. log ใน `evaluation/results/logs/` มี URL ของ PDF 1 รายการที่ชื่อไฟล์มีนามสกุลผู้แต่ง (`eval_query_adaptation.log`) → **ไม่ควร commit โฟลเดอร์ logs** หรือเพิ่มใน .gitignore
5. ไฟล์ที่ commit อยู่แล้ว `evaluation/data/ranking_set.json` มี "(Gruzitis, et al., 2018)" ใน abstract (มีอยู่บน GitHub ก่อนรอบนี้)
6. log T1 ไม่มีบรรทัด "=== T1 end" (บั๊กใน .cmd ของผม `exit=%RC%>>`; แก้แล้วใน T3–T5; ไม่กระทบผล exit code = 0)
7. มีโฟลเดอร์ `Claude outputs/` (untracked) ที่แอปบันทึกไฟล์ .pptx ไว้ — ไม่ได้แตะ

## 6. ตรวจชื่อผู้แต่ง (ขั้น 8)

ดึงชื่อผู้แต่ง 149 ชื่อของ 32 paper (ที่เลือกและที่ gap อ้าง ใน T1–T5) จาก API (2 ฉบับดึงไม่ได้เพราะ 429) แล้วค้นในทุกไฟล์ใน `evaluation/results` และ `evaluation/data`: **ไม่พบชื่อในไฟล์ที่สร้างรอบนี้** · คำที่ match มีแค่ "will" (นามสกุลที่ตรงกับคำภาษาอังกฤษ = false positive) · รายชื่อเก็บที่ `_live_test/author_check_names.txt` (git-ignored) · ข้อจำกัด: ไม่ครอบชื่อผู้แต่งของ paper ในรอบ query adaptation

## 7. ไฟล์ที่สร้าง

| ไฟล์ | สถานะ |
|---|---|
| `evaluation/results/runs_summary.csv` | เสร็จ |
| `evaluation/results/eval_citations_20261121_165841.md` (+ .json git-ignored) | เสร็จ · รอคนตรวจ 13 แถว mismatch |
| `evaluation/results/eval_query_adaptation_20261121_172211.md` (+ .json git-ignored) | เสร็จ · รอคนตัดสินคำค้น |
| `evaluation/results/logs/` (pytest, T1–T5, eval_citations, eval_query_adaptation) | เสร็จ · ไม่แนะนำให้ commit |
| `evaluation/data/extraction_gold_<run>.csv` × 5 | **รอคนกรอก** (T5 มีแต่หัวคอลัมน์ เพราะไม่มี paper) |
| `evaluation/data/gap_rating_TODO.csv` (15 gaps) | **รอคนกรอก** คะแนน |
| `evaluation/data/gap_reference_TEMPLATE.csv` | **รอคนเขียน** gap อ้างอิงต่อหัวข้อ ก่อนดูผลระบบ |
| `evaluation/results/eval_summary_2026-11-21.md` | ไฟล์นี้ |

ยังไม่ได้ทำ: consistency test (ขั้น 7) — รอคำตอบ (ประมาณ 5 × 7–10 นาที ≈ 40–50 นาที)
