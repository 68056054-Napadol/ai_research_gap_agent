"""
What the user is told about the evidence behind an answer - shared by the Markdown report
(main.render_report), the web UI (app.py), the Dashboard and the Langfuse trace, so all four say
the same thing.

Teacher feedback this module answers: the system must not claim to have *discovered* a research
gap. It proposes *Potential Research Gaps* from the papers it happened to find and analyse, and
says how wide that search was.
"""

ANSWER_TITLE = "Potential Research Gaps"

# Shown under every "Potential Research Gaps" heading (report, UI, Dashboard).
LIMITATION_NOTE_TH = ("เสนอจากหลักฐานในชุด paper ที่ระบบค้นและวิเคราะห์ได้เท่านั้น "
                      "ไม่ใช่การทบทวนวรรณกรรมอย่างเป็นระบบ ผู้ใช้ควรตรวจสอบต่อ")
LIMITATION_NOTE_EN = ("Proposed only from the papers this run found and analysed - not a systematic "
                      "literature review. Check each one before relying on it.")
LIMITATION_NOTE = f"{LIMITATION_NOTE_TH} ({LIMITATION_NOTE_EN})"
