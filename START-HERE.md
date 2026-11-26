# Start here

You have a 48-hour case from CGPRR group for the AI Developer role in Rajarhat,
Kolkata. Deadline: **28 November 2026, 6:00 PM IST**. Reply to the email from
`weholdp@gmail.com` with three things (plus your CTC details).

This folder is your submission-in-progress.

## What the case is actually asking

They want a plan for running two AI products on **one GPU they already own**:

- **Counsellor Assist** for NEET.info - a chat assistant that answers families'
  questions about MBBS fees, bonds, seats and counselling rules, and must cite a
  document for every number.
- **Hirrx Match** - ranks doctors' resumes against hospital job descriptions and
  explains the ranking.

The founder's three rules: no student or doctor data leaves the building, a
wrong fee figure is worse than no answer, and "show me the rupees".

The trap in the pack is a consultant's plan that looks reasonable and is wrong
in four places. Your job is to do the arithmetic, write the code, and make a
call a founder could act on Monday morning. They explicitly reward naming
trade-offs, stating assumptions, and refusing to invent legal rules.

**The one finding that wins or loses this:** a single 48 GB card cannot serve
the June-September peak, and the reason is *prefill* (reading 3,200 tokens of
context before generating a word), not output speed. One card covers 26-45% of
season. Everything in the plan follows from that.

## What is in this folder

| File | What it is | Where it goes |
|---|---|---|
| `design-note.pdf` | The 6-page design note, one-page executive summary on top. Parts B, C, D, F, G2. | Submission item 1 |
| `design-note.md` | Same, editable source | Keep |
| `calculations.xlsx` | Every calculation in 6 sheets: KV cache, throughput, worst-hour demand, batch plan, 3-year TCO, GPU rental | Submission item 3 |
| `prototype/` | The Part E code: grounded Q&A, 40+12 question evaluation, resume matcher, 27 tests | Submission item 2 |
| `prototype.zip` | Zipped code for the email attachment | Submission item 2 |
| `build_calcs.py`, `build_pdf.py` | Scripts that generate the spreadsheet and PDF from the numbers | Keep |

## How to check it yourself

```bash
cd prototype
pip install -r requirements.txt
python run_all.py          # everything: Q&A demo, evaluation, matcher
python -m pytest -q tests  # 27 unit tests
```

The code runs on a laptop CPU in under a second. If a line in the design note
ever looks wrong, `calculations.xlsx` has the arithmetic behind it and
`build_calcs.py` regenerates both.

## Before you send it

1. **Put your real numbers in.** The cost model assumes an engineer CTC of
   INR 18 lakh a year and a junior at INR 55,000 a month. Replace yours in
   `build_calcs.py` (`eng`, `junior`) and rerun both scripts, or the ₹95.5 lakh
   three-year figure is not your plan.
2. **The email also asks for current CTC, expected CTC and notice period.**
   That goes in the covering reply, not the PDF.
3. **Say what is not verified.** The note already does this - no GPU was
   available, so the capacity numbers are roofline estimates calibrated against
   the measured decode speeds in Exhibit 3. Keep that honesty; the panel will
   probe it.
4. **Be ready for the 45-minute defence.** They will change one number (VRAM,
   traffic or price) and ask you to re-plan on the spot. The fastest way to be
   ready: open `build_calcs.py`, change `CARD_GB = 48` to `96`, and rerun. The
   note's claim that drops out - Model XL fits at 96 GB and holds 40 sequences -
   is exactly the kind of answer they will ask for.

## The weakest parts, if you want to improve them

- Retrieval weights and two intent constants were tuned on the 40-question set,
  so the 1.000 scores there are an upper bound, not performance.
- The generator is extractive (a stub), so no answer here is a real LLM answer.
- `nDCG@10` is on the release-gate list but not implemented in the code.
- The roman-Hindi/Bengali lexicon covers the 10 questions in the set and will
  not cover a family who spells it a fourth way. The note says so.
