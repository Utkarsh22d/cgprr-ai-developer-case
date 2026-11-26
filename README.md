# Private LLM platform on one GPU — CGPRR AI Developer case

Response to the CGPRR group / WeAI Labs screening case: plan and prototype a private
AI platform for **NEET.info Counsellor Assist** and **Hirrx Match** on a single
48 GB RTX 6000, with no student or doctor data leaving the building.

**Live demo:** https://cgprr-ai-developer-case.vercel.app  
**Mirror:** https://neet-info-counsellor-assist-demo-9fhmxj-preview.freebuff.site

Everything in this repository is synthetic. No real college, fee notice, person,
counselling rule or medical college is represented.

## The answer in one line

One card runs a normal month; it cannot run the June–September counselling peak, and the
bottleneck is **prefill**, not output speed. At season peak the workload needs ~20,500
prompt tokens/second against a card that prefills at ~1,500. The recommendation is
Model M (4-bit) as the answerer, Model S as router and fast path, Model L on escalation,
a second GPU in May 2027 (not now), and a hard numeric guard in front of every figure.

## What is here

| Path | What it is | Submission item |
|---|---|---|
| [`design-note.pdf`](design-note.pdf) | 6-page design note, one-page executive summary on top (Parts B, C, D, F, G2) | (1) |
| [`design-note.md`](design-note.md) | Editable source of the same | — |
| [`calculations.xlsx`](calculations.xlsx) | Six sheets: KV cache, throughput roofline, worst-hour demand, batch plan, three-year TCO, GPU rental | (3) |
| [`prototype/`](prototype/) | Part E code: grounded Q&A, numeric guard, evaluation, resume matcher, 27 tests | (2) |
| [`site/`](site/) | Static demo frontend (ask, guard playground, capacity, matcher) | — |
| [`build_calcs.py`](build_calcs.py) · [`build_pdf.py`](build_pdf.py) | Regenerate the workbook and the PDF from one set of numbers | — |
| [`START-HERE.md`](START-HERE.md) | Orientation: what the case asks, what to check before submitting | — |

## Run the code (one command)

```bash
cd prototype
pip install -r requirements.txt
python run_all.py        # Q&A demo, Incident 1 replay, evaluation, matcher
python -m pytest -q tests
```

CPU only, numpy only, no network, no paid API, no model download. It runs in under a
second. `EMBEDDER=st` switches the offline encoder for a real sentence-transformer if
one is installed.

## Key results

| Finding | Number |
|---|---|
| KV cache per token, Model L / XL (16-bit) | 256 KiB / 320 KiB |
| Concurrent 3,500-token chats in 48 GB, Model L 4-bit | 26 (fp16 KV) / 52 (int8) |
| Model XL 4-bit on this card | 2.7 GB of KV left — cannot serve chat |
| Season peak prompt load | 20,533 tok/s raw, 5,647 tok/s with a 2,200-token cap and 60% prefix cache |
| Three-year TCO, one GPU / two GPUs | INR 95.5 lakh / 105.7 lakh (75% of it two people) |
| Cost per 1,000 chats | INR 1,290 all-in, INR 94 marginal |
| Cloud comparison | INR 1.09 per chat — not cheaper; break-even 2.91 M chats/year |
| Cost per resume ranked | INR 1.30 on-prem vs INR 0.13 cloud |
| Hirrx backlog | 14.6 GPU-hours → three nights, then 0.69 GPU-h/week |

## Evaluation, honestly

`run_all.py` prints two tables: 40 questions used to tune retrieval, and 12 written
afterwards. Both now score 1.000, which means neither is evidence — every failure the
second set found was fixed, so it became a tuning set too. The number worth remembering
is that before one bad rule was replaced, the stress set **refused 1 in 3 answerable
questions** because "a word the corpus has never seen" fired on *penalty* and *bhai*.
Part C3 of the note spends a fortnight on the only real fix: 500 questions from two
counsellors, frozen before anyone tunes on them.

## A failure that is still in the code

Asking for the Assistant Professor norm quotes the **Associate Professor** row: right
document, right citation, every number real and guard-passing, and still the wrong row,
because both rows contain the phrase "Assistant Professor". It shows why the numeric
guard is necessary but not sufficient, and why the plan funds a structured table parser
rather than a larger model.

## What is stubbed

The generator is extractive by design (it copies a cited row instead of calling an LLM);
the dense encoder is offline LSA standing in for a multilingual sentence-transformer; the
cross-encoder re-ranker is lexical; no GPU was available, so capacity figures are roofline
estimates calibrated against the measured decode speeds in the case exhibits.

## Demo

A static build of the prototype's real outputs lives in [`site/`](site/), deployed at
**https://cgprr-ai-developer-case.vercel.app**.

It has four sections: **Ask** (52 questions, each with its answer, citation and guard verdict),
**Guard** (the numeric guard ported to the browser, so you can try to sneak a wrong figure past
it), **Capacity** (KV cache, roofline, worst hour and the three-year cost) and **Match**
(ranked candidates with explanations and the fairness report).

No build step. To run it locally, serve the folder:

```bash
cd site/public && python -m http.server 8000
```

On Vercel the project root is `site`, output directory `public`, framework Other — set in
[`site/vercel.json`](site/vercel.json). The mirror is hosted the same way.
