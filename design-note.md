# Private AI platform on one RTX 6000: what I would build, and what it costs

**CGPRR group / WeAI Labs · NEET.info Counsellor Assist and Hirrx Match**
Case response · 26 November 2026 · Utkarsh Dwivedi

## Executive summary

**The short version.** One card can run this platform for a normal month, and it cannot run the June-September counselling peak. The reason is not output speed. It is *prefill*: every family question drags 3,200 tokens of context through the GPU before a single word is generated. At season peak that is about 20,500 prompt tokens per second against a card that prefills at roughly 1,500. Result day is 2.7x worse again. One card covers 26-45% of the season and about 26% of result day even with a second card and a rewritten prompt.

**Five decisions I am asking for this week.**

1. **Build on one card, with three models, not one.** Model M at 4-bit answers; Model S routes and answers the high-volume identical questions; Model L is an escalation path, not the default. Zero extra capex, and it is the difference between 26% and 45% of season peak on the card we already own.
2. **Approve the second GPU in May 2027 (INR 8.9 lakh), not now.** The arithmetic says buy it; the calendar says buy it after the March launch is stable. One card is enough for the pilot and for a normal day (12% load).
3. **Approve a hard numeric guard and accept the refusals it causes.** No number reaches a family unless it appears in the cited chunk of a current-session notice. In the pilot this would have blocked Incident 1 outright. It will refuse more questions than the consultant's design, and that is the point.
4. **Approve a structured "cut-off card" fast path for result day.** If 70% of result-day questions are answered by a deterministic lookup against the seat and cut-off tables with no LLM in the path, two cards hold. Without it, result day needs four to six cards.
5. **Approve the DPDP erasure runbook before the first resume is ranked.** Erasure has to reach the vector index, the cache, the logs and the backups, and that has to be designed in, not retrofitted.

**The rupees, in one paragraph.** Three years, all in: **INR 95.5 lakh** with one card and **INR 105.7 lakh** with two. Of that, **75% is two people**, which is the finding that matters - the GPU is not the expensive part of this plan. All-in cost is INR 1.29 per chat, or **INR 1,290 per 1,000 chats**; the marginal cost of one more chat is nine paise. A cloud plan at the prices in Exhibit 1 would cost INR 1.09 per chat, so it is **not cheaper**: break-even is 2.91 million chats a year and we forecast 2.5-3.1 million. The founder's constraint is therefore not costing him money - but he should know that the privacy argument, not the rupee argument, is the one carrying this decision. Resume ranking is the opposite: INR 1.30 per resume on-premises against 13 paise in the cloud, a tenfold premium we pay for the DPDP obligation, not for the compute.

**What quality to expect, honestly.** Raw model quality is 72% for Model M against 81% for Model L on the consultant's 300 questions. That 9-point gap is not the quality gap that will hurt us. In my prototype the retrieval was correct, every number passed the guard, and the answer was still wrong: asked for the Assistant Professor norm it quoted the Associate Professor row, because both rows contain the phrase "Assistant Professor" (Q18, reproduced in the code and the evaluation output). **The residual risk after retrieval and the guard is row-level error, not model-level error, and a bigger model does not fix it.** A structured table does. That is why the plan puts the engineering week into the table parser and not into a bigger model.

**What I am not claiming.** I have not run a model on a GPU; there is no GPU in this exercise. Every capacity number in Part B is a roofline estimate with stated efficiency factors, calibrated against the one thing Exhibit 3 does give me - measured single-stream decode - and the calibration closes within 63-71% for all four models, which is what makes me trust the rest of it. The prototype in Part E runs on CPU with an extractive stub generator, no paid API, and its evaluation numbers are reported twice: once on the 40 questions I built first and tuned against, and once on 12 questions written afterwards. Both are now perfect, which means neither is evidence. The honest number is that before I replaced one bad rule, the stress set refused **1 in 3 answerable questions**; 40 questions cannot measure anything to three decimal places, and Part C3 spends a fortnight on the only fix that works - 500 questions from two counsellors.

## Part B: will one GPU do it?

**B1. KV cache.** Per token the cache holds, for every layer, one key vector and one value vector per KV head:

`bytes/token = 2 x layers x kv_heads x head_dim x bytes_per_element`

At fp16 (2 bytes): Model L `2 x 64 x 8 x 128 x 2 = 262,144 B` = **256 KiB/token**; Model XL `2 x 80 x 8 x 128 x 2 = 327,680 B` = **320 KiB/token**. At int8 exactly half: **128 KiB** and **160 KiB**.

Budget: 48 GB card, 10% reserved for kernels, activations and fragmentation = 43.2 GB usable. A 3,500-token sequence therefore costs 0.918 GB on Model L and 1.147 GB on XL.

| Model, 4-bit weights | Weights | KV pool | Seqs @3,500 tok, fp16 | Seqs @3,500 tok, int8 |
|---|---|---|---|---|
| L (32B) | 19.0 GB | 24.2 GB | **26** | **52** |
| XL (70B) | 40.5 GB | 2.7 GB | **2** | **4** |
| L, 8-bit weights | 34.5 GB | 8.7 GB | 9 | 18 |
| XL, 8-bit weights | 73.0 GB | -29.8 GB | 0 | 0 |

The first finding is the one that changes the model choice: **Model XL at 4-bit leaves 2.7 GB for KV cache on a 48 GB card. It fits twice. It cannot serve chat**, and at 8-bit it does not load at all. Any plan that names XL as the answerer is a plan for a 96 GB card. On a 96 GB card XL would leave 46.5 GB of KV and hold 40 fp16 sequences, which is usable but still worse per rupee than L.

**B2. The worst hour.** 6 turns per chat, 70% of traffic between 7 and 11 PM. Model L, 4-bit, fp16 KV.

| Scenario | Chats/day | Turns/s in peak hour | Prompt tok/s (raw) | Prompt tok/s (our design) | Output tok/s |
|---|---|---|---|---|---|
| Normal day | 1,500 | 0.44 | 1,400 | 385 | 122 |
| Season | 22,000 | 6.42 | 20,533 | 5,647 | 1,797 |
| Result day | 60,000 | 17.50 | 56,000 | 15,400 | 4,900 |

*Our design* = a 2,200-token context cap (system and rules 800, three chunks at 450, question) with a 60% prefix-cache hit, so about 880 new tokens per turn. The 3,200 in Exhibit 2 is what the consultant assumed; it is the single largest cost lever in the plan.

**Why single-stream decode is the wrong number for capacity.** Exhibit 3 measures a batch of one. At batch one the GPU re-reads every weight for every token: 19 GB of weights at 960 GB/s gives 50 tokens/s ceiling, and the measured 36 is 72% of that. That agrees for all four models (63-71%), which is how I know the card is memory-bound at batch one. Under continuous batching, one weight read serves all resident sequences, and the limit becomes total bytes per step: weights plus the whole KV pool. On Model L that is 43.2 GB per step, 45 ms, and 26 sequences per 45 ms = 582 tokens/s ideal, **408 tokens/s after a 70% efficiency factor**. Batching buys 11x on decode. It does nothing for the first token, because the first token is a prefill: compute-bound, 2 x params x tokens FLOPs, 64 GFLOP per token on Model L, **640 tokens/s** at 41 effective TFLOPS (45% of the card's 91 dense).

| Model | Decode, one stream | Decode, batched real | Prefill |
|---|---|---|---|
| S (8B) | 118 | 1,287 | 2,559 |
| M (14B) | 74 | 928 | 1,462 |
| L (32B) | 36 | 408 | 640 |
| XL (70B) | 15 | 31 | 292 |

Prefix caching is what makes the plan affordable: the system prompt, the counselling rules and the top chunks repeat across thousands of questions in the same hour, so 60% of the 3,200 prompt tokens never run through the GPU twice.

**Can Model L meet p95 on one card?** No, and it fails at the first token, not the last. One turn's prefill at 880 new tokens costs 1.4 s on an idle card; at season peak the queue alone puts p95 time-to-first-token past the 2.5 s target before generation starts. Model M at 1,462 tokens/s covers 26% of season demand; Model S at 2,559 covers 45%. The assumption I am least sure of is the **60% prefix-cache hit rate** - it is the number the whole cost model hangs on and it can only be measured on real traffic. If it is 30%, season prefill demand rises from 5,647 to 9,900 tokens/s and even two cards fail.

**B3. Hirrx backlog.** 240,000 resumes against 3,500 JDs, overnight 23:00-07:00, is **not** a GPU problem - it is 14.6 GPU-hours, and most of that is the LLM explanation step, not the ranking.

| Stage | GPU-hours |
|---|---|
| Embeddings, 240k x 1,900 tokens, 0.3B encoder | 1.9 |
| Cross-encoder re-rank, 3,500 JDs x top-50 | 1.9 |
| LLM explanations, 3,500 JDs x top-10, Model S | 10.8 |
| OCR of the 72,000 scanned PDFs (CPU, 6-way) | 26.7 CPU-hours, run in parallel |
| **Total GPU per full backlog** | **14.6** |
| Weekly increment, 4,000 new resumes | 0.69 |

Schedule: **three nights for the backlog** (23:00-03:00 embeddings and re-rank, 03:00-07:00 explanations, OCR on CPU throughout), then a fixed 01:00-02:00 nightly slot for new resumes. That leaves the GPU free from 19:00 to 23:00, which is the hour that actually matters. The backlog job is pre-emptible, so it yields to chat traffic by design.

**B4. Recommendation.** **Model M at 4-bit as the answerer, Model S at 4-bit as router and fast path, Model L at 4-bit on escalation only** (guard trip, low confidence, multi-document question). int8 KV cache on M and S; we gain 2x concurrency and lose a measurable amount of long-context quality that I would want to test before committing.

What we give up: 9 quality points versus L on raw answers (72% vs 81%). What we gain: 2.3x the prefill throughput and 2.3x the concurrency, which is the difference between serving 1,600 and 3,700 season questions an hour, and the guard recovers most of the quality gap on the questions that matter because a grounded extract-and-cite task is easier than open generation. I would ask for the second GPU on any one of three triggers: season forecast above 8,000 chats a day, p95 TTFT above 2.5 s for three consecutive days, or the fast path covering less than half of result-day traffic.

## Part C: retrieval and evaluation a regulator could audit

**C1. Document pipeline.** Five stages, each with a rule that maps to a failure in Exhibit 4.

*Ingest.* Every notice gets a source URL, a fetch date, the issuing authority, a SHA-256 of the PDF and the session year. Nothing enters without all five; a document with no session year cannot be cited for a current question.

*Parse.* Text PDFs through a layout parser that preserves table cells; 30% are scanned, so OCR with a confidence threshold and a human review queue below 0.90. Tables are parsed to rows, never to flowing text, because the failure we are defending against is a number losing its column.

*Chunk.* One row per chunk, with the header row glued on: "Annual tuition fee | INR 10,40,000" is a chunk; "INR 10,40,000" alone is not. Prose is chunked in two-sentence windows. Every chunk inherits `session`, `state`, `college`, `programme`, `doc_type`, `status`.

*Version by session.* Documents are immutable and versioned: a 2026-27 notice does not overwrite the 2025-26 one, it supersedes it. `status: superseded` is set on the old one by the ingestion job, which is idempotent and can be replayed. The retriever filters to the question's session; a superseded chunk is reachable only by a question that names its session explicitly, which is how "what was the fee in 2025-26" still works.

*Retire.* Never delete. A superseded document stays for audit and for historical questions, is excluded from every default retrieval, and is listed in the index manifest with the reason and the approving user. **Incident 1 becomes impossible at the index layer, not the prompt layer:** the 2025-26 fee PDF can no longer be retrieved for a 2026-27 question, whatever the model does, because the filter runs before the model sees anything.

Metadata filters: `session` (hard), `state` (hard when the question names a state), `programme` (hard: MBBS vs BSc Nursing vs MD), `doc_type` (hard for bond: a bond question can only be answered by a bond rule), `college` (soft boost).

**C2. Bengali and Hindi in Roman script.** The pilot's recall@5 is 0.88 English and 0.52 Bengali-in-Roman. Four approaches, cheapest first.

| Approach | Effect on recall@5 | Latency | Verdict |
|---|---|---|---|
| Transliteration before retrieval (rule-based ITRANS-style map) | +0.05 to +0.12 on Bengali; brittle on inconsistently spelled words ("sulkho", "shulk") | +5 ms, no model | Necessary but not sufficient |
| Multilingual embeddings (Indic-tuned encoder or a multilingual 0.6B) | +0.10 to +0.20; the real gain, because it handles spelling variation natively | +15 ms per query, 0.6B in VRAM | **Adopt** |
| Hybrid BM25 + dense with a cross-encoder reranker | +0.08 on top of either; the reranker fixes ordering, not recall | +40-60 ms for 50 pairs on the GPU | **Adopt** |
| LLM query rewriting into English before retrieval | +0.15 when the rewrite is right; destroys recall when it is wrong, and adds 400-800 ms | +600 ms, and a second hallucination surface | Only as a fallback for queries that score low |

My prototype takes a fifth, cheaper route that is available to us because the domain vocabulary is small: a hand-built lexicon mapping roman-Hindi/Bengali function words and domain words onto the corpus vocabulary ("kitni/koto" to "how much", "shulk" to "fee"), plus character 4-grams in the embedding so spelling variants collide without a transliteration table. On 10 roman-script questions it retrieves the right document first every time. That is a floor, not a result: 10 questions cannot separate 0.52 from 0.88, and the lexicon will not cover a family who spells it a fourth way. The production answer is a multilingual encoder plus the lexicon plus the reranker, and a golden set stratified 62/21/14/3 by language so the number is measured, not asserted.

**C3. The release gate.** If any line fails, the product does not go live.

| Metric | Threshold | My prototype |
|---|---|---|
| recall@5 (500-question golden set) | >= 0.90 overall, >= 0.85 on roman script | 1.00 on 52 questions, not credible - see below |
| nDCG@10 | >= 0.85 | not implemented |
| citation precision | >= 0.98 | 1.00 |
| numeric faithfulness after the guard | 100% - zero untraceable numbers in 500 | 1.00, and 100% of injected wrong figures blocked |
| refusal accuracy on unanswerable | >= 0.95 | 1.00 |
| false refusal on answerable | <= 0.10 | 0.00 dev, **0.33** on the stress set before I replaced one rule |
| p95 TTFT / p95 full answer, at season load | <= 2.5 s / <= 9 s | not measurable without the GPU |
| session-year correctness | 100% - no current answer cites a superseded notice | 100% on 52 questions |
| human audit, 2 counsellors, 500 answers | **zero** wrong fee or bond figures | not yet run |

**The 500-question golden set in three weeks, two counsellors.** Week 1: freeze the schema and the answer keys (document ID, chunk ID, the exact figure, the session), seed 60 questions from real chat logs, and agree the adjudication rule. Week 2: each counsellor writes 220 questions from the notices without seeing the other's, stratified by state and by document type; 40 disagreements are adjudicated together and the adjudication becomes a written rule. Week 3: balance to 62/21/14/3 by language, add 150 unanswerable questions drawn from real out-of-scope asks (salary, cut-off, BDS, other states), double-annotate 100 for agreement and report kappa, then freeze and split 60/40 dev/test **before** anyone tunes anything. The reason my own numbers cannot be trusted is exactly this: I wrote 40 questions, tuned against them, then wrote 12 more and fixed every failure they found, which means I have now tuned against those too. Only a frozen set built by two other people measures anything.

**How a number is stopped from being invented.** Four gates in series, all cheap, all auditable: (1) the model may only answer from chunks, and every claim carries a chunk ID; (2) a regex-and-normaliser pass extracts every number from the answer and from the cited chunks, folds "5 lakh" and "INR 5,00,000" to the same value, ignores doc IDs and session ranges, and blocks the answer if any number is missing; (3) a second pass blocks programme names that the cited document mentions only in order to exclude them ("does not apply to BDS seats"); (4) an attribute gate refuses questions no document type can answer. In the prototype the guard blocked a deliberately injected "INR 5 lakh" for Haryana - the exact Incident 1 figure, which is real but belongs to Uttar Pradesh - on 100% of attempted injections.

**C4. The founder's "closest answer".** 148 words.

I would argue against him, and I would not say "I don't know". Two different failures are being merged. Tone is the family's problem: a flat refusal feels like being turned away at a counter, and he is right about that. Content is our problem: a wrong bond figure cost one Patna family a near-miss with a broker; the case does not say what it cost, but a five lakh bond paid on the strength of an answer is five lakh gone and a complaint we cannot defend. At 22,000 chats a day, a 0.5% wrong-number rate is 110 families a day in season. So the answer to "be less blunt" is not "be less accurate": it is **"the notice says 10 lakh for Haryana; I cannot confirm your college's figure, here is the number to call"** - a specific, cited, actionable near-answer, which is what families actually want and what the guard permits. The compromise I would refuse is the model's own paraphrase of an uncertain figure.

## Part D: Hirrx Match, fairness and the law

**D1. Architecture.** Rules where the law is a rule, learned models where judgement is required.

*Extract.* Structured parse of each resume into degree, specialty, registration, years of service, procedures and skills, with every field pointing back to a page and an offset. Unparseable resumes go to a human queue, not to a low score.

*Screen - rules.* NMC faculty norms (Professor 8 years with 4 as Associate, Associate 5 with 2 as Assistant, Assistant 3), the post's minimum experience, and degree eligibility are **hard gates**, not features. A hospital cannot hire outside them, so a soft score would only produce candidates who cannot be appointed. This is also the cheapest fairness win available: every candidate sees the same rule, applied identically, and can be told why they were excluded.

*Rank - learned.* Embedding similarity between resume and JD, plus explicit skill coverage, plus experience fit, combined in the weights in my code (experience 0.35, specialty 0.30, skills 0.25, recency 0.10, embedding 0.10). A cross-encoder re-ranks the top 50. Rules decide *who is eligible*; the model decides *the order inside the eligible pool*.

*Explain.* Two lines generated from the same feature vector that produced the score, with the data quoted. Explanation is not a separate model, because a separate model will contradict the score.

The one thing deliberately absent: **any feature derived from employment gaps or dates.** My matcher computes experience from *recorded service years*, never from date arithmetic on a CV, and the test suite asserts that changing a gap label from none to maternity, illness or caregiving leaves the score bit-identical and the rank unchanged.

**D2. The two gaps.** Women are 27% of the top 10 against 41% of the pool: a selection-rate ratio of 0.66, which fails the four-fifths rule (0.80). Twelve "top" colleges are 58% of the top 10 against 19% of the pool: 3.1x over-representation, which is exposure concentration.

I reproduced the defect rather than assuming it. My own matcher, with college name absent from every feature and the gap excluded by construction, still lands at a ratio of **0.64** on the applicant pool for the Assistant Professor post - women are 25% of the top 10 against 40% of the applicants - because specialty and skill coverage correlate with who applies. A monitor that would have caught the first two gaps catches this one too, which is the argument for shipping the monitor before the second model.

Measurement: **selection rate at top-k by group, per JD, per quarter**, reported as min/max ratio, with a threshold of 0.80; below 0.80 for two consecutive cycles freezes ranking for that JD and triggers a manual audit. Second measure: **exposure parity** - the share of each group in the top-10 should sit within +/-5 points of the eligible pool. Third: a **counterfactual swap test** - change only the college name on 200 synthetic resumes and require the score to move by less than 0.02; if it moves more, the feature is acting as a proxy for caste, class or gender.

Two fixes I would ship first. (1) **Remove college name and institution prestige from the feature set entirely** and replace them with the signals that actually predict performance - procedures performed, publications, years of supervised service - because a college name in India is a proxy for everything we are not allowed to use. (2) **Two-stage ranking with a rule gate before the learned score**, plus a per-JD disparity monitor on the top-10 that blocks publication of a shortlist whose ratio is below 0.80. On maternity: score from recorded service, ignore gaps, and stop the explanation generator from having access to the gap field at all - Incident 2 happened because the explanation had a field it should never have had.

**D3. Incident 2 under the DPDP Act, 2023.** What Hirrx must be able to do: give **notice** of what is collected and why; obtain and record **consent** with a purpose and an expiry, and honour withdrawal; answer an **access** request with the data and, in practice, an explanation of the ranking; **correct** wrong facts; **erase** on request; and publish a **grievance** route with a named officer and a response clock.

What erasure means for our stack. An embedding is derived from her resume and remains personal data under a "relates to an identified individual" test - I would treat it as personal data. So erasure is: delete the resume row and every parsed field; delete every chunk; delete the vector index entry (and rebuild or tombstone the index segment, because most vector stores leave the vector in a segment until merge); delete cached prompts and answers keyed to her candidate ID; delete or pseudonymise application logs, with a **30-day** default retention instead of the usual 12 months; and handle backups by tombstone-and-reindex - a restore must replay the erasure list before serving, and the restore procedure is tested quarterly. The proof we give her is an erasure receipt listing each system touched and a timestamp.

Where I am not sure and would check with counsel, not invent: (a) whether an automated ranking decision triggers an explanation duty under DPDP - as I read it, the Act's provisions on automated decision-making are not yet in force in the form GDPR Article 22 takes, and the Rules are still being notified; (b) the exact retention obligation for recruitment records under labour law, which may exceed the DPDP minimum; (c) whether a hospital customer is a data fiduciary and Hirrx a processor, or whether Hirrx is a fiduciary in its own right for the ranking layer - this changes who answers the data principal's request; (d) cross-border storage, if any part of the backup lands outside India. I would not build on my reading of any of these; I would get a written opinion before the first resume is processed.

## Part F: build or buy, in rupees

**F1. Three-year TCO, December 2026 to November 2029.**

| Line (INR) | One GPU | Two GPUs |
|---|---|---|
| Workstation (sunk) | 14,80,000 | 14,80,000 |
| UPS and cooling | 2,10,000 | 2,10,000 |
| Second GPU | 0 | 8,90,000 |
| People: engineer at 18 lakh/yr + junior from Apr 2027 | 71,60,000 | 71,60,000 |
| Power: 15.2 kWh normal day, 16.9 kWh season, at 9.80 | 1,69,110 | 2,26,000 |
| UPS/AC upkeep | 2,34,000 | 2,34,000 |
| AMC at 5% plus a spare NVMe | 2,93,500 | 4,27,000 |
| **Total** | **95,46,609** | **1,05,67,588** |
| Cash excluding the sunk workstation | 78,56,609 | 90,87,588 |

**Cost per 1,000 chats: INR 1,290 all-in, INR 94 marginal.** The gap between those two numbers is the whole argument: 75% of the cost is two people, so the machine decision is a small decision and the headcount decision is the big one.

**Cost per resume ranked: INR 1.30 on-premises against INR 0.13 in the cloud.** A tenfold premium. I would say this out loud rather than hide it: for Hirrx, on-premises is *not* justified by cost at any volume, and the whole justification is the founder's own rule that doctor data does not leave the building. If he ever relaxes that rule for anonymised ranking, Hirrx should move first and the savings are real.

**Break-even: 2.91 million chats a year.** We forecast 2.5-3.1 million, so the cloud and the GPU are within 20% of each other on rupees. The decision is the privacy constraint, and it is affordable - but it is not free, and he should hear that from me rather than discover it in year two.

**F2. Renting spare hours.** 5,658 spare hours a year, 4,526 sellable after a 20% maintenance and peak reserve, at INR 60 per GPU-hour (Indian retail for a comparable 48 GB card is INR 90-130) and 40% utilisation: **INR 1.09 lakh a year, about 1.1% of the three-year cost.** My recommendation is **no**. That is 1% of revenue in exchange for a network path into a machine that holds student and doctor data, a client who will ask what else is on the card, and an operations burden on a two-person team in its first season. Conditions if he overrules me: a physically separate second card, its own NVMe and its own network namespace, no shared model weights with the production service, no personal data ever on that card, a signed DPA with an audit right, and a hard kill switch.

**F3. Fourteen weeks, 1 December 2026 to 15 March 2027.**

| Weeks | Milestone |
|---|---|
| 1-2 | Corpus v1: 3 states, 2 sessions, 6 colleges. Ingestion, versioning and supersession working. Numeric guard + unit tests. |
| 3-4 | Hybrid retrieval, session and programme filters, roman-script lexicon. Golden-set schema frozen; counsellors start writing. |
| 5-6 | Model M and S served on the card; 2,200-token context cap; prefix caching measured on synthetic traffic. |
| 7-8 | Guard end-to-end, refusal policy, counsellor review UI with the citation beside every number. |
| 9-10 | 500-question golden set complete; release gate run; **fail-and-fix fortnight**, not feature work. |
| 11-12 | Hirrx: parse, rule gates, two-stage rank, explanation, fairness monitor, DPDP erasure runbook drilled. |
| 13 | Load test at season volume, result-day fast path, 24-hour soak, incident runbook. |
| 14 | Go/no-go, 10-counsellor pilot, launch on 15 March. |

Three risks, in order: **the prefix-cache hit rate is lower than 60%**, which turns a two-card plan into a four-card plan; **the counsellors cannot produce 500 questions in three weeks** while doing their day jobs, which is why they start in week 3 and not week 9; and **the fee gazettes are worse than the case data** - undated, scanned, amended in place - which is why session-year correctness is a hard release gate and not a nice-to-have.

Go/no-go on 12 March: golden set at or above every threshold in C3; zero wrong fee or bond figures in the 500-question human audit; p95 inside target at season load; erasure drill passed; and one named counsellor able to explain any answer to a parent from the citation alone. If any one fails, launch four of the six states, not all of them.

## G2: the one thing I think is wrong

The founder's brief says a wrong fee figure is worse than no answer, and then says families hate "I don't know". Those two sentences cannot both be the design principle, and the consultant quietly resolved the conflict in favour of the model - "good enough, the model will fill gaps" - which is how Incident 1 reached a parent. I would tell the founder this: **the metric to manage is not answer rate, it is the number of untraceable figures that reach a family, and that number should be zero.** Then give him the trade in his own currency: the guard will refuse roughly one question in ten that the old design would have guessed at, and at 22,000 chats a day that is 2,200 families a day told "here is the notice, here is the number to call" instead of being given a figure we cannot stand behind.

The second thing I would tell him is about the consultant's quality table. Model L scores 81%, which reads as "one in five answers wrong" - at result-day volume that is 12,000 wrong answers in a day. **The model is not the product. The guard, the citation and the session filter are the product**, and they are what I would spend the first four weeks on, before a single token is generated on the card.

## What I would do with two more weeks

Run the guard against a real 500-question counsellor set and publish the failure list; replace the LSA stand-in with a multilingual encoder and measure the roman-script gap properly; build the structured table parser that fixes Q18; and load-test the prefix-cache assumption on the actual card, because it is the one number in this note that can move the whole plan.
