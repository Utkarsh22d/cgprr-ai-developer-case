# Part E: grounded counselling Q&A and doctor-resume matching

A runnable prototype for the CGPRR / NEET.info case. CPU only, numpy only, no
network, no paid API, no model download.

## Run it (one command)

```bash
pip install -r requirements.txt
python run_all.py
```

That prints: the grounded-Q&A demo with citations, the Incident 1 replay, the
evaluation tables, and the resume matcher with its fairness report.

```bash
python -m pytest -q tests     # 27 unit tests, <1 second
python tools/build_dataset.py # regenerate data/*.json from scratch
```

## What is real, what is stubbed

**Real.** Hybrid retrieval (BM25 + a dense vector space, reciprocal-rank fusion,
a cross-encoder stand-in), the session-year and programme filters, table-aware
chunking, the numeric guard and its unit tests, the exclusion and attribute
gates, the evaluation harness and metrics, the resume rule gates and ranking,
the fairness report, and the maternity-gap invariance test.

**Stubbed, deliberately.** (1) The generator is extractive: it copies a table
row or sentence out of the cited chunk instead of calling an LLM. This is the
weakest generator that can work, which makes it the right thing to build first
- if retrieval, citation and the guard are right, dropping Model M in is a
one-function change; if they are wrong, a better model only hides it. (2) The
dense encoder is latent-semantic indexing (truncated SVD over tf-idf) built
from the corpus, standing in for a multilingual sentence-transformer;
`EMBEDDER=st` switches to a real one if `sentence-transformers` is installed.
(3) There is no GPU, so no latency number here is a measurement. (4) The
cross-encoder re-ranker is a lexical stand-in, not a trained model.

## The evaluation numbers, honestly

`run_all.py` prints two tables. The first is 40 questions I wrote and tuned
retrieval against. The second is 12 questions written afterwards and not used
to set any threshold. **Both are now 1.000**, which means neither is evidence -
I fixed every failure the second set found, so I have tuned against it too.

The number worth remembering is what the stress set showed before one rule was
replaced: **1 in 3 answerable questions refused**, because the "a word the
corpus has never seen means we cannot answer" rule fired on *penalty*, *bhai*,
*including*. Replacing it with an entity check plus a document-type attribute
gate took false refusals to zero on 52 questions - and 52 questions cannot
distinguish 0.00 from 0.10. Part C3 of the design note spends a fortnight
building the only thing that can: 500 questions from two counsellors, frozen
before anyone tunes on them.

## What the fairness report actually found

The report compares top-10 selection rates for women and men against the
applicant pool, and separately against the *eligible* pool, because the two
answer different questions (a biased ranker versus biased eligibility rules).
On the Assistant Professor post the applicant-pool ratio is **0.64** - women are
25% of the top 10 against 40% of the applicants - which fails the four-fifths
rule even though college name is not a feature and the employment gap is
structurally excluded from scoring. That is the honest result: removing the
obvious proxy does not by itself produce parity, which is why the design note
asks for a per-JD parity monitor that blocks a shortlist below 0.80.

## A failure I could not fix

`Q18`: *"What is the minimum teaching experience for an Assistant Professor
under NMC norms?"* The right document is retrieved, the citation is correct,
and every number passes the guard - and the answer quotes the **Associate
Professor** row ("5 years, of which 2 years as Assistant Professor"), because
both rows contain the phrase "Assistant Professor". This is the important
lesson of the whole prototype: after retrieval and the numeric guard, the
residual error is *row selection*, not model hallucination, and it is invisible
to a numeric guard because the number it quotes is real. The fix is a
structured table parse with a key column, not a larger model.

## Layout

```
run_all.py            single entry point
src/neetqa/
  textnorm.py         tokens, roman-Hindi/Bengali hints, Indian money parsing
  guard.py            numeric guard (block / rewrite / report)
  corpus.py           documents, table-aware chunking, loaders
  retrieval.py        BM25 + LSA dense + RRF + rerank + metadata filters
  answer.py           grounding, citations, refusal, exclusion and attribute gates
  evaluate.py         recall@5, MRR, citation precision, faithfulness, refusals
  matcher.py          rule gates, ranking, explanations, fairness report
tools/build_dataset.py  28 documents, 40 + 12 questions, 30 resumes, 5 JDs
tests/                27 tests (guard, retrieval, matcher)
data/                  the synthetic dataset, committed
```

## Two more weeks

Replace the LSA stand-in with a multilingual encoder and measure the
roman-script recall gap on a stratified set; build the structured table parser
that fixes Q18; add a second citation only when the question genuinely spans
two documents; and load-test the prefix-cache assumption on the real card,
since that one number moves the whole capacity plan.
