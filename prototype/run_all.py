#!/usr/bin/env python3
"""Single entry point: `python run_all.py` runs everything the case asks for.

E1  grounded Q&A demo with citations, session filter and the numeric guard
E1b Incident 1 replay: a wrong figure injected into the answer is blocked
E2  evaluation table (recall@5, MRR, citation precision, numeric faithfulness,
    refusal accuracy) plus one honest, unfixed failure
E3  resume ranking with two-line explanations and a fairness report
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from neetqa.answer import Answerer  # noqa: E402
from neetqa.corpus import (  # noqa: E402
    load_documents,
    load_heldout,
    load_jobs,
    load_questions,
    load_resumes,
)
from neetqa.evaluate import format_table, run_evaluation  # noqa: E402
from neetqa.guard import check_answer, enforce  # noqa: E402
from neetqa.matcher import Ranker  # noqa: E402
from neetqa.retrieval import HybridIndex  # noqa: E402

RULE = "=" * 78


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def part_e1(index: HybridIndex) -> None:
    banner("E1  Grounded Q&A  (cite, refuse, and block any untraceable number)")
    answerer = Answerer(index)
    demo = [
        ("What is the annual MBBS tuition fee at Shanti Medical College, Patna for 2026-27?", "2026-27"),
        ("patna me mbbs ki fees kitni hai 2026-27 me", "2026-27"),
        ("kolkata medical college er mbbs fees koto 2026-27", "2026-27"),
        ("What is the MBBS bond amount in Haryana for 2026-27?", None),
        ("What is the MBBS tuition fee at a medical college in Assam?", None),
        ("What is the tuition fee for session 2029-30 at Shanti Medical College, Patna?", "2029-30"),
    ]
    for question, session in demo:
        ans = answerer.answer(question, session=session)
        print(f"\nQ: {question}")
        if session:
            print(f"   filter: session={session}")
        print(f"   A: {ans.text}")
        if ans.citations:
            print(f"   {ans.cite_line()}")
        if ans.refused:
            print(f"   refused: {ans.reason}")
        else:
            print(f"   guard: {'PASS' if ans.guard and ans.guard.ok else 'BLOCKED'}")


def part_e1b(index: HybridIndex) -> None:
    banner("E1b  Incident 1 replay -- Haryana bond, wrong figure from the UP notice")
    hits = index.search("What is the MBBS bond amount in Haryana for 2026-27?", top_k=3)
    cited = {h.chunk.doc_id for h in hits}
    evidence = "\n".join(h.chunk.text for h in hits if h.chunk.doc_id in cited)
    # This is exactly what the pilot's generator did: it answered "INR 5 lakh",
    # a figure that exists in the corpus -- but in Uttar Pradesh's notice.
    bad = "The bond amount for Haryana is INR 5 lakh. [DOC-BOND-HR-2627]"
    raw = check_answer(bad, evidence)
    blocked, _ = enforce(bad, evidence, mode="block")
    rewritten, _ = enforce(bad, evidence, mode="rewrite")
    print(f"retrieved: {sorted(cited)}")
    print(f"generator said: {bad}")
    print(f"guard verdict: {'PASS' if raw.ok else 'REJECT'} -- {raw.reason()}")
    print(f"block mode  -> {blocked}")
    print(f"rewrite mode-> {rewritten}")


def _report(name: str, index: HybridIndex, questions: list[dict]) -> None:
    metrics, rows = run_evaluation(index, questions)
    print(format_table(metrics))
    by_lang: dict[str, list[float]] = {}
    for r in rows:
        if r["recall5"] is not None:
            by_lang.setdefault(r["lang"], []).append(r["recall5"])
    print("recall@5 by language:")
    for lang, vals in sorted(by_lang.items()):
        print(f"  {lang:8s} n={len(vals):2d}  recall@5={sum(vals)/len(vals):.3f}")

    for r in rows:
        if r["answerable"] and r["recall5"] is not None and r["recall5"] < 1.0:
            print(f"  MISS {r['qid']}: gold={r['gold']} cited={r['citations']}")
        if r["answerable"] and r["refused"]:
            print(f"  FALSE-REFUSAL {r['qid']}: {r['question']}\n      -> {r['reason']}")
        if not r["answerable"] and not r["refused"]:
            print(f"  LEAK {r['qid']}: {r['question']}\n      -> {r['answered']}")
    return rows  # type: ignore[return-value]


def part_e2(index: HybridIndex, questions: list[dict], heldout: list[dict]) -> None:
    banner("E2  Evaluation")
    print("\n-- dev set: 40 questions, 10 roman-script, 8 unanswerable --")
    print("(retrieval weights and two intent constants were tuned on this set; treat")
    print(" the numbers below as an upper bound, not as performance.)")
    _report("dev", index, questions)

    print("\n-- held-out set: 12 questions written AFTER tuning, never used --")
    print("-- to set a threshold. These are the numbers we would defend.      --")
    _report("heldout", index, heldout)


def part_e3(resumes: list[dict], jobs: list[dict]) -> None:
    banner("E3  Hirrx Match -- rule gates, explanations, fairness report")
    ranker = Ranker(resumes, jobs)
    for job in jobs[:2]:
        print(f"\n{job['post']} -- {job['hospital']}  ({job['degree_level']}, {job['min_years']}+ years)")
        for item in ranker.rank(job["jid"], top_k=5):
            tag = "" if item.eligible else "  [INELIGIBLE]"
            print(f"  {item.rank:2d}. {item.name:22s} {item.score:6.2f}{tag}")
            for line in item.explanation:
                print(f"      {line}")

    print("\nFairness report: top-10 selection rate by gender, 4/5ths rule (0.80)")
    for job in jobs:
        for eligible_only in (False, True):
            rep = ranker.fairness_report(job["jid"], attribute="gender", eligible_only=eligible_only)
            label = "eligible" if eligible_only else "applicant"
            print(
                f"  {job['jid']} [{label:9s}]: pool={rep['pool']} top10={rep['top_k']} "
                f"rates={rep['selection_rate']} ratio={rep['min_max_ratio']:.2f} "
                f"{'PASS' if rep['passes_four_fifths'] else 'FAIL'}  (eligible candidates: {rep['n_eligible']})"
            )
    rep = ranker.fairness_report(jobs[1]["jid"], attribute="gender")
    print(f"  college tiers in top10 {rep['tier_top_k']} vs pool {rep['tier_pool']}  (J2, applicant pool)")

    print("\nMaternity-gap invariance (the Incident 2 defect):")
    jd = jobs[1]["jid"]
    top = next(r for r in ranker.rank(jd, top_k=len(resumes)) if r.eligible)
    base = next(r for r in resumes if r["rid"] == top.rid)
    scores = {}
    for label in ("none", "maternity", "illness", "caregiving"):
        probe = dict(base, gap_type=label, gap_months=0 if label == "none" else 12)
        r = Ranker([probe] + resumes[1:], jobs).rank(jd, top_k=30)
        scores[label] = next(i.score for i in r if i.rid == probe["rid"])
    print(f"  score with gap_type=none       : {scores['none']}")
    print(f"  score with gap_type=maternity  : {scores['maternity']}")
    print(f"  score with gap_type=illness    : {scores['illness']}")
    print(f"  score with gap_type=caregiving : {scores['caregiving']}")
    print(f"  identical: {len(set(scores.values())) == 1}")


def main() -> None:
    documents = load_documents()
    index = HybridIndex(documents)
    print(f"indexed {len(documents)} documents into {len(index.chunks)} chunks "
          f"({sum(1 for d in documents if d.status == 'superseded')} superseded, filtered out by default)")
    part_e1(index)
    part_e1b(index)
    part_e2(index, load_questions(), load_heldout())
    part_e3(load_resumes(), load_jobs())


if __name__ == "__main__":
    main()
