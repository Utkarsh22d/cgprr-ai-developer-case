"""Export the prototype's real outputs to JSON for the demo frontend.

The site is not a mock: every answer, ranking, fairness number and capacity
figure below is produced by the same code that run_all.py and build_calcs.py run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "prototype" / "src"))

import build_calcs  # noqa: E402
from neetqa.answer import Answerer  # noqa: E402
from neetqa.corpus import load_documents, load_jobs, load_heldout, load_questions, load_resumes  # noqa: E402
from neetqa.matcher import Ranker  # noqa: E402
from neetqa.retrieval import HybridIndex  # noqa: E402
from neetqa.textnorm import detect_language  # noqa: E402

OUT = HERE / "site" / "public" / "data" / "demo.json"


def main() -> None:
    docs = load_documents()
    index = HybridIndex(docs)
    answerer = Answerer(index)

    questions = []
    for tag, qs in (("dev", load_questions()), ("stress", load_heldout())):
        for q in qs:
            ans = answerer.answer(q["question"], session=q.get("session"))
            questions.append(
                {
                    "qid": q["qid"],
                    "set": tag,
                    "question": q["question"],
                    "lang": q.get("lang") or detect_language(q["question"]),
                    "session": q.get("session"),
                    "answerable": q["answerable"],
                    "gold": q["gold_docs"],
                    "answer": ans.text,
                    "citations": ans.citations,
                    "refused": ans.refused,
                    "reason": ans.reason,
                    "guardOk": bool(ans.guard and ans.guard.ok),
                }
            )

    ranker = Ranker(load_resumes(), load_jobs())
    jobs = []
    for job in load_jobs():
        ranked = ranker.rank(job["jid"], top_k=10)
        fair = ranker.fairness_report(job["jid"], eligible_only=False)
        fair_elig = ranker.fairness_report(job["jid"], eligible_only=True)
        jobs.append(
            {
                "jid": job["jid"],
                "post": job["post"],
                "hospital": job["hospital"],
                "min_years": job["min_years"],
                "degree_level": job["degree_level"],
                "ranked": [
                    {
                        "rank": r.rank,
                        "name": r.name,
                        "score": r.score,
                        "eligible": r.eligible,
                        "explanation": r.explanation,
                    }
                    for r in ranked
                ],
                "fairness": {
                    "applicant": {
                        "pool": fair["pool"],
                        "top": fair["top_k"],
                        "rates": fair["selection_rate"],
                        "ratio": fair["min_max_ratio"],
                        "passes": fair["passes_four_fifths"],
                    },
                    "eligible": {
                        "pool": fair_elig["pool"],
                        "top": fair_elig["top_k"],
                        "rates": fair_elig["selection_rate"],
                        "ratio": fair_elig["min_max_ratio"],
                        "passes": fair_elig["passes_four_fifths"],
                    },
                    "tierTop": fair["tier_top_k"],
                    "tierPool": fair["tier_pool"],
                },
            }
        )

    cap = build_calcs.capacity()
    thr = build_calcs.throughput()
    traf = build_calcs.traffic()
    tco = build_calcs.tco()
    batch = build_calcs.batch_plan()
    rent = build_calcs.rental()

    corpus = [
        {
            "doc_id": d.doc_id,
            "title": d.title,
            "session": d.session,
            "status": d.status,
            "doc_type": d.doc_type,
            "text": d.text,
        }
        for d in docs
    ]

    payload = {
        "meta": {
            "documents": len(docs),
            "chunks": len(index.chunks),
            "superseded": sum(1 for d in docs if d.status == "superseded"),
            "questions": len(questions),
            "resumes": len(load_resumes()),
            "jobs": len(load_jobs()),
        },
        "questions": questions,
        "jobs": jobs,
        "capacity": {
            "kv": cap,
            "throughput": thr,
            "traffic": traf,
            "tco": tco,
            "batch": batch,
            "rental": rent,
        },
        "corpus": corpus,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"questions={len(questions)} jobs={len(jobs)} docs={len(docs)}")


if __name__ == "__main__":
    main()
