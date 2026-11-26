"""Evaluation harness for the grounded Q&A path."""

from __future__ import annotations

from dataclasses import dataclass

from .answer import Answerer
from .corpus import load_documents, load_questions
from .retrieval import HybridIndex


@dataclass
class Metrics:
    recall_at_5: float = 0.0
    mrr: float = 0.0
    citation_precision: float = 0.0
    numeric_faithfulness: float = 0.0
    refusal_accuracy: float = 0.0
    false_refusal_rate: float = 0.0
    injection_block_rate: float = 0.0
    answered: int = 0
    total: int = 0


def _ranked_docs(hits) -> list[str]:
    out: list[str] = []
    for h in hits:
        if h.chunk.doc_id not in out:
            out.append(h.chunk.doc_id)
    return out


def injection_check(index: HybridIndex, questions: list[dict]) -> tuple[int, int]:
    """Inject a plausible wrong figure into a correct answer; must be blocked.

    This is the honest way to measure numeric faithfulness for an extractive
    stub: by construction the stub cannot hallucinate, so we measure the guard
    on a simulated hallucination instead of scoring a tautology.
    """
    from .guard import check_answer

    answerer = Answerer(index)
    tried = caught = 0
    for q in questions:
        if not q["answerable"]:
            continue
        hits = index.search(q["question"], top_k=3, session=q.get("session"))
        if not hits:
            continue
        evidence = "\n".join(h.chunk.text for h in hits)
        poisoned = f"According to the notice, the figure is INR 7,77,777 [{hits[0].chunk.doc_id}]."
        tried += 1
        if not check_answer(poisoned, evidence).ok:
            caught += 1
    del answerer
    return caught, tried


def run_evaluation(index: HybridIndex, questions: list[dict]) -> tuple[Metrics, list[dict]]:
    answerer = Answerer(index)
    m = Metrics(total=len(questions))
    recalls: list[float] = []
    rrs: list[float] = []
    cps: list[float] = []
    faith: list[float] = []
    unanswerable = [q for q in questions if not q["answerable"]]
    answerable = [q for q in questions if q["answerable"]]
    refused_unanswerable = 0
    false_refusals = 0
    rows: list[dict] = []

    for q in questions:
        gold: list[str] = q["gold_docs"]
        # Retrieve with the question's session so the filter is exercised.
        hits = index.search(q["question"], top_k=5, session=q.get("session"))
        ranked = _ranked_docs(hits)

        if gold:
            hit = len(set(gold) & set(ranked))
            recalls.append(hit / len(gold))
            rr = 0.0
            for rank, doc in enumerate(ranked, start=1):
                if doc in gold:
                    rr = 1.0 / rank
                    break
            rrs.append(rr)

        ans = answerer.answer(q["question"], session=q.get("session"), guard_mode="report")
        answered = not ans.refused
        if answered:
            m.answered += 1
            if gold:
                cited = set(ans.citations)
                cps.append(len(cited & set(gold)) / max(len(cited), 1))
            faith.append(1.0 if ans.guard is not None and ans.guard.ok else 0.0)
        elif q["answerable"]:
            false_refusals += 1

        if not q["answerable"] and ans.refused:
            refused_unanswerable += 1

        rows.append(
            {
                "qid": q["qid"],
                "lang": q["lang"],
                "question": q["question"],
                "answerable": q["answerable"],
                "refused": ans.refused,
                "reason": ans.reason,
                "citations": ans.citations,
                "gold": gold,
                "recall5": (len(set(gold) & set(ranked)) / len(gold)) if gold else None,
                "answered": ans.text,
            }
        )

    m.recall_at_5 = sum(recalls) / len(recalls) if recalls else 0.0
    m.mrr = sum(rrs) / len(rrs) if rrs else 0.0
    m.citation_precision = sum(cps) / len(cps) if cps else 0.0
    m.numeric_faithfulness = sum(faith) / len(faith) if faith else 0.0
    m.refusal_accuracy = (
        refused_unanswerable / len(unanswerable) if unanswerable else 0.0
    )
    m.false_refusal_rate = false_refusals / len(answerable) if answerable else 0.0
    caught, tried = injection_check(index, questions)
    m.injection_block_rate = caught / tried if tried else 0.0
    return m, rows


def format_table(m: Metrics) -> str:
    rows = [
        ("recall@5", f"{m.recall_at_5:.3f}"),
        ("MRR", f"{m.mrr:.3f}"),
        ("citation precision", f"{m.citation_precision:.3f}"),
        ("numeric faithfulness (unguarded)", f"{m.numeric_faithfulness:.3f}"),
        ("refusal accuracy (unanswerable)", f"{m.refusal_accuracy:.3f}"),
        ("false-refusal rate (answerable)", f"{m.false_refusal_rate:.3f}"),
        ("injected wrong figure blocked", f"{m.injection_block_rate:.3f}"),
        ("answered / total", f"{m.answered}/{m.total}"),
    ]
    width = max(len(k) for k, _ in rows) + 2
    lines = ["| metric | value |", "|" + "-" * (width + 2) + "|-------|"]
    for k, v in rows:
        lines.append(f"| {k.ljust(width)} | {v} |")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - convenience entry point
    index = HybridIndex(load_documents())
    questions = load_questions()
    m, _ = run_evaluation(index, questions)
    print(format_table(m))


if __name__ == "__main__":  # pragma: no cover
    main()
