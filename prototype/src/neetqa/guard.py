"""The numeric guard.

Rule enforced: every number that reaches a family must appear in a chunk that
the answer cites. Anything else is blocked, or (in ``rewrite`` mode) the
offending sentences are dropped.

This is the cheap, auditable half of the answer path. It is deliberately not a
model: a model cannot be audited line by line by a counsellor, and Incident 1
in the case pack is exactly a number that a model produced from the wrong
document in the retrieved set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .textnorm import Amount, amount_set, extract_amounts

_SENTENCE_SPLIT = __import__("re").compile(r"(?<=[.!?])\s+(?=[A-Z0-9₹])")


@dataclass
class GuardResult:
    """Outcome of checking one answer against its cited evidence."""

    ok: bool
    unsupported: list[Amount] = field(default_factory=list)
    checked: int = 0

    def reason(self) -> str:
        if self.ok:
            return f"all {self.checked} numbers traceable to cited chunks"
        bad = ", ".join(sorted({a.raw for a in self.unsupported}))
        return f"{len(self.unsupported)} number(s) not in cited sources: {bad}"


def check_answer(answer: str, evidence: str) -> GuardResult:
    """Check every number in ``answer`` against the concatenated ``evidence``."""
    supported = amount_set(evidence)
    amounts = extract_amounts(answer)
    unsupported: list[Amount] = []
    for amount in amounts:
        # A year range ("2026-27") is stripped by the extractor, so bare years
        # are compared as counts, which is what we want here.
        if amount.key not in supported:
            unsupported.append(amount)
    return GuardResult(ok=not unsupported, unsupported=unsupported, checked=len(amounts))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def enforce(
    answer: str,
    evidence: str,
    mode: str = "block",
    refusal: str = "I cannot give that figure from the official notices I hold.",
) -> tuple[str, GuardResult]:
    """Apply the guard.

    mode="block"   -> raise the whole answer, return a refusal + citations.
    mode="rewrite" -> drop only the sentences carrying unsupported numbers.
    mode="report"  -> leave the answer alone (used by the evaluation harness to
                      measure how often the generator would have been wrong).
    """
    result = check_answer(answer, evidence)
    if result.ok or mode == "report":
        return answer, result

    if mode == "block":
        return f"{refusal} See the cited notice.", result

    kept: list[str] = []
    for sentence in split_sentences(answer):
        sub = check_answer(sentence, evidence)
        if sub.ok:
            kept.append(sentence)
    if not kept:
        return f"{refusal} See the cited notice.", result
    return " ".join(kept), result
