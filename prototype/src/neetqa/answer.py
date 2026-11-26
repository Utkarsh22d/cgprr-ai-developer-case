"""Grounded answering: a stub extractive generator behind a citation + number guard.

No paid API is called and no model is required. The "generator" here copies a
table row or sentence out of the cited document. That is a *stub* for the LLM
that would run on the RTX 6000, and it is deliberately the weakest possible
generator, because it is the right thing to build first: if the retrieval, the
citation contract and the numeric guard are correct, swapping in Model M later
is a one-function change. If they are wrong, a better model only hides it.

Three safety gates sit between retrieval and the family:

1. session filter   -- a 2025-26 notice can never answer a 2026-27 question;
2. coverage refusal -- if the question's terms are not in the evidence, refuse;
3. numeric guard    -- every number must appear in the cited chunk.
   plus an exclusion guard: a programme name that the source only mentions in
   order to *exclude* it ("does not apply to BDS seats") is a refusal, not an
   answer. That single rule is what stops Incident 1's cousin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .guard import GuardResult, enforce
from .retrieval import HybridIndex, ScoredChunk, content_tokens
from .textnorm import detect_language, extract_amounts, tokenize

REFUSAL_TEXT = {
    "en": "I do not have an official notice that answers this. Please check with the "
    "counselling authority before acting.",
    "hi-rom": "Is sawaal ka jawab kisi official notice me nahi mila. Counselling "
    "authority se confirm kariye.",
    "bn-rom": "Ei prosner uttor kono official notice e nei. Counselling authority "
    "theke nishchit kore nin.",
    "hi": "इस प्रश्न का उत्तर किसी आधिकारिक सूचना में नहीं मिला।",
    "bn": "এই প্রশ্নের উত্তর কোনো সরকারি বিজ্ঞপ্তিতে নেই।",
}

MIN_COVERAGE = 0.5

# Head nouns: the word that says *which* figure is being asked for. A fee
# question must not be answered with the seat count just because both are rows
# with a number in them.
HEAD_NOUNS = {
    "tuition", "fee", "hostel", "mess", "deposit", "total", "payable",
    "seat", "seats", "bond", "period", "forfeit", "forfeiture", "last",
    "date", "minimum", "experience", "ceiling", "caution", "admission",
    "round", "waiver", "resign", "report",
}

# Questions that ask for a figure should be answered from a row that has a figure.
AMOUNT_INTENT = {
    "fee", "fees", "tuition", "amount", "cost", "charge", "charges", "hostel",
    "deposit", "total", "payable", "much", "kitni", "kitna", "kitne", "koto",
    "taka", "rupees", "salary", "date", "minimum", "years",
}

# Programme names are guarded against exclusion clauses.
PROGRAMME_TERMS = {"mbbs", "bds", "md", "ms", "pg", "dm", "mch", "bsc", "nursing"}

# Attributes that no document in this corpus can answer. Asking for them is a
# refusal, not a retrieval problem: no amount of better search will find a
# salary figure in a fee gazette.
UNSUPPORTED_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    (r"\bcut[\s-]?off\b", "cut-off"),
    (r"\bsalary\b|\bstipend\b", "salary"),
    (r"\bplacement\b|\bpackage\b", "placement record"),
    (r"\bdirector\b|\bprincipal\b|\bdean\b", "office-holder"),
    (r"\bpercentile\b|\ball india rank\b", "rank or percentile"),
)

COMPARISON = r"\b(highest|lowest|cheapest|most expensive|best|compare|versus|vs)\b"

NEGATION_CUES = (
    "does not", "do not", "not covered", "not covered by", "not apply",
    "not applicable", "must not", "is not",
)

# Interrogative scaffolding. These words carry no evidence, so a term that the
# corpus has never seen is only a refusal signal once they are removed.
SCAFFOLD = {
    "give", "gives", "given", "require", "requires", "required", "set", "sets",
    "under", "wise", "tell", "show", "says", "say", "mention", "mentions",
    "mean", "means", "provide", "provides", "available", "according",
    "information", "detail", "details", "document", "documents", "please",
    "kindly", "exactly",
}


@dataclass
class Answer:
    question: str
    text: str
    citations: list[str] = field(default_factory=list)
    guard: GuardResult | None = None
    refused: bool = False
    reason: str = ""
    hits: list[ScoredChunk] = field(default_factory=list)
    language: str = "en"

    def cite_line(self) -> str:
        return "Sources: " + ", ".join(self.citations) if self.citations else ""


class Answerer:
    def __init__(self, index: HybridIndex, top_k: int = 5) -> None:
        self.index = index
        self.top_k = top_k

    # -- rendering -------------------------------------------------------
    @staticmethod
    def _render(chunk_text: str, section: str) -> str:
        lines = [ln.strip() for ln in chunk_text.split("\n") if ln.strip()]
        if section == "table-row" and len(lines) >= 2:
            cells = [c.strip() for c in lines[-1].strip("|").split("|")]
            if len(cells) == 2:
                return f"{cells[0]} {cells[1]}"
            if cells:
                return "; ".join(cells)
        # prose: first sentence is enough and is never truncated mid-number
        from .guard import split_sentences

        parts = split_sentences(lines[0] if lines else chunk_text)
        return parts[0] if parts else chunk_text

    # -- selection -------------------------------------------------------
    def _scored_chunks(self, doc_id: str, question: str, wants_amount: bool):
        q = set(content_tokens(question))
        ranked: list[tuple[float, int, ScoredChunk]] = []
        for pos, chunk in enumerate(c for c in self.index.chunks if c.doc_id == doc_id):
            toks = set(content_tokens(chunk.text))
            overlap = float(len(q & toks))
            head = float(len((q & toks) & HEAD_NOUNS))
            has_amount = bool(extract_amounts(chunk.text))
            score = overlap + 1.5 * head
            if wants_amount:
                score += (2.0 if has_amount else -4.0)
                if has_amount and chunk.section == "table-row":
                    score += 1.0
            else:
                score += 0.3 if has_amount else 0.0
            if chunk.section == "table-row":
                score += 0.2
            ranked.append((score, pos, ScoredChunk(chunk, 0.0, 0.0, 0.0, score)))
        ranked.sort(key=lambda t: (-t[0], t[1]))
        return [r[2] for r in ranked[:2]]

    # -- guards ----------------------------------------------------------
    def _unknown_entities(self, question: str) -> list[str]:
        """Capitalised words the corpus has never seen: Assam, AIIMS, Rajasthan.

        Only entities count. An unknown lowercase word ("penalty", "bhai",
        "including") is vocabulary, not evidence of a gap -- refusing on those
        was the single biggest source of false refusals on the held-out set.
        """
        words = question.split()
        capitalised = {
            w.strip(".,?!;:()\"'").lower()
            for w in words[1:]
            if w[:1].isupper()
        }
        return [
            tok
            for tok in content_tokens(question)
            if tok in capitalised and self.index.bm25.df_of(tok) == 0
        ]

    def _corpus_states(self) -> set[str]:
        return {d.state for d in self.index.documents if d.state}

    def _attribute_gate(self, question: str, doc) -> str | None:
        """Refuse when no document of this kind can hold the answer."""
        import re

        q = question.lower()
        for pattern, label in UNSUPPORTED_ATTRIBUTES:
            if re.search(pattern, q):
                return f"no notice in the corpus states a {label}"

        states = [s for s in self._corpus_states() if s.lower() in q]
        if states and doc.state and doc.state not in states:
            return (
                f"the closest notice is for {doc.state}, but the question asks about "
                f"{', '.join(states)}"
            )
        if "bond" in set(tokenize(question)) and doc.doc_type != "bond_rule":
            return "no bond rule is published for this state or college"
        if re.search(COMPARISON, q) and not states:
            return "a comparison across colleges cannot be cited to a single notice"
        return None

    @staticmethod
    def _excluded_terms(question: str, evidence: str) -> list[str]:
        """Programme names that the evidence mentions only in order to exclude.

        Tight by design: the term must sit inside the negation clause itself
        ("does not apply to ... BDS seats"). A loose "sentence contains a
        negation" rule would refuse a correct PG fee answer because the same
        notice says "does not change the MBBS fee".
        """
        import re

        out: list[str] = []
        for term in set(tokenize(question)) & PROGRAMME_TERMS:
            occurrences = re.findall(
                rf"[^.]*\b{re.escape(term)}\b[^.]*\.", evidence, flags=re.IGNORECASE
            )
            if not occurrences:
                continue
            negated = all(
                re.search(
                    r"(does not|do not|must not|is not)\s+\w+(\s+\w+){0,3}[^.]{0,80}$",
                    re.split(rf"\b{re.escape(term)}\b", sentence, flags=re.IGNORECASE)[0],
                    flags=re.IGNORECASE,
                )
                for sentence in occurrences
            )
            if negated:
                out.append(term)
        return out

    def _refuse(self, question: str, lang: str, reason: str, hits=None) -> Answer:
        return Answer(
            question,
            REFUSAL_TEXT.get(lang, REFUSAL_TEXT["en"]),
            refused=True,
            reason=reason,
            hits=hits or [],
            language=lang,
        )

    # -- main ------------------------------------------------------------
    def answer(
        self,
        question: str,
        session: str | None = None,
        guard_mode: str = "block",
        top_k: int | None = None,
    ) -> Answer:
        lang = detect_language(question)
        hits = self.index.search(question, top_k=top_k or self.top_k, session=session)
        if not hits:
            return self._refuse(question, lang, "no document matches this session")

        coverage = hits[0].parts.get("coverage", 0.0)
        evidence_pool = "\n".join(h.chunk.text for h in hits[:3])

        if coverage < MIN_COVERAGE:
            return self._refuse(question, lang, f"coverage {coverage:.2f} < {MIN_COVERAGE}", hits)

        missing = self._unknown_entities(question)
        if missing:
            return self._refuse(question, lang, "no notice mentions: " + ", ".join(sorted(missing)), hits)

        doc_id = hits[0].chunk.doc_id
        doc = self.index.document(doc_id)
        blocked = self._attribute_gate(question, doc)
        if blocked:
            return self._refuse(question, lang, blocked, hits)
        # Exclusions are checked against the whole document we are about to
        # cite, not only the chunks that happened to be retrieved: the notice
        # that says "does not apply to BDS seats" may say it in a paragraph the
        # retriever did not pick.
        excluded = self._excluded_terms(question, doc.text)
        if excluded:
            return self._refuse(
                question,
                lang,
                "programme only appears as an exclusion: " + ", ".join(sorted(excluded)),
                hits,
            )
        wants_amount = bool(set(tokenize(question)) & AMOUNT_INTENT)
        picked = self._scored_chunks(doc_id, question, wants_amount)
        if not picked:
            return self._refuse(question, lang, "no usable chunk in the best document", hits)

        rendered = [self._render(p.chunk.text, p.chunk.section) for p in picked]
        body = " — ".join(rendered)
        text = f"{doc.title}: {body} [{doc_id}]."
        cited_text = "\n".join(p.chunk.text for p in picked)

        final, guard = enforce(text, cited_text, mode=guard_mode)
        refused = not guard.ok and guard_mode == "block"
        return Answer(
            question,
            final,
            citations=[doc_id],
            guard=guard,
            refused=refused,
            reason=guard.reason() if refused else "",
            hits=hits,
            language=lang,
        )
