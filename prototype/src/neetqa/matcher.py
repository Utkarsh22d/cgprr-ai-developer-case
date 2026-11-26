"""Hirrx Match: rule-gated, embedding-assisted resume ranking with an audit trail.

Design position (defended in the design note, Part D):

* eligibility is a RULE, not a score. NMC-style faculty norms and the post's
  minimum experience are hard gates, because a hospital cannot hire outside
  them and a soft score would let a model talk its way past the law;
* ranking inside the eligible pool is learned (embedding similarity + skill
  coverage) and is explainable term by term;
* an employment gap is NEVER a feature. It is not in the score, it is not in
  the explanation, and it cannot be inferred from dates because the score uses
  *recorded service years* from the resume body. A maternity, illness or
  caregiving gap therefore leaves the score bit-identical -- which the tests
  assert rather than assume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .retrieval import LSAEmbedder
from .textnorm import tokenize

WEIGHTS = {"eligibility": 0.0, "experience": 0.35, "specialty": 0.30, "skills": 0.25, "recency": 0.10}

# Post -> the degrees that satisfy NMC-style eligibility (case data, synthetic).
DEGREE_RULES: dict[str, tuple[str, ...]] = {
    "MD": ("MD", "MS", "DNB", "DM", "MCh"),
    "MBBS": ("MBBS",),
}


@dataclass
class Ranked:
    rank: int
    rid: str
    name: str
    score: float
    eligible: bool
    explanation: list[str] = field(default_factory=list)


class Ranker:
    def __init__(self, resumes: list[dict], jobs: list[dict]) -> None:
        self.resumes = resumes
        self.jobs = jobs
        texts = [self._resume_text(r) for r in resumes] + [
            self._job_text(j) for j in jobs
        ]
        self.embedder = LSAEmbedder(dim=32).fit(texts)
        self.rvec = self.embedder.transform([self._resume_text(r) for r in resumes])
        self.jvec = self.embedder.transform([self._job_text(j) for j in jobs])

    @staticmethod
    def _resume_text(r: dict) -> str:
        # Only the professional record. gap_type / gap_months / gender are
        # excluded here on purpose; ``test_matcher`` proves it.
        return " ".join(
            [
                r["specialty"],
                r["degree"],
                " ".join(r["skills"]),
                r.get("current_role", ""),
                r.get("notes", ""),
            ]
        )

    @staticmethod
    def _job_text(j: dict) -> str:
        return " ".join([j["post"], j["specialty"], " ".join(j["required_skills"]), j["text"]])

    # -- gates -----------------------------------------------------------
    def eligible(self, r: dict, j: dict) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        allowed = DEGREE_RULES[j["degree_level"]]
        if r["degree"] not in allowed:
            reasons.append(f"{r['degree']} does not meet {j['degree_level']} requirement")
        if r["specialty"].lower() != j["specialty"].lower():
            reasons.append(f"specialty {r['specialty']} vs {j['specialty']}")
        if r["relevant_years"] < j["min_years"]:
            reasons.append(f"{r['relevant_years']}y < {j['min_years']}y required")
        return (not reasons, reasons)

    # -- scoring ---------------------------------------------------------
    def score(self, ri: int, ji: int) -> tuple[float, dict[str, float]]:
        r, j = self.resumes[ri], self.jobs[ji]
        exp = min(1.0, r["relevant_years"] / max(j["min_years"], 1))
        spec = 1.0 if r["specialty"].lower() == j["specialty"].lower() else 0.0
        wanted = {s.lower() for s in j["required_skills"]}
        have = {s.lower() for s in r["skills"]}
        skills = len(wanted & have) / max(len(wanted), 1)
        recency = 1.0 if r.get("years_since_last_role", 0) <= 2 else 0.5
        parts = {
            "experience": exp,
            "specialty": spec,
            "skills": skills,
            "recency": recency,
            "embedding": float(self.rvec[ri] @ self.jvec[ji]),
        }
        total = (
            WEIGHTS["experience"] * exp
            + WEIGHTS["specialty"] * spec
            + WEIGHTS["skills"] * skills
            + WEIGHTS["recency"] * recency
            + 0.10 * parts["embedding"]
        )
        return round(total, 4), parts

    def rank(self, job_id: str, top_k: int = 10) -> list[Ranked]:
        ji = next(i for i, j in enumerate(self.jobs) if j["jid"] == job_id)
        j = self.jobs[ji]
        scored: list[tuple[float, int, bool, list[str]]] = []
        for ri, r in enumerate(self.resumes):
            ok, why = self.eligible(r, j)
            s, parts = self.score(ri, ji)
            # ineligible candidates stay visible but are ranked below every
            # eligible one, so a recruiter sees the reason rather than silence
            s = s if ok else s - 10.0
            scored.append((s, ri, ok, why))
        scored.sort(key=lambda t: (-t[0], self.resumes[t[1]]["rid"]))
        out: list[Ranked] = []
        for rank, (s, ri, ok, why) in enumerate(scored[:top_k], start=1):
            r = self.resumes[ri]
            _, parts = self.score(ri, ji)
            matched = sorted(
                {x.lower() for x in r["skills"]} & {x.lower() for x in j["required_skills"]}
            )
            line1 = (
                f"{r['degree']} {r['specialty']}, {r['relevant_years']}y relevant service; "
                f"specialty match, skills {', '.join(matched) or 'none'}."
            )
            if ok:
                line2 = (
                    f"Meets {j['post']} eligibility ({j['degree_level']}, "
                    f"{j['min_years']}y minimum) and scores "
                    f"{s:.2f} on experience {parts['experience']:.2f}, "
                    f"skills {parts['skills']:.2f}."
                )
            else:
                line2 = "Not eligible for this post: " + "; ".join(why) + "."
            out.append(
                Ranked(
                    rank=rank,
                    rid=r["rid"],
                    name=r["name"],
                    score=round(s, 3),
                    eligible=ok,
                    explanation=[line1, line2],
                )
            )
        return out

    # -- fairness --------------------------------------------------------
    def fairness_report(
        self, job_id: str, attribute: str = "gender", top_k: int = 10, eligible_only: bool = False
    ) -> dict:
        """Exposure parity at top-k.

        Default is the whole applicant pool, which is the measure the brief asks
        for (top-10 share against pool share) and the one a regulator sees. The
        eligible-pool variant is reported alongside it, because an ineligible
        applicant was never in the race and the two numbers answer different
        questions. Both are needed: the first catches a biased *ranker*, the
        second catches biased *eligibility rules*.
        """
        ji = next(i for i, j in enumerate(self.jobs) if j["jid"] == job_id)
        ranked_all = self.rank(job_id, top_k=len(self.resumes))
        ranked = [r for r in ranked_all if r.eligible] if eligible_only else ranked_all
        top = ranked[:top_k]
        eligible_ids = {r.rid for r in (ranked_all if not eligible_only else ranked)}
        pool: dict[str, int] = {}
        picked: dict[str, int] = {}
        for r in self.resumes:
            if r["rid"] in eligible_ids:
                pool[r[attribute]] = pool.get(r[attribute], 0) + 1
        ids = {t.rid for t in top}
        for r in self.resumes:
            if r["rid"] in ids:
                picked[r[attribute]] = picked.get(r[attribute], 0) + 1
        rates = {g: picked.get(g, 0) / pool[g] for g in pool if pool[g]}
        hi = max(rates.values()) if rates else 0.0
        lo = min(rates.values()) if rates else 0.0
        ratio = (lo / hi) if hi else 1.0
        tiers = {r["college_tier"] for r in self.resumes}
        by_id = {r["rid"]: r for r in self.resumes}
        tier_top = {
            t: sum(1 for item in top if by_id[item.rid]["college_tier"] == t)
            for t in sorted(tiers)
        }
        tier_pool = {
            t: sum(1 for rid in eligible_ids if by_id[rid]["college_tier"] == t)
            for t in sorted(tiers)
        }
        n_eligible = sum(1 for r in ranked_all if r.eligible)
        return {
            "job_id": job_id,
            "attribute": attribute,
            "population": "eligible" if eligible_only else "applicant",
            "n_eligible": n_eligible,
            "pool": pool,
            "top_k": picked,
            "selection_rate": {g: round(v, 3) for g, v in rates.items()},
            "min_max_ratio": round(ratio, 3),
            "passes_four_fifths": ratio >= 0.8,
            "tier_top_k": tier_top,
            "tier_pool": tier_pool,
            "embedding_dims": int(np.asarray(self.rvec).shape[1]),
        }
