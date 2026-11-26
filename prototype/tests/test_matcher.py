"""Ranking tests, including the one the DPDP Act forces us to prove."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neetqa.corpus import load_jobs, load_resumes  # noqa: E402
from neetqa.matcher import Ranker  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return load_resumes(), load_jobs()


def _score_for(ranker: Ranker, resumes: list[dict], rid: str, jid: str) -> float:
    return next(r.score for r in ranker.rank(jid, top_k=len(resumes)) if r.rid == rid)


def test_thirty_resumes_five_jobs(data):
    resumes, jobs = data
    assert len(resumes) == 30
    assert len(jobs) == 5


@pytest.mark.parametrize("label", ["maternity", "illness", "caregiving"])
def test_gap_label_never_changes_the_score(data, label):
    """Incident 2: a maternity gap must not move a candidate's rank."""
    resumes, jobs = data
    base = dict(resumes[0], gap_type="none", gap_months=0)
    probe = dict(resumes[0], gap_type=label, gap_months=12)
    a = Ranker([base] + resumes[1:], jobs).rank(jobs[1]["jid"], top_k=30)
    b = Ranker([probe] + resumes[1:], jobs).rank(jobs[1]["jid"], top_k=30)
    sa = next(x.score for x in a if x.rid == base["rid"])
    sb = next(x.score for x in b if x.rid == probe["rid"])
    assert sa == sb
    ranks_a = [x.rid for x in a]
    ranks_b = [x.rid for x in b]
    assert ranks_a.index(base["rid"]) == ranks_b.index(probe["rid"])


def test_ineligible_candidates_are_ranked_below_eligible_ones(data):
    resumes, jobs = data
    ranker = Ranker(resumes, jobs)
    ranked = ranker.rank(jobs[0]["jid"], top_k=30)
    last_eligible = max((i for i, r in enumerate(ranked) if r.eligible), default=-1)
    first_ineligible = min((i for i, r in enumerate(ranked) if not r.eligible), default=len(ranked))
    assert last_eligible < first_ineligible


def test_explanations_are_two_lines(data):
    resumes, jobs = data
    ranker = Ranker(resumes, jobs)
    for item in ranker.rank(jobs[2]["jid"], top_k=5):
        assert len(item.explanation) == 2
        assert all(line.strip() for line in item.explanation)


def test_fairness_report_shape(data):
    resumes, jobs = data
    rep = Ranker(resumes, jobs).fairness_report(jobs[1]["jid"])
    assert rep["attribute"] == "gender"
    assert set(rep["pool"]) <= {"woman", "man"}
    assert 0.0 <= rep["min_max_ratio"] <= 1.0
    assert isinstance(rep["passes_four_fifths"], bool)
