"""Unit tests for the numeric guard -- the piece that must never be wrong."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neetqa.guard import check_answer, enforce  # noqa: E402
from neetqa.textnorm import amount_set, extract_amounts  # noqa: E402

HARYANA = "| Bond amount | INR 10,00,000 |\n| Service period | 2 years in a designated rural or district hospital in Haryana |"
UP = "| Bond amount | INR 5,00,000 |\n| Service period | 5 years |"


def test_lakh_parses_to_rupees():
    (a,) = extract_amounts("The bond is INR 5 lakh.")
    assert a.value == 500000
    assert a.kind == "money"


def test_no_space_money_equals_word_form():
    assert amount_set("INR 5,00,000") == amount_set("INR 5 lakh")


def test_crore_and_percent():
    assert extract_amounts("INR 1.5 crore")[0].value == 15_000_000
    assert extract_amounts("10 percent")[0].kind == "percent"
    assert extract_amounts("10%")[0].key == ("percent", 10.0)


def test_doc_ids_are_not_amounts():
    # 2627 must never be read as a rupee figure
    assert amount_set("DOC-SMC-PAT-FEE-2627") == set()


def test_session_range_is_not_an_amount():
    assert amount_set("Session 2026-27. Issued in 2026.") == {("year", 2026.0)}


def test_answer_numbers_must_come_from_cited_text():
    ok = check_answer("The bond for Haryana is INR 10 lakh. [DOC-BOND-HR-2627]", HARYANA)
    assert ok.ok, ok.reason()


def test_incident_one_is_caught():
    """A figure that is real but belongs to another state must be rejected."""
    bad = "The bond amount for Haryana is INR 5 lakh. [DOC-BOND-HR-2627]"
    result = check_answer(bad, HARYANA)
    assert not result.ok
    assert result.unsupported[0].value == 500000


def test_block_mode_raises_a_refusal():
    bad = "The bond amount for Haryana is INR 5 lakh. [DOC-BOND-HR-2627]"
    text, result = enforce(bad, HARYANA, mode="block")
    assert not result.ok
    assert "cannot give that figure" in text
    assert "5 lakh" not in text


def test_rewrite_mode_drops_only_the_poisoned_sentence():
    text, result = enforce(
        "The bond is INR 5 lakh. The service period is 2 years.", "| Service period | 2 years |", mode="rewrite"
    )
    assert not result.ok
    assert "5 lakh" not in text
    assert "2 years" in text


def test_no_numbers_is_trivially_faithful():
    assert check_answer("The notice does not state a bond for this state.", "").ok


def test_guard_ignores_hallucinated_decimal_precision():
    result = check_answer("The fee is INR 10,40,000.50 per year.", "| Annual tuition fee | INR 10,40,000 |")
    assert not result.ok
