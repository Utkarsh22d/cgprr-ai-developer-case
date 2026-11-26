"""Retrieval tests: chunking, session filtering, superseded retirement."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neetqa.corpus import chunk_document, load_documents  # noqa: E402
from neetqa.retrieval import HybridIndex  # noqa: E402


@pytest.fixture(scope="module")
def index() -> HybridIndex:
    return HybridIndex(load_documents())


def test_dataset_is_at_least_25_documents(index):
    assert len(index.documents) >= 25


def test_table_rows_keep_their_header(index):
    chunk = next(c for c in index.chunks if "Annual tuition fee" in c.text and c.section == "table-row")
    assert "| Head |" in chunk.text or "| Item |" in chunk.text or "| Bond amount |" in chunk.text


def test_superseded_notice_is_not_retrieved_for_current_session(index):
    hits = index.search("Shanti Medical College Patna MBBS tuition fee", top_k=5, session="2026-27")
    assert hits
    assert all(h.chunk.status == "active" for h in hits)
    assert hits[0].chunk.doc_id == "DOC-SMC-PAT-FEE-2627"


def test_past_session_is_retrievable_when_asked(index):
    hits = index.search("Shanti Medical College Patna MBBS tuition fee", top_k=5, session="2025-26")
    assert hits[0].chunk.doc_id == "DOC-SMC-PAT-FEE-2526"


def test_future_session_has_no_documents(index):
    assert index.search("Shanti Medical College tuition fee", session="2029-30") == []


def test_roman_hindi_query_finds_the_bihar_notice(index):
    hits = index.search("patna me mbbs ki fees kitni hai 2026-27 me", session="2026-27")
    assert hits[0].chunk.doc_id == "DOC-SMC-PAT-FEE-2627"


def test_roman_bengali_query_finds_the_kolkata_notice(index):
    hits = index.search("kolkata medical college er mbbs fees koto 2026-27", session="2026-27")
    assert hits[0].chunk.doc_id in {"DOC-RMCH-KOL-FEE-2627", "DOC-WB-FEE-ORDER-2627"}


def test_bond_question_does_not_return_the_fee_notice_first(index):
    hits = index.search("What is the MBBS bond amount in Haryana for 2026-27?", top_k=5)
    assert hits[0].chunk.doc_id == "DOC-BOND-HR-2627"


def test_chunker_splits_each_table_row():
    docs = load_documents()
    doc = next(d for d in docs if d.doc_id == "DOC-BOND-BR-2627")
    chunks = chunk_document(doc)
    rows = [c for c in chunks if c.section == "table-row"]
    assert len(rows) == 4  # amount, service period, invocation, waiver
    assert all(c.session == "2026-27" for c in chunks)
