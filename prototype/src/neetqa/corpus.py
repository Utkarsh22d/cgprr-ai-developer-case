"""Document model, chunking (including table-aware chunking) and loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    session: str | None
    state: str | None
    college: str | None
    doc_type: str
    status: str  # "active" | "superseded"
    effective_date: str
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    session: str | None
    state: str | None
    college: str | None
    doc_type: str
    status: str
    section: str


def chunk_document(doc: Document) -> list[Chunk]:
    """Split a document into retrieval units.

    Two rules matter for the audit trail:

    * a table is never split across chunks -- each *row* becomes a chunk with
      the header row glued on, so "Annual tuition fee | INR 10,40,000" can never
      be retrieved without the column it belongs to;
    * every chunk inherits the document's session year, state and college, so a
      metadata filter can retire a whole session without re-indexing.
    """
    chunks: list[Chunk] = []
    blocks = [b for b in doc.text.strip().split("\n\n") if b.strip()]

    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        table_lines = [ln for ln in lines if ln.lstrip().startswith("|")]
        prose_lines = [ln for ln in lines if not ln.lstrip().startswith("|")]

        if table_lines:
            header = table_lines[0]
            for row in table_lines[1:]:
                body = f"{header}\n{row}"
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}#c{len(chunks)}",
                        doc_id=doc.doc_id,
                        text=body,
                        session=doc.session,
                        state=doc.state,
                        college=doc.college,
                        doc_type=doc.doc_type,
                        status=doc.status,
                        section="table-row",
                    )
                )
        if prose_lines:
            # Prose is chunked in two-sentence windows. A paragraph that mixes
            # two topics ("seats approved ... 150" and "the tuition above is for
            # MBBS only") would otherwise be quoted whole, and the seat count
            # would end up answering a fee question.
            from .guard import split_sentences

            sentences: list[str] = []
            for ln in prose_lines:
                sentences.extend(split_sentences(ln.strip()))
            for start in range(0, max(len(sentences), 1), 2):
                body = " ".join(sentences[start : start + 2]) or " ".join(
                    ln.strip() for ln in prose_lines
                )
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}#c{len(chunks)}",
                        doc_id=doc.doc_id,
                        text=body,
                        session=doc.session,
                        state=doc.state,
                        college=doc.college,
                        doc_type=doc.doc_type,
                        status=doc.status,
                        section="prose",
                    )
                )
    return chunks


def load_documents(path: Path | None = None) -> list[Document]:
    path = path or DATA_DIR / "documents.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Document(**d) for d in raw]


def load_questions(path: Path | None = None) -> list[dict]:
    path = path or DATA_DIR / "questions.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_heldout(path: Path | None = None) -> list[dict]:
    path = path or DATA_DIR / "questions_heldout.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_resumes(path: Path | None = None) -> list[dict]:
    path = path or DATA_DIR / "resumes.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_jobs(path: Path | None = None) -> list[dict]:
    path = path or DATA_DIR / "jobs.json"
    return json.loads(path.read_text(encoding="utf-8"))
