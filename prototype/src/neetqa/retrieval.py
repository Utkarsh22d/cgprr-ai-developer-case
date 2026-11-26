"""Hybrid retrieval: BM25 + a dense vector space, fused, then re-ranked.

The dense half is a latent-semantic (LSA) encoder built from the corpus itself
with numpy. That is a *stand-in* for a sentence-transformer of the same size
budget (a 22M-parameter MiniLM is well under the 8B limit); it is here so the
prototype has zero download and zero network dependency. ``EMBEDDER=st`` in the
environment switches to sentence-transformers when it is installed.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .corpus import Chunk, Document, chunk_document
from .textnorm import tokenize

# Programme a document is about. Getting this wrong is how a BSc Nursing fee
# gets quoted to an MBBS parent, so it is metadata, not a model decision.
PROGRAMME: dict[str, str] = {
    "fee_notice": "MBBS",
    "fee_notice_pg": "MD",
    "fee_notice_nursing": "BSc Nursing",
    "bond_rule": "MBBS",
    "fee_order": "MBBS",
    "seat_matrix": "MBBS",
}

# High-precision document-type cues. Deliberately only the two where a
# confusion is dangerous (bond vs fee, and counselling rules vs fee).
DOCTYPE_CUES: dict[str, tuple[str, ...]] = {
    "bond_rule": ("bond", "bondh"),
    "counselling_rules": ("forfeit", "forfeiture", "resign", "round", "counselling", "reporting"),
}

STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it of on or that the
    to was what when where which who will with you your me my do does did i we
    much many please tell give know need want""".split()
)


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


class BM25:
    """Okapi BM25 over a fixed chunk collection."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [content_tokens(d) for d in docs]
        self.n = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.n) if self.n else 0.0
        self.tf: list[Counter[str]] = [Counter(d) for d in self.docs]
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        self.df = df
        self.idf = {
            t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }

    def score(self, query: str, i: int) -> float:
        q = content_tokens(query)
        if not q or not self.avgdl:
            return 0.0
        tf = self.tf[i]
        dl = len(self.docs[i])
        total = 0.0
        for term in q:
            f = tf.get(term, 0)
            if not f:
                continue
            idf = self.idf.get(term, 0.0)
            total += idf * f * (self.k1 + 1) / (
                f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
        return total

    def df_of(self, term: str) -> int:
        return self.df.get(term, 0)


class LSAEmbedder:
    """Truncated-SVD embedding of a tf-idf matrix. Deterministic, offline."""

    def __init__(self, dim: int = 48) -> None:
        self.dim = dim
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.components: np.ndarray | None = None

    @staticmethod
    def _features(text: str) -> list[str]:
        toks = content_tokens(text)
        feats = list(toks)
        # character 4-grams give roman-Hindi spellings ("shulk"/"sulk") a
        # chance to collide without a transliteration table.
        for t in toks:
            if len(t) >= 5 and t.isalpha():
                feats.extend(t[i : i + 4] for i in range(len(t) - 3))
        return feats

    def _matrix(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            c = Counter(self._features(t))
            rows.append([c.get(w, 0) for w in self.vocab])
        return np.asarray(rows, dtype=np.float32)

    def fit(self, texts: list[str]) -> "LSAEmbedder":
        counts = Counter()
        for t in texts:
            counts.update(set(self._features(t)))
        self.vocab = {w: i for i, w in enumerate(sorted(counts))}
        n = max(len(texts), 1)
        self.idf = np.asarray(
            [math.log((1 + n) / (1 + counts[w])) + 1.0 for w in self.vocab],
            dtype=np.float32,
        )
        x = self._matrix(texts) * self.idf
        # log-normalise term frequency
        x = np.log1p(x)
        k = min(self.dim, min(x.shape) - 1) if min(x.shape) > 1 else 1
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        self.components = vt[:k].T.astype(np.float32)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        assert self.idf is not None and self.components is not None
        x = np.log1p(self._matrix(texts) * self.idf)
        v = x @ self.components
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return v / norms


class SentenceTransformerEmbedder:  # pragma: no cover - optional backend
    """Optional real encoder; only used when sentence-transformers is present."""

    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(name)

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        v = np.asarray(self.model.encode(texts, normalize_embeddings=True))
        return v


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    bm25: float
    dense: float
    rerank: float
    parts: dict[str, float] = field(default_factory=dict)


class HybridIndex:
    """BM25 + dense, reciprocal-rank fusion, then a cross-encoder stand-in."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.chunks: list[Chunk] = []
        for doc in documents:
            self.chunks.extend(chunk_document(doc))
        self._by_doc = {d.doc_id: d for d in documents}
        # Chunk text is what we CITE; index text adds the document title so that
        # "deposit at Ganges Institute, Varanasi" matches the Varanasi row and
        # not an identical-looking row from four other colleges.
        self.index_texts = [
            f"{self._by_doc[c.doc_id].title}. {c.text}" for c in self.chunks
        ]
        self.bm25 = BM25(self.index_texts)
        backend = os.environ.get("EMBEDDER", "lsa").lower()
        self.embedder = (
            SentenceTransformerEmbedder() if backend == "st" else LSAEmbedder()
        )
        self.matrix = self.embedder.fit(self.index_texts).transform(self.index_texts)
        self.programme = {c.chunk_id: PROGRAMME.get(c.doc_type) for c in self.chunks}

    # -- filters ---------------------------------------------------------
    def _allowed(self, chunk: Chunk, session: str | None, include_superseded: bool) -> bool:
        if session is None:
            if chunk.status != "active":
                return False
            return True
        # A question about a past session may read that session's superseded
        # notice -- but only that session's.
        if chunk.session is not None and chunk.session != session:
            return False
        if chunk.session is None:
            return True
        return chunk.status == "active" or chunk.session == session

    # -- query -----------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        session: str | None = None,
        include_superseded: bool = False,
    ) -> list[ScoredChunk]:
        allowed = [
            i
            for i, c in enumerate(self.chunks)
            if self._allowed(c, session, include_superseded)
        ]
        if not allowed:
            return []

        bm = np.array([self.bm25.score(query, i) for i in allowed], dtype=np.float32)
        qv = self.embedder.transform([query])[0]
        dense = self.matrix[allowed] @ qv

        # reciprocal rank fusion of the two candidate lists
        rr = np.zeros(len(allowed), dtype=np.float32)
        for rank, pos in enumerate(np.argsort(-bm)[:50]):
            rr[pos] += 1.0 / (60 + rank + 1)
        for rank, pos in enumerate(np.argsort(-dense)[:50]):
            rr[pos] += 1.0 / (60 + rank + 1)

        # cross-encoder stand-in: idf-weighted term coverage + numeric match +
        # dense cosine. A real 22M cross-encoder would replace this exactly.
        q_terms = set(content_tokens(query))
        q_amounts = {
            t
            for t in tokenize(query)
            if any(ch.isdigit() for ch in t)
        }
        rerank = np.zeros(len(allowed), dtype=np.float32)
        raw_coverage = np.zeros(len(allowed), dtype=np.float32)
        for pos, i in enumerate(allowed):
            toks = set(content_tokens(self.index_texts[i]))
            hit = q_terms & toks
            coverage = (
                sum(self.bm25.idf.get(t, 0.0) for t in hit)
                / max(sum(self.bm25.idf.get(t, 0.0) for t in q_terms), 1e-6)
                if q_terms
                else 0.0
            )
            numeric = 0.0
            if q_amounts:
                raw_tokens = set(tokenize(self.index_texts[i]))
                numeric = len(q_amounts & raw_tokens) / len(q_amounts)
            raw_coverage[pos] = coverage
            rerank[pos] = 0.55 * coverage + 0.25 * float(dense[pos]) + 0.20 * numeric

        # Intent adjustment: the programme the family asked about and the
        # document type that can answer it. This is what stops a nursing notice
        # or a PG notice from answering an MBBS fee question.
        qset = set(tokenize(query))
        q_prog = (
            "MBBS"
            if "mbbs" in qset
            else "MD"
            if qset & {"md", "ms", "pg", "postgraduate"}
            else "BSc Nursing"
            if qset & {"nursing", "bsc"}
            else None
        )
        q_doctype = next(
            (dt for dt, cues in DOCTYPE_CUES.items() if qset & set(cues)), None
        )
        for pos, i in enumerate(allowed):
            chunk = self.chunks[i]
            prog = self.programme.get(chunk.chunk_id)
            if q_prog and prog:
                rerank[pos] += 0.35 if prog == q_prog else -0.25
            if q_doctype and chunk.doc_type == q_doctype:
                rerank[pos] += 0.45

        fused = 0.35 * (rr / (rr.max() or 1.0)) + 0.65 * (rerank / (rerank.max() or 1.0))
        order = np.argsort(-fused)[: max(top_k * 3, top_k)]
        out = [
            ScoredChunk(
                chunk=self.chunks[allowed[p]],
                score=float(fused[p]),
                bm25=float(bm[p]),
                dense=float(dense[p]),
                rerank=float(rerank[p]),
                parts={"coverage": float(raw_coverage[p])},
            )
            for p in order
        ]
        # de-duplicate: at most two chunks per document, keep best first
        seen: Counter[str] = Counter()
        deduped: list[ScoredChunk] = []
        for sc in out:
            if seen[sc.chunk.doc_id] >= 2:
                continue
            seen[sc.chunk.doc_id] += 1
            deduped.append(sc)
        return deduped[:top_k]

    def document(self, doc_id: str) -> Document:
        return self._by_doc[doc_id]
