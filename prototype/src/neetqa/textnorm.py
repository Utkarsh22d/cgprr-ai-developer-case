"""Normalisation helpers: tokens, Indian number words, roman-Hindi/Bengali slang.

These functions are deliberately pure and deterministic so that the numeric
guard can be unit tested without a model in the loop.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Roman-script Hindi / Bengali helpers.
#
# Families type "patna me mbbs ki fees kitni hai" or "kolkata college er bond
# koto". Neither is Hindi or Bengali in Unicode; it is the local language in
# Latin letters. We do not try to translate. We map the high-frequency function
# words and question words onto the English vocabulary the corpus uses, which
# is enough to make BM25 behave on these queries.
# --------------------------------------------------------------------------

_ROMAN_HINTS: dict[str, str] = {
    # question words / copulas (dropped, they carry no retrieval signal)
    "kitni": "how much",
    "kitna": "how much",
    "kitne": "how much",
    "koto": "how much",
    "koto": "how much",
    "konta": "which",
    "konti": "which",
    "kemon": "how",
    "kaise": "how",
    "kya": "what",
    "ki": "",
    "ka": "",
    "ke": "",
    "er": "",
    "me": "",
    "mein": "",
    "hai": "is",
    "hain": "is",
    "ache": "is",
    "achhe": "is",
    "kore": "",
    "kori": "",
    "karna": "",
    "karni": "",
    "karo": "",
    "theke": "",
    "jonno": "",
    "liye": "for",
    "re": "",
    "te": "",
    "me": "",
    "lagta": "",
    "lagbe": "",
    "hobe": "will be",
    "hoy": "is",
    "chai": "want",
    "batao": "tell",
    "bolo": "tell",
    # domain words, roman spellings -> corpus spelling
    "fees": "fee",
    "fis": "fee",
    "phis": "fee",
    "shulk": "fee",
    "sulkho": "fee",
    "bondh": "bond",
    "band": "bond",
    "bond": "bond",
    "seat": "seat",
    "siton": "seat",
    "asoon": "seat",
    "sanghya": "seat",
    "hostel": "hostel",
    "chhatrabas": "hostel",
    "medical": "medical",
    "college": "college",
    "bharti": "admission",
    "bhurti": "admission",
    "pravesh": "admission",
    "sarkari": "government",
    "govt": "government",
    "private": "private",
    "sesan": "session",
    "bochor": "year",
    "saal": "year",
    "barsa": "year",
    "bharna": "",
    "padta": "",
    "taka": "",
    "rupaye": "",
    "lakh": "lakh",
    "lac": "lakh",
    "crore": "crore",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalise_unicode(text: str) -> str:
    """Fold Devanagari/Bengali digits and full-width forms to ASCII."""
    text = unicodedata.normalize("NFKC", text)
    devanagari = "०१२३४५६७८९"
    bengali = "০১২৩৪৫৬৭৮৯"
    trans = {ord(d): str(i) for i, d in enumerate(devanagari + bengali)}
    return text.translate(trans)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, with roman-Hindi/Bengali hints applied."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(normalise_unicode(text).lower()):
        mapped = _ROMAN_HINTS.get(raw, raw)
        if mapped:
            out.extend(_TOKEN_RE.findall(mapped))
    # Light plural folding: "seats" and "seat" must be the same term, otherwise
    # "how many seats" cannot find "Seats approved: 100".
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in out]


def detect_language(question: str) -> str:
    """Crude language tag used only for reporting and reply scaffolding."""
    q = normalise_unicode(question).lower()
    bengali_markers = ("koto", "kemon", "er ", "ache", "hobe", "chhatrabas", "bochor")
    hindi_markers = ("kitni", "kitna", "kaise", "ki fee", "hai", "saal", "kya", "batao")
    if any(m in q for m in bengali_markers):
        return "bn-rom"
    if any(m in q for m in hindi_markers):
        return "hi-rom"
    if re.search(r"[\u0900-\u097F]", question):
        return "hi"
    if re.search(r"[\u0980-\u09FF]", question):
        return "bn"
    return "en"


# --------------------------------------------------------------------------
# Indian money parsing: "INR 10 lakh", "Rs. 5,00,000", "1.5 crore", "50%".
# --------------------------------------------------------------------------

_MULTIPLIERS: dict[str, float] = {
    "lakh": 1e5,
    "lakhs": 1e5,
    "lac": 1e5,
    "lacs": 1e5,
    "crore": 1e7,
    "crores": 1e7,
    "thousand": 1e3,
}

# Doc IDs such as DOC-SMC-PAT-FEE-2627 carry digits that are not amounts.
_DOC_ID_RE = re.compile(r"\b[A-Z]{2,}-[A-Z0-9-]{2,}\b")
_SESSION_RE = re.compile(r"\b(20\d{2})\s*[-/]\s*(\d{2,4})\b")

_NUM_RE = re.compile(
    r"(?P<cur>₹|Rs\.?|INR)?\s*"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"\s*(?P<unit>lakhs?|lacs?|crores?|thousand|%|percent)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Amount:
    """A number lifted out of text, in canonical form."""

    raw: str
    value: float
    kind: str  # "money" | "percent" | "count" | "year"
    unit: str  # "", "lakh", "crore", "percent"

    @property
    def key(self) -> tuple[str, float]:
        return (self.kind, round(self.value, 2))


def strip_identifiers(text: str) -> str:
    """Remove doc IDs and session-year ranges so their digits are not amounts."""
    text = _DOC_ID_RE.sub(" ", text)
    text = _SESSION_RE.sub(" ", text)
    return text


def extract_amounts(text: str) -> list[Amount]:
    """Pull every number out of ``text`` in canonical form."""
    text = strip_identifiers(normalise_unicode(text))
    found: list[Amount] = []
    for m in _NUM_RE.finditer(text):
        raw = m.group(0).strip()
        digits = m.group("num").replace(",", "")
        if not digits:
            continue
        try:
            value = float(digits)
        except ValueError:  # pragma: no cover - regex already guarantees digits
            continue
        unit = (m.group("unit") or "").lower()
        if unit in _MULTIPLIERS:
            value *= _MULTIPLIERS[unit]
            kind = "money" if m.group("cur") or unit != "thousand" else "count"
            unit = unit.rstrip("s")
        elif unit in {"%", "percent"}:
            kind = "percent"
            unit = "percent"
        elif m.group("cur"):
            kind = "money"
            unit = ""
        else:
            kind = "count"
            unit = ""
        # Bare 4-digit tokens in 1950..2099 are almost always years.
        if kind == "count" and value.is_integer() and 1950 <= value <= 2099:
            kind = "year"
        found.append(Amount(raw=raw, value=value, kind=kind, unit=unit))
    return found


def amount_set(text: str) -> set[tuple[str, float]]:
    return {a.key for a in extract_amounts(text)}


def money_inr(value: float) -> str:
    """Format a rupee amount the way the notices print it (Indian grouping)."""
    n = int(round(value))
    s = str(n)
    if len(s) <= 3:
        return "INR " + s
    head, tail = s[:-3], s[-3:]
    parts: list[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return "INR " + ",".join(parts + [tail])
