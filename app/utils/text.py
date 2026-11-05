"""Text normalisation and sentence-aware chunking for TTS."""

from __future__ import annotations

import re
import unicodedata

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_BOUNDARY = re.compile(r"(?<=[,;:])\s+")
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")


def normalize_text(text: str) -> str:
    """Normalise text before speech synthesis.

    - Unicode NFC
    - CRLF/CR to LF
    - remove control/format characters (tabs become spaces)
    - collapse repeated spaces and newlines
    - strip line edges

    Punctuation is intentionally kept because edge-tts uses it for pauses.
    Numbers and abbreviations are deliberately not expanded.
    """

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned: list[str] = []
    for char in text:
        if char == "\n":
            cleaned.append("\n")
        elif char == "\t":
            cleaned.append(" ")
        elif unicodedata.category(char).startswith("C"):
            # Control, format (zero-width), surrogate, private-use, unassigned.
            continue
        else:
            cleaned.append(char)

    text = "".join(cleaned)
    text = _HORIZONTAL_SPACE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)

    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    return _HORIZONTAL_SPACE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split normalised text into sentence-like units.

    Newlines and sentence-ending punctuation are treated as boundaries. This
    is intentionally conservative; it never splits inside a word.
    """

    sentences: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_BOUNDARY.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
    return sentences


def _split_by_words(segment: str, max_chars: int) -> list[str]:
    """Split an over-long segment at spaces.

    A single token longer than ``max_chars`` has to be hard-split; this is a
    pathological case (for example a 2,000-character URL) and is documented in
    the README.
    """

    pieces: list[str] = []
    current = ""

    for word in segment.split(" "):
        if not word:
            continue
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            pieces.append(current)

        if len(word) <= max_chars:
            current = word
            continue

        # A single word that cannot fit: split it into max-sized pieces.
        for start in range(0, len(word), max_chars):
            piece = word[start : start + max_chars]
            if len(piece) == max_chars:
                pieces.append(piece)
            else:
                current = piece

    if current:
        pieces.append(current)
    return pieces


def _split_long_segment(segment: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    for clause in _CLAUSE_BOUNDARY.split(segment):
        clause = clause.strip()
        if not clause:
            continue
        if len(clause) <= max_chars:
            pieces.append(clause)
        else:
            pieces.extend(_split_by_words(clause, max_chars))
    return pieces


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most ``max_chars`` characters.

    The function packs whole sentences (or clauses, for very long sentences)
    up to the limit. It never splits mid-word unless a single token is itself
    longer than ``max_chars``.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    segments: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence) <= max_chars:
            segments.append(sentence)
        else:
            segments.extend(_split_long_segment(sentence, max_chars))

    chunks: list[str] = []
    current = ""
    for segment in segments:
        candidate = f"{current} {segment}" if current else segment
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = segment
    if current:
        chunks.append(current)

    return [chunk.strip() for chunk in chunks if chunk.strip()]
