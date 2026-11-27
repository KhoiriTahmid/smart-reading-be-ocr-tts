"""Text normalisation and sentence-aware chunking for TTS."""

from __future__ import annotations

import re
import unicodedata

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_BOUNDARY = re.compile(r"(?<=[,;:])\s+")
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_ALLOWED_ALNUM_TOKEN = re.compile(
    r"^(?:\d+(?:st|nd|rd|th|s|km|cm|mm|kg|mg|ml|m|g|h|x)|[A-Z]{1,4}\d{1,3})$"
)
_VOWELS = frozenset("aeiouyAEIOUY")


def _is_noise_token(token: str) -> bool:
    """Return True if a single stripped token looks like OCR margin gibberish."""
    core = token.strip(".,;:!?\"'()[]{}-–—/\\")
    if not core:
        return False

    has_alpha = any(ch.isalpha() for ch in core)
    has_digit = any(ch.isdigit() for ch in core)
    if not has_alpha:
        return False

    # Mixed digits and letters inside a single word (e.g. 197ov9, 3onslq, 1OSE2, g79q,oyA)
    if has_digit:
        if _ALLOWED_ALNUM_TOKEN.match(core) or _ALLOWED_ALNUM_TOKEN.match(core.lower() if core[0].isdigit() else core):
            return False
        return True

    # Purely alphabetic lowercase/mixed-case token >= 3 chars with no vowels (e.g. sbgq),
    # or ending in 'q' / containing 'q' not followed by 'u' (e.g. insq, pitaprq).
    # Uppercase acronyms (e.g. SMP, DPR, UUD) are preserved.
    if len(core) >= 3 and not core.isupper():
        if not any(ch in _VOWELS for ch in core):
            return True
        lower = core.lower()
        if lower.endswith("q"):
            return True

    return False


def is_gibberish_line(line: str) -> bool:
    """Return True if a short isolated OCR line is likely spine/margin noise.

    Targets short 1-2 token lines containing mixed digit/alpha gibberish
    (e.g. ``197ov9``, ``3onslq``, ``1OSE2``, ``g79q,oyA``) or vowelless
    consonant clusters (e.g. ``insq sbgq``), while preserving legitimate
    body text, numbers, ordinals, units, and normal short words.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 24:
        return False

    tokens = stripped.split()
    if not tokens or len(tokens) > 2:
        return False

    if any(_is_noise_token(tok) for tok in tokens):
        if len(tokens) == 1:
            return True
        has_mixed_alnum = any(
            any(c.isalpha() for c in tok) and any(c.isdigit() for c in tok) and _is_noise_token(tok)
            for tok in tokens
        )
        if has_mixed_alnum or all(_is_noise_token(tok) or len(tok) <= 2 for tok in tokens):
            return True

    return False


def clean_ocr_line(line: str) -> str:
    """Drop isolated gibberish lines and strip leading margin-noise tokens."""
    stripped = line.strip()
    if not stripped or is_gibberish_line(stripped):
        return ""

    tokens = stripped.split()
    start = 0
    while start < len(tokens) - 1 and _is_noise_token(tokens[start]):
        start += 1
    if start > 0:
        cleaned = " ".join(tokens[start:]).strip()
        if is_gibberish_line(cleaned):
            return ""
        return cleaned
    return stripped


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
