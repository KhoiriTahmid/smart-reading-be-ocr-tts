"""Text normalisation and chunking tests."""

from __future__ import annotations

import unicodedata

import pytest

from app.utils.text import chunk_text, normalize_text, split_sentences


def test_normalize_nfc() -> None:
    decomposed = "Cafe\u0301"
    result = normalize_text(decomposed)
    assert result == unicodedata.normalize("NFC", decomposed)
    assert result == "Café"


def test_normalize_collapses_whitespace_and_newlines() -> None:
    assert normalize_text("Halo   dunia\r\n\r\n\r\nini\ttest") == (
        "Halo dunia\nini test"
    )


def test_normalize_removes_control_and_zero_width_characters() -> None:
    assert normalize_text("Halo\u200b\u0000 dunia\u0007") == "Halo dunia"


def test_normalize_keeps_punctuation_numbers_and_contacts() -> None:
    text = "Harga Rp 25.000, email: budi@example.com, telp: 0812-3456."
    assert normalize_text(text) == text


def test_normalize_empty_and_whitespace() -> None:
    assert normalize_text("   \n\t ") == ""


def test_split_sentences() -> None:
    assert split_sentences("Halo. Ini uji! Apakah benar?") == [
        "Halo.",
        "Ini uji!",
        "Apakah benar?",
    ]


def test_chunk_short_text_is_single_chunk() -> None:
    assert chunk_text("Halo dunia.", 900) == ["Halo dunia."]


def test_chunk_empty_text() -> None:
    assert chunk_text("   ", 900) == []


def test_chunk_prefers_sentence_boundaries() -> None:
    sentence_one = "A" * 400 + "."
    sentence_two = "B" * 400 + "."
    text = f"{sentence_one} {sentence_two}"
    assert chunk_text(text, 500) == [sentence_one, sentence_two]


def test_chunk_5000_chars() -> None:
    text = " ".join(
        f"Ini adalah kalimat contoh nomor {index} yang cukup panjang."
        for index in range(120)
    )
    assert len(text) > 5_000

    chunks = chunk_text(text, 900)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 900 for chunk in chunks)
    assert " ".join(chunks) == normalize_text(text)


def test_chunk_never_splits_words_when_possible() -> None:
    text = " ".join(["kata"] * 1_000)
    chunks = chunk_text(text, 100)
    assert all(len(chunk) <= 100 for chunk in chunks)
    for chunk in chunks:
        assert chunk.split(" ")[0] in {"kata"}
        assert chunk.split(" ")[-1] in {"kata"}


def test_chunk_hard_splits_pathological_single_token() -> None:
    text = "x" * 5_000
    chunks = chunk_text(text, 900)
    assert all(len(chunk) <= 900 for chunk in chunks)
    assert "".join(chunks) == text


def test_chunk_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError):
        chunk_text("Halo", 0)
