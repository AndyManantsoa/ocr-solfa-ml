"""Tonic sol-fa grammar behavior independent of OCR model confidence."""

import pytest

from sn_ml.grammar import validate_tonic_solfa


@pytest.mark.parametrize("text,notes", [
    ("d", ["d"]), ("di", ["di"]), ("drm", ["d", "r", "m"]),
    ("didi", ["di", "di"]), ("d,d", ["d", "d"]), ("di.r", ["di", "r"]),
    ("mfs", ["m", "f", "s"]), ("d,,", ["d"]), ("fi,,,", ["fi"]),
    ("d'''", ["d"]), (",d", ["d"]), ("d,r", ["d", "r"]),
    ("d.,r", ["d", "r"]), ("(d)", ["d"]), ("", []),
])
def test_valid_tonic_solfa_sequences(text: str, notes: list[str]) -> None:
    result = validate_tonic_solfa(text)
    assert result.valid, result.errors
    assert [token.text for token in result.note_tokens] == notes


def test_octaves_rhythm_and_parentheses_are_structured() -> None:
    octave = validate_tonic_solfa("fi,,,")
    assert octave.note_tokens[0].octave == ",,,"
    leading = validate_tonic_solfa(",d")
    assert leading.note_tokens[0].leading_lower_octave is True
    rhythm = validate_tonic_solfa("d.,r")
    assert rhythm.rhythmic_symbols == [".,"]
    assert rhythm.requires_musical_context is True
    assert validate_tonic_solfa("(d)").parentheses == ["(", ")"]


@pytest.mark.parametrize("text", ["da", "mi", "fa", "sa", "ti", "d,'", "d',", "d,,'", ",d'", ",d,", ",,d", "d("])
def test_invalid_combinations_are_not_rewritten(text: str) -> None:
    result = validate_tonic_solfa(text)
    assert not result.valid
    assert result.errors
