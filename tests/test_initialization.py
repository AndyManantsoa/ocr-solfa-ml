"""Initialization checks; these intentionally exercise no OCR behavior."""

import importlib

from sn_ml import __version__
from sn_ml.vocabulary import (
    CHAR_TO_INDEX,
    CTC_BLANK_INDEX,
    CTC_CLASS_COUNT,
    INDEX_TO_CHAR,
    VOCAB,
)


def test_package_imports() -> None:
    assert __version__ == "0.1.0"


def test_vocabulary_is_exactly_the_configured_visible_characters() -> None:
    assert VOCAB == (
        "d", "r", "m", "f", "s", "l", "t",
        "i", "e", "a",
        ",", "'", ".", ";", "-", "(", ")",
    )
    assert len(VOCAB) == 17
    assert len(set(VOCAB)) == 17
    assert "0" not in VOCAB
    assert ":" not in VOCAB
    assert "|" not in VOCAB


def test_ctc_blank_is_internal_and_distinct() -> None:
    assert CTC_BLANK_INDEX == 0
    assert CTC_CLASS_COUNT == 18
    assert set(CHAR_TO_INDEX) == set(VOCAB)
    assert set(INDEX_TO_CHAR.values()) == set(VOCAB)
    assert CTC_BLANK_INDEX not in INDEX_TO_CHAR


def test_declared_modules_import() -> None:
    for module in ("dataset", "model", "train", "evaluate", "grammar", "inference"):
        importlib.import_module(f"sn_ml.{module}")
