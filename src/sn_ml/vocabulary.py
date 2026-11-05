"""Closed character vocabulary for future CTC OCR work."""

from typing import Final

# Visible OCR characters only. CTC's blank is an internal class, not a token.
VOCAB: Final[tuple[str, ...]] = (
    "d", "r", "m", "f", "s", "l", "t",
    "i", "e", "a",
    ",", "'", ".", ";", "-",
    "(", ")",
)

CTC_BLANK_INDEX: Final[int] = 0
CTC_CLASS_COUNT: Final[int] = len(VOCAB) + 1
CHAR_TO_INDEX: Final[dict[str, int]] = {
    character: index for index, character in enumerate(VOCAB, start=1)
}
INDEX_TO_CHAR: Final[dict[int, str]] = {
    index: character for character, index in CHAR_TO_INDEX.items()
}


def encode_text(text: str) -> list[int]:
    """Encode visible OCR characters; CTC blank is never accepted as text."""
    try:
        return [CHAR_TO_INDEX[character] for character in text]
    except KeyError as error:
        raise ValueError(f"Unsupported OCR character: {error.args[0]!r}") from error


def decode_indices(indices: list[int]) -> str:
    """Decode visible classes only, rejecting the internal CTC blank class."""
    try:
        return "".join(INDEX_TO_CHAR[index] for index in indices)
    except KeyError as error:
        raise ValueError(f"Index is not a visible OCR class: {error.args[0]!r}") from error


def ctc_greedy_decode(indices: list[int], blank_index: int = CTC_BLANK_INDEX) -> str:
    """Collapse repeats and remove CTC blanks from one argmax sequence."""
    collapsed: list[int] = []
    previous: int | None = None
    for index in indices:
        if index != previous and index != blank_index:
            collapsed.append(index)
        previous = index
    return decode_indices(collapsed)
