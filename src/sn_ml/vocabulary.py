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
