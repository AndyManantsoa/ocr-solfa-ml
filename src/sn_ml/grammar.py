"""Closed-vocabulary tonic sol-fa syntax validation, separate from OCR decoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

NATURAL_NOTES = frozenset("drmfslt")
CHROMATIC_NOTES = ("di", "ri", "fi", "si", "li", "de", "ra", "ma", "fe", "se", "le", "ta")
RHYTHMIC_MARKERS = (".,", ".", ";", "-")


@dataclass(frozen=True)
class NoteToken:
    text: str
    octave: str = ""
    leading_lower_octave: bool = False


@dataclass
class GrammarResult:
    valid: bool
    note_tokens: list[NoteToken] = field(default_factory=list)
    octave_markers: list[str] = field(default_factory=list)
    rhythmic_symbols: list[str] = field(default_factory=list)
    parentheses: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    requires_musical_context: bool = False

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "note_tokens": [asdict(token) for token in self.note_tokens]}


def validate_tonic_solfa(text: str) -> GrammarResult:
    """Validate a cell transcription without correcting or timing it.

    A lone leading comma is a documented lower-octave prefix. Other commas
    immediately following a note form its lower-octave suffix, except the comma
    in the ``.,`` rhythmic marker. Rhythm is recognized but flagged as requiring
    meter/context for full musical validation.
    """
    result = GrammarResult(valid=True)
    if text == "":
        return result
    index = 0
    open_parentheses = 0
    pending_leading_lower = False
    chromatics = sorted(CHROMATIC_NOTES, key=len, reverse=True)
    while index < len(text):
        remainder = text[index:]
        if remainder.startswith(".,"):
            result.rhythmic_symbols.append(".,")
            result.requires_musical_context = True
            index += 2
            continue
        character = text[index]
        if character in ".;-":
            result.rhythmic_symbols.append(character)
            result.requires_musical_context = True
            index += 1
            continue
        if character == "(":
            open_parentheses += 1
            result.parentheses.append("(")
            index += 1
            continue
        if character == ")":
            result.parentheses.append(")")
            if open_parentheses == 0:
                result.errors.append("unmatched_closing_parenthesis")
            else:
                open_parentheses -= 1
            index += 1
            continue
        if character == ",":
            if pending_leading_lower or index + 1 >= len(text) or text[index + 1] not in NATURAL_NOTES:
                result.errors.append("invalid_leading_comma")
                index += 1
                continue
            pending_leading_lower = True
            index += 1
            continue
        if character in "'iea":
            result.errors.append(f"independent_or_invalid_modifier:{character}")
            index += 1
            continue
        note = next((candidate for candidate in chromatics if remainder.startswith(candidate)), None)
        if note is None and character in NATURAL_NOTES:
            note = character
        if note is None:
            result.errors.append(f"unsupported_character:{character}")
            index += 1
            continue
        index += len(note)
        octave_start = index
        if index < len(text) and text[index] in ",'":
            octave_symbol = text[index]
            while index < len(text) and text[index] == octave_symbol:
                index += 1
            octave = text[octave_start:index]
            if index < len(text) and text[index] in ",'":
                result.errors.append("mixed_or_invalid_octave_markers")
        else:
            octave = ""
        if pending_leading_lower and octave:
            result.errors.append("leading_comma_cannot_combine_with_suffix_octave")
        result.note_tokens.append(NoteToken(note, octave, pending_leading_lower))
        if pending_leading_lower:
            result.octave_markers.append(",")
        if octave:
            result.octave_markers.append(octave)
        pending_leading_lower = False
    if pending_leading_lower:
        result.errors.append("leading_comma_without_note")
    if open_parentheses:
        result.errors.append("unclosed_parenthesis")
    result.valid = not result.errors
    return result
