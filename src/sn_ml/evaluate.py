"""Recognition metrics for validation only; no split mutation or model training."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from sn_ml.grammar import validate_tonic_solfa


def levenshtein_distance(reference: str, prediction: str) -> int:
    row = list(range(len(prediction) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        next_row = [reference_index]
        for prediction_index, prediction_character in enumerate(prediction, start=1):
            next_row.append(min(
                next_row[-1] + 1,
                row[prediction_index] + 1,
                row[prediction_index - 1] + (reference_character != prediction_character),
            ))
        row = next_row
    return row[-1]


@dataclass
class RecognitionMetrics:
    samples: int = 0
    character_errors: int = 0
    reference_characters: int = 0
    exact_cells: int = 0
    blank_predictions: int = 0
    invalid_grammar_predictions: int = 0
    per_character_reference: Counter[str] = field(default_factory=Counter)
    per_character_exact_position: Counter[str] = field(default_factory=Counter)

    def add(self, reference: str, prediction: str) -> None:
        self.samples += 1
        self.character_errors += levenshtein_distance(reference, prediction)
        self.reference_characters += len(reference)
        self.exact_cells += reference == prediction
        self.blank_predictions += prediction == ""
        self.invalid_grammar_predictions += not validate_tonic_solfa(prediction).valid
        self.per_character_reference.update(reference)
        if len(reference) == len(prediction):
            self.per_character_exact_position.update(
                expected for expected, observed in zip(reference, prediction) if expected == observed
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "character_errors": self.character_errors,
            "reference_characters": self.reference_characters,
            "exact_cells": self.exact_cells,
            "blank_predictions": self.blank_predictions,
            "invalid_grammar_predictions": self.invalid_grammar_predictions,
            "cer": self.character_errors / max(self.reference_characters, 1),
            "exact_cell_accuracy": self.exact_cells / max(self.samples, 1),
            "blank_prediction_frequency": self.blank_predictions / max(self.samples, 1),
            "invalid_grammar_prediction_frequency": self.invalid_grammar_predictions / max(self.samples, 1),
            "per_character": {
            character: {
                "reference_count": count,
                "exact_position_correct": self.per_character_exact_position[character],
                "exact_position_accuracy_when_equal_length": self.per_character_exact_position[character] / count,
            }
            for character, count in sorted(self.per_character_reference.items())
            },
        }


def recognition_metrics(pairs: Iterable[tuple[str, str]]) -> dict[str, object]:
    metrics = RecognitionMetrics()
    for reference, prediction in pairs:
        metrics.add(reference, prediction)
    return metrics.to_dict()


def character_error_summary(reference: str, prediction: str) -> dict[str, object]:
    """Compact diagnostic without applying grammar-based correction to prediction."""
    return {
        "reference": reference,
        "prediction": prediction,
        "character_errors": levenshtein_distance(reference, prediction),
        "prediction_grammar": validate_tonic_solfa(prediction).to_dict(),
    }


def alignment_errors(reference: str, prediction: str) -> dict[str, object]:
    """Return deterministic Levenshtein operations without changing either string."""
    rows, columns = len(reference), len(prediction)
    costs = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows + 1):
        costs[row][0] = row
    for column in range(columns + 1):
        costs[0][column] = column
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            costs[row][column] = min(
                costs[row - 1][column] + 1,
                costs[row][column - 1] + 1,
                costs[row - 1][column - 1] + (reference[row - 1] != prediction[column - 1]),
            )
    operations: list[dict[str, object]] = []
    row, column = rows, columns
    while row or column:
        if row and column and costs[row][column] == costs[row - 1][column - 1] and reference[row - 1] == prediction[column - 1]:
            row, column = row - 1, column - 1
        elif row and column and costs[row][column] == costs[row - 1][column - 1] + 1:
            operations.append({"kind": "substitution", "reference": reference[row - 1], "prediction": prediction[column - 1]})
            row, column = row - 1, column - 1
        elif column and costs[row][column] == costs[row][column - 1] + 1:
            operations.append({"kind": "insertion", "reference": "", "prediction": prediction[column - 1]})
            column -= 1
        else:
            operations.append({"kind": "deletion", "reference": reference[row - 1], "prediction": ""})
            row -= 1
    operations.reverse()
    counts = Counter(operation["kind"] for operation in operations)
    return {"operations": operations, "insertions": counts["insertion"], "deletions": counts["deletion"], "substitutions": counts["substitution"]}
