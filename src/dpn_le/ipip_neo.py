"""IPIP-NEO-300 data loading, prompting, and scoring utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TRAIT_ABBR_TO_NAME = {
    "A": "Agreeableness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "N": "Neuroticism",
    "O": "Openness",
}
TRAIT_NAME_TO_ABBR = {name: abbr for abbr, name in TRAIT_ABBR_TO_NAME.items()}
CHOICE_TO_SCORE = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}

IPIP_PROMPT_TEMPLATE = """Given a statement of you: "You {item}."
Please choose from the following options to identify how accurately this statement describes you.
Options:
(A). Very Accurate
(B). Moderately Accurate
(C). Neither Accurate Nor Inaccurate
(D). Moderately Inaccurate
(E). Very Inaccurate

You must respond with ONLY ONE LETTER: A, B, C, D, or E.
Do not include any explanation, reasoning, or other text.
Answer:"""


def normalize_trait(value: str) -> str:
    value = value.strip()
    upper = value.upper()
    if upper in TRAIT_ABBR_TO_NAME:
        return upper
    title = value[:1].upper() + value[1:]
    if title in TRAIT_NAME_TO_ABBR:
        return TRAIT_NAME_TO_ABBR[title]
    raise ValueError(f"Unknown trait '{value}'. Use A/C/E/N/O or a Big Five trait name.")


def format_ipip_prompt(item_text: str) -> str:
    return IPIP_PROMPT_TEMPLATE.format(item=item_text.strip().lower())


def parse_choice(response_text: str) -> str:
    text = response_text.upper().strip()
    for pattern in [r"\(([ABCDE])\)", r"ANSWER[:\s]+([ABCDE])", r"\b([ABCDE])\b"]:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    if text and text[0] in CHOICE_TO_SCORE:
        return text[0]
    return "C"


def parse_individual_responses(individual: dict[str, Any]) -> dict[int, int]:
    responses = {}
    for key, value in individual.items():
        if key.startswith("i") and key[1:].isdigit():
            responses[int(key[1:])] = int(value)
    return responses


class IPIPNEOScorer:
    """Load IPIP item metadata and calculate Big Five/IPIP MAE scores."""

    def __init__(self, item_key_path: str | Path):
        self.item_key_path = Path(item_key_path)
        self.item_info = self._load_item_info(self.item_key_path)

    @staticmethod
    def _load_item_info(path: Path) -> dict[int, dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"IPIP item key not found: {path}")
        if path.suffix.lower() in {".xls", ".xlsx"}:
            frame = pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                frame = pd.DataFrame(json.load(f))
        else:
            raise ValueError("Item key must be .xls, .xlsx, .csv, or .json")

        required = {"Full#", "Key", "Sign", "Item"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Item key missing columns: {sorted(missing)}")

        item_info = {}
        for _, row in frame.iterrows():
            item_id = int(row["Full#"])
            trait = str(row["Key"])[0].upper()
            if trait not in TRAIT_ABBR_TO_NAME:
                continue
            item_info[item_id] = {
                "trait": trait,
                "key": 1 if str(row["Sign"]).strip() == "+" else -1,
                "text": str(row["Item"]),
            }
        return item_info

    def calculate_big_five_scores(
        self,
        responses: dict[int, int],
        item_indices: list[int],
    ) -> dict[str, float]:
        trait_scores: dict[str, list[int]] = {abbr: [] for abbr in TRAIT_ABBR_TO_NAME}
        for item_id in item_indices:
            item_id = int(item_id)
            info = self.item_info.get(item_id)
            if info is None or item_id not in responses:
                continue
            raw = responses[item_id]
            score = raw if info["key"] == 1 else 6 - raw
            trait_scores[info["trait"]].append(score)
        return {
            trait: float(np.mean(scores)) if scores else 3.0
            for trait, scores in trait_scores.items()
        }

    def item_text(self, item_id: int) -> str:
        return self.item_info[int(item_id)]["text"]

    def item_trait(self, item_id: int) -> str:
        return self.item_info[int(item_id)]["trait"]

    def trait_test_indices(self, test_indices: list[int], trait: str) -> list[int]:
        trait = normalize_trait(trait)
        return [int(idx) for idx in test_indices if self.item_trait(int(idx)) == trait]


def score_choices(
    choices: list[str],
    individual: dict[str, Any],
    item_indices: list[int],
) -> dict[str, Any]:
    responses = parse_individual_responses(individual)
    pred_scores = [CHOICE_TO_SCORE[parse_choice(choice)] for choice in choices]
    true_scores = [responses[int(idx)] for idx in item_indices]
    errors = [abs(pred - true) for pred, true in zip(pred_scores, true_scores)]
    return {
        "mae": float(np.mean(errors)) if errors else float("nan"),
        "pred_scores": pred_scores,
        "true_scores": true_scores,
        "errors": errors,
    }


def load_ipip_dataset(data_dir: str | Path) -> tuple[list[dict[str, Any]], list[int], list[int], Path]:
    data_dir = Path(data_dir)
    test_set_path = data_dir / "Test-set.json"
    split_path = data_dir / "mpi_300_split.json"
    item_key_path = data_dir / "IPIP-NEO-ItemKey.xls"
    if not item_key_path.exists():
        for candidate in ["IPIP-NEO-ItemKey.xlsx", "IPIP-NEO-ItemKey.csv", "IPIP-NEO-ItemKey.json"]:
            candidate_path = data_dir / candidate
            if candidate_path.exists():
                item_key_path = candidate_path
                break

    with open(test_set_path, "r", encoding="utf-8") as f:
        individuals = json.load(f)
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    return individuals, split["train_index"], split["test_index"], item_key_path
