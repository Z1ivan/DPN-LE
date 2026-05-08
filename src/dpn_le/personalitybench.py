"""PersonalityBench/NPTI data loading helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .utils import BIG_FIVE_TRAITS


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_descriptions(dataset_dir: str | Path, trait: str) -> tuple[list[str], list[str]]:
    """Load the 80 high and 80 low/reversed descriptions for a trait."""

    dataset_dir = Path(dataset_dir)
    path = dataset_dir / "description.json"
    if not path.exists():
        raise FileNotFoundError(f"PersonalityBench description file not found: {path}")

    high: list[str] = []
    low: list[str] = []
    reversed_key = f"{trait}_reversed"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if trait in row:
                high.append(row[trait])
            if reversed_key in row:
                low.append(row[reversed_key])

    if not high or not low:
        raise ValueError(f"No descriptions found for {trait} and {reversed_key}")
    return high, low


def load_search_questions(
    dataset_dir: str | Path,
    trait: str,
    *,
    num_samples: int = 1000,
) -> list[str]:
    """Load the first N PersonalityBench search questions for a trait."""

    path = Path(dataset_dir) / "search" / f"{trait}.json"
    if not path.exists():
        raise FileNotFoundError(f"PersonalityBench search split not found: {path}")
    rows = _read_jsonl(path)
    return [row["question"] for row in rows[:num_samples]]


def build_contrastive_samples(
    dataset_dir: str | Path,
    trait: str,
    *,
    num_samples: int = 1000,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build the 1,000 high/low contrastive pairs described in the appendix."""

    if trait not in BIG_FIVE_TRAITS:
        raise ValueError(f"Unknown trait {trait}. Expected one of {BIG_FIVE_TRAITS}")
    questions = load_search_questions(dataset_dir, trait, num_samples=num_samples)
    high_desc, low_desc = load_descriptions(dataset_dir, trait)
    rng = random.Random(seed)
    high_samples = [
        {"description": rng.choice(high_desc), "question": question}
        for question in questions
    ]
    low_samples = [
        {"description": rng.choice(low_desc), "question": question}
        for question in questions
    ]
    return high_samples, low_samples


def load_test_questions(dataset_dir: str | Path, trait: str) -> list[str]:
    path = Path(dataset_dir) / "test" / f"{trait}.json"
    if not path.exists():
        raise FileNotFoundError(f"PersonalityBench test split not found: {path}")
    return [row["question"] for row in _read_jsonl(path)]
