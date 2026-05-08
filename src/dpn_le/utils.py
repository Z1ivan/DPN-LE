"""Utilities shared by DPN-LE scripts."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

import torch


BIG_FIVE_TRAITS = [
    "Agreeableness",
    "Conscientiousness",
    "Extraversion",
    "Neuroticism",
    "Openness",
]


PERSONALITY_PROMPT_TEMPLATE = """You will find a personality description followed by a question below. I want you to fully immerse yourself in the persona described.

###Personality description: {description}

###Question: {question}

###Response:"""


PERSONALITYBENCH_TEST_TEMPLATE = """Imagine you are a real person rather than a language model, and you're asked by the following question. Write your response based on your authentic thoughts and emotions. 

Do not overthink your answer—let your thoughts flow naturally as you write. Focus on expressing your genuine feelings and reactions. Aim to write no more than 300 words.

### Question:
{question}

### Response:"""


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a list of records from a JSON list or a JSONL file."""

    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "records"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Expected JSON list or JSONL records in {path}")


def save_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_personality_prompt(description: str, question: str) -> str:
    return PERSONALITY_PROMPT_TEMPLATE.format(description=description, question=question)


def format_personalitybench_test_prompt(question: str) -> str:
    return PERSONALITYBENCH_TEST_TEMPLATE.format(question=question)


def normalize_answer(text: str) -> str:
    """SQuAD-style answer normalization used by QA evaluations."""

    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    return " ".join(text.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_torch_dtype(value: str | torch.dtype | None) -> torch.dtype | str | None:
    if value is None or isinstance(value, torch.dtype):
        return value
    normalized = value.lower()
    if normalized == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return mapping[normalized]


def validate_samples(samples: list[dict[str, str]]) -> None:
    if not samples:
        raise ValueError("samples is empty")
    required = {"description", "question"}
    for i, sample in enumerate(samples):
        missing = required.difference(sample)
        if missing:
            raise ValueError(f"sample {i} missing keys: {sorted(missing)}")
