#!/usr/bin/env python3
"""Score PersonalityBench generations with an OpenAI-compatible chat API."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

TRAIT_FACTORS = {
    "Openness": "imagination, artistic interests, emotionality, adventurousness, intellect, liberalism",
    "Conscientiousness": "self-efficacy, orderliness, dutifulness, achievement-striving, self-discipline, cautiousness",
    "Extraversion": "friendliness, gregariousness, assertiveness, activity level, excitement-seeking, cheerfulness",
    "Agreeableness": "trust, morality, altruism, cooperation, modesty, sympathy",
    "Neuroticism": "anxiety, anger, depression, self-consciousness, immoderation, vulnerability",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load rows from evaluate.py JSON summaries or JSONL response files."""

    if path.suffix.lower() == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("responses"), list):
        return data["responses"]
    raise ValueError("Input must be a JSONL file, a JSON list, or an evaluate.py JSON summary")


def trait_prompt(trait: str, question: str, answer: str) -> str:
    factors = TRAIT_FACTORS[trait]
    return f"""[System]
Please act as an impartial judge and evaluate the personality trait of {trait} exhibited in the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as {factors} demonstrated in the response. Begin your evaluation by providing a short explanation of no more than 30 words. Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 5, where a higher score indicates a stronger reflection of the specified trait.
Please follow this format for your rating: "[[rating]]", for example: "Rating: [[5]]".
[Question]
{question}
[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""


def fluency_prompt(answer: str) -> str:
    return f"""[System]
Please act as an impartial judge and evaluate the fluency of the provided answer within a conversation. Focus solely on how smoothly and naturally the thoughts and emotions are conveyed, ignoring aspects like informal language use and the length of the response. Begin with a concise, objective explanation (no more than 30 words), and then assign a rating on a scale of 1 to 5, where a higher score indicates better fluency.
Format your rating as follows: "Rating: [[rating]]" (e.g., "Rating: [[5]]").
[The Start of the Answer]
{answer}
[The End of the Answer]"""


def parse_rating(text: str) -> int | None:
    match = re.search(r"Rating:\s*\[\[(\d+)\]\]", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    matches = re.findall(r"\[\[(\d+)\]\]", text)
    return int(matches[-1]) if matches else None


def make_client(api_key_env: str, base_url: str | None):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install scoring dependencies with: pip install -e '.[scoring]'") from exc

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {api_key_env}")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def score_rows(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(args.input)
    client = make_client(args.api_key_env, args.base_url)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if args.resume and args.output.exists():
        with open(args.output, "r", encoding="utf-8") as f:
            completed = sum(1 for line in f if line.strip())

    mode = "a" if completed else "w"
    ratings = []
    with open(args.output, mode, encoding="utf-8") as out:
        for index, row in enumerate(rows[completed:], start=completed):
            question = row.get("question", "")
            answer = row.get("answer", "")
            if args.mode == "trait":
                if args.trait is None:
                    raise ValueError("--trait is required for trait scoring")
                prompt = trait_prompt(args.trait, question, answer)
            else:
                prompt = fluency_prompt(answer)

            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            score_text = response.choices[0].message.content or ""
            rating = parse_rating(score_text)
            ratings.append(rating)
            out.write(
                json.dumps(
                    {
                        "index": index,
                        "question": question,
                        "answer": answer,
                        "score_text": score_text,
                        "rating": rating,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()

    valid = [value for value in ratings if value is not None]
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "mode": args.mode,
        "trait": args.trait,
        "model": args.model,
        "count_scored_this_run": len(ratings),
        "valid_ratings_this_run": len(valid),
        "mean_rating_this_run": sum(valid) / len(valid) if valid else None,
    }
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary_output", type=Path)
    parser.add_argument("--mode", choices=["trait", "fluency"], default="trait")
    parser.add_argument("--trait", choices=sorted(TRAIT_FACTORS))
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    summary = score_rows(build_parser().parse_args())
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
