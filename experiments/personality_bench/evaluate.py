#!/usr/bin/env python3
"""Generate DPN-LE responses for PersonalityBench.

The paper reports GPT-4o trait/fluency scores for these generations. This
script produces the model responses and records the exact DPN-LE settings; it
does not call a proprietary scorer by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dpn_le import DPNLEInference, get_model_config
from dpn_le.utils import load_jsonl


def evaluate_personalitybench(args: argparse.Namespace) -> dict:
    test_data = load_jsonl(args.data)
    questions = [item["question"] for item in test_data]
    config = get_model_config(args.model)
    inference = DPNLEInference(args.model, config, torch_dtype=args.torch_dtype)
    steering_summary = inference.apply_steering(
        args.steering_data_dir,
        trait=args.trait,
        gamma=args.gamma,
        direction=args.direction,
        method=args.method,
        neuron_mode=args.neuron_mode,
        weight_range=(args.weight_min, args.weight_max),
        quantile=args.quantile,
        cohens_d_threshold=args.cohens_d,
    )
    results = inference.generate_questions(
        questions,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        output_path=args.output_jsonl,
    )
    payload = {
        "model": args.model,
        "trait": args.trait,
        "direction": args.direction,
        "method": args.method,
        "neuron_mode": args.neuron_mode,
        "gamma": args.gamma,
        "quantile": args.quantile,
        "cohens_d": args.cohens_d,
        "steering_summary": steering_summary,
        "total_questions": len(results),
        "responses": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(results)} responses")
    print(f"Saved JSON summary: {args.output}")
    if args.output_jsonl:
        print(f"Saved response JSONL: {args.output_jsonl}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path, help="PersonalityBench test JSONL")
    parser.add_argument("--steering_data_dir", required=True, type=Path)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--direction", default="increase", choices=["increase", "decrease"])
    parser.add_argument("--method", default="weighted", choices=["linear", "weighted"])
    parser.add_argument("--neuron_mode", default="both", choices=["both", "directional"])
    parser.add_argument("--quantile", type=float, default=None)
    parser.add_argument("--cohens_d", type=float, default=None)
    parser.add_argument("--weight_min", type=float, default=0.75)
    parser.add_argument("--weight_max", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output_jsonl", type=Path)
    return parser


def main() -> None:
    evaluate_personalitybench(build_parser().parse_args())


if __name__ == "__main__":
    main()
