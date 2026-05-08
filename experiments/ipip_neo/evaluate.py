#!/usr/bin/env python3
"""Run the IPIP-NEO-300 single-trait evaluation for DPN-LE.

Expected data directory:
  data_dir/
    Test-set.json
    mpi_300_split.json
    IPIP-NEO-ItemKey.xls  (or .xlsx/.csv/.json with the same columns)

The single-trait protocol first estimates each individual's Big Five scores
from the 120 train items. For a target trait, it increases that trait when the
individual's train score is above the threshold, otherwise decreases it, and
then evaluates only the held-out items for that same trait.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dpn_le import DPNLEInference, get_model_config
from dpn_le.ipip_neo import (
    IPIPNEOScorer,
    TRAIT_ABBR_TO_NAME,
    format_ipip_prompt,
    load_ipip_dataset,
    normalize_trait,
    parse_choice,
    parse_individual_responses,
    score_choices,
)


def parse_traits(values: list[str]) -> list[str]:
    if len(values) == 1 and values[0].lower() == "all":
        return list(TRAIT_ABBR_TO_NAME)
    return [normalize_trait(value) for value in values]


def direction_for_score(score: float, threshold: float) -> str:
    return "increase" if score > threshold else "decrease"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    individuals, train_indices, test_indices, item_key_path = load_ipip_dataset(args.data_dir)
    if args.max_individuals:
        individuals = individuals[: args.max_individuals]

    scorer = IPIPNEOScorer(item_key_path)
    traits = parse_traits(args.traits)

    config = get_model_config(args.model)
    effective_quantile = config.quantile if args.quantile is None else args.quantile
    effective_cohens_d = config.cohens_d_threshold if args.cohens_d is None else args.cohens_d
    inference = DPNLEInference(
        args.model,
        config,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )

    all_results: dict[str, list[dict[str, Any]]] = {trait: [] for trait in traits}
    generation_cache: dict[tuple[str, str], dict[str, Any]] = {}

    for trait in traits:
        trait_name = TRAIT_ABBR_TO_NAME[trait]
        item_indices = scorer.trait_test_indices(test_indices, trait)
        prompts = [format_ipip_prompt(scorer.item_text(item_id)) for item_id in item_indices]

        individual_scores = []
        directions_needed = set()
        for individual in individuals:
            responses = parse_individual_responses(individual)
            big5_scores = scorer.calculate_big_five_scores(responses, train_indices)
            trait_score = big5_scores[trait]
            direction = direction_for_score(trait_score, args.threshold)
            individual_scores.append((individual, trait_score, direction))
            directions_needed.add(direction)

        for direction in sorted(directions_needed):
            steering_summary = inference.apply_steering(
                args.steering_data_dir,
                trait=trait_name,
                gamma=args.gamma,
                direction=direction,
                method=args.method,
                neuron_mode=args.neuron_mode,
                weight_range=(args.weight_min, args.weight_max),
                apply_on_prompt=args.apply_on_prompt,
                quantile=effective_quantile,
                cohens_d_threshold=effective_cohens_d,
            )
            raw_outputs = inference.generate_prompts(
                prompts,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
                use_chat_template=args.use_chat_template,
            )
            choices = [parse_choice(output) for output in raw_outputs]
            generation_cache[(trait, direction)] = {
                "choices": choices,
                "raw_outputs": raw_outputs if args.save_raw_outputs else None,
                "steering_summary": steering_summary,
            }

        for individual, trait_score, direction in tqdm(
            individual_scores,
            desc=f"score {trait}",
            leave=False,
        ):
            cache = generation_cache[(trait, direction)]
            scored = score_choices(cache["choices"], individual, item_indices)
            row = {
                "individual_id": individual.get("case"),
                "trait": trait,
                "trait_name": trait_name,
                "train_trait_score": trait_score,
                "direction": direction,
                "mae": scored["mae"],
                "item_indices": item_indices,
                "pred_choices": cache["choices"],
                "pred_scores": scored["pred_scores"],
                "true_scores": scored["true_scores"],
            }
            all_results[trait].append(row)

        trait_mae = float(np.mean([row["mae"] for row in all_results[trait]]))
        print(f"{trait} ({trait_name}): MAE={trait_mae:.4f}, n={len(all_results[trait])}")

    mae_per_trait = {
        trait: float(np.mean([row["mae"] for row in rows]))
        for trait, rows in all_results.items()
    }
    overall_mae = float(np.mean(list(mae_per_trait.values())))
    serializable_generation_cache = {
        f"{trait}_{direction}": value
        for (trait, direction), value in generation_cache.items()
    }
    output = {
        "mode": "single_trait",
        "model": args.model,
        "parameters": {
            "method": args.method,
            "neuron_mode": args.neuron_mode,
            "gamma": args.gamma,
            "quantile": effective_quantile,
            "cohens_d": effective_cohens_d,
            "threshold": args.threshold,
            "layers": [config.target_layers[0], config.target_layers[-1]],
            "weight_range": [args.weight_min, args.weight_max],
            "apply_on_prompt": args.apply_on_prompt,
            "use_chat_template": args.use_chat_template,
        },
        "data": {
            "data_dir": str(args.data_dir),
            "item_key": str(item_key_path),
            "num_individuals": len(individuals),
            "num_train_items": len(train_indices),
            "num_test_items": len(test_indices),
        },
        "mae_per_trait": mae_per_trait,
        "overall_mae": overall_mae,
        "all_results": all_results,
        "generation_cache": serializable_generation_cache if args.save_generation_cache else None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Overall MAE: {overall_mae:.4f}")
    print(f"Saved: {args.output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path")
    parser.add_argument("--steering_data_dir", required=True, type=Path)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--traits", nargs="+", default=["all"], help="all or any of A C E N O")
    parser.add_argument("--method", choices=["linear", "weighted"], default="weighted")
    parser.add_argument(
        "--neuron_mode",
        choices=["directional", "both"],
        default="directional",
        help="IPIP single-trait reproduction uses directional; paper/general capability use both.",
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--quantile", type=float, default=None)
    parser.add_argument("--cohens_d", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=2.8)
    parser.add_argument("--weight_min", type=float, default=0.75)
    parser.add_argument("--weight_max", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=15)
    parser.add_argument("--max_individuals", type=int, default=None)
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--apply_on_prompt", action="store_true")
    parser.add_argument("--save_raw_outputs", action="store_true")
    parser.add_argument("--save_generation_cache", action="store_true")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
