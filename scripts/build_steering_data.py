#!/usr/bin/env python3
"""Build DPN-LE steering data from PersonalityBench/NPTI data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dpn_le import DPNLEPipeline, get_model_config
from dpn_le.personalitybench import build_contrastive_samples
from dpn_le.utils import BIG_FIVE_TRAITS, set_seed


def parse_traits(values: list[str]) -> list[str]:
    if len(values) == 1 and values[0].lower() == "all":
        return BIG_FIVE_TRAITS
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--personalitybench_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--traits", nargs="+", default=["all"])
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--quantile", type=float, default=None)
    parser.add_argument("--cohens_d", type=float, default=None)
    parser.add_argument("--torch_dtype", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    config = get_model_config(args.model)
    pipeline = DPNLEPipeline(args.model, config, torch_dtype=args.torch_dtype)
    activations_dir = args.output_dir / "activations"
    steering_dir = args.output_dir / "steering_data"

    for trait in parse_traits(args.traits):
        print(f"\n=== {trait} ===")
        high_samples, low_samples = build_contrastive_samples(
            args.personalitybench_dir,
            trait,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        pipeline.extract_activations(
            high_samples,
            low_samples,
            trait=trait,
            output_dir=activations_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_chat_template=args.use_chat_template,
            system_prompt=args.system_prompt,
        )
        summary = pipeline.prepare_steering(
            activations_dir,
            trait=trait,
            output_dir=steering_dir,
            quantile=args.quantile,
            cohens_d_threshold=args.cohens_d,
        )
        print(
            f"Selected {summary['total_neurons']} neurons "
            f"({summary['percentage_of_target_mlp_neurons']:.3f}% of target MLP neurons)"
        )

    print(f"\nSteering data saved to: {steering_dir}")


if __name__ == "__main__":
    main()
