#!/usr/bin/env python3
"""Paper-aligned vLLM general-capability evaluation for DPN-LE."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpn_le.model_configs import get_model_config
from dpn_le.vllm_general_capability import (
    default_general_capability_data_path,
    run_vllm_general_capability,
)


def build_parser(default_benchmark: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paper-aligned vLLM general-capability evaluation."
    )
    if default_benchmark is None:
        parser.add_argument(
            "--benchmark",
            required=True,
            choices=["gsm8k", "hotpotqa", "triviaqa"],
            help="Benchmark to evaluate.",
        )
    else:
        parser.add_argument(
            "--benchmark",
            default=default_benchmark,
            choices=["gsm8k", "hotpotqa", "triviaqa"],
            help="Benchmark to evaluate.",
        )
    parser.add_argument("--model", required=True, help="Model name or local model path.")
    parser.add_argument("--data", help="Benchmark data path. Defaults to data/general_capability/.")
    parser.add_argument("--output", help="Output JSON path.")
    parser.add_argument("--baseline_only", action="store_true", help="Run without DPN-LE intervention.")

    parser.add_argument("--steering_data_dir", help="Directory containing DPN-LE steering .pt files.")
    parser.add_argument("--trait", help="Big Five trait name.")
    parser.add_argument("--direction", default="increase", choices=["increase", "decrease"])
    parser.add_argument("--method", default="weighted", choices=["linear", "weighted"])
    parser.add_argument("--neuron_mode", default="both", choices=["both", "directional"])
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--quantile", type=float)
    parser.add_argument("--cohens_d", type=float)
    parser.add_argument("--layer_start", type=int)
    parser.add_argument("--layer_end", type=int, help="Inclusive target layer end.")

    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--no_enforce_eager", action="store_true")
    parser.add_argument("--no_trust_remote_code", action="store_true")
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--repetition_penalty", type=float)
    return parser


def _model_config_from_args(args: argparse.Namespace):
    config = get_model_config(args.model)
    if args.layer_start is None and args.layer_end is None:
        return config
    start = config.target_layers[0] if args.layer_start is None else args.layer_start
    end = config.target_layers[-1] if args.layer_end is None else args.layer_end
    if start > end:
        raise ValueError("--layer_start must be <= --layer_end")
    return replace(config, target_layers=list(range(start, end + 1)))


def main(default_benchmark: str | None = None) -> None:
    parser = build_parser(default_benchmark)
    args = parser.parse_args()

    intervention = "baseline" if args.baseline_only else "dpn"
    if intervention == "dpn":
        if not args.steering_data_dir:
            parser.error("--steering_data_dir is required unless --baseline_only is set")
        if not args.trait:
            parser.error("--trait is required unless --baseline_only is set")

    data_path = Path(args.data) if args.data else default_general_capability_data_path(
        REPO_ROOT,
        args.benchmark,
    )
    output_path = Path(args.output) if args.output else (
        REPO_ROOT
        / "outputs"
        / "general_capability"
        / f"{args.benchmark}_{intervention}.json"
    )

    config = _model_config_from_args(args) if intervention == "dpn" else None
    result = run_vllm_general_capability(
        benchmark=args.benchmark,
        model_name=args.model,
        data_path=data_path,
        output_path=output_path,
        intervention=intervention,
        model_config=config,
        steering_data_dir=args.steering_data_dir,
        trait=args.trait,
        direction=args.direction,
        method=args.method,
        neuron_mode=args.neuron_mode,
        gamma=args.gamma,
        quantile=args.quantile,
        cohens_d_threshold=args.cohens_d,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=not args.no_enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=not args.no_trust_remote_code,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    print(f"Saved results to {output_path}")
    print(result["metrics"])


if __name__ == "__main__":
    main()
