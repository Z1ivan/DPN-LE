#!/usr/bin/env python3
"""Paper-aligned vLLM NPTI preliminary general-capability evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpn_le.vllm_general_capability import (
    default_general_capability_data_path,
    run_vllm_general_capability,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paper-aligned vLLM NPTI preliminary capability evaluation."
    )
    parser.add_argument("--benchmark", required=True, choices=["gsm8k", "hotpotqa", "triviaqa"])
    parser.add_argument("--model", required=True, help="Model name or local model path.")
    parser.add_argument("--data", help="Benchmark data path. Defaults to data/general_capability/.")
    parser.add_argument("--output", help="Output JSON path.")
    parser.add_argument("--baseline_only", action="store_true", help="Run without NPTI intervention.")

    parser.add_argument(
        "--neuron_dir",
        default=str(REPO_ROOT / "data" / "npti_neuron_results"),
        help="Directory containing NPTI *_dict.json neuron dictionaries.",
    )
    parser.add_argument("--trait", help="Big Five trait name.")
    parser.add_argument("--direction", default="increase", choices=["increase", "decrease"])
    parser.add_argument("--gamma", type=float, default=1.4)

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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    intervention = "baseline" if args.baseline_only else "npti"
    if intervention == "npti" and not args.trait:
        parser.error("--trait is required unless --baseline_only is set")

    data_path = Path(args.data) if args.data else default_general_capability_data_path(
        REPO_ROOT,
        args.benchmark,
    )
    output_path = Path(args.output) if args.output else (
        REPO_ROOT
        / "outputs"
        / "npti_general_capability"
        / f"{args.benchmark}_{intervention}.json"
    )

    result = run_vllm_general_capability(
        benchmark=args.benchmark,
        model_name=args.model,
        data_path=data_path,
        output_path=output_path,
        intervention=intervention,
        neuron_dir=args.neuron_dir,
        trait=args.trait,
        direction=args.direction,
        gamma=args.gamma,
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
