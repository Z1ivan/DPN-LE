"""Paper-aligned vLLM evaluation helpers for general-capability benchmarks."""

from __future__ import annotations

import gc
import json
import re
import string
from collections import Counter
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from tqdm import tqdm

from .model_configs import ModelConfig, get_model_config
from .prepare_steering import compute_exclusive_indices, load_steering_data
from .utils import load_records, save_json


BenchmarkName = str

BENCHMARK_DEFAULTS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "data_file": "gsm8k_test.json",
        "max_samples": 1319,
        "max_tokens": 512,
        "repetition_penalty": 1.1,
        "stop": None,
    },
    "hotpotqa": {
        "data_file": "hotpotqa_validation.json",
        "max_samples": 1000,
        "max_tokens": 50,
        "repetition_penalty": 1.1,
        "stop": None,
    },
    "triviaqa": {
        "data_file": "triviaqa_validation.json",
        "max_samples": 1000,
        "max_tokens": 50,
        "repetition_penalty": 1.1,
        "stop": ["\n", "Question:", "Q:"],
    },
}


def validate_benchmark(benchmark: str) -> str:
    normalized = benchmark.lower()
    if normalized not in BENCHMARK_DEFAULTS:
        raise ValueError(f"Unsupported benchmark '{benchmark}'. Use gsm8k, hotpotqa, or triviaqa.")
    return normalized


def default_general_capability_data_path(repo_root: str | Path, benchmark: str) -> Path:
    benchmark = validate_benchmark(benchmark)
    return Path(repo_root) / "data" / "general_capability" / BENCHMARK_DEFAULTS[benchmark]["data_file"]


def benchmark_prompts(benchmark: str, records: list[dict[str, Any]]) -> list[str]:
    benchmark = validate_benchmark(benchmark)
    if benchmark == "gsm8k":
        return [
            "Solve the following math problem step by step. Show your reasoning and provide "
            "the final numerical answer.\n\n"
            f"Question: {item['question']}\n\n"
            "Solution:"
            for item in records
        ]
    if benchmark == "hotpotqa":
        return [
            "Answer the following question based on the given context. Give a short and "
            "direct answer.\n\n"
            f"Context:\n{str(item.get('context', ''))[:2000]}\n\n"
            f"Question: {item['question']}\n\n"
            "Answer:"
            for item in records
        ]
    return [
        "Answer the following question with a short answer.\n\n"
        f"Question: {item['question']}\n\n"
        "Answer:"
        for item in records
    ]


def normalize_number(num_str: str | None) -> str | None:
    if not num_str:
        return None
    num_str = num_str.replace(",", "")
    try:
        num = float(num_str)
    except ValueError:
        return num_str
    return str(int(num)) if num.is_integer() else str(num)


def extract_gsm8k_gold(text: str) -> str | None:
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    return normalize_number(match.group(1)) if match else None


def extract_gsm8k_prediction(text: str) -> str | None:
    number_pattern = r"-?\d+(?:,\d+)*(?:\.\d+)?"
    match = re.search(rf"####\s*({number_pattern})", text)
    if match:
        return normalize_number(match.group(1))
    match = re.search(rf"(?:answer|result)\s+is\s+({number_pattern})", text, re.IGNORECASE)
    if match:
        return normalize_number(match.group(1))
    numbers = re.findall(rf"({number_pattern})", text)
    return normalize_number(numbers[-1]) if numbers else None


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_hotpotqa_answer(text: str) -> str:
    text = text.strip()
    for prefix in ("Answer:", "The answer is", "A:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    first_line = text.split("\n")[0].strip()
    first_sentence = re.split(r"[.!?]", first_line)[0].strip()
    return first_sentence if first_sentence else first_line


def extract_triviaqa_answer(text: str) -> str:
    first_line = text.strip().split("\n")[0].strip()
    match = re.match(r"^([^.!?(]+)[.!?(]", first_line)
    if match and match.group(1).strip():
        return match.group(1).strip()
    first_sentence = re.split(r"[.!?]", first_line)[0].strip()
    if len(first_sentence) > 100:
        comma_split = first_sentence.split(",")[0].strip()
        if 0 < len(comma_split) < 100:
            return comma_split
    return first_sentence if first_sentence else first_line


def triviaqa_gold_answers(item: dict[str, Any]) -> list[str]:
    answer = item["answer"]
    if isinstance(answer, dict):
        answers = [answer["value"]]
        answers.extend(answer.get("aliases", []))
        return answers
    answers = [answer]
    answers.extend(item.get("aliases", []))
    return answers


def triviaqa_exact_match(prediction: str, ground_truths: list[str]) -> float:
    pred_norm = normalize_answer(prediction)
    for ground_truth in ground_truths:
        gold_norm = normalize_answer(ground_truth)
        if pred_norm == gold_norm:
            return 1.0
        if len(gold_norm) >= 3 and gold_norm in pred_norm:
            return 1.0
        if len(pred_norm) >= 3 and pred_norm in gold_norm:
            return 1.0
    return 0.0


def evaluate_outputs(
    benchmark: str,
    records: list[dict[str, Any]],
    outputs: list[str],
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    benchmark = validate_benchmark(benchmark)
    if len(records) != len(outputs):
        raise ValueError("records and outputs must have the same length")

    if benchmark == "gsm8k":
        correct = 0
        details = []
        for item, output in zip(records, outputs):
            ground_truth = extract_gsm8k_gold(item["answer"])
            predicted = extract_gsm8k_prediction(output)
            is_correct = bool(ground_truth and predicted and ground_truth == predicted)
            correct += int(is_correct)
            details.append(
                {
                    "question": item["question"],
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "correct": is_correct,
                    "model_output": output,
                }
            )
        return {"accuracy": correct / len(records) * 100, "correct": correct, "total": len(records)}, details

    if benchmark == "hotpotqa":
        total_em = 0.0
        total_f1 = 0.0
        details = []
        for item, output in zip(records, outputs):
            ground_truth = item["answer"]
            predicted = extract_hotpotqa_answer(output)
            em = float(normalize_answer(predicted) == normalize_answer(ground_truth))
            f1 = compute_token_f1(predicted, ground_truth)
            total_em += em
            total_f1 += f1
            details.append(
                {
                    "question": item["question"],
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "em": em,
                    "f1": f1,
                    "model_output": output,
                }
            )
        return {
            "exact_match": total_em / len(records) * 100,
            "f1_score": total_f1 / len(records) * 100,
            "total": len(records),
        }, details

    total_em = 0.0
    total_f1 = 0.0
    details = []
    for item, output in zip(records, outputs):
        predicted = extract_triviaqa_answer(output)
        ground_truths = triviaqa_gold_answers(item)
        em = triviaqa_exact_match(predicted, ground_truths)
        f1 = max(compute_token_f1(predicted, answer) for answer in ground_truths)
        total_em += em
        total_f1 += f1
        details.append(
            {
                "question": item["question"],
                "ground_truth": ground_truths[0],
                "aliases": ground_truths[1:],
                "predicted": predicted,
                "em": em,
                "f1": f1,
                "model_output": output,
            }
        )
    return {
        "exact_match": total_em / len(records) * 100,
        "f1_score": total_f1 / len(records) * 100,
        "total": len(records),
    }, details


def _resolve_attr_chain(obj: Any, attrs: tuple[str, ...]) -> Any | None:
    current = obj
    for attr in attrs:
        if not hasattr(current, attr):
            return None
        current = getattr(current, attr)
    return current


def vllm_model_layers(model: Any) -> Any:
    candidates = (
        ("llm_engine", "model_executor", "driver_worker", "model_runner", "model", "model", "layers"),
        ("llm_engine", "model_executor", "model_runner", "model", "model", "layers"),
        ("llm_engine", "model_executor", "model", "model", "layers"),
    )
    for attrs in candidates:
        layers = _resolve_attr_chain(model, attrs)
        if layers is not None:
            return layers
    raise AttributeError("Could not locate decoder layers inside the vLLM LLM object.")


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_cohens_d_weights(
    cohens_d_values: torch.Tensor,
    neuron_indices: torch.Tensor,
    *,
    weight_range: tuple[float, float] = (0.75, 1.0),
) -> torch.Tensor:
    if neuron_indices.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    min_weight, max_weight = weight_range
    abs_d = cohens_d_values[neuron_indices].abs().float()
    if abs_d.numel() == 1:
        return torch.tensor([max_weight], dtype=torch.float32)
    ranks = abs_d.argsort().argsort().float()
    return min_weight + (max_weight - min_weight) * ranks / (abs_d.numel() - 1)


def build_sparse_dpn_delta(
    steering_data: dict[str, Any],
    *,
    intermediate_size: int,
    direction: str,
    method: str,
    neuron_mode: str,
    gamma: float,
    quantile: float,
    cohens_d_threshold: float,
    weight_range: tuple[float, float] = (0.75, 1.0),
) -> torch.Tensor:
    if direction not in {"increase", "decrease"}:
        raise ValueError("direction must be 'increase' or 'decrease'")
    if method not in {"linear", "weighted"}:
        raise ValueError("method must be 'linear' or 'weighted'")
    if neuron_mode == "symmetry":
        neuron_mode = "directional"
    if neuron_mode not in {"both", "directional"}:
        raise ValueError("neuron_mode must be 'both' or 'directional'")

    steering_vector = steering_data["steering_vector"].float()
    cohens_d_values = steering_data.get("cohens_d_values")
    if cohens_d_values is not None:
        high_indices, low_indices = compute_exclusive_indices(
            steering_vector,
            cohens_d_values.float(),
            quantile=quantile,
            cohens_d_threshold=cohens_d_threshold,
        )
    else:
        high_indices = steering_data.get("high_exclusive_indices", torch.tensor([], dtype=torch.long)).long()
        low_indices = steering_data.get("low_exclusive_indices", torch.tensor([], dtype=torch.long)).long()

    if neuron_mode == "both":
        neuron_indices = torch.cat([high_indices.long(), low_indices.long()])
    elif direction == "increase":
        neuron_indices = high_indices.long()
    else:
        neuron_indices = low_indices.long()

    sparse = torch.zeros(intermediate_size, dtype=torch.float32)
    if neuron_indices.numel() == 0:
        return sparse

    signed_vector = steering_vector if direction == "increase" else -steering_vector
    values = signed_vector[neuron_indices]
    if method == "weighted" and cohens_d_values is not None:
        weights = compute_cohens_d_weights(
            cohens_d_values.float(),
            neuron_indices,
            weight_range=weight_range,
        )
        values = values * weights
    sparse[neuron_indices] = gamma * values
    return sparse


def create_dpn_mlp_forward(sparse_delta: torch.Tensor, batch_size: int):
    def forward_with_steering(self, x):
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)

        gate_up_new = gate_up.clone()
        gate_up_new[:, : i // 2] = torch.nn.SiLU()(gate_up[:, : i // 2])
        hidden = gate_up_new[:, : i // 2] * gate_up_new[:, i // 2 :]

        if hidden.shape[0] <= batch_size:
            hidden = hidden + sparse_delta.to(device=hidden.device, dtype=hidden.dtype)

        x, _ = self.down_proj(hidden.contiguous())
        return x

    return forward_with_steering


def apply_dpn_intervention(
    model: Any,
    steering_data_dir: str | Path,
    config: ModelConfig,
    *,
    trait: str,
    gamma: float,
    direction: str = "increase",
    method: str = "weighted",
    neuron_mode: str = "both",
    batch_size: int = 32,
    quantile: float | None = None,
    cohens_d_threshold: float | None = None,
    weight_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    quantile = config.quantile if quantile is None else quantile
    cohens_d_threshold = config.cohens_d_threshold if cohens_d_threshold is None else cohens_d_threshold
    weight_range = config.weight_range if weight_range is None else weight_range

    layers = vllm_model_layers(model)
    per_layer = []
    total_neurons = 0
    for layer_idx in config.target_layers:
        steering_data = load_steering_data(steering_data_dir, trait, layer_idx)
        sparse_delta = build_sparse_dpn_delta(
            steering_data,
            intermediate_size=config.intermediate_size,
            direction=direction,
            method=method,
            neuron_mode=neuron_mode,
            gamma=gamma,
            quantile=quantile,
            cohens_d_threshold=cohens_d_threshold,
            weight_range=weight_range,
        )
        neuron_count = int((sparse_delta != 0).sum().item())
        total_neurons += neuron_count
        mlp = layers[layer_idx].mlp
        sparse_delta = sparse_delta.to(_module_device(mlp))
        mlp.forward = MethodType(create_dpn_mlp_forward(sparse_delta, batch_size), mlp)
        per_layer.append({"layer": layer_idx, "neurons": neuron_count})

    denominator = len(config.target_layers) * config.intermediate_size
    return {
        "intervention": "dpn",
        "trait": trait,
        "direction": direction,
        "method": method,
        "neuron_mode": neuron_mode,
        "gamma": gamma,
        "quantile": quantile,
        "cohens_d_threshold": cohens_d_threshold,
        "weight_range": list(weight_range),
        "target_layers": config.target_layers,
        "total_neurons": total_neurons,
        "percentage_of_target_mlp_neurons": 100.0 * total_neurons / denominator if denominator else 0.0,
        "per_layer": per_layer,
    }


def load_npti_neurons(neuron_path: str | Path) -> dict[str, torch.Tensor]:
    neuron_path = Path(neuron_path)
    if not neuron_path.exists():
        raise FileNotFoundError(f"NPTI neuron file not found: {neuron_path}")
    with open(neuron_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {layer: torch.tensor(values) for layer, values in raw.items()}


def create_npti_neuron_modifier(
    layer_idx: int,
    neurons_to_activate: dict[str, torch.Tensor],
    neurons_to_deactivate: dict[str, torch.Tensor],
    gamma: float,
    batch_size: int,
):
    def forward_with_modification(self, x):
        def custom_function(delta):
            return 1 / (1 + torch.exp(-10 * (delta - 0.15)))

        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)
        max_idx = i // 2

        gate_up[:, :max_idx] = torch.nn.SiLU()(gate_up[:, :max_idx])

        if x.shape[0] <= batch_size and str(layer_idx) in neurons_to_activate:
            elements = neurons_to_activate[str(layer_idx)]
            indices_all = elements[:, 1].long()
            values_all = elements[:, 4]
            difference_all = elements[:, 2]
            valid_positions = torch.nonzero(
                (indices_all >= 0) & (indices_all < max_idx),
                as_tuple=False,
            ).squeeze(-1)

            if valid_positions.numel() > 0:
                indices = torch.index_select(indices_all, 0, valid_positions).to(gate_up.device)
                values = torch.index_select(values_all, 0, valid_positions).to(
                    device=gate_up.device,
                    dtype=gate_up.dtype,
                )
                difference = torch.index_select(difference_all, 0, valid_positions).to(
                    device=gate_up.device,
                    dtype=gate_up.dtype,
                )

                if str(layer_idx) in neurons_to_deactivate:
                    deactivation = neurons_to_deactivate[str(layer_idx)]
                    deactivation_indices_all = deactivation[:, 1].long()
                    valid_deactivation = torch.nonzero(
                        (deactivation_indices_all >= 0) & (deactivation_indices_all < max_idx),
                        as_tuple=False,
                    ).squeeze(-1)
                    if valid_deactivation.numel() > 0:
                        deactivation_indices = torch.index_select(
                            deactivation_indices_all,
                            0,
                            valid_deactivation,
                        ).to(gate_up.device)
                        gate_up[:, deactivation_indices] = torch.minimum(
                            gate_up[:, deactivation_indices],
                            torch.tensor(0.0, device=gate_up.device, dtype=gate_up.dtype),
                        )

                delta_vals = values * gamma * custom_function(difference)
                gate_up[:, indices] += delta_vals.unsqueeze(0).expand(gate_up.size(0), -1)

        hidden = gate_up[:, :max_idx] * gate_up[:, max_idx:]
        x, _ = self.down_proj(hidden)
        return x

    return forward_with_modification


def apply_npti_intervention(
    model: Any,
    neuron_dir: str | Path,
    *,
    trait: str,
    direction: str = "increase",
    gamma: float = 1.4,
    batch_size: int = 32,
) -> dict[str, Any]:
    if direction not in {"increase", "decrease"}:
        raise ValueError("direction must be 'increase' or 'decrease'")
    neuron_dir = Path(neuron_dir)
    if direction == "increase":
        activate_name = trait
        deactivate_name = f"{trait}_reversed"
    else:
        activate_name = f"{trait}_reversed"
        deactivate_name = trait

    neurons_to_activate = load_npti_neurons(neuron_dir / f"{activate_name}_dict.json")
    neurons_to_deactivate = load_npti_neurons(neuron_dir / f"{deactivate_name}_dict.json")
    layers = vllm_model_layers(model)
    for layer_idx, layer in enumerate(layers):
        layer.mlp.forward = MethodType(
            create_npti_neuron_modifier(
                layer_idx,
                neurons_to_activate,
                neurons_to_deactivate,
                gamma,
                batch_size,
            ),
            layer.mlp,
        )

    return {
        "intervention": "npti",
        "trait": trait,
        "direction": direction,
        "gamma": gamma,
        "activate_file": f"{activate_name}_dict.json",
        "deactivate_file": f"{deactivate_name}_dict.json",
        "patched_layers": len(layers),
    }


def create_vllm_model(
    model_name: str,
    *,
    tensor_parallel_size: int = 1,
    enforce_eager: bool = True,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int | None = None,
    dtype: str = "auto",
    trust_remote_code: bool = True,
):
    try:
        from vllm import LLM
    except ImportError as exc:
        raise ImportError(
            "vLLM is required for paper-aligned general-capability evaluation. "
            "Install with: pip install -e '.[vllm]'"
        ) from exc

    kwargs: dict[str, Any] = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "enforce_eager": enforce_eager,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if dtype:
        kwargs["dtype"] = dtype
    return LLM(**kwargs)


def create_sampling_params(
    benchmark: str,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int | None = None,
    repetition_penalty: float | None = None,
):
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise ImportError(
            "vLLM is required for paper-aligned general-capability evaluation. "
            "Install with: pip install -e '.[vllm]'"
        ) from exc

    benchmark = validate_benchmark(benchmark)
    defaults = BENCHMARK_DEFAULTS[benchmark]
    kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": defaults["max_tokens"] if max_tokens is None else max_tokens,
        "repetition_penalty": (
            defaults["repetition_penalty"] if repetition_penalty is None else repetition_penalty
        ),
    }
    if defaults["stop"] is not None:
        kwargs["stop"] = defaults["stop"]
    return SamplingParams(**kwargs)


def generate_vllm_outputs(
    model: Any,
    benchmark: str,
    records: list[dict[str, Any]],
    sampling_params: Any,
    *,
    batch_size: int = 32,
) -> list[str]:
    prompts = benchmark_prompts(benchmark, records)
    outputs: list[str] = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="inference"):
        batch_prompts = prompts[start : start + batch_size]
        batch_outputs = model.generate(batch_prompts, sampling_params)
        outputs.extend([output.outputs[0].text for output in batch_outputs])
    return outputs


def run_vllm_general_capability(
    *,
    benchmark: str,
    model_name: str,
    data_path: str | Path,
    output_path: str | Path | None = None,
    intervention: str = "baseline",
    model_config: ModelConfig | None = None,
    steering_data_dir: str | Path | None = None,
    neuron_dir: str | Path | None = None,
    trait: str | None = None,
    direction: str = "increase",
    method: str = "weighted",
    neuron_mode: str = "both",
    gamma: float | None = None,
    quantile: float | None = None,
    cohens_d_threshold: float | None = None,
    weight_range: tuple[float, float] = (0.75, 1.0),
    num_samples: int | None = None,
    batch_size: int = 32,
    tensor_parallel_size: int = 1,
    enforce_eager: bool = True,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int | None = None,
    dtype: str = "auto",
    trust_remote_code: bool = True,
    max_tokens: int | None = None,
    repetition_penalty: float | None = None,
) -> dict[str, Any]:
    benchmark = validate_benchmark(benchmark)
    intervention = intervention.lower()
    if intervention not in {"baseline", "dpn", "npti"}:
        raise ValueError("intervention must be baseline, dpn, or npti")

    defaults = BENCHMARK_DEFAULTS[benchmark]
    limit = defaults["max_samples"] if num_samples is None else num_samples
    records = load_records(data_path)[:limit]

    model = create_vllm_model(
        model_name,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    intervention_config: dict[str, Any]
    if intervention == "baseline":
        intervention_config = {"intervention": "baseline"}
    elif intervention == "dpn":
        if steering_data_dir is None:
            raise ValueError("steering_data_dir is required for DPN intervention")
        if trait is None:
            raise ValueError("trait is required for DPN intervention")
        model_config = model_config or get_model_config(model_name)
        intervention_config = apply_dpn_intervention(
            model,
            steering_data_dir,
            model_config,
            trait=trait,
            gamma=0.8 if gamma is None else gamma,
            direction=direction,
            method=method,
            neuron_mode=neuron_mode,
            batch_size=batch_size,
            quantile=quantile,
            cohens_d_threshold=cohens_d_threshold,
            weight_range=weight_range,
        )
    else:
        if neuron_dir is None:
            raise ValueError("neuron_dir is required for NPTI intervention")
        if trait is None:
            raise ValueError("trait is required for NPTI intervention")
        intervention_config = apply_npti_intervention(
            model,
            neuron_dir,
            trait=trait,
            direction=direction,
            gamma=1.4 if gamma is None else gamma,
            batch_size=batch_size,
        )

    sampling_params = create_sampling_params(
        benchmark,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )
    outputs = generate_vllm_outputs(model, benchmark, records, sampling_params, batch_size=batch_size)
    metrics, details = evaluate_outputs(benchmark, records, outputs)

    summary = {
        "benchmark": benchmark,
        "model": model_name,
        "data_path": str(data_path),
        "intervention": intervention_config,
        "metrics": metrics,
        "results": details,
    }
    if output_path is not None:
        save_json(summary, output_path)

    del model
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    return summary
