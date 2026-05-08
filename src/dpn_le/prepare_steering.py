"""Steering-vector construction and dual-criterion neuron selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .model_configs import ModelConfig


def compute_cohens_d(high_acts: torch.Tensor, low_acts: torch.Tensor) -> torch.Tensor:
    """Compute per-neuron Cohen's d as defined in the paper."""

    high = high_acts.float()
    low = low_acts.float()
    if high.ndim != 2 or low.ndim != 2:
        raise ValueError("high_acts and low_acts must be [n_samples, n_neurons]")
    if high.shape[1] != low.shape[1]:
        raise ValueError("high_acts and low_acts must have the same neuron dimension")
    if high.shape[0] < 2 or low.shape[0] < 2:
        raise ValueError("Cohen's d requires at least two samples per group")

    high_mean = high.mean(dim=0)
    low_mean = low.mean(dim=0)
    high_var = high.var(dim=0, unbiased=True)
    low_var = low.var(dim=0, unbiased=True)
    n_high = high.shape[0]
    n_low = low.shape[0]
    pooled_var = ((n_high - 1) * high_var + (n_low - 1) * low_var) / (n_high + n_low - 2)
    pooled_std = torch.sqrt(pooled_var.clamp_min(0.0) + 1e-8)
    return torch.nan_to_num((high_mean - low_mean) / pooled_std)


def compute_exclusive_indices(
    steering_vector: torch.Tensor,
    cohens_d: torch.Tensor,
    *,
    quantile: float = 0.995,
    cohens_d_threshold: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select high- and low-trait exclusive neurons.

    A selected neuron must satisfy both criteria from the paper:
    ``abs(steering_vector) > tau_q`` and ``abs(Cohen's d) > tau_d``.
    """

    if steering_vector.ndim != 1 or cohens_d.ndim != 1:
        raise ValueError("steering_vector and cohens_d must be one-dimensional")
    if steering_vector.shape != cohens_d.shape:
        raise ValueError("steering_vector and cohens_d must have the same shape")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if cohens_d_threshold < 0:
        raise ValueError("cohens_d_threshold must be non-negative")

    abs_steering = steering_vector.abs()
    magnitude_threshold = torch.quantile(abs_steering.float(), quantile)
    high_mask = (abs_steering > magnitude_threshold) & (cohens_d > cohens_d_threshold)
    low_mask = (abs_steering > magnitude_threshold) & (cohens_d < -cohens_d_threshold)
    return torch.where(high_mask)[0], torch.where(low_mask)[0]


class SteeringDataPreparer:
    """Prepare DPN-LE steering data from cached MLP activations."""

    def __init__(self, model_config: ModelConfig):
        self.config = model_config

    def prepare_layer(
        self,
        layer_idx: int,
        high_acts: torch.Tensor,
        low_acts: torch.Tensor,
        *,
        quantile: float | None = None,
        cohens_d_threshold: float | None = None,
    ) -> dict[str, Any]:
        quantile = self.config.quantile if quantile is None else quantile
        cohens_d_threshold = (
            self.config.cohens_d_threshold if cohens_d_threshold is None else cohens_d_threshold
        )

        steering_vector = high_acts.float().mean(dim=0) - low_acts.float().mean(dim=0)
        cohens_d_values = compute_cohens_d(high_acts, low_acts)
        high_indices, low_indices = compute_exclusive_indices(
            steering_vector,
            cohens_d_values,
            quantile=quantile,
            cohens_d_threshold=cohens_d_threshold,
        )

        return {
            "layer": layer_idx,
            "steering_vector": steering_vector.cpu(),
            "cohens_d_values": cohens_d_values.cpu(),
            "high_exclusive_indices": high_indices.cpu(),
            "low_exclusive_indices": low_indices.cpu(),
            "quantile": float(quantile),
            "cohens_d_threshold": float(cohens_d_threshold),
            "num_high_exclusive": int(high_indices.numel()),
            "num_low_exclusive": int(low_indices.numel()),
            "num_total_exclusive": int(high_indices.numel() + low_indices.numel()),
        }

    def prepare(
        self,
        activations_dir: str | Path,
        trait: str,
        output_dir: str | Path,
        *,
        quantile: float | None = None,
        cohens_d_threshold: float | None = None,
    ) -> dict[str, Any]:
        activations_dir = Path(activations_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "trait": trait,
            "model": self.config.model_name,
            "layers": self.config.target_layers,
            "per_layer": [],
            "total_neurons": 0,
        }

        for layer_idx in self.config.target_layers:
            high_path = activations_dir / trait / "high" / f"layer_{layer_idx}.pt"
            low_path = activations_dir / trait / "low" / f"layer_{layer_idx}.pt"
            if not high_path.exists() or not low_path.exists():
                raise FileNotFoundError(
                    f"Missing activations for layer {layer_idx}: {high_path} / {low_path}"
                )

            high_acts = torch.load(high_path, map_location="cpu")
            low_acts = torch.load(low_path, map_location="cpu")
            data = self.prepare_layer(
                layer_idx,
                high_acts,
                low_acts,
                quantile=quantile,
                cohens_d_threshold=cohens_d_threshold,
            )

            torch_path = output_dir / f"layer{layer_idx}_{trait}_steering_data.pt"
            torch.save(data, torch_path)

            inspection = {
                "trait": trait,
                "layer": layer_idx,
                "steering_vector_norm": float(data["steering_vector"].norm().item()),
                "cohens_d_abs_mean": float(data["cohens_d_values"].abs().mean().item()),
                "cohens_d_abs_max": float(data["cohens_d_values"].abs().max().item()),
                "high_exclusive_indices": data["high_exclusive_indices"].tolist(),
                "low_exclusive_indices": data["low_exclusive_indices"].tolist(),
                "num_high_exclusive": data["num_high_exclusive"],
                "num_low_exclusive": data["num_low_exclusive"],
                "num_total_exclusive": data["num_total_exclusive"],
                "quantile": data["quantile"],
                "cohens_d_threshold": data["cohens_d_threshold"],
            }
            json_path = output_dir / f"layer{layer_idx}_{trait}_steering_data.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(inspection, f, ensure_ascii=False, indent=2)

            summary["per_layer"].append(inspection)
            summary["total_neurons"] += data["num_total_exclusive"]

        denominator = len(self.config.target_layers) * self.config.intermediate_size
        summary["percentage_of_target_mlp_neurons"] = (
            100.0 * summary["total_neurons"] / denominator if denominator else 0.0
        )
        with open(output_dir / f"{trait}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary


def _tensorize_steering_json(data: dict[str, Any]) -> dict[str, Any]:
    tensor_keys = {
        "steering_vector": torch.float32,
        "cohens_d_values": torch.float32,
        "high_exclusive_indices": torch.long,
        "low_exclusive_indices": torch.long,
    }
    for key, dtype in tensor_keys.items():
        if key in data and not isinstance(data[key], torch.Tensor):
            data[key] = torch.tensor(data[key], dtype=dtype)
    return data


def load_steering_data(steering_data_dir: str | Path, trait: str, layer: int) -> dict[str, Any]:
    steering_data_dir = Path(steering_data_dir)
    candidates = [
        steering_data_dir / f"layer{layer}_{trait}_steering_data.pt",
        steering_data_dir / f"layer{layer}_{trait}_steering_data_origin.pt",
    ]
    for path in candidates:
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=False)

    json_candidates = [
        steering_data_dir / f"layer{layer}_{trait}_steering_data.json",
        steering_data_dir / f"layer{layer}_{trait}_steering_data_origin.json",
    ]
    for path in json_candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return _tensorize_steering_json(json.load(f))

    searched = ", ".join(str(path) for path in candidates + json_candidates)
    raise FileNotFoundError(f"Steering data not found. Searched: {searched}")
