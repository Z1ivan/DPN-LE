"""Inference-time sparse MLP intervention for DPN-LE."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .extract_activations import _model_layers
from .model_configs import ModelConfig, get_model_config
from .prepare_steering import compute_exclusive_indices, load_steering_data
from .utils import (
    format_personalitybench_test_prompt,
    parse_torch_dtype,
    save_jsonl,
)


def compute_cohens_d_weights(
    cohens_d_values: torch.Tensor,
    neuron_indices: torch.Tensor,
    *,
    weight_range: tuple[float, float] = (0.75, 1.0),
) -> torch.Tensor:
    """Rank selected neurons by abs(Cohen's d) and map ranks to weights."""

    if neuron_indices.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    selected_d = cohens_d_values[neuron_indices].abs().float()
    min_w, max_w = weight_range
    if selected_d.numel() == 1:
        return torch.full_like(selected_d, max_w)
    ranks = selected_d.argsort().argsort().float()
    return min_w + (max_w - min_w) * ranks / (selected_d.numel() - 1)


class DPNLEInference:
    """Apply DPN-LE or DPN-LE_w during Transformers generation."""

    def __init__(
        self,
        model_name: str,
        model_config: Optional[ModelConfig] = None,
        *,
        model=None,
        tokenizer=None,
        device: str | None = None,
        device_map: str | dict[str, int] | None = "auto",
        torch_dtype: str | torch.dtype | None = "auto",
        trust_remote_code: bool = True,
        model_kwargs: Optional[dict[str, Any]] = None,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model_name = model_name
        self.config = model_config or get_model_config(model_name)
        self._hooks: list[Any] = []

        if tokenizer is None:
            tokenizer_kwargs = tokenizer_kwargs or {}
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code, **tokenizer_kwargs
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer

        if model is None:
            model_kwargs = model_kwargs or {}
            if device is not None and device_map == "auto":
                device_map = {"": device}
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=parse_torch_dtype(torch_dtype),
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
        self.model = model.eval()

    def reset(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _down_proj(self, layer_idx: int):
        layer = _model_layers(self.model)[layer_idx]
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "down_proj"):
            raise AttributeError(f"Layer {layer_idx} does not expose mlp.down_proj")
        return layer.mlp.down_proj

    def _build_sparse_delta(
        self,
        steering_data: dict[str, Any],
        *,
        direction: str,
        method: str,
        neuron_mode: str,
        weight_range: tuple[float, float],
        quantile: float | None = None,
        cohens_d_threshold: float | None = None,
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
        if quantile is not None or cohens_d_threshold is not None:
            high_indices, low_indices = compute_exclusive_indices(
                steering_vector,
                steering_data["cohens_d_values"].float(),
                quantile=quantile if quantile is not None else steering_data["quantile"],
                cohens_d_threshold=(
                    cohens_d_threshold
                    if cohens_d_threshold is not None
                    else steering_data["cohens_d_threshold"]
                ),
            )
        else:
            high_indices = steering_data["high_exclusive_indices"].long()
            low_indices = steering_data["low_exclusive_indices"].long()

        signed_vector = steering_vector if direction == "increase" else -steering_vector
        if neuron_mode == "both":
            neuron_indices = torch.cat([high_indices.long(), low_indices.long()])
        elif direction == "increase":
            neuron_indices = high_indices.long()
        else:
            neuron_indices = low_indices.long()

        sparse = torch.zeros(self.config.intermediate_size, dtype=torch.float32)
        if neuron_indices.numel() == 0:
            return sparse

        values = signed_vector[neuron_indices]
        if method == "weighted":
            weights = compute_cohens_d_weights(
                steering_data["cohens_d_values"].float(),
                neuron_indices,
                weight_range=weight_range,
            )
            values = values * weights
        sparse[neuron_indices] = values
        return sparse

    def _make_pre_hook(self, sparse_delta: torch.Tensor, gamma: float, apply_on_prompt: bool):
        def hook(_module, inputs):
            hidden = inputs[0]
            if hidden.ndim != 3:
                raise RuntimeError(
                    "Expected down_proj input with shape [batch, seq_len, intermediate_size]; "
                    f"got {tuple(hidden.shape)}"
                )
            if not apply_on_prompt and hidden.shape[1] != 1:
                return inputs
            delta = (gamma * sparse_delta).to(device=hidden.device, dtype=hidden.dtype)
            edited = hidden + delta.view(1, 1, -1)
            return (edited, *inputs[1:])

        return hook

    def apply_steering(
        self,
        steering_data_dir: str | Path,
        *,
        trait: str,
        gamma: float,
        direction: str = "increase",
        method: str = "weighted",
        neuron_mode: str = "both",
        weight_range: Optional[tuple[float, float]] = None,
        apply_on_prompt: bool = False,
        quantile: float | None = None,
        cohens_d_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Install forward pre-hooks on each target layer's ``down_proj`` input.

        ``neuron_mode="both"`` uses the union of high- and low-exclusive
        neurons and flips the steering vector for decrease, matching the paper
        and the general-capability scripts. ``"directional"`` uses high
        neurons for increase and low neurons for decrease, matching the
        single-trait IPIP protocol.
        """

        self.reset()
        if neuron_mode == "symmetry":
            neuron_mode = "directional"
        weight_range = weight_range or self.config.weight_range
        effective_quantile = self.config.quantile if quantile is None else quantile
        effective_cohens_d_threshold = (
            self.config.cohens_d_threshold if cohens_d_threshold is None else cohens_d_threshold
        )
        total = 0
        per_layer = []

        for layer_idx in self.config.target_layers:
            data = load_steering_data(steering_data_dir, trait, layer_idx)
            sparse_delta = self._build_sparse_delta(
                data,
                direction=direction,
                method=method,
                neuron_mode=neuron_mode,
                weight_range=weight_range,
                quantile=effective_quantile,
                cohens_d_threshold=effective_cohens_d_threshold,
            )
            total += int((sparse_delta != 0).sum().item())
            hook = self._down_proj(layer_idx).register_forward_pre_hook(
                self._make_pre_hook(sparse_delta, gamma, apply_on_prompt)
            )
            self._hooks.append(hook)
            per_layer.append({"layer": layer_idx, "neurons": int((sparse_delta != 0).sum().item())})

        denominator = len(self.config.target_layers) * self.config.intermediate_size
        return {
            "trait": trait,
            "direction": direction,
            "method": method,
            "neuron_mode": neuron_mode,
            "gamma": gamma,
            "apply_on_prompt": apply_on_prompt,
            "quantile": effective_quantile,
            "cohens_d_threshold": effective_cohens_d_threshold,
            "total_neurons": total,
            "percentage_of_target_mlp_neurons": 100.0 * total / denominator,
            "per_layer": per_layer,
        }

    def generate_prompts(
        self,
        prompts: list[str],
        *,
        batch_size: int = 8,
        max_new_tokens: int = 400,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.15,
        use_chat_template: bool = False,
        system_prompt: str = "You are a helpful assistant.",
    ) -> list[str]:
        outputs: list[str] = []
        for start in tqdm(range(0, len(prompts), batch_size), desc="generate"):
            batch = prompts[start : start + batch_size]
            if use_chat_template and getattr(self.tokenizer, "chat_template", None):
                batch = [
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for prompt in batch
                ]
            device = next(self.model.parameters()).device
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True).to(device)
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if temperature > 0:
                generate_kwargs["temperature"] = temperature
            with torch.no_grad():
                generated = self.model.generate(**encoded, **generate_kwargs)
            prompt_len = encoded["input_ids"].shape[1]
            new_tokens = generated[:, prompt_len:]
            outputs.extend(
                self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            )
        return [text.strip() for text in outputs]

    def generate_questions(
        self,
        questions: list[str],
        *,
        batch_size: int = 8,
        output_path: str | Path | None = None,
        **generation_kwargs,
    ) -> list[dict[str, str]]:
        prompts = [format_personalitybench_test_prompt(question) for question in questions]
        answers = self.generate_prompts(prompts, batch_size=batch_size, **generation_kwargs)
        rows = [{"question": question, "answer": answer} for question, answer in zip(questions, answers)]
        if output_path is not None:
            save_jsonl(rows, output_path)
        return rows

    def generate(
        self,
        prompts: list[str],
        *,
        temperature: float = 0.0,
        max_tokens: int = 400,
        top_p: float = 1.0,
        repetition_penalty: float = 1.15,
        batch_size: int = 8,
        **kwargs,
    ) -> list[str]:
        """Backward-compatible alias for generating from already-formatted prompts."""

        return self.generate_prompts(
            prompts,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )

    def generate_with_steering(
        self,
        questions: list[str],
        steering_data_dir: str | Path,
        *,
        trait: str,
        gamma: float,
        direction: str = "increase",
        method: str = "weighted",
        neuron_mode: str = "both",
        weight_range: Optional[tuple[float, float]] = None,
        apply_on_prompt: bool = False,
        quantile: float | None = None,
        cohens_d_threshold: float | None = None,
        batch_size: int = 8,
        output_path: str | Path | None = None,
        **generation_kwargs,
    ) -> list[dict[str, str]]:
        self.apply_steering(
            steering_data_dir,
            trait=trait,
            gamma=gamma,
            direction=direction,
            method=method,
            neuron_mode=neuron_mode,
            weight_range=weight_range,
            apply_on_prompt=apply_on_prompt,
            quantile=quantile,
            cohens_d_threshold=cohens_d_threshold,
        )
        return self.generate_questions(
            questions,
            batch_size=batch_size,
            output_path=output_path,
            **generation_kwargs,
        )
