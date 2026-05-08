"""Activation extraction for DPN-LE.

This module captures the MLP activation at the input of ``down_proj`` for the
last prompt token.  That is the activation described in the paper as the
post-gated MLP hidden state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_configs import ModelConfig, get_model_config
from .utils import format_personality_prompt, parse_torch_dtype, validate_samples


def _model_layers(model) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not locate decoder layers on the loaded model")


class ActivationExtractor:
    """Extract DPN-LE activations from contrastive high/low samples."""

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
        self._last_token_indices: Optional[torch.Tensor] = None
        self._layer_outputs: dict[int, torch.Tensor] = {}

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
            dtype = parse_torch_dtype(torch_dtype)
            if device is not None and device_map == "auto":
                device_map = {"": device}
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
        self.model = model.eval()

    def _down_proj(self, layer_idx: int):
        layer = _model_layers(self.model)[layer_idx]
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "down_proj"):
            raise AttributeError(f"Layer {layer_idx} does not expose mlp.down_proj")
        return layer.mlp.down_proj

    def _make_hook(self, layer_idx: int):
        def hook(_module, inputs, _output):
            hidden = inputs[0]
            if hidden.ndim != 3:
                raise RuntimeError(
                    "Expected down_proj input with shape [batch, seq_len, intermediate_size]; "
                    f"got {tuple(hidden.shape)}"
                )
            if self._last_token_indices is None:
                raise RuntimeError("last token indices were not set before forward")
            row_idx = torch.arange(hidden.shape[0], device=hidden.device)
            token_idx = self._last_token_indices.to(hidden.device)
            self._layer_outputs[layer_idx] = hidden[row_idx, token_idx, :].detach().float().cpu()

        return hook

    @staticmethod
    def _last_nonpad_indices(attention_mask: torch.Tensor) -> torch.Tensor:
        flipped = attention_mask.flip(dims=[1])
        distance_from_end = flipped.long().argmax(dim=1)
        return attention_mask.shape[1] - 1 - distance_from_end

    def extract_prompts(
        self,
        prompts: list[str],
        *,
        batch_size: int = 8,
        max_length: int = 2048,
        use_chat_template: bool = False,
        system_prompt: str | None = None,
        show_progress: bool = True,
    ) -> dict[int, torch.Tensor]:
        """Extract last-token down_proj-input activations for prompts."""

        if use_chat_template:
            if not getattr(self.tokenizer, "chat_template", None):
                raise ValueError("Tokenizer does not provide a chat_template")
            messages = []
            for prompt in prompts:
                turns = []
                if system_prompt is not None:
                    turns.append({"role": "system", "content": system_prompt})
                turns.append({"role": "user", "content": prompt})
                messages.append(turns)
            prompts = [
                self.tokenizer.apply_chat_template(
                    turns,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for turns in messages
            ]

        activations: dict[int, list[torch.Tensor]] = {layer: [] for layer in self.config.target_layers}
        iterator = range(0, len(prompts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="extract activations")

        for start in iterator:
            batch = prompts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            device = next(self.model.parameters()).device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            self._last_token_indices = self._last_nonpad_indices(encoded["attention_mask"]).cpu()
            self._layer_outputs = {}

            hooks = [
                self._down_proj(layer_idx).register_forward_hook(self._make_hook(layer_idx))
                for layer_idx in self.config.target_layers
            ]
            try:
                with torch.no_grad():
                    self.model(**encoded, use_cache=False)
            finally:
                for hook in hooks:
                    hook.remove()

            for layer_idx in self.config.target_layers:
                if layer_idx not in self._layer_outputs:
                    raise RuntimeError(f"No activation captured for layer {layer_idx}")
                activations[layer_idx].append(self._layer_outputs[layer_idx])

        return {layer: torch.cat(parts, dim=0) for layer, parts in activations.items()}

    def extract_from_samples(
        self,
        samples: list[dict[str, str]],
        *,
        trait: str,
        direction: str,
        output_dir: str | Path,
        batch_size: int = 8,
        max_length: int = 2048,
        use_chat_template: bool = False,
        system_prompt: str | None = None,
    ) -> dict[int, torch.Tensor]:
        validate_samples(samples)
        prompts = [format_personality_prompt(s["description"], s["question"]) for s in samples]
        activations = self.extract_prompts(
            prompts,
            batch_size=batch_size,
            max_length=max_length,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
        )
        save_dir = Path(output_dir) / trait / direction
        save_dir.mkdir(parents=True, exist_ok=True)
        for layer_idx, tensor in activations.items():
            torch.save(tensor, save_dir / f"layer_{layer_idx}.pt")
        return activations

    def extract(
        self,
        high_samples: list[dict[str, str]],
        low_samples: list[dict[str, str]],
        *,
        trait: str,
        output_dir: str | Path,
        batch_size: int = 8,
        max_length: int = 2048,
        use_chat_template: bool = False,
        system_prompt: str | None = None,
    ) -> None:
        self.extract_from_samples(
            high_samples,
            trait=trait,
            direction="high",
            output_dir=output_dir,
            batch_size=batch_size,
            max_length=max_length,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
        )
        self.extract_from_samples(
            low_samples,
            trait=trait,
            direction="low",
            output_dir=output_dir,
            batch_size=batch_size,
            max_length=max_length,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
        )


def load_contrastive_samples(
    data_path: str | Path,
    *,
    num_samples: int = 1000,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load high/low samples from a JSON or JSONL file."""

    path = Path(data_path)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        high = data.get("high_samples", data.get("high", []))
        low = data.get("low_samples", data.get("low", []))
    else:
        high = []
        low = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                direction = row.get("direction")
                if direction == "high":
                    high.append(row)
                elif direction == "low":
                    low.append(row)
                else:
                    raise ValueError("JSONL rows must contain direction='high' or direction='low'")
    return high[:num_samples], low[:num_samples]
