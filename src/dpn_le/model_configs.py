"""Model configuration helpers for DPN-LE.

The paper uses MLP activations at the input of ``down_proj``.  For LLaMA and
Qwen-style decoder-only models this activation has ``intermediate_size``
dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a model architecture used by DPN-LE."""

    model_name: str
    target_layers: list[int]
    intermediate_size: int
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    quantile: float = 0.995
    cohens_d_threshold: float = 0.8
    weight_range: tuple[float, float] = (0.75, 1.0)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target_layers:
            raise ValueError("target_layers cannot be empty")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("quantile must be between 0 and 1")
        if self.cohens_d_threshold < 0:
            raise ValueError("cohens_d_threshold must be non-negative")


LLAMA_3_8B_CONFIG = ModelConfig(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    target_layers=list(range(12, 32)),
    intermediate_size=14336,
    num_layers=32,
    hidden_size=4096,
    quantile=0.995,
    cohens_d_threshold=0.8,
    metadata={"paper_layers": "12-31", "paper_model": "LLaMA-3-8B-Instruct"},
)


QWEN_25_7B_CONFIG = ModelConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    target_layers=list(range(14, 28)),
    intermediate_size=18944,
    num_layers=28,
    hidden_size=3584,
    quantile=0.995,
    cohens_d_threshold=0.3,
    metadata={"paper_layers": "14-27", "paper_model": "Qwen2.5-7B-Instruct"},
)


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "llama-3-8b": LLAMA_3_8B_CONFIG,
    "meta-llama-3-8b-instruct": LLAMA_3_8B_CONFIG,
    "qwen2.5-7b": QWEN_25_7B_CONFIG,
    "qwen2.5-7b-instruct": QWEN_25_7B_CONFIG,
}


def get_model_config(model_name: str) -> ModelConfig:
    """Return the paper-aligned config for a supported model name."""

    normalized = model_name.lower().replace("/", "-").replace("_", "-")
    for key, config in MODEL_REGISTRY.items():
        if key in normalized:
            return config
    raise ValueError(
        f"Unsupported model '{model_name}'. Use one of {list_supported_models()} "
        "or pass a custom ModelConfig."
    )


def create_custom_config(
    model_name: str,
    num_layers: int,
    intermediate_size: int,
    separation_start_layer: int,
    *,
    hidden_size: Optional[int] = None,
    quantile: float = 0.995,
    cohens_d_threshold: float = 0.8,
    weight_range: tuple[float, float] = (0.75, 1.0),
) -> ModelConfig:
    """Create a config for a model after identifying its separation layer."""

    if not 0 <= separation_start_layer < num_layers:
        raise ValueError("separation_start_layer must be in [0, num_layers)")
    return ModelConfig(
        model_name=model_name,
        target_layers=list(range(separation_start_layer, num_layers)),
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        hidden_size=hidden_size,
        quantile=quantile,
        cohens_d_threshold=cohens_d_threshold,
        weight_range=weight_range,
    )


def config_from_model(
    model,
    model_name: str,
    target_layers: Optional[Iterable[int]] = None,
    *,
    separation_start_layer: Optional[int] = None,
    quantile: float = 0.995,
    cohens_d_threshold: float = 0.8,
    weight_range: tuple[float, float] = (0.75, 1.0),
) -> ModelConfig:
    """Infer dimensions from a loaded Transformers causal LM."""

    hf_config = getattr(model, "config", None)
    if hf_config is None:
        raise ValueError("model must expose a HuggingFace-style .config")

    num_layers = getattr(hf_config, "num_hidden_layers", None)
    intermediate_size = getattr(hf_config, "intermediate_size", None)
    hidden_size = getattr(hf_config, "hidden_size", None)
    if num_layers is None or intermediate_size is None:
        raise ValueError("model.config lacks num_hidden_layers or intermediate_size")

    if target_layers is None:
        if separation_start_layer is None:
            raise ValueError("Provide target_layers or separation_start_layer")
        target_layers = range(separation_start_layer, num_layers)

    return ModelConfig(
        model_name=model_name,
        target_layers=list(target_layers),
        intermediate_size=int(intermediate_size),
        num_layers=int(num_layers),
        hidden_size=int(hidden_size) if hidden_size is not None else None,
        quantile=quantile,
        cohens_d_threshold=cohens_d_threshold,
        weight_range=weight_range,
    )


def list_supported_models() -> list[str]:
    """List short registry names for built-in configs."""

    return sorted(MODEL_REGISTRY)
