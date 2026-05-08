"""DPN-LE: Dual Personality Neuron Localization and Editing."""

from .extract_activations import ActivationExtractor, load_contrastive_samples
from .inference import DPNLEInference
from .ipip_neo import IPIPNEOScorer, format_ipip_prompt, parse_choice
from .utils import load_records
from .model_configs import (
    LLAMA_3_8B_CONFIG,
    QWEN_25_7B_CONFIG,
    ModelConfig,
    create_custom_config,
    get_model_config,
    list_supported_models,
)
from .personalitybench import (
    build_contrastive_samples,
    load_descriptions,
    load_search_questions,
    load_test_questions,
)
from .pipeline import DPNLEPipeline
from .prepare_steering import (
    SteeringDataPreparer,
    compute_cohens_d,
    compute_exclusive_indices,
    load_steering_data,
)

__version__ = "0.1.0"

__all__ = [
    "ActivationExtractor",
    "DPNLEInference",
    "DPNLEPipeline",
    "IPIPNEOScorer",
    "LLAMA_3_8B_CONFIG",
    "ModelConfig",
    "QWEN_25_7B_CONFIG",
    "SteeringDataPreparer",
    "compute_cohens_d",
    "compute_exclusive_indices",
    "create_custom_config",
    "format_ipip_prompt",
    "get_model_config",
    "build_contrastive_samples",
    "list_supported_models",
    "load_contrastive_samples",
    "load_descriptions",
    "load_search_questions",
    "load_steering_data",
    "load_test_questions",
    "load_records",
    "parse_choice",
]
