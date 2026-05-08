"""High-level DPN-LE workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .extract_activations import ActivationExtractor
from .inference import DPNLEInference
from .model_configs import ModelConfig, get_model_config
from .prepare_steering import SteeringDataPreparer


class DPNLEPipeline:
    """One object for extraction, steering preparation, and generation."""

    def __init__(
        self,
        model_name: str,
        model_config: Optional[ModelConfig] = None,
        **loader_kwargs,
    ) -> None:
        self.model_name = model_name
        self.config = model_config or get_model_config(model_name)
        self.loader_kwargs = loader_kwargs
        self._extractor: Optional[ActivationExtractor] = None
        self._preparer: Optional[SteeringDataPreparer] = None
        self._inference: Optional[DPNLEInference] = None

    @property
    def extractor(self) -> ActivationExtractor:
        if self._extractor is None:
            self._extractor = ActivationExtractor(
                self.model_name,
                self.config,
                **self.loader_kwargs,
            )
        return self._extractor

    @property
    def preparer(self) -> SteeringDataPreparer:
        if self._preparer is None:
            self._preparer = SteeringDataPreparer(self.config)
        return self._preparer

    @property
    def inference(self) -> DPNLEInference:
        if self._inference is None:
            self._inference = DPNLEInference(
                self.model_name,
                self.config,
                **self.loader_kwargs,
            )
        return self._inference

    def extract_activations(
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
        self.extractor.extract(
            high_samples,
            low_samples,
            trait=trait,
            output_dir=output_dir,
            batch_size=batch_size,
            max_length=max_length,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
        )

    def prepare_steering(
        self,
        activations_dir: str | Path,
        *,
        trait: str,
        output_dir: str | Path,
        quantile: float | None = None,
        cohens_d_threshold: float | None = None,
    ):
        return self.preparer.prepare(
            activations_dir,
            trait,
            output_dir,
            quantile=quantile,
            cohens_d_threshold=cohens_d_threshold,
        )

    def generate(
        self,
        questions: list[str],
        steering_data_dir: str | Path,
        *,
        trait: str,
        gamma: float,
        direction: str = "increase",
        method: str = "weighted",
        neuron_mode: str = "both",
        batch_size: int = 8,
        output_path: str | Path | None = None,
        **generation_kwargs,
    ):
        return self.inference.generate_with_steering(
            questions,
            steering_data_dir,
            trait=trait,
            gamma=gamma,
            direction=direction,
            method=method,
            neuron_mode=neuron_mode,
            batch_size=batch_size,
            output_path=output_path,
            **generation_kwargs,
        )

    def run_complete_pipeline(
        self,
        high_samples: list[dict[str, str]],
        low_samples: list[dict[str, str]],
        test_questions: list[str],
        *,
        trait: str,
        work_dir: str | Path = "./dpn_le_output",
        quantile: float | None = None,
        cohens_d_threshold: float | None = None,
        gamma: float = 1.0,
        direction: str = "increase",
        method: str = "weighted",
        neuron_mode: str = "both",
        batch_size: int = 8,
        max_length: int = 2048,
        use_chat_template: bool = False,
        system_prompt: str | None = None,
        **generation_kwargs,
    ):
        work_dir = Path(work_dir)
        activations_dir = work_dir / "activations"
        steering_dir = work_dir / "steering_data"
        results_path = work_dir / "results" / f"{trait}_{direction}_{method}_g{gamma}.jsonl"

        self.extract_activations(
            high_samples,
            low_samples,
            trait=trait,
            output_dir=activations_dir,
            batch_size=batch_size,
            max_length=max_length,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
        )
        self.prepare_steering(
            activations_dir,
            trait=trait,
            output_dir=steering_dir,
            quantile=quantile,
            cohens_d_threshold=cohens_d_threshold,
        )
        return self.generate(
            test_questions,
            steering_dir,
            trait=trait,
            gamma=gamma,
            direction=direction,
            method=method,
            neuron_mode=neuron_mode,
            batch_size=batch_size,
            output_path=results_path,
            **generation_kwargs,
        )
