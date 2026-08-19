from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from machiningfm.fewshot.few_shot import FewShotPredictor
from machiningfm.fewshot.one_shot import OneShotPredictor
from machiningfm.fewshot.zero_shot import ZeroShotPredictor
from machiningfm.inference.predictor import MachiningPredictor
from machiningfm.utils.config import load_config
from machiningfm.utils.io import read_json
from machiningfm.utils.paths import project_root
from .preprocessing_service import PreprocessingService

LOGGER = logging.getLogger(__name__)


class InferenceService:
    def __init__(
        self,
        config_path: str | Path | None = None,
        auto_discover_checkpoint: bool = True,
    ) -> None:
        self.config = load_config(config_path) if config_path and Path(config_path).exists() else {}
        self.auto_discover_checkpoint = auto_discover_checkpoint
        checkpoint = self._checkpoint_from_config()
        latent_context_path = self._latent_context_path_from_config()
        device = self._inference_device_from_config()
        try:
            self.predictor = MachiningPredictor(
                checkpoint_path=checkpoint,
                latent_context_path=latent_context_path,
                device=device,
            )
        except ValueError:
            if self.config or checkpoint is None:
                raise
            LOGGER.warning("Ignoring auto-discovered checkpoint incompatible with the current channel schema: %s", checkpoint)
            self.predictor = MachiningPredictor(latent_context_path=latent_context_path, device=device)
        self.preprocessing = PreprocessingService(int(self.config.get("max_sequence_length", 8192)))
        max_support = int(self.config.get("max_support_samples", 20))
        self.zero_shot_predictor = ZeroShotPredictor(self.predictor)
        self.one_shot_predictor = OneShotPredictor(self.predictor)
        self.few_shot_predictor = FewShotPredictor(self.predictor, max_support)

    def predict(self, task: str, sample: Any) -> dict[str, Any]:
        normalized = self.preprocessing.normalize(sample)
        checkpoint = normalized.get("model_checkpoint_path")
        if checkpoint:
            return MachiningPredictor(
                checkpoint_path=checkpoint,
                latent_context_path=self._latent_context_path_from_config(),
                device=self._inference_device_from_config(),
            ).predict(task, normalized)
        return self.predictor.predict(task, normalized)

    def embed(self, sample: Any) -> dict[str, Any]:
        return {"embedding": self.predictor.embed(self.preprocessing.normalize(sample)), "model_version": self.predictor.model_version}

    def zero_shot(self, task: str, query: Any) -> dict[str, Any]:
        return self.zero_shot_predictor.predict(task, self.preprocessing.normalize(query))

    def one_shot(self, task: str, support_set: list[Any], query: Any) -> dict[str, Any]:
        return self.one_shot_predictor.predict(
            task,
            [self.preprocessing.normalize(sample) for sample in support_set],
            self.preprocessing.normalize(query),
        )

    def few_shot(self, task: str, support_set: list[Any], query: Any, method: str) -> dict[str, Any]:
        return self.few_shot_predictor.predict(
            task,
            [self.preprocessing.normalize(sample) for sample in support_set],
            self.preprocessing.normalize(query),
            method,
        )

    def metadata(self) -> dict[str, Any]:
        return self.predictor.metadata()

    def _checkpoint_from_config(self) -> Path | None:
        direct = self.config.get("checkpoint_path")
        if direct:
            path = Path(direct)
            checkpoint = path if path.is_absolute() else project_root() / path
            if not checkpoint.exists():
                raise FileNotFoundError(f"Configured API checkpoint does not exist: {checkpoint}")
            return checkpoint
        registry_value = self.config.get("model_registry_path")
        if registry_value:
            registry_path = Path(registry_value)
            if not registry_path.is_absolute():
                registry_path = project_root() / registry_path
            registry = read_json(registry_path, {"models": []})
            stage = self.config.get("serving_stage", "staging")
            staged = [
                model
                for model in registry.get("models", [])
                if model.get("promotion_status") == stage and model.get("api_serving")
            ]
            if staged:
                checkpoint = Path(staged[-1]["checkpoint_path"])
                if checkpoint.exists():
                    return checkpoint
            fallback_version = self.config.get("fallback_model_version")
            fallback = next(
                (model for model in registry.get("models", []) if model.get("model_version") == fallback_version),
                None,
            )
            if fallback:
                checkpoint = Path(fallback["checkpoint_path"])
                if checkpoint.exists():
                    return checkpoint
        if not getattr(self, "auto_discover_checkpoint", True):
            return None
        for candidate in (
            project_root() / "outputs/checkpoints/full_pretrain_multimodal_no_phm2010/machiningfm_full_pretrain_best.pt",
            project_root() / "outputs/checkpoints/full_pretrain_multimodal_no_phm2010/machiningfm_full_pretrain_latest.pt",
            project_root() / "outputs/checkpoints/full_pretrain_no_phm2010/machiningfm_full_pretrain_best.pt",
            project_root() / "outputs/checkpoints/full_pretrain_no_phm2010/machiningfm_full_pretrain_latest.pt",
            project_root() / "outputs/checkpoints/full_pretrain_all_numeric/machiningfm_full_pretrain_best.pt",
            project_root() / "outputs/checkpoints/full_pretrain_all_numeric/machiningfm_full_pretrain_latest.pt",
            project_root() / "outputs/checkpoints/full_pretrain_stage1/machiningfm_full_pretrain_best.pt",
            project_root() / "outputs/checkpoints/full_pretrain_stage1/machiningfm_full_pretrain_latest.pt",
            project_root() / "outputs/checkpoints/machiningfm_latest.pt",
        ):
            if candidate.exists():
                return candidate
        return None

    def _latent_context_path_from_config(self) -> Path | None:
        value = self.config.get("latent_context_path")
        if not value:
            return None
        path = Path(value)
        return path if path.is_absolute() else project_root() / path

    def _inference_device_from_config(self) -> str:
        requested = str(self.config.get("inference_device", "cpu")).lower()
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("API inference_device is set to CUDA, but torch.cuda.is_available() is False")
        return requested
