from __future__ import annotations

from typing import Any

from .base import expected_label_keys, is_regression, support_embeddings
from .forecasting import forecast_adapter, forecast_target_from_sample
from .knn import EmbeddingKNNAdapter
from .linear_probe import LinearProbeAdapter
from .prototype import PrototypicalAdapter
from .ridge import (
    GaussianProcessRegressionAdapter,
    KernelRidgeRegressionAdapter,
    RidgeRegressionAdapter,
    SVRRegressionAdapter,
)

VIRTUAL_SENSOR_TASKS = {"virtual_sensor", "virtual_vibration"}


class FewShotPredictor:
    def __init__(self, predictor: Any, max_support_samples: int = 20) -> None:
        self.predictor = predictor
        self.max_support_samples = max_support_samples

    def predict(
        self,
        task: str,
        support_set: list[dict[str, Any]],
        query: dict[str, Any],
        method: str = "prototype",
    ) -> dict[str, Any]:
        support_set = support_set[: self.max_support_samples]
        if task == "future_forecasting":
            return self._predict_forecasting(support_set, query, method)
        if task in VIRTUAL_SENSOR_TASKS:
            return self._predict_virtual_sensor(support_set, query, method, task)
        embeddings, labels = support_embeddings(self.predictor, support_set, task)
        if not labels:
            raise ValueError(
                "Few-shot prediction requires labeled support samples. "
                f"Expected label keys for task {task!r}: {list(expected_label_keys(task))}"
            )
        adapter = self._adapter(method, labels)
        adapter.fit(embeddings, labels)
        result = adapter.predict(self.predictor.embed(query))
        return {
            "mode": "few-shot",
            "task": task,
            **result,
            "adaptation_method": method,
            "support_size": len(labels),
            "model_version": self.predictor.model_version,
        }

    def _predict_forecasting(
        self,
        support_set: list[dict[str, Any]],
        query: dict[str, Any],
        method: str,
    ) -> dict[str, Any]:
        embeddings: list[list[float]] = []
        labels: list[Any] = []
        base_forecasts: list[Any] = []
        for sample in support_set:
            target = forecast_target_from_sample(sample)
            if target is None:
                continue
            embeddings.append(self.predictor.embed(sample))
            labels.append(target)
            base_forecasts.append(self.predictor.predict("future_forecasting", sample)["prediction"])
        if not labels:
            raise ValueError(
                "Few-shot forecasting requires labeled support samples. "
                "Expected label keys: ['future_series', 'forecast', 'target_series', 'future']"
            )

        query_embedding = self.predictor.embed(query)
        query_base_prediction = self.predictor.predict("future_forecasting", query)["prediction"]
        adapter = forecast_adapter(method)
        normalized_method = method.lower().replace("-", "_")
        if normalized_method in {"ridge", "ridge_forecast"}:
            adapter.fit(embeddings, labels, query_base_prediction)
        elif normalized_method in {"knn", "embedding_knn", "knn_forecast"}:
            adapter.fit(embeddings, labels)
        else:
            adapter.fit(embeddings, labels, base_forecasts)
        result = adapter.predict(query_embedding, query_base_prediction)
        return {
            "mode": "few-shot",
            "task": "future_forecasting",
            **result,
            "adaptation_method": method,
            "support_size": len(labels),
            "model_version": self.predictor.model_version,
        }

    def _predict_virtual_sensor(
        self,
        support_set: list[dict[str, Any]],
        query: dict[str, Any],
        method: str,
        task: str,
    ) -> dict[str, Any]:
        embeddings: list[list[float]] = []
        labels: list[Any] = []
        base_predictions: list[Any] = []
        for sample in support_set:
            target = forecast_target_from_sample(sample)
            if target is None:
                continue
            embeddings.append(self._virtual_sensor_embedding(sample))
            labels.append(target)
            base_predictions.append(self.predictor.predict("virtual_sensor", sample)["prediction"])
        if not labels:
            raise ValueError(
                "Few-shot virtual sensor prediction requires labeled support samples. "
                "Expected label keys: ['target_series', 'future_series', 'forecast', 'future']"
            )

        query_embedding = self._virtual_sensor_embedding(query)
        query_base_prediction = self.predictor.predict("virtual_sensor", query)["prediction"]
        adapter = forecast_adapter(method)
        normalized_method = method.lower().replace("-", "_")
        if normalized_method in {"ridge", "ridge_forecast"}:
            adapter.fit(embeddings, labels, query_base_prediction)
        elif normalized_method in {"knn", "embedding_knn", "knn_forecast"}:
            adapter.fit(embeddings, labels)
        else:
            adapter.fit(embeddings, labels, base_predictions)
        result = adapter.predict(query_embedding, query_base_prediction)
        return {
            "mode": "few-shot",
            "task": task,
            **result,
            "adaptation_method": method,
            "support_size": len(labels),
            "model_version": self.predictor.model_version,
        }

    def _virtual_sensor_embedding(self, sample: dict[str, Any]) -> list[float]:
        if hasattr(self.predictor, "embed_virtual_sensor_source"):
            return self.predictor.embed_virtual_sensor_source(sample)
        return self.predictor.embed(sample)

    @staticmethod
    def _adapter(method: str, labels: list[Any]) -> Any:
        normalized = method.lower().replace("-", "_")
        if normalized in {"prototype", "prototypical_classifier"}:
            return PrototypicalAdapter()
        if normalized in {"knn", "embedding_knn"}:
            return EmbeddingKNNAdapter()
        if normalized in {"ridge", "ridge_regression"}:
            if not is_regression(labels):
                return PrototypicalAdapter()
            return RidgeRegressionAdapter()
        if normalized in {"kernel_ridge", "krr"}:
            _require_regression_labels(normalized, labels)
            return KernelRidgeRegressionAdapter()
        if normalized in {"svr", "support_vector_regression"}:
            _require_regression_labels(normalized, labels)
            return SVRRegressionAdapter()
        if normalized in {"gaussian_process", "gp", "gpr", "gaussian_process_regression"}:
            _require_regression_labels(normalized, labels)
            return GaussianProcessRegressionAdapter()
        if normalized in {"linear_probe", "linear"}:
            return LinearProbeAdapter()
        raise ValueError(f"Unknown few-shot method: {method}")


def _require_regression_labels(method: str, labels: list[Any]) -> None:
    if not is_regression(labels):
        raise ValueError(f"Few-shot method {method!r} requires numeric regression labels")
