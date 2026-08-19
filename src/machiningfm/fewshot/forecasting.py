from __future__ import annotations

from typing import Any

from .base import euclidean


FORECAST_LABEL_KEYS = ("future_series", "forecast", "target_series", "future")


class ResidualForecastAdapter:
    """Correct a base forecast with embedding-weighted support residuals."""

    def __init__(self) -> None:
        self.embeddings: list[list[float]] = []
        self.residuals: list[list[list[float]]] = []

    def fit(
        self,
        embeddings: list[list[float]],
        targets: list[Any],
        base_forecasts: list[Any],
    ) -> "ResidualForecastAdapter":
        if not embeddings:
            raise ValueError("At least one labeled support sample is required")
        self.embeddings = embeddings
        self.residuals = []
        for target, base in zip(targets, base_forecasts):
            base_matrix = forecast_matrix(base)
            channels = len(base_matrix)
            horizon = len(base_matrix[0]) if base_matrix else 1
            target_matrix = resize_forecast_matrix(forecast_matrix(target), channels, horizon)
            self.residuals.append(_subtract(target_matrix, base_matrix))
        return self

    def predict(self, embedding: list[float], base_forecast: Any) -> dict[str, Any]:
        base_matrix = forecast_matrix(base_forecast)
        channels = len(base_matrix)
        horizon = len(base_matrix[0]) if base_matrix else 1
        residual = _weighted_average_matrices(
            embedding,
            self.embeddings,
            [resize_forecast_matrix(value, channels, horizon) for value in self.residuals],
        )
        prediction = _add(base_matrix, residual)
        return {
            "prediction": prediction,
            "base_prediction": base_matrix,
            "few_shot_correction": residual,
            "uncertainty": _mean_abs(residual),
        }


class KNNForecastAdapter:
    """Predict the future directly by weighted averaging nearby support futures."""

    def __init__(self) -> None:
        self.embeddings: list[list[float]] = []
        self.targets: list[list[list[float]]] = []

    def fit(self, embeddings: list[list[float]], targets: list[Any]) -> "KNNForecastAdapter":
        if not embeddings:
            raise ValueError("At least one labeled support sample is required")
        self.embeddings = embeddings
        self.targets = [forecast_matrix(target) for target in targets]
        return self

    def predict(self, embedding: list[float], base_forecast: Any) -> dict[str, Any]:
        base_matrix = forecast_matrix(base_forecast)
        channels = len(base_matrix)
        horizon = len(base_matrix[0]) if base_matrix else 1
        prediction = _weighted_average_matrices(
            embedding,
            self.embeddings,
            [resize_forecast_matrix(value, channels, horizon) for value in self.targets],
        )
        return {
            "prediction": prediction,
            "base_prediction": base_matrix,
            "uncertainty": _mean_abs(_subtract(prediction, base_matrix)),
        }


class RidgeForecastAdapter:
    """Map embeddings to a flattened future trajectory with multi-output ridge."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.model: Any = None
        self.channels = 1
        self.horizon = 1

    def fit(
        self,
        embeddings: list[list[float]],
        targets: list[Any],
        base_forecast: Any,
    ) -> "RidgeForecastAdapter":
        if not embeddings:
            raise ValueError("At least one labeled support sample is required")
        from sklearn.linear_model import Ridge

        base_matrix = forecast_matrix(base_forecast)
        self.channels = len(base_matrix)
        self.horizon = len(base_matrix[0]) if base_matrix else 1
        flattened_targets = [
            _flatten(resize_forecast_matrix(forecast_matrix(target), self.channels, self.horizon))
            for target in targets
        ]
        self.model = Ridge(alpha=self.alpha).fit(embeddings, flattened_targets)
        return self

    def predict(self, embedding: list[float], base_forecast: Any) -> dict[str, Any]:
        prediction = self.model.predict([embedding])[0].tolist()
        matrix = _unflatten(prediction, self.channels, self.horizon)
        return {
            "prediction": matrix,
            "base_prediction": resize_forecast_matrix(forecast_matrix(base_forecast), self.channels, self.horizon),
            "uncertainty": None,
        }


def forecast_target_from_sample(sample: dict[str, Any]) -> Any | None:
    label = sample.get("label")
    if isinstance(label, dict):
        for key in FORECAST_LABEL_KEYS:
            if key in label:
                return label[key]
        return None
    return label


def forecast_matrix(value: Any) -> list[list[float]]:
    if value is None:
        raise ValueError("Forecast value is required")
    if isinstance(value, dict):
        target = forecast_target_from_sample({"label": value})
        if target is None:
            raise ValueError(f"Forecast label must contain one of {FORECAST_LABEL_KEYS}")
        return forecast_matrix(target)
    if not isinstance(value, list):
        return [[float(value)]]
    if not value:
        raise ValueError("Forecast value cannot be empty")
    first = value[0]
    if isinstance(first, list):
        return [[float(item) for item in channel] for channel in value]
    return [[float(item) for item in value]]


def resize_forecast_matrix(matrix: list[list[float]], channels: int, horizon: int) -> list[list[float]]:
    if channels <= 0 or horizon <= 0:
        raise ValueError("Forecast channels and horizon must be positive")
    if not matrix:
        matrix = [[0.0]]
    output: list[list[float]] = []
    for channel_index in range(channels):
        source = matrix[channel_index] if channel_index < len(matrix) else matrix[-1]
        output.append(_resize_vector(source, horizon))
    return output


def forecast_adapter(method: str) -> Any:
    normalized = method.lower().replace("-", "_")
    if normalized in {"prototype", "residual", "residual_forecast", "forecast_residual"}:
        return ResidualForecastAdapter()
    if normalized in {"knn", "embedding_knn", "knn_forecast"}:
        return KNNForecastAdapter()
    if normalized in {"ridge", "ridge_forecast"}:
        return RidgeForecastAdapter()
    raise ValueError(f"Unknown few-shot forecasting method: {method}")


def _resize_vector(values: list[float], horizon: int) -> list[float]:
    if not values:
        return [0.0 for _ in range(horizon)]
    if len(values) == horizon:
        return [float(value) for value in values]
    if len(values) == 1:
        return [float(values[0]) for _ in range(horizon)]
    scale = (len(values) - 1) / max(1, horizon - 1)
    output = []
    for index in range(horizon):
        position = index * scale
        left = int(position)
        right = min(left + 1, len(values) - 1)
        weight = position - left
        output.append(float(values[left]) * (1.0 - weight) + float(values[right]) * weight)
    return output


def _weighted_average_matrices(
    query_embedding: list[float],
    support_embeddings: list[list[float]],
    matrices: list[list[list[float]]],
) -> list[list[float]]:
    distances = [euclidean(query_embedding, support) for support in support_embeddings]
    weights = [1.0 / (distance + 1e-8) for distance in distances]
    total = sum(weights)
    channels = len(matrices[0])
    horizon = len(matrices[0][0])
    output = [[0.0 for _ in range(horizon)] for _ in range(channels)]
    for weight, matrix in zip(weights, matrices):
        normalized_weight = weight / total
        for channel_index in range(channels):
            for step_index in range(horizon):
                output[channel_index][step_index] += normalized_weight * matrix[channel_index][step_index]
    return output


def _add(first: list[list[float]], second: list[list[float]]) -> list[list[float]]:
    return [
        [left + right for left, right in zip(first_channel, second_channel)]
        for first_channel, second_channel in zip(first, second)
    ]


def _subtract(first: list[list[float]], second: list[list[float]]) -> list[list[float]]:
    return [
        [left - right for left, right in zip(first_channel, second_channel)]
        for first_channel, second_channel in zip(first, second)
    ]


def _mean_abs(matrix: list[list[float]]) -> float:
    values = [abs(value) for channel in matrix for value in channel]
    return sum(values) / len(values) if values else 0.0


def _flatten(matrix: list[list[float]]) -> list[float]:
    return [value for channel in matrix for value in channel]


def _unflatten(values: list[float], channels: int, horizon: int) -> list[list[float]]:
    return [values[index * horizon : (index + 1) * horizon] for index in range(channels)]
