from __future__ import annotations

from typing import Any


class SklearnScalarRegressionAdapter:
    estimator_import: tuple[str, str] = ("sklearn.linear_model", "Ridge")

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.model: Any = None

    def fit(self, embeddings: list[list[float]], labels: list[Any]) -> "SklearnScalarRegressionAdapter":
        if not embeddings:
            raise ValueError("At least one labeled support sample is required")
        module_name, class_name = self.estimator_import
        module = __import__(module_name, fromlist=[class_name])
        estimator_class = getattr(module, class_name)
        self.model = estimator_class(**self.kwargs).fit(embeddings, [float(value) for value in labels])
        return self

    def predict(self, embedding: list[float]) -> dict[str, Any]:
        return {"prediction": float(self.model.predict([embedding])[0]), "uncertainty": None}


class RidgeRegressionAdapter(SklearnScalarRegressionAdapter):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__(alpha=alpha)


class KernelRidgeRegressionAdapter(SklearnScalarRegressionAdapter):
    estimator_import = ("sklearn.kernel_ridge", "KernelRidge")

    def __init__(self, alpha: float = 1.0, kernel: str = "rbf") -> None:
        super().__init__(alpha=alpha, kernel=kernel)


class SVRRegressionAdapter(SklearnScalarRegressionAdapter):
    estimator_import = ("sklearn.svm", "SVR")

    def __init__(self, kernel: str = "rbf", c: float = 1.0, epsilon: float = 0.1) -> None:
        super().__init__(kernel=kernel, C=c, epsilon=epsilon)


class GaussianProcessRegressionAdapter(SklearnScalarRegressionAdapter):
    estimator_import = ("sklearn.gaussian_process", "GaussianProcessRegressor")

    def __init__(self, normalize_y: bool = True) -> None:
        super().__init__(normalize_y=normalize_y)
