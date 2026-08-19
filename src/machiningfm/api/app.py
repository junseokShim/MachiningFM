from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from machiningfm.utils.paths import project_root
from .inference_service import InferenceService
from .schemas import FewShotRequest, PredictRequest, SampleInput, ZeroShotRequest


def create_app(config_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="MachiningFM", version="0.1.0")
    service: InferenceService | None = None
    auto_discover_checkpoint = config_path is not None

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def key_error_handler(_request: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    def get_service() -> InferenceService:
        nonlocal service
        if service is None:
            service = InferenceService(config_path, auto_discover_checkpoint=auto_discover_checkpoint)
        return service

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_version": get_service().predictor.model_version}

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, Any]:
        return get_service().predict(request.task, request)

    @app.post("/embed")
    def embed(request: SampleInput) -> dict[str, Any]:
        return get_service().embed(request)

    @app.post("/forecast")
    def forecast(request: SampleInput) -> dict[str, Any]:
        return get_service().predict("future_forecasting", request)

    @app.get("/metadata")
    @app.post("/metadata")
    def metadata() -> dict[str, Any]:
        return get_service().metadata()

    @app.post("/predict/zero-shot")
    def zero_shot(request: ZeroShotRequest) -> dict[str, Any]:
        return get_service().zero_shot(request.task, request.query)

    @app.post("/predict/one-shot")
    def one_shot(request: FewShotRequest) -> dict[str, Any]:
        return get_service().one_shot(request.task, request.support_set[:1], request.query)

    @app.post("/predict/few-shot")
    def few_shot(request: FewShotRequest) -> dict[str, Any]:
        return get_service().few_shot(request.task, request.support_set, request.query, request.adaptation_method)

    @app.post("/adapt/few-shot")
    def adapt_few_shot(request: FewShotRequest) -> dict[str, Any]:
        return get_service().few_shot(request.task, request.support_set, request.query, request.adaptation_method)

    @app.post("/evaluate/few-shot")
    def evaluate_few_shot(request: FewShotRequest) -> dict[str, Any]:
        return {"result": get_service().few_shot(request.task, request.support_set, request.query, request.adaptation_method)}

    @app.post("/embed/support")
    def embed_support(request: SampleInput) -> dict[str, Any]:
        return get_service().embed(request)

    @app.post("/embed/query")
    def embed_query(request: SampleInput) -> dict[str, Any]:
        return get_service().embed(request)

    return app


def _default_config_path() -> Path | None:
    configured = os.environ.get("MACHININGFM_API_CONFIG")
    if configured:
        return Path(configured)
    default = project_root() / "configs/api/api_config.yaml"
    return default if default.exists() else None


app = create_app(_default_config_path())
