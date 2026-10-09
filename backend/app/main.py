import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.app.config import Settings
from backend.app.history import History
from backend.app.images import decode_image, image_data_url
from backend.app.schemas import Prediction
from ml.data.transforms import Letterbox
from ml.inference.predictor import Predictor

logger = logging.getLogger(__name__)


class BodyLimitMiddleware:
    """Limit request size before multipart parsing, including chunked uploads."""

    def __init__(self, app, limit):
        self.app, self.limit = app, limit
        self.slots = asyncio.Semaphore(2)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Invalid content length."}, 400)(
                scope, receive, send
            )
        if length > self.limit:
            return await JSONResponse({"detail": "Upload is too large."}, 413)(scope, receive, send)
        async with self.slots:
            data = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                data.extend(message.get("body", b""))
                if len(data) > self.limit:
                    return await JSONResponse({"detail": "Upload is too large."}, 413)(
                        scope, receive, send
                    )
                if not message.get("more_body", False):
                    break
            consumed = False

            async def bounded_receive():
                nonlocal consumed
                if not consumed:
                    consumed = True
                    return {"type": "http.request", "body": bytes(data), "more_body": False}
                return await receive()

            await self.app(scope, bounded_receive, send)


def create_app(settings: Settings | None = None, predictor_factory=Predictor) -> FastAPI:
    settings = settings or Settings()
    inference_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = None
        app.state.history = None
        if settings.checkpoint.is_file():
            try:
                predictor = predictor_factory(settings.checkpoint)
                if not predictor.metadata.get("selection_locked") or not all(
                    predictor.metadata.get(key)
                    for key in ("model_version", "training_checkpoint_sha256")
                ):
                    raise ValueError("Model selection has not been locked")
                app.state.predictor = predictor
            except Exception as exc:
                logger.error("Model could not be loaded (%s)", type(exc).__name__)
        if settings.history_enabled:
            from pathlib import Path

            if not settings.database_url.startswith("sqlite:///"):
                raise ValueError("Only local SQLite history is supported")
            Path(settings.database_url.removeprefix("sqlite:///")).parent.mkdir(
                parents=True, exist_ok=True
            )
            app.state.history = History(settings.database_url)
        yield
        if app.state.history:
            app.state.history.engine.dispose()

    app = FastAPI(title="Chest X-ray Research API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware, limit=settings.max_upload_bytes + 65536)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def private_responses(request, call_next):
        origin = request.headers.get("origin")
        if request.method in {"POST", "DELETE"} and origin and origin not in settings.origins:
            return JSONResponse({"detail": "Origin is not allowed."}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(Exception)
    async def unexpected_error(_request, exc):
        logger.error("Request failed (%s)", type(exc).__name__)
        return JSONResponse({"detail": "The request could not be completed."}, status_code=500)

    def predictor():
        if app.state.predictor is None:
            raise HTTPException(
                503, "Model unavailable. Complete training, evaluation, and export first."
            )
        return app.state.predictor

    def history():
        if app.state.history is None:
            raise HTTPException(409, "Local prediction history is disabled.")
        return app.state.history

    def matching_metrics():
        model = app.state.predictor
        path = settings.reports / "evaluation.json"
        if model is None or not path.is_file():
            return None
        try:
            report = json.loads(path.read_text())
            if report.get("checkpoint_sha256") != model.metadata.get("training_checkpoint_sha256"):
                return None
            return report
        except (OSError, ValueError):
            return None

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "device": "cpu",
            "model_available": app.state.predictor is not None,
            "history_enabled": settings.history_enabled,
        }

    @app.get("/api/model/info")
    def info():
        model = app.state.predictor
        return {
            "available": model is not None,
            "architecture": "ResNet18",
            "device": "cpu",
            "version": model.metadata["model_version"] if model else None,
            "threshold": model.metadata["threshold"] if model else None,
            "preprocessing": model.metadata["preprocessing"] if model else None,
            "uncertainty_policy": model.metadata.get("uncertainty_policy") if model else None,
            "dataset": "Kaggle Chest X-Ray Images (Pneumonia)",
            "research_only": True,
            "history_enabled": settings.history_enabled,
        }

    @app.get("/api/model/metrics")
    def metrics():
        report = matching_metrics()
        plots = [
            name
            for name in (
                "confusion_matrix",
                "roc_curve",
                "precision_recall_curve",
                "training_curves",
            )
            if (settings.reports / f"{name}.png").is_file()
        ]
        return {
            "available": report is not None,
            "metrics": report,
            "plots": plots if report else [],
        }

    @app.get("/api/model/plots/{name}")
    def plot(
        name: Literal["confusion_matrix", "roc_curve", "precision_recall_curve", "training_curves"],
    ):
        path = settings.reports / f"{name}.png"
        if matching_metrics() is None or not path.is_file():
            raise HTTPException(404, "Evaluation figure unavailable.")
        return FileResponse(path, media_type="image/png")

    def classify(data, explain, save_history):
        image = decode_image(data, settings)
        model = predictor()
        if save_history and not settings.history_enabled:
            raise HTTPException(409, "Local prediction history is disabled.")
        if not inference_lock.acquire(blocking=False):
            raise HTTPException(503, "The CPU model is busy. Please retry shortly.")
        try:
            result, heatmap, combined = model.predict(image, explain)
        finally:
            inference_lock.release()
        result.pop("smoke", None)
        result.update(
            prediction_id=str(uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            model_version=model.metadata["model_version"],
            saved=save_history,
            images=None,
        )
        if explain:
            from PIL import Image

            result["images"] = {
                "model_input": image_data_url(Letterbox(model.size)(image)),
                "heatmap": image_data_url(Image.fromarray((heatmap * 255).astype("uint8"))),
                "overlay": image_data_url(combined),
            }
        validated = Prediction.model_validate(result).model_dump()
        if save_history:
            history().save(validated)
        return validated

    @app.post("/api/predict", response_model=Prediction)
    async def predict(
        file: UploadFile = File(...), explain: bool = Form(True), save_history: bool = Form(False)
    ):
        try:
            data = await file.read(settings.max_upload_bytes + 1)
        finally:
            await file.close()
        return await run_in_threadpool(classify, data, explain, save_history)

    @app.get("/api/predictions")
    def predictions(
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
        label: Literal["NORMAL", "PNEUMONIA", "INCONCLUSIVE"] | None = None,
        search: str = Query("", max_length=80),
    ):
        if not settings.history_enabled:
            return {"enabled": False, "items": [], "total": 0}
        return history().list(limit, offset, label, search)

    @app.get("/api/predictions/{identifier}", response_model=Prediction)
    def prediction(identifier: UUID):
        record = history().get(str(identifier))
        if record is None:
            raise HTTPException(404, "Prediction not found.")
        return record

    @app.delete("/api/predictions/{identifier}", status_code=204)
    def delete_prediction(identifier: UUID):
        if not history().delete(str(identifier)):
            raise HTTPException(404, "Prediction not found.")
        return Response(status_code=204)

    return app


app = create_app()
