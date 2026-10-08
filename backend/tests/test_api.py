import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.schemas import Prediction
from ml.data.transforms import specification


class FixturePredictor:
    """Predictable responses for API tests."""

    size = 64
    metadata = {
        "selection_locked": True,
        "model_version": "test-model",
        "threshold": 0.5,
        "preprocessing": specification(64),
        "training_checkpoint_sha256": "test-checkpoint",
    }

    def __init__(self, _path):
        pass

    def predict(self, image, explain):
        return (
            {
                "predicted_class": "NORMAL",
                "probabilities": {"NORMAL": 0.7, "PNEUMONIA": 0.3},
                "threshold": 0.5,
                "architecture": "resnet18",
                "smoke": False,
                "gradcam_available": explain,
                "gradcam_target": "NORMAL",
                "uncertainty_policy": None,
                "seconds": 0.01,
            },
            np.zeros((64, 64), dtype=np.float32),
            image.resize((64, 64)),
        )


def encoded(format="PNG", size=(80, 100)):
    output = io.BytesIO()
    Image.new("RGB", size, "gray").save(output, format=format)
    return output.getvalue()


@pytest.fixture
def settings(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    return Settings(
        checkpoint=checkpoint,
        reports=tmp_path / "reports",
        database_url=f"sqlite:///{tmp_path / 'history.sqlite'}",
        history_enabled=True,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, FixturePredictor), base_url="http://localhost") as client:
        yield client


def test_health_and_metadata(client):
    assert client.get("/api/health").json()["device"] == "cpu"
    assert client.get("/api/model/info").json()["version"] == "test-model"
    assert client.get("/api/model/metrics").json() == {
        "available": False,
        "metrics": None,
        "plots": [],
    }


def test_image_response_and_history(client):
    response = client.post(
        "/api/predict",
        files={"file": ("untrusted.txt", encoded(), "text/plain")},
        data={"explain": "true", "save_history": "true"},
    )
    assert response.status_code == 200, response.text
    result = Prediction.model_validate(response.json())
    assert result.images.overlay.startswith("data:image/png;base64,")
    assert response.headers["cache-control"] == "no-store"
    assert result.saved
    listing = client.get("/api/predictions?label=NORMAL").json()
    assert listing["total"] == 1
    assert "images" not in listing["items"][0]
    assert listing["items"][0]["gradcam_available"] is False
    assert client.get("/api/predictions?label=PNEUMONIA").json()["total"] == 0
    assert client.get("/api/predictions?search=no-match").json()["total"] == 0
    detail = client.get(f"/api/predictions/{result.prediction_id}")
    assert detail.status_code == 200
    assert detail.json()["images"] is None
    assert client.delete(f"/api/predictions/{result.prediction_id}").status_code == 204
    assert client.get(f"/api/predictions/{result.prediction_id}").status_code == 404
    assert client.get("/api/predictions").json()["total"] == 0


@pytest.mark.parametrize(
    ("data", "status"),
    [(b"corrupt", 400), (b"", 400), (encoded("GIF"), 415), (encoded(size=(4097, 1)), 413)],
)
def test_bad_content(client, data, status):
    response = client.post("/api/predict", files={"file": ("x.png", data, "image/png")})
    assert response.status_code == status


def test_upload_limits(client, settings):
    response = client.post(
        "/api/predict", files={"file": ("x.png", b"x" * (settings.max_upload_bytes + 1))}
    )
    assert response.status_code == 413
    response = client.post("/api/predict", content=b"tiny", headers={"content-length": "999999999"})
    assert response.status_code == 413
    chunks = (b"x" * 65536 for _ in range(140))
    response = client.post("/api/predict", content=chunks)
    assert response.status_code == 413


def test_missing_model_and_disabled_history(settings):
    settings.checkpoint.unlink()
    settings.history_enabled = False
    with TestClient(create_app(settings), base_url="http://localhost") as client:
        assert client.get("/api/health").json()["model_available"] is False
        response = client.post("/api/predict", files={"file": ("x.png", encoded())})
        assert response.status_code == 503
        assert "Model unavailable" in response.json()["detail"]
        assert client.get("/api/predictions").json() == {"enabled": False, "items": [], "total": 0}
    assert not settings.checkpoint.parent.joinpath("history.sqlite").exists()


def test_no_history_without_opt_in(client):
    assert client.post("/api/predict", files={"file": ("x.png", encoded())}).status_code == 200
    assert client.get("/api/predictions").json()["total"] == 0


def test_metrics_must_match_model(client, settings):
    settings.reports.mkdir()
    report = settings.reports / "evaluation.json"
    report.write_text(json.dumps({"checkpoint_sha256": "other-checkpoint"}))
    assert client.get("/api/model/metrics").json()["available"] is False
    report.write_text(json.dumps({"checkpoint_sha256": "test-checkpoint", "accuracy": 0.5}))
    assert client.get("/api/model/metrics").json()["available"] is True
    assert client.get("/api/model/plots/../../.env").status_code != 200


def test_cors_and_host(client):
    assert (
        client.post(
            "/api/predict",
            files={"file": ("x.png", encoded())},
            headers={"Origin": "https://untrusted.example"},
        ).status_code
        == 403
    )
    response = client.options(
        "/api/predict",
        headers={"Origin": "https://untrusted.example", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
    assert client.get("/api/health", headers={"host": "untrusted.example"}).status_code == 400
