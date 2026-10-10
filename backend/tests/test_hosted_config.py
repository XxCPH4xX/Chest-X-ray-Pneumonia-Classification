import runpy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app

ENTRYPOINT = Path(__file__).resolve().parents[2] / "deploy/vercel/app.py"


def test_hosted_entrypoint_restricts_hosts_origins_and_history(monkeypatch):
    monkeypatch.setenv("VERCEL_URL", "preview.example.com")
    monkeypatch.setenv("VERCEL_PROJECT_PRODUCTION_URL", "research.example.com")
    monkeypatch.delenv("VERCEL_BRANCH_URL", raising=False)
    monkeypatch.delenv("ALLOWED_HOSTS", raising=False)
    monkeypatch.setenv("HISTORY_ENABLED", "true")
    monkeypatch.setenv("MODEL_CHECKPOINT", "/nonexistent/model.pt")
    app = runpy.run_path(str(ENTRYPOINT))["app"]
    with TestClient(app, base_url="https://research.example.com") as client:
        assert client.get("/api/health").json()["history_enabled"] is False
        assert (
            client.get("/api/health", headers={"host": "untrusted.example.com"}).status_code == 400
        )
        assert (
            client.post(
                "/api/predict", headers={"origin": "https://untrusted.example.com"}
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/predict", headers={"origin": "https://research.example.com"}
            ).status_code
            == 422
        )
        assert (
            client.post("/api/predict", content=b"x" * (3 * 1024 * 1024 + 65537)).status_code == 413
        )


def test_hosted_entrypoint_requires_explicit_hostname(monkeypatch):
    for name in (
        "VERCEL_URL",
        "VERCEL_PROJECT_PRODUCTION_URL",
        "VERCEL_BRANCH_URL",
        "ALLOWED_HOSTS",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="deployment hostname"):
        runpy.run_path(str(ENTRYPOINT))


def test_local_defaults_reject_public_hosts(tmp_path):
    settings = Settings(checkpoint=tmp_path / "missing.pt", history_enabled=False)
    assert settings.max_upload_bytes == 8 * 1024 * 1024
    with TestClient(create_app(settings), base_url="http://localhost") as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/health", headers={"host": "public.example.com"}).status_code == 400
