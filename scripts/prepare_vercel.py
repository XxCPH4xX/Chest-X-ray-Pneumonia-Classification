"""Build the hosted frontend and collect the files needed for CPU inference."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from ml.training.checkpoint import read_checkpoint

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "artifacts/deploy/vercel"
REPORTS = (
    "evaluation.json",
    "confusion_matrix.png",
    "roc_curve.png",
    "precision_recall_curve.png",
    "training_curves.png",
)


def main():
    model_path = ROOT / "artifacts/model/model.pt"
    model = read_checkpoint(model_path)
    report = json.loads((ROOT / "artifacts/reports/evaluation.json").read_text())
    if model.get("smoke") or not model.get("selection_locked") or not model.get("model_version"):
        raise ValueError("An exported, selected research model is required")
    if report["checkpoint_sha256"] != model["training_checkpoint_sha256"]:
        raise ValueError("Evaluation does not match the exported model")
    for name in REPORTS:
        if not (ROOT / "artifacts/reports" / name).is_file():
            raise FileNotFoundError(name)

    environment = {**os.environ, "STATIC_EXPORT": "true", "NEXT_PUBLIC_HOSTED": "true"}
    environment["NEXT_PUBLIC_API_URL"] = ""
    subprocess.run(["npm", "run", "build"], cwd=ROOT / "frontend", env=environment, check=True)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    # Keep the Vercel project link when refreshing an existing bundle.
    for name in ("backend", "ml", "public", "artifacts"):
        path = DESTINATION / name
        if path.exists():
            shutil.rmtree(path)
    for package in ("backend", "ml"):
        for source in (ROOT / package).rglob("*.py"):
            if "tests" in source.parts:
                continue
            target = DESTINATION / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    shutil.copytree(ROOT / "frontend/out", DESTINATION / "public")
    for source in (ROOT / "deploy/vercel").iterdir():
        if source.is_file():
            shutil.copyfile(source, DESTINATION / source.name)
    model_target = DESTINATION / "artifacts/model/model.pt"
    model_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_path, model_target)
    (DESTINATION / "artifacts/reports").mkdir(parents=True, exist_ok=True)
    for name in REPORTS:
        shutil.copyfile(ROOT / "artifacts/reports" / name, DESTINATION / "artifacts/reports" / name)
    (DESTINATION / ".vercelignore").write_text(".vercel\n__pycache__\n*.pyc\n")
    inventory = {
        "model_version": model["model_version"],
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "files": sorted(
            str(path.relative_to(DESTINATION))
            for path in DESTINATION.rglob("*")
            if path.is_file() and ".vercel" not in path.relative_to(DESTINATION).parts
        ),
    }
    (DESTINATION.parent / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(f"Deployment bundle: {DESTINATION}")


if __name__ == "__main__":
    main()
