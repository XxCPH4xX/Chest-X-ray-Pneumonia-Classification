"""Small CPU training runs with epoch-boundary recovery and strict test isolation."""

import argparse
import hashlib
import json
import random
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ml.data.dataset import CLASSES, XRayDataset
from ml.data.transforms import preprocessing, specification
from ml.evaluation.metrics import classification_metrics
from ml.models.resnet import build_model, set_stage, training_mode
from ml.runtime import configure
from ml.training.checkpoint import read_checkpoint, save_checkpoint


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def make_loader(dataset, config, training=False, generator=None):
    workers = config["num_workers"]
    if not 0 <= workers <= 2 or not 1 <= config["batch_size"] <= 8:
        raise ValueError("CPU profile supports 0–2 workers and batch sizes 1–8")
    options = {"prefetch_factor": 1} if workers else {}
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=training,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
        **options,
    )


def run_epoch(model, loader, criterion, optimizer=None, stage="head", threshold=0.5):
    training = optimizer is not None
    training_mode(model, stage) if training else model.eval()
    total_loss, labels, scores = 0.0, [], []
    start = time.perf_counter()
    with torch.set_grad_enabled(training):
        for images, targets in tqdm(loader, desc="Train" if training else "Validate", leave=False):
            images, targets = images.to("cpu"), targets.to("cpu", dtype=torch.float32)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images).flatten()
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; checkpoint not advanced")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(targets)
            labels.extend(targets.tolist())
            scores.extend(logits.detach().sigmoid().tolist())
    return {
        "loss": total_loss / len(labels),
        "seconds": time.perf_counter() - start,
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        **classification_metrics(labels, scores, threshold),
    }


def rng_state(generator):
    state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "loader": generator.get_state(),
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
    }


def restore_rng(state, generator):
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    random.setstate(state["python"])
    name, values, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(values, dtype=np.uint32), position, has_gauss, cached))


def plot_history(history, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ("train", "val"):
        axes[0].plot([h[split]["loss"] for h in history], marker="o", label=split)
        axes[1].plot([h[split]["roc_auc"] for h in history], marker="o", label=split)
    for ax, title in zip(axes, ("BCE loss", "ROC-AUC")):
        ax.set(xlabel="Epoch index", title=title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train(
    config: dict,
    smoke: bool = True,
    approved_full_training: bool = False,
    resume: str | None = None,
):
    """Run a capped smoke test or explicitly approved training. Never reads test images."""
    if not smoke and not approved_full_training:
        raise PermissionError("Full training requires explicit user approval")
    if config.get("device") != "cpu":
        raise ValueError("This project only supports CPU execution")
    env = configure(config["seed"], config["threads"])
    manifest_path = Path(config["manifest"])
    manifest = pd.read_csv(manifest_path).fillna("")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if smoke:
        manifest = pd.concat(
            [
                g.head(16 if split == "train" else 8)
                for (split, _), g in manifest.groupby(["split", "target"])
                if split in ("train", "val")
            ]
        )
    datasets = {
        s: XRayDataset(
            config["dataset_root"], manifest, s, preprocessing(config["image_size"], s == "train")
        )
        for s in ("train", "val")
    }
    generator = torch.Generator().manual_seed(config["seed"])
    loaders = {s: make_loader(datasets[s], config, s == "train", generator) for s in datasets}
    counts = datasets["train"].rows.target.value_counts()
    if set(counts.index) != {0, 1}:
        raise ValueError("Training requires both classes")
    # Use loss weighting only, never combine with an oversampling sampler.
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([counts[0] / counts[1]]))
    state = read_checkpoint(resume) if resume else None
    if state:
        if state["manifest_sha256"] != manifest_hash or state["smoke"] != smoke:
            raise ValueError("Resume requires the same manifest and run type")
        if state["config"] != config:
            raise ValueError("Resume configuration must match the checkpoint")
    model = build_model(pretrained=config["pretrained"] and not state)
    if state:
        model.load_state_dict(state["model_state"])
    out = Path(config["checkpoint_dir"]) / ("smoke" if smoke else "run")
    out.mkdir(parents=True, exist_ok=True)
    history = state["history"] if state else []
    best_score = state["best_score"] if state else -1.0
    stale = state["stale_epochs"] if state else 0
    epochs = 1 if smoke else config["head_epochs"] + config["fine_tune_epochs"]
    start_epoch = state["next_epoch"] if state else 0
    optimizer, scheduler, previous_stage = None, None, None
    if state and state.get("early_stopped"):
        return history
    try:
        for epoch in range(start_epoch, epochs):
            stage = "head" if epoch < config["head_epochs"] or smoke else "finetune"
            set_stage(model, stage)
            if stage != previous_stage:
                lr = config["learning_rate_head" if stage == "head" else "learning_rate_finetune"]
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=lr,
                    weight_decay=config["weight_decay"],
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=1
                )
                if state and state["stage"] == stage:
                    optimizer.load_state_dict(state["optimizer_state"])
                    scheduler.load_state_dict(state["scheduler_state"])
            if state:
                restore_rng(state["rng"], generator)
                state = None

            def payload(next_epoch, early_stopped=False):
                return {
                    "format_version": 1,
                    "architecture": "resnet18",
                    "smoke": smoke,
                    "class_mapping": CLASSES,
                    "threshold": config["threshold"],
                    "preprocessing": specification(config["image_size"]),
                    "config": config,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "next_epoch": next_epoch,
                    "stage": stage,
                    "best_score": best_score,
                    "stale_epochs": stale,
                    "early_stopped": early_stopped,
                    "history": history,
                    "manifest_sha256": manifest_hash,
                    "environment": env,
                    "rng": rng_state(generator),
                }

            # Interrupting an epoch replays it from this consistent model/optimizer/RNG boundary.
            save_checkpoint(payload(epoch), out / "last.pt")
            started = time.perf_counter()
            training = run_epoch(
                model, loaders["train"], criterion, optimizer, stage, config["threshold"]
            )
            validation = run_epoch(model, loaders["val"], criterion, threshold=config["threshold"])
            score = validation["roc_auc"]
            if score is None:
                raise ValueError("Validation ROC-AUC undefined; inspect the split")
            scheduler.step(score)
            improved = score > best_score
            best_score, stale = (score, 0) if improved else (best_score, stale + 1)
            history.append(
                {
                    "epoch": epoch + 1,
                    "stage": stage,
                    "train": training,
                    "val": validation,
                    "seconds": time.perf_counter() - started,
                }
            )
            early_stop = stale >= config["early_stopping_patience"]
            checkpoint = payload(epoch + 1, early_stop)
            save_checkpoint(checkpoint, out / "last.pt")
            if improved:
                save_checkpoint(checkpoint, out / "best.pt")
            (out / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False))
            plot_history(history, out / "training_curves.png")
            print(
                f"Epoch {epoch + 1}/{epochs}: val AUC={score:.4f}, "
                f"loss={validation['loss']:.4f}, elapsed={history[-1]['seconds']:.1f}s"
            )
            previous_stage = stage
            if early_stop:
                break
    except KeyboardInterrupt:
        print(f"Interrupted. Resume at the saved epoch boundary: {out / 'last.pt'}")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu.yaml")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--approved-full-training", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    for key in ("batch_size", "image_size", "num_workers"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    train(
        config,
        smoke=not args.full,
        approved_full_training=args.approved_full_training,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
