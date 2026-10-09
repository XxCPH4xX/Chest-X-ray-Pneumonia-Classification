"""Audit the source dataset and build explicit, reproducible split manifests."""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from tqdm.auto import tqdm

CLASSES = {"NORMAL": 0, "PNEUMONIA": 1}
SPLITS = ("train", "val", "test")


def roots_under(path: str | Path) -> list[Path]:
    """Find complete dataset roots, including duplicate nested extractions."""
    path = Path(path)
    candidates = [path, *sorted(p for p in path.rglob("*") if p.is_dir())]
    return [p for p in candidates if all((p / s / c).is_dir() for s in SPLITS for c in CLASSES)]


def patient_proxy(name: str) -> str | None:
    """Infer filename groups; these are proxies, not verified patient identifiers."""
    if match := re.match(r"(person\d+)_", name):
        return match[1]
    if match := re.match(r"((?:NORMAL2-)?IM-\d+)-", name):
        return match[1]
    return None


def audit_dataset(root: str | Path, output: str | Path = "artifacts/data") -> pd.DataFrame:
    """Decode every image and record errors and duplicates without silently omitting files."""
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    roots = roots_under(root)
    if not roots:
        raise FileNotFoundError(f"No train/val/test root with both classes under {root}")
    if root not in roots:
        if len(roots) != 1:
            raise ValueError(f"Select an explicit dataset root: {roots}")
        root = roots[0]
    rows = []
    ignored = []
    for split in SPLITS:
        for label, target in CLASSES.items():
            for path in tqdm(
                sorted((root / split / label).iterdir()), desc=f"Audit {split}/{label}"
            ):
                if path.name.startswith(".") or path.is_dir():
                    ignored.append(str(path.relative_to(root)))
                    continue
                row = {
                    "path": str(path.relative_to(root)),
                    "source_split": split,
                    "label": label,
                    "target": target,
                    "patient_proxy": patient_proxy(path.name),
                    "error": "",
                    "sha256": "",
                    "pixel_sha256": "",
                }
                try:
                    row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    with Image.open(path) as im:
                        im.load()
                        row.update(width=im.width, height=im.height, format=im.format)
                        rgb = im.convert("RGB")
                        row["pixel_sha256"] = hashlib.sha256(
                            str(rgb.size).encode() + rgb.tobytes()
                        ).hexdigest()
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    row["error"] = str(exc)
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "inventory.csv", index=False)
    valid = frame[frame.error.eq("")]
    duplicates = valid[valid.duplicated("pixel_sha256", keep=False)]
    duplicates.to_csv(output / "duplicates.csv", index=False)
    overlap = valid.groupby("patient_proxy").source_split.nunique()
    copy_checks = []
    for other in roots:
        if other == root:
            continue
        same = all(
            (other / r.path).is_file()
            and hashlib.sha256((other / r.path).read_bytes()).hexdigest() == r.sha256
            for r in frame.itertuples()
        )
        files = {
            str(p.relative_to(other))
            for s in SPLITS
            for c in CLASSES
            for p in (other / s / c).iterdir()
            if p.is_file() and not p.name.startswith(".")
        }
        copy_checks.append(
            {"root": str(other), "identical_copy": same and files == set(frame.path)}
        )
    report = {
        "selected_root": str(root),
        "images": len(frame),
        "unreadable": int(frame.error.ne("").sum()),
        "duplicate_image_rows": len(duplicates),
        "filename_groups_across_source_splits": overlap[overlap > 1].index.tolist(),
        "unknown_patient_proxies": int(frame.patient_proxy.isna().sum()),
        "nested_copies": copy_checks,
        "ignored_metadata": ignored,
        "patient_separation_verified": False,
        "limitation": "Filename groups are heuristics; no authoritative patient metadata.",
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2))
    return frame


def build_manifest(inventory: pd.DataFrame, output: str | Path, seed: int = 42) -> pd.DataFrame:
    """Group by filename proxy and pixel identity; reserve original test unchanged."""
    df = inventory.copy().reset_index(drop=True)
    if df.error.fillna("").ne("").any():
        raise ValueError("Unreadable images found; review inventory.csv before creating splits")
    if df.patient_proxy.fillna("").eq("").any():
        raise ValueError("Unknown patient proxies; supply reviewed grouping metadata first")
    if (df.groupby("pixel_sha256").target.nunique() > 1).any():
        raise ValueError("Identical images have conflicting labels; review duplicates.csv")
    # Union filename groups linked by identical decoded pixels.
    parent = {x: x for x in df.patient_proxy}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, group in df.groupby("pixel_sha256"):
        first = group.patient_proxy.iloc[0]
        for patient in group.patient_proxy:
            parent[find(patient)] = find(first)
    df["group"] = df.patient_proxy.map(find)
    test_groups = set(df.loc[df.source_split.eq("test"), "group"])
    df["split"] = df.source_split
    df["exclusion_reason"] = ""
    overlap = ~df.source_split.eq("test") & df.group.isin(test_groups)
    df.loc[overlap, ["split", "exclusion_reason"]] = ["excluded", "group_overlaps_test"]
    # Original 16 validation images are retained as an auxiliary set, not used for selection.
    df.loc[df.source_split.eq("val") & ~overlap, "split"] = "auxiliary"
    pool = df[df.split.eq("train")]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, val_indices = next(splitter.split(pool, pool.target, pool.group))
    df.loc[pool.iloc[val_indices].index, "split"] = "val"
    # Every training-side duplicate stays in the audit trail with an explicit exclusion reason.
    dev = df[df.split.isin(["train", "val"])]
    dup_idx = dev[dev.duplicated("pixel_sha256")].index
    df.loc[dup_idx, ["split", "exclusion_reason"]] = ["excluded", "duplicate_pixels"]
    groups = [set(df.loc[df.split.eq(s), "group"]) for s in ("train", "val", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    for split in ("train", "val", "test"):
        if set(df.loc[df.split.eq(split), "target"]) != {0, 1}:
            raise ValueError(f"Both classes required in {split}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


class XRayDataset(Dataset):
    def __init__(self, root, manifest: pd.DataFrame, split: str, transform):
        self.root = Path(root)
        self.rows = manifest[manifest.split.eq(split)].reset_index(drop=True)
        self.transform = transform
        if self.rows.empty:
            raise ValueError(f"Empty split: {split}")
        if not set(self.rows.target).issubset({0, 1}):
            raise ValueError("Unknown class target")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(self.root / row.path) as im:
            image = self.transform(im.convert("RGB"))
        return image, float(row.target)
