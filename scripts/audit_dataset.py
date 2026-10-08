import argparse
from pathlib import Path

from ml.data.dataset import audit_dataset, build_manifest

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="chest_xray")
parser.add_argument("--output", default="artifacts/data")
args = parser.parse_args()
inventory = audit_dataset(args.root, args.output)
manifest = build_manifest(inventory, Path(args.output) / "manifest.csv")
print(manifest.groupby(["split", "label"]).size().to_string())
