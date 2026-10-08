"""Save executed copies without changing the source notebooks."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

parser = argparse.ArgumentParser()
parser.add_argument("numbers", nargs="*", default=["01", "02", "03"])
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--approved-full-training", action="store_true")
parser.add_argument("--selection-locked", action="store_true")
args = parser.parse_args()
if args.smoke and args.approved_full_training:
    parser.error("Choose smoke or approved full training")
destination = Path("artifacts/notebooks")
destination.mkdir(parents=True, exist_ok=True)
for number in args.numbers:
    path = next(Path("notebooks").glob(f"{number}_*.ipynb"))
    notebook = nbformat.read(path, as_version=4)
    if args.smoke and number == "04":
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.source = cell.source.replace("RUN_SMOKE_TEST = False", "RUN_SMOKE_TEST = True")
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        if args.approved_full_training and number == "04":
            cell.source = cell.source.replace(
                "FULL_TRAINING_APPROVED = False", "FULL_TRAINING_APPROVED = True"
            )
        if args.selection_locked and number == "05":
            cell.source = cell.source.replace(
                "MODEL_SELECTION_LOCKED = False", "MODEL_SELECTION_LOCKED = True"
            )
    print(f"Executing {path.name}", flush=True)
    NotebookClient(
        notebook,
        timeout=14400 if number == "04" else 1200,
        kernel_name="chest-xray-cpu",
        resources={"metadata": {"path": str(Path.cwd())}},
    ).execute()
    nbformat.write(notebook, destination / path.name)
    print(f"Saved {destination / path.name}", flush=True)
