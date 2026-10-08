"""Execute local notebook copies; keep source notebooks sanitized for version control."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

parser = argparse.ArgumentParser()
parser.add_argument("numbers", nargs="*", default=["01", "02", "03"])
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
destination = Path("artifacts/notebooks")
destination.mkdir(parents=True, exist_ok=True)
for number in args.numbers:
    path = next(Path("notebooks").glob(f"{number}_*.ipynb"))
    notebook = nbformat.read(path, as_version=4)
    if args.smoke and number == "04":
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.source = cell.source.replace("RUN_SMOKE_TEST = False", "RUN_SMOKE_TEST = True")
    print(f"Executing {path.name}", flush=True)
    NotebookClient(
        notebook,
        timeout=1200,
        kernel_name="chest-xray-cpu",
        resources={"metadata": {"path": str(Path.cwd())}},
    ).execute()
    nbformat.write(notebook, destination / path.name)
    print(f"Saved {destination / path.name}", flush=True)
