# Repository handling

The workspace already has a Git repository on `main`, with origin:

`https://github.com/XxCPH4xX/Chest-X-ray-Pneumonia-Classification.git`

The GitHub API confirmed this repository is **private**, with `main` as its default branch.
It differs from the requested preferred name `Chest-XRay-Pneumonia-Detection`. No visibility
change or public deployment has been performed.

Work is on `codex/cpu-only-environment`. The dataset is untracked in the new source snapshot
but remains on disk. The original local `main` and its dataset-containing history are preserved.
An existing transfer process was observed before these changes; it was not started, stopped,
or otherwise controlled by this work. No new push was issued.

The existing local initial commit includes 11,720 dataset and metadata files. Ignoring or
untracking the dataset does not remove those blobs from history. A normal push of this local
history would still transfer the images. Preserve the original repository as a local backup;
publish a clean source snapshot separately instead of force-pushing rewritten history.

The available GitHub connector supports reading and editing existing repositories but exposes
no repository-creation tool, and the GitHub CLI is not installed. A new preferred-name repository
has therefore not been created. Once `gh` is available and authenticated, these commands create
a separate source-only repository from the verified commit (run from this project root):

```bash
mkdir ../Chest-XRay-Pneumonia-Detection-source
git archive HEAD | tar -x -C ../Chest-XRay-Pneumonia-Detection-source
cd ../Chest-XRay-Pneumonia-Detection-source
git init -b main
git add .
git commit -m "Add CPU-only notebook research pipeline"
gh auth status
gh repo create XxCPH4xX/Chest-XRay-Pneumonia-Detection --private --source=. --remote=origin
gh repo view XxCPH4xX/Chest-XRay-Pneumonia-Detection --json visibility
# Confirm PRIVATE before pushing:
git push -u origin main
```

This workflow preserves the existing local history and dataset files. Do not run the export
until the current commit contains source only (`git ls-tree -r --name-only HEAD` should not
include `chest_xray/` or `artifacts/`). No commits use fabricated authors or timestamps.
