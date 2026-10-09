#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
conda env create -f environment.yml
conda run -n chest-xray-cpu python -m pip install -r requirements-cpu.txt
conda run -n chest-xray-cpu python -m pip install -r requirements-lock.txt
conda run -n chest-xray-cpu python -m pip install --no-deps -e .
conda run -n chest-xray-cpu python -m ipykernel install --user --name chest-xray-cpu --display-name 'Chest X-ray (CPU)'
conda run -n chest-xray-cpu python -m ml.runtime
