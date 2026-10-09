#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../frontend"
exec conda run --no-capture-output -n chest-xray-web npm run start
