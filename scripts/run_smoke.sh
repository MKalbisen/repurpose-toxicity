#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python environment_check.py
repurpose --config configs/smoke.yaml run "$@"
python -m pytest -q
