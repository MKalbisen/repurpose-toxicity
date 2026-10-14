#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Install Python 3.11 and an appropriate CUDA PyTorch wheel before running.
python -m repurpose.academic.experiment freeze
python -m repurpose.academic.experiment train-toxicity
python -m repurpose.academic.experiment train-kg
# Test evaluation is intentionally an explicit later command after development finishes.
echo 'Development training complete. Return reports/academic/runs/academic-v2 for locked evaluation.'
