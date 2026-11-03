#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .tools/bin/uv ]]; then
  echo "Run bash scripts/setup.sh first" >&2
  exit 1
fi
.tools/bin/uv venv --python 3.12 data/ibl-venv
.tools/bin/uv pip sync --python data/ibl-venv/bin/python edge/requirements-ibl-lock.txt
echo "IBL environment: data/ibl-venv/bin/python"
