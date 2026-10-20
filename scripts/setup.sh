#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
if [[ ! -x .tools/bin/uv ]]; then
  python3 -m pip install --target .tools 'uv==0.13.3'
fi
.tools/bin/uv python install 3.12.15 --no-bin
.tools/bin/uv venv .venv --python 3.12.15 --allow-existing
.tools/bin/uv pip sync --python .venv/bin/python requirements-lock.txt
.tools/bin/uv pip check --python .venv/bin/python
