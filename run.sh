#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/.venv/bin/python"
[[ -x "$python_bin" ]] || python_bin="${PYTHON:-python3}"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m fdtdx_check "$@"
