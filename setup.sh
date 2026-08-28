#!/usr/bin/env bash
set -euo pipefail

backend="${1:-}"
[[ "$backend" == "cuda" || "$backend" == "rocm" ]] || {
  printf 'usage: %s cuda|rocm\n' "$0" >&2
  exit 2
}
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_cmd="${PYTHON_CMD:-python3.12}"

"$python_cmd" -m venv "$root/.venv"
python_bin="$root/.venv/bin/python"
"$python_bin" -m pip install --upgrade pip wheel

if [[ "$backend" == "cuda" ]]; then
  "$python_bin" -m pip install --upgrade 'jax[cuda12]==0.11.0'
  constraints="$root/requirements/cuda.txt"
else
  "$python_bin" -m pip install \
    'jax==0.8.2' \
    'jax-rocm7-pjrt==0.8.2' \
    'jax-rocm7-plugin==0.8.2' \
    'https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.8.2/jaxlib-0.8.2+rocm7-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
  constraints="$root/requirements/rocm.txt"
fi

"$python_bin" -m pip install --constraint "$constraints" --editable "$root"
"$python_bin" -m pip check
printf 'Ready. Run: ./run.sh inventory configs/%s.yaml\n' \
  "$([[ "$backend" == cuda ]] && printf cuda || printf rocm-mi250x-profile-fp64)"
