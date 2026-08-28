#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${FDTDX_CHECK_IMAGE:-fdtdx-check:rocm7.2.4}"
command="${1:-}"
shift || true

[[ -n "$command" ]] || {
  printf 'usage: %s build|inventory|profile|science [config]\n' "$0" >&2
  exit 2
}

if [[ "$command" == build ]]; then
  exec docker build --tag "$image" "$root"
fi

[[ -e /dev/kfd ]] || { printf '%s\n' 'Missing /dev/kfd.' >&2; exit 1; }
render="$(find /dev/dri -maxdepth 1 -name 'renderD*' -print -quit)"
[[ -n "$render" ]] || { printf '%s\n' 'Missing a render node.' >&2; exit 1; }
mkdir -p "$root/results"

args=("$command")
if (( $# )); then
  args+=("$@")
else
  case "$command" in
    science) args+=(configs/rocm-mi250x-science-fp64.yaml) ;;
    *) args+=(configs/rocm-mi250x-profile-fp64.yaml) ;;
  esac
fi

exec docker run --rm \
  --network none \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add "$(stat -c '%g' /dev/kfd)" \
  --group-add "$(stat -c '%g' "$render")" \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --ipc private \
  --shm-size 64g \
  --user "$(id -u):$(id -g)" \
  --volume "$root/results:/app/results" \
  "$image" "${args[@]}"
