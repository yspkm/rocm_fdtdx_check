#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
python3 -m compileall -q src
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
python3 - <<'PY'
from pathlib import Path
import tomllib
import yaml
for path in sorted(Path("configs").glob("*.yaml")):
    data = yaml.safe_load(path.read_text())
    assert data["schema"] == 1, path
    numerics = data["numerics"]
    assert numerics["precision"] == "float64", path
    assert numerics["enable_x64"] is True, path
    pool = data["device_pool"]
    assert isinstance(pool.get("hardware_id"), str) and pool["hardware_id"], path
    expected_counts = pool.get("expected_logical_device_counts", [])
    assert isinstance(expected_counts, list), path
    assert all(isinstance(value, int) and value > 0 for value in expected_counts), path
    accepted_kinds = pool.get("accepted_device_kind_substrings", [])
    assert isinstance(accepted_kinds, list), path
    assert all(isinstance(value, str) and value for value in accepted_kinds), path
    if "memory_gib_per_physical_accelerator" in pool:
        assert pool["aggregate_hbm_gib"] == (
            pool["expected_physical_accelerators"]
            * pool["memory_gib_per_physical_accelerator"]
        ), path
    if "science" in data:
        science = data["science"]
        assert isinstance(science.get("device_counts"), list) and science["device_counts"], path
        times = [float(value) for value in science["time_samples_fs"]]
        assert len(times) >= 3 and times == sorted(set(times)), path
        assert float(science["pml_thickness_um"]) > 0, path
        validation = science["validation"]
        required = {
            "minimum_input_overlap_magnitude",
            "minimum_total_transmission",
            "maximum_total_transmission",
            "maximum_imbalance_db",
            "maximum_material_mirror_relative_error",
            "maximum_output_neff_relative_difference",
            "maximum_time_magnitude_relative_drift",
        }
        assert required <= validation.keys(), path
        assert float(validation["maximum_total_transmission"]) <= 1.10, path
source = Path("src/fdtdx_check/simulation.py").read_text()
assert "jnp.complex128 if dtype == jnp.float64" in source
assert source.count("dtype=detector_dtype") >= 3
assert '"detector_precision"' in source
assert '"propagation_precision"' in source
assert '"output_waveguide_positive_x_pml_contact"' in source
assert '"field_norm2"' in source and "field_energy" not in source
metadata = tomllib.loads(Path("pyproject.toml").read_text())
assert metadata["tool"]["hatch"]["metadata"]["allow-direct-references"] is True
print("YAML PASS")
PY
bash -n run.sh setup.sh docker.sh qa.sh
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git diff --check
fi
if rg -n --hidden --glob '!.git/**' --glob '!LICENSE' \
  '(/mnt/[a-z]/|[A-Za-z]:\\\\|@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|BEGIN (RSA |OPENSSH )?PRIVATE KEY|[0-9]{1,3}(\.[0-9]{1,3}){3})' .; then
  printf '%s\n' 'Privacy scan found a path, email, key, or IP-like value.' >&2
  exit 1
fi
printf '%s\n' 'QA PASS'
