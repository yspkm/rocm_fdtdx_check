from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .geometry import (
    canonical_sharding_multiple,
    science_device_milestones,
    science_grid,
)
from .reporting import profile_html, science_suite_html


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("schema") != 1 or cfg.get("backend") not in {"cuda", "rocm"}:
        raise ValueError("Expected schema=1 and backend=cuda|rocm")
    precision = cfg["numerics"]["precision"]
    if precision not in {"float32", "float64"}:
        raise ValueError("precision must be float32 or float64")
    if precision == "float64" and not cfg["numerics"].get("enable_x64"):
        raise ValueError("float64 requires numerics.enable_x64=true")
    pool = cfg.get("device_pool", {})
    if not str(pool.get("hardware_id", "")).strip():
        raise ValueError("device_pool.hardware_id must be a non-empty string")
    expected_counts = pool.get("expected_logical_device_counts", [])
    if expected_counts and (
        not isinstance(expected_counts, list)
        or any(int(value) < 1 for value in expected_counts)
    ):
        raise ValueError("device_pool.expected_logical_device_counts must be positive integers")
    accepted_kinds = pool.get("accepted_device_kind_substrings", [])
    if accepted_kinds and (
        not isinstance(accepted_kinds, list)
        or any(not str(value).strip() for value in accepted_kinds)
    ):
        raise ValueError("device_pool.accepted_device_kind_substrings must be non-empty strings")
    if "science" in cfg:
        science = cfg["science"]
        times = [float(value) for value in science["time_samples_fs"]]
        if len(times) < 2 or times != sorted(set(times)):
            raise ValueError("science.time_samples_fs must contain at least two increasing values")
        if float(science["pml_thickness_um"]) <= 0:
            raise ValueError("science.pml_thickness_um must be positive")
        if not isinstance(science.get("device_counts"), list) or not science["device_counts"]:
            raise ValueError("science.device_counts must contain stable sharding milestones")
    return cfg


def child_env(cfg: dict[str, Any], devices: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    backend = cfg["backend"]
    env.update(
        JAX_PLATFORMS=f"{backend},cpu",
        JAX_ENABLE_X64=str(bool(cfg["numerics"]["enable_x64"])).lower(),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        MPLCONFIGDIR=env.get("MPLCONFIGDIR", "/tmp/fdtdx-check-mpl"),
        XDG_CACHE_HOME=env.get("XDG_CACHE_HOME", "/tmp/fdtdx-check-cache"),
    )
    if devices is not None:
        visible = ",".join(str(i) for i in range(devices))
        env["FDTDX_CHECK_DEVICE_COUNT"] = str(devices)
        env["CUDA_VISIBLE_DEVICES" if backend == "cuda" else "ROCR_VISIBLE_DEVICES"] = visible
    return env


def invoke(args: list[str], cfg: dict[str, Any], devices: int | None = None) -> int:
    command = [sys.executable, "-m", "fdtdx_check", *args]
    return subprocess.run(command, env=child_env(cfg, devices), check=False).returncode


def inventory(config_path: Path, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    code = invoke(["_inventory", str(config_path), str(output)], cfg)
    if code or not output.exists():
        raise RuntimeError("Accelerator inventory failed")
    return json.loads(output.read_text())


def resolve_counts(raw: list[Any], available: int) -> list[int]:
    values = [available if value == "all" else int(value) for value in raw]
    return sorted({value for value in values if 1 <= value <= available})


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")


def run_profile(config_path: Path, cfg: dict[str, Any]) -> int:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = Path("results") / f"profile-{stamp}"
    inv = inventory(config_path, cfg, root / "inventory.json")
    counts = resolve_counts(cfg["profile"]["device_counts"], inv["logical_device_count"])
    attempts: list[dict[str, Any]] = []
    last_pass: dict[int, dict[str, Any]] = {}

    for count in counts:
        for case in cfg["profile"]["cases"]:
            local = [int(v) for v in case["local_shape"]]
            shape = [local[0] * count, local[1], local[2]]
            output = root / f"{count}d-{case['id']}.json"
            code = invoke(
                ["_profile_case", str(config_path), json.dumps(shape), str(output)], cfg, count
            )
            report = json.loads(output.read_text()) if output.exists() else {"status": "FAIL"}
            if not output.exists() and code in {-9, 137}:
                report = {"status": "FAIL", "failure": {"kind": "OOM"}}
            cells_total = int(report.get("cells_total") or math.prod(shape))
            row = {
                "devices": count,
                "case": case["id"],
                "status": report.get("status"),
                "cells_total": cells_total,
                "cells_per_device": int(report.get("cells_per_device") or cells_total // count),
                "elapsed_seconds": report.get("elapsed_seconds"),
                "failure_kind": report.get("failure", {}).get("kind"),
            }
            attempts.append(row)
            if code == 0:
                last_pass[count] = dict(row)
            else:
                if row["failure_kind"] != "OOM":
                    raise RuntimeError(
                        f"Profile case {case['id']} failed for a non-memory reason; inspect {output}"
                    )
                break

    if not last_pass:
        raise RuntimeError(f"No profile case passed; inspect {root}")
    winner = max(last_pass.values(), key=lambda item: int(item["cells_total"]))
    boundary_observed = any(
        row["devices"] == winner["devices"] and row["failure_kind"] == "OOM"
        for row in attempts
    )
    safe_fraction = float(cfg["profile"]["safe_fraction"])
    capacity = {
        "schema": 1,
        "backend": cfg["backend"],
        "hardware_id": cfg["device_pool"]["hardware_id"],
        "precision": cfg["numerics"]["precision"],
        "logical_devices_available": inv["logical_device_count"],
        "device_kinds": inv["device_kinds"],
        "hardware_contract": inv["hardware_contract"],
        "expected_physical_accelerators": cfg["device_pool"].get(
            "expected_physical_accelerators"
        ),
        "declared_aggregate_hbm_gib": cfg["device_pool"].get("aggregate_hbm_gib"),
        "last_pass_by_device_count": {str(k): dict(v) for k, v in last_pass.items()},
        "largest_tested_cells": int(winner["cells_total"]),
        "capacity_boundary_observed": boundary_observed,
        "capacity_interpretation": (
            "bounded_last_success_before_oom" if boundary_observed else "tested_lower_bound_only"
        ),
        "recommended_safe_cells": int(int(winner["cells_total"]) * safe_fraction),
        "safe_fraction": safe_fraction,
        "attempts": attempts,
    }
    save_yaml(Path("results/capacity.yaml"), capacity)
    (root / "report.html").write_text(profile_html(capacity))
    print(f"capacity=results/capacity.yaml\nreport={root / 'report.html'}")
    return 0


def science_cells(cfg: dict[str, Any], resolution_nm: int, sharding_multiple: int = 1) -> int:
    return int(
        science_grid(cfg["science"], resolution_nm, sharding_multiple)["cells_total"]
    )


def _magnitude(metrics: dict[str, Any], port: str) -> float:
    return math.hypot(float(metrics[f"{port}_real"]), float(metrics[f"{port}_imag"]))


def run_science(config_path: Path, cfg: dict[str, Any]) -> int:
    capacity_path = Path("results/capacity.yaml")
    if not capacity_path.exists():
        raise RuntimeError("Run the profile stage first: ./run.sh profile CONFIG")
    capacity = yaml.safe_load(capacity_path.read_text())
    if (
        capacity["backend"] != cfg["backend"]
        or capacity["precision"] != cfg["numerics"]["precision"]
        or capacity.get("hardware_id") != cfg["device_pool"]["hardware_id"]
    ):
        raise RuntimeError("Capacity hardware/backend/precision does not match the science config")

    safe = int(capacity["recommended_safe_cells"])
    available = int(capacity["logical_devices_available"])
    maximum_raw = cfg["device_pool"]["max_devices"]
    maximum = available if maximum_raw == "all" else min(available, int(maximum_raw))
    device_counts = science_device_milestones(cfg["science"], available, maximum)
    sharding_multiple = canonical_sharding_multiple(device_counts)

    candidates = sorted({int(value) for value in cfg["science"]["resolution_candidates_nm"]})
    viable = [
        value
        for value in candidates
        if science_cells(cfg, value, sharding_multiple) <= safe
    ]
    if not viable:
        raise RuntimeError(
            f"Even the coarsest science grid ({max(candidates)} nm) exceeds the profiled safe capacity"
        )

    memory_raw = cfg["device_pool"]["memory_gib_per_logical_device"]
    memory_per_logical = (
        float(cfg["device_pool"]["aggregate_hbm_gib"]) / available
        if memory_raw == "auto"
        else float(memory_raw)
    )
    budget = memory_per_logical * 1024**3 * float(cfg["device_pool"]["usable_fraction"])
    bytes_per_cell = int(cfg["numerics"]["bytes_per_cell"])
    retry_devices = bool(cfg["device_pool"].get("retry_on_oom"))
    time_samples = [float(value) for value in cfg["science"]["time_samples_fs"]]

    plan: dict[str, Any] = {
        "schema": 1,
        "backend": cfg["backend"],
        "hardware_id": cfg["device_pool"]["hardware_id"],
        "precision": cfg["numerics"]["precision"],
        "validation_level": cfg["science"]["validation_level"],
        "capacity_file": "results/capacity.yaml",
        "profiled_safe_cells": safe,
        "resolution_candidates_nm": candidates,
        "capacity_viable_resolutions_nm": viable,
        "maximum_logical_devices": maximum,
        "science_device_milestones": device_counts,
        "canonical_sharding_multiple": sharding_multiple,
        "expected_physical_accelerators": cfg["device_pool"].get("expected_physical_accelerators"),
        "declared_aggregate_hbm_gib": cfg["device_pool"].get("aggregate_hbm_gib"),
        "derived_hbm_gib_per_logical_device": memory_per_logical,
        "retry_policy": (
            "fresh_process_next_milestone_rerun_all_time_windows_on_oom"
            if retry_devices
            else "fixed_device_count_then_coarsen_resolution_on_oom"
        ),
        "time_samples_fs": time_samples,
        "wavelength_um": cfg["science"]["wavelength_um"],
        "pml_requested_um": cfg["science"]["pml_thickness_um"],
        "geometry": dict(cfg["science"]["geometry"]),
        "validation": dict(cfg["science"]["validation"]),
        "resolution_attempts": [],
    }
    save_yaml(Path("results/science-plan.yaml"), plan)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = Path("results") / f"science-{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    resolution_attempts: list[dict[str, Any]] = []
    case_reports: list[dict[str, Any]] = []
    selected_resolution: int | None = None
    selected_grid: dict[str, Any] | None = None
    selected_device_count: int | None = None
    selected_prefix = ""

    for resolution in viable:
        grid = science_grid(cfg["science"], resolution, sharding_multiple)
        planned_cells = int(grid["cells_total"])
        required = max(
            int(cfg["device_pool"]["min_devices"]),
            math.ceil(planned_cells * bytes_per_cell / budget),
        )
        candidate_counts = [count for count in device_counts if count >= required]
        if not candidate_counts:
            candidate_counts = [device_counts[-1]]
        if not retry_devices:
            candidate_counts = candidate_counts[:1]

        prefix = f"resolution-{resolution:04d}nm"
        device_attempts: list[dict[str, Any]] = []
        successful_reports: list[dict[str, Any]] | None = None
        successful_count: int | None = None
        failure_kind: str | None = None
        failure_time: float | None = None

        for count in candidate_counts:
            local_reports: list[dict[str, Any]] = []
            failure_kind = None
            failure_time = None
            attempt_prefix = f"{prefix}/devices-{count:02d}"
            for time_fs in time_samples:
                case_root = root / attempt_prefix / f"time-{int(time_fs):04d}fs"
                output = case_root / "attempt.json"
                code = invoke(
                    [
                        "_science_case",
                        str(config_path),
                        str(resolution),
                        str(time_fs),
                        str(output),
                        str(case_root),
                        str(sharding_multiple),
                    ],
                    cfg,
                    count,
                )
                if code == 0:
                    local_reports.append(json.loads(output.read_text()))
                    continue
                failure = json.loads(output.read_text()) if output.exists() else {}
                failure_kind = failure.get("failure", {}).get("kind")
                if not output.exists() and code in {-9, 137}:
                    failure_kind = "OOM"
                failure_time = time_fs
                break

            device_attempts.append(
                {
                    "logical_devices": count,
                    "completed_time_samples": len(local_reports),
                    "status": (
                        "PASS_EXECUTION"
                        if len(local_reports) == len(time_samples)
                        else "FAIL"
                    ),
                    "failure_kind": failure_kind,
                    "failure_time_fs": failure_time,
                }
            )
            if len(local_reports) == len(time_samples):
                successful_reports = local_reports
                successful_count = count
                selected_prefix = attempt_prefix
                break
            if failure_kind != "OOM":
                break

        attempt = {
            "resolution_nm": resolution,
            "planned_cells": planned_cells,
            "required_logical_devices": required,
            "candidate_device_milestones": candidate_counts,
            "device_attempts": device_attempts,
            "status": "PASS_EXECUTION" if successful_reports is not None else "FAIL",
            "selected_logical_devices": successful_count,
            "failure_kind": failure_kind,
            "failure_time_fs": failure_time,
        }
        resolution_attempts.append(attempt)
        plan["resolution_attempts"] = resolution_attempts
        save_yaml(Path("results/science-plan.yaml"), plan)

        if successful_reports is not None and successful_count is not None:
            selected_resolution = resolution
            selected_grid = grid
            selected_device_count = successful_count
            case_reports = successful_reports
            break
        if failure_kind != "OOM":
            print(
                f"science run failed for a non-memory reason at {resolution} nm; inspect {root / prefix}",
                file=sys.stderr,
            )
            return 1

    if selected_resolution is None or selected_grid is None or selected_device_count is None:
        print(f"all capacity-viable resolutions exhausted by OOM; inspect {root}", file=sys.stderr)
        return 1

    previous, final = case_reports[-2], case_reports[-1]
    drift_s21 = abs(_magnitude(final["metrics"], "S21") - _magnitude(previous["metrics"], "S21")) / max(
        _magnitude(final["metrics"], "S21"), 1e-30
    )
    drift_s31 = abs(_magnitude(final["metrics"], "S31") - _magnitude(previous["metrics"], "S31")) / max(
        _magnitude(final["metrics"], "S31"), 1e-30
    )
    contract_hashes = {str(report["grid_contract_hash"]) for report in case_reports}
    grid_shapes = {tuple(report["grid_shape"]) for report in case_reports}
    actual_device_counts = {int(report["logical_devices"]) for report in case_reports}
    identical_grid_contract = len(contract_hashes) == 1 and len(grid_shapes) == 1
    fixed_device_count = len(actual_device_counts) == 1
    drift_limit = float(cfg["science"]["validation"]["maximum_time_magnitude_relative_drift"])
    convergence = {
        "comparison_fs": [previous["time_fs"], final["time_fs"]],
        "S21_magnitude_relative_drift": drift_s21,
        "S31_magnitude_relative_drift": drift_s31,
        "maximum_observed_drift": max(drift_s21, drift_s31),
        "maximum_allowed_drift": drift_limit,
        "identical_grid_contract": identical_grid_contract,
        "fixed_device_count": fixed_device_count,
        "grid_contract_hash": final["grid_contract_hash"],
        "pass": (
            max(drift_s21, drift_s31) <= drift_limit
            and identical_grid_contract
            and fixed_device_count
        ),
    }
    case_sanity = all(bool(report["validation"]["all_checks_pass"]) for report in case_reports)
    suite_valid = case_sanity and bool(convergence["pass"])
    cases = [
        {
            "time_fs": report["time_fs"],
            "status": report["status"],
            "logical_devices": report["logical_devices"],
            "grid_contract_hash": report["grid_contract_hash"],
            "grid_shape": report["grid_shape"],
            "steps": report["steps"],
            "elapsed_seconds": report["elapsed_seconds"],
            "metrics": report["metrics"],
            "validation": report["validation"],
            "report_path": f"{selected_prefix}/time-{int(report['time_fs']):04d}fs/report.html",
            "field_path": f"{selected_prefix}/time-{int(report['time_fs']):04d}fs/field.png",
        }
        for report in case_reports
    ]
    summary = {
        "schema": 1,
        "status": "PASS" if suite_valid else "FAIL",
        "validation_level": cfg["science"]["validation_level"],
        "backend": cfg["backend"],
        "hardware_id": cfg["device_pool"]["hardware_id"],
        "precision": cfg["numerics"]["precision"],
        "detector_dtype": final["detector_dtype"],
        "field_state_dtypes": final["field_state_dtypes"],
        "resolution_nm": selected_resolution,
        "resolution_attempts": resolution_attempts,
        "logical_devices": selected_device_count,
        "science_device_milestones": device_counts,
        "canonical_sharding_multiple": sharding_multiple,
        "grid_contract_hash": final["grid_contract_hash"],
        "grid_shape": final["grid_shape"],
        "cells_total": final["cells_total"],
        "pml_cells": final["pml_cells"],
        "pml_actual_um": final["pml_actual_um"],
        "output_extension_cells": final["output_extension_cells"],
        "output_extension_um": final["output_extension_um"],
        "geometry": dict(cfg["science"]["geometry"]),
        "rasterized_geometry": final["rasterized_geometry"],
        "cases": cases,
        "convergence": convergence,
        "validation": {
            "all_case_sanity_checks_pass": case_sanity,
            "identical_grid_contract": identical_grid_contract,
            "fixed_device_count": fixed_device_count,
            "time_convergence_pass": bool(convergence["pass"]),
            "all_checks_pass": suite_valid,
        },
        "software": final["software"],
    }
    save_json(root / "report.json", summary)
    (root / "report.html").write_text(science_suite_html(summary))
    plan.update(
        selected_resolution_nm=selected_resolution,
        planned_cells=selected_grid["cells_total"],
        selected_logical_devices=selected_device_count,
        pml_cells=selected_grid["pml_cells"],
        pml_actual_um=selected_grid["pml_actual_um"],
        grid_shape=selected_grid["total_shape"],
        grid_contract_hash=selected_grid["grid_contract_hash"],
        output_extension_cells=selected_grid["output_extension_cells"],
        output_extension_um=selected_grid["output_extension_um"],
        actual_logical_devices=[report["logical_devices"] for report in case_reports],
        science_status=summary["status"],
        science_report=str(root / "report.json"),
    )
    save_yaml(Path("results/science-plan.yaml"), plan)
    print(f"report={root / 'report.html'}\nfield={root / cases[-1]['field_path']}")
    if not suite_valid:
        print("science validation failed; inspect the report", file=sys.stderr)
        return 1
    return 0


def write_failure(path: Path, exc: Exception) -> int:
    text = str(exc)
    kind = "OOM" if any(token in text.lower() for token in ("out of memory", "resource_exhausted", "allocator")) else "ERROR"
    payload = {"status": "FAIL", "failure": {"kind": kind, "type": type(exc).__name__, "message": text[:1000]}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return 42 if kind == "OOM" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="FDTDX accelerator capacity and science check")
    parser.add_argument("command", choices=("inventory", "profile", "science", "_inventory", "_profile_case", "_science_case"))
    parser.add_argument("config", type=Path)
    parser.add_argument("extra", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "inventory":
        data = inventory(args.config, cfg, Path("results/inventory.json"))
        print(json.dumps(data, indent=2))
        return 0
    if args.command == "profile":
        return run_profile(args.config, cfg)
    if args.command == "science":
        return run_science(args.config, cfg)

    from . import simulation

    try:
        if args.command == "_inventory":
            simulation.write_inventory(cfg, Path(args.extra[0]))
        elif args.command == "_profile_case":
            simulation.run_profile_case(cfg, json.loads(args.extra[0]), Path(args.extra[1]))
        else:
            simulation.run_science_case(
                cfg,
                int(args.extra[0]),
                float(args.extra[1]),
                Path(args.extra[2]),
                Path(args.extra[3]),
                int(args.extra[4]),
            )
        return 0
    except Exception as exc:
        output_index = {"_inventory": 0, "_profile_case": 1, "_science_case": 2}[args.command]
        return write_failure(Path(args.extra[output_index]), exc)


if __name__ == "__main__":
    raise SystemExit(main())
