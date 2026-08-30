from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdtdx_check import __main__ as cli  # noqa: E402
from fdtdx_check.geometry import science_grid  # noqa: E402


class SchedulerContractTests(unittest.TestCase):
    def test_oom_moves_to_milestone_and_reruns_every_window(self) -> None:
        cfg = yaml.safe_load((ROOT / "configs" / "cuda.yaml").read_text())
        cfg["science"]["resolution_candidates_nm"] = [100]
        cfg["science"]["device_counts"] = [1, 2]
        cfg["device_pool"]["aggregate_hbm_gib"] = 32
        calls: list[tuple[int, int]] = []
        grid = science_grid(cfg["science"], 100, 2)
        raster = grid["rasterized_cells"]
        dx_um = 0.1

        def fake_invoke(args: list[str], _cfg: dict, devices: int | None = None) -> int:
            self.assertEqual(args[0], "_science_case")
            assert devices is not None
            time_fs = int(float(args[3]))
            calls.append((devices, time_fs))
            output = Path(args[4])
            output.parent.mkdir(parents=True, exist_ok=True)
            if devices == 1 and time_fs == 480:
                output.write_text(
                    json.dumps({"status": "FAIL", "failure": {"kind": "OOM"}})
                )
                return 42
            report = {
                "status": "PASS",
                "time_fs": float(time_fs),
                "logical_devices": devices,
                "grid_contract_hash": grid["grid_contract_hash"],
                "grid_shape": grid["total_shape"],
                "steps": time_fs,
                "elapsed_seconds": 0.01,
                "metrics": {
                    "S21_real": 0.5,
                    "S21_imag": 0.0,
                    "S31_real": 0.5,
                    "S31_imag": 0.0,
                    "T1": 0.25,
                    "T2": 0.25,
                    "total_transmission": 0.5,
                    "imbalance_db": 0.0,
                },
                "validation": {"all_checks_pass": True, "checks": {}},
                "detector_dtype": "complex128",
                "field_state_dtypes": {"E": "float64", "H": "float64"},
                "cells_total": grid["cells_total"],
                "pml_cells": grid["pml_cells"],
                "pml_actual_um": grid["pml_actual_um"],
                "output_extension_cells": grid["output_extension_cells"],
                "output_extension_um": grid["output_extension_um"],
                "rasterized_geometry": {
                    "input_length_um": raster["input_length"] * dx_um,
                    "body_length_um": raster["body_length"] * dx_um,
                    "output_length_um": raster["declared_output_length"] * dx_um,
                    "output_pml_extension_um": grid["output_extension_um"],
                    "effective_output_length_um": raster["effective_output_length"] * dx_um,
                },
                "software": {"jax": "test", "jaxlib": "test", "fdtdx": "test"},
            }
            output.write_text(json.dumps(report))
            return 0

        with tempfile.TemporaryDirectory() as tmp, patch.object(cli, "invoke", fake_invoke):
            previous = Path.cwd()
            os.chdir(tmp)
            try:
                Path("results").mkdir()
                Path("results/capacity.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "backend": "cuda",
                            "hardware_id": cfg["device_pool"]["hardware_id"],
                            "precision": "float64",
                            "recommended_safe_cells": 100_000_000,
                            "logical_devices_available": 2,
                        }
                    )
                )
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.run_science(Path("config.yaml"), cfg), 0)
                suite_path = next(Path("results").glob("science-*/report.json"))
                suite = json.loads(suite_path.read_text())
                plan = yaml.safe_load(Path("results/science-plan.yaml").read_text())
            finally:
                os.chdir(previous)

        self.assertEqual(calls, [(1, 320), (1, 480), (2, 320), (2, 480), (2, 640)])
        self.assertEqual({case["logical_devices"] for case in suite["cases"]}, {2})
        self.assertTrue(suite["convergence"]["identical_grid_contract"])
        self.assertTrue(suite["convergence"]["fixed_device_count"])
        self.assertEqual(plan["selected_logical_devices"], 2)
        self.assertEqual(plan["grid_shape"], grid["total_shape"])

    def test_science_rejects_capacity_from_another_hardware_pool(self) -> None:
        cfg = yaml.safe_load((ROOT / "configs" / "rocm-mi350p-science-fp64.yaml").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            os.chdir(tmp)
            try:
                Path("results").mkdir()
                Path("results/capacity.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "backend": "rocm",
                            "hardware_id": "amd-instinct-mi250x-4x",
                            "precision": "float64",
                            "recommended_safe_cells": 100_000_000,
                            "logical_devices_available": 4,
                        }
                    )
                )
                with self.assertRaisesRegex(RuntimeError, "Capacity hardware"):
                    cli.run_science(Path("config.yaml"), cfg)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
