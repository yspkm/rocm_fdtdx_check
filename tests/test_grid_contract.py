from __future__ import annotations

from pathlib import Path
import sys
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdtdx_check.geometry import (  # noqa: E402
    canonical_sharding_multiple,
    science_device_milestones,
    science_grid,
)


class GridContractTests(unittest.TestCase):
    def config(self, name: str) -> dict:
        return yaml.safe_load((ROOT / "configs" / name).read_text())

    def test_cuda_80_nm_uses_rasterized_segment_sum(self) -> None:
        science = self.config("cuda.yaml")["science"]
        grid = science_grid(science, 80, 1)
        raster = grid["rasterized_cells"]
        self.assertEqual(raster["declared_geometry_x"], 196)
        self.assertEqual(grid["interior_shape"][0], 196)
        self.assertEqual(grid["total_shape"][0], 216)
        self.assertEqual(grid["output_extension_cells"], 0)

    def test_mi250x_20_nm_padding_extends_output_waveguide(self) -> None:
        science = self.config("rocm-mi250x-science-fp64.yaml")["science"]
        counts = science_device_milestones(science, available=8)
        multiple = canonical_sharding_multiple(counts)
        grid = science_grid(science, 20, multiple)
        raster = grid["rasterized_cells"]
        self.assertEqual(counts, [1, 2, 4, 8])
        self.assertEqual(multiple, 8)
        self.assertEqual(raster["declared_geometry_x"], 780)
        self.assertEqual(grid["interior_shape"][0], 784)
        self.assertEqual(grid["output_extension_cells"], 4)
        self.assertEqual(raster["effective_output_length"], 154)

    def test_every_configured_grid_is_divisible_and_filled(self) -> None:
        scenarios = (
            ("cuda.yaml", (1,)),
            ("rocm-mi250x-science-fp64.yaml", (4, 8)),
        )
        for name, available_values in scenarios:
            science = self.config(name)["science"]
            for available in available_values:
                counts = science_device_milestones(science, available)
                multiple = canonical_sharding_multiple(counts)
                for resolution in science["resolution_candidates_nm"]:
                    with self.subTest(name=name, available=available, resolution=resolution):
                        first = science_grid(science, int(resolution), multiple)
                        second = science_grid(science, int(resolution), multiple)
                        raster = first["rasterized_cells"]
                        self.assertEqual(first["grid_contract_hash"], second["grid_contract_hash"])
                        self.assertEqual(
                            raster["effective_geometry_x"], first["interior_shape"][0]
                        )
                        self.assertEqual(
                            raster["declared_geometry_x"] + first["output_extension_cells"],
                            first["interior_shape"][0],
                        )
                        for count in counts:
                            self.assertEqual(first["total_shape"][0] % count, 0)


if __name__ == "__main__":
    unittest.main()
