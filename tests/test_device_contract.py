from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdtdx_check.simulation import _device_contract  # noqa: E402


class FakeDevice:
    def __init__(self, device_id: int, device_kind: str) -> None:
        self.id = device_id
        self.platform = "rocm"
        self.device_kind = device_kind
        self.process_index = 0


class DeviceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = yaml.safe_load(
            (ROOT / "configs" / "rocm-mi350p-profile-fp64.yaml").read_text()
        )

    @staticmethod
    def devices(count: int, kind: str = "AMD Instinct MI350P") -> list[FakeDevice]:
        return [FakeDevice(index, kind) for index in range(count)]

    def test_full_mi350p_inventory_accepts_four_devices(self) -> None:
        contract = _device_contract(self.cfg, self.devices(4), full_inventory=True)
        self.assertEqual(contract["hardware_id"], "amd-instinct-mi350p-4x")
        self.assertEqual(contract["expected_logical_device_counts"], [4])

    def test_full_inventory_rejects_wrong_count(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected logical device count"):
            _device_contract(self.cfg, self.devices(3), full_inventory=True)

    def test_inventory_rejects_wrong_accelerator(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "rejected device kinds"):
            _device_contract(
                self.cfg,
                self.devices(4, kind="AMD Instinct MI250X"),
                full_inventory=True,
            )

    def test_worker_accepts_requested_visible_subset(self) -> None:
        with patch.dict(os.environ, {"FDTDX_CHECK_DEVICE_COUNT": "2"}):
            contract = _device_contract(self.cfg, self.devices(2), full_inventory=False)
        self.assertFalse(contract["full_inventory"])

    def test_worker_rejects_visibility_mismatch(self) -> None:
        with patch.dict(os.environ, {"FDTDX_CHECK_DEVICE_COUNT": "2"}):
            with self.assertRaisesRegex(RuntimeError, "Requested 2 visible devices"):
                _device_contract(self.cfg, self.devices(1), full_inventory=False)


if __name__ == "__main__":
    unittest.main()
