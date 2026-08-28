from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def nearest_cells(value_um: float, dx_um: float, minimum: int = 1) -> int:
    """Rasterize one declared length with the same rule used by FDTDX placement."""
    return max(minimum, int(round(float(value_um) / dx_um)))


def nearest_even_cells(value_um: float, dx_um: float) -> int:
    """Rasterize a symmetric width while keeping the mirror plane on a grid boundary."""
    raw = float(value_um) / dx_um
    lower = max(2, 2 * math.floor(raw / 2))
    upper = lower + 2
    return min((lower, upper), key=lambda value: (abs(value - raw), -value))


def science_device_milestones(
    science: dict[str, Any], available: int, maximum: int | None = None
) -> list[int]:
    """Resolve stable sharding milestones such as 1, 2, 4, and all devices."""
    limit = max(1, min(available, maximum if maximum is not None else available))
    raw = science.get("device_counts", [1, 2, 4, "all"])
    values = [limit if value == "all" else int(value) for value in raw]
    counts = sorted({value for value in values if 1 <= value <= limit})
    if not counts:
        raise ValueError("science.device_counts does not select an available logical device")
    return counts


def canonical_sharding_multiple(counts: list[int]) -> int:
    """Return one x-grid multiple divisible by every permitted device milestone."""
    if not counts or any(value < 1 for value in counts):
        raise ValueError("At least one positive science device milestone is required")
    return math.lcm(*counts)


def science_rasterization(science: dict[str, Any], resolution_nm: int) -> dict[str, int]:
    """Return the authoritative integer-cell representation of the science geometry."""
    g = science["geometry"]
    dx_um = resolution_nm * 1e-3
    raster = {
        "input_length": nearest_cells(g["input_length_um"], dx_um, 2),
        "body_length": nearest_cells(g["body_length_um"], dx_um, 2),
        "declared_output_length": nearest_cells(g["output_length_um"], dx_um, 2),
        "access_width": nearest_even_cells(g["access_width_um"], dx_um),
        "body_width": nearest_even_cells(g["body_width_um"], dx_um),
        "output_offset": nearest_cells(g["output_offset_um"], dx_um),
        "core_height": nearest_cells(g["core_height_um"], dx_um),
    }
    raster["declared_geometry_x"] = (
        raster["input_length"]
        + raster["body_length"]
        + raster["declared_output_length"]
    )
    return raster


def science_grid(
    science: dict[str, Any], resolution_nm: int, sharding_multiple: int = 1
) -> dict[str, Any]:
    """Return a canonical grid independent of one attempt's device count.

    Segment rasterization is authoritative. Any x padding needed for sharding is
    assigned to a straight output-waveguide extension, never to background between
    the waveguide and the positive-x PML.
    """
    if sharding_multiple < 1:
        raise ValueError("sharding_multiple must be positive")
    g = science["geometry"]
    dx_um = resolution_nm * 1e-3
    raster = science_rasterization(science, resolution_nm)
    pml = math.ceil(float(science["pml_thickness_um"]) / dx_um)

    y_requested = float(g["body_width_um"]) + 2 * float(g["side_padding_um"])
    z_requested = float(g["core_height_um"]) + 2 * float(g["z_padding_um"])
    interior_y = math.ceil(y_requested / dx_um - 1e-12)
    if interior_y % 2:
        interior_y += 1
    interior_z = math.ceil(z_requested / dx_um - 1e-12)

    raw_total_x = raster["declared_geometry_x"] + 2 * pml
    total_x = math.ceil(raw_total_x / sharding_multiple) * sharding_multiple
    interior_x = total_x - 2 * pml
    output_extension = interior_x - raster["declared_geometry_x"]
    if output_extension < 0:
        raise RuntimeError("Canonical x grid is smaller than its rasterized geometry")
    raster["output_extension"] = output_extension
    raster["effective_output_length"] = raster["declared_output_length"] + output_extension
    raster["effective_geometry_x"] = interior_x

    interior = [interior_x, interior_y, interior_z]
    total = [total_x, interior_y + 2 * pml, interior_z + 2 * pml]
    contract = {
        "schema": 1,
        "resolution_nm": int(resolution_nm),
        "sharding_multiple": int(sharding_multiple),
        "pml_cells": pml,
        "interior_shape": interior,
        "total_shape": total,
        "rasterized_cells": raster,
    }
    contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "resolution_nm": resolution_nm,
        "sharding_multiple": sharding_multiple,
        "grid_contract_hash": contract_hash,
        "pml_cells": pml,
        "pml_actual_um": pml * dx_um,
        "interior_shape": interior,
        "interior_size_um": [value * dx_um for value in interior],
        "total_shape": total,
        "total_size_um": [value * dx_um for value in total],
        "cells_total": math.prod(total),
        "rasterized_cells": raster,
        "output_extension_cells": output_extension,
        "output_extension_um": output_extension * dx_um,
    }
