from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from .geometry import nearest_cells, science_grid
from .reporting import science_html


def _software(jax: Any) -> dict[str, str]:
    return {
        "jax": jax.__version__,
        "jaxlib": importlib.metadata.version("jaxlib"),
        "fdtdx": importlib.metadata.version("fdtdx"),
    }


def _device_contract(
    cfg: dict[str, Any], devices: list[Any], *, full_inventory: bool
) -> dict[str, Any]:
    pool = cfg["device_pool"]
    hardware_id = str(pool["hardware_id"])
    expected_counts = [int(value) for value in pool.get("expected_logical_device_counts", [])]
    accepted_kinds = [
        str(value).lower() for value in pool.get("accepted_device_kind_substrings", [])
    ]
    observed_kinds = [str(device.device_kind) for device in devices]

    unexpected_kinds = [
        kind
        for kind in observed_kinds
        if accepted_kinds and not any(token in kind.lower() for token in accepted_kinds)
    ]
    if unexpected_kinds:
        raise RuntimeError(
            f"Hardware {hardware_id!r} rejected device kinds {unexpected_kinds!r}; "
            f"expected one of {accepted_kinds!r}"
        )

    if full_inventory and expected_counts and len(devices) not in expected_counts:
        raise RuntimeError(
            f"Hardware {hardware_id!r} expected logical device count in {expected_counts!r}, "
            f"observed {len(devices)}"
        )

    requested_visible = os.environ.get("FDTDX_CHECK_DEVICE_COUNT")
    if requested_visible is not None and len(devices) != int(requested_visible):
        raise RuntimeError(
            f"Requested {requested_visible} visible devices, observed {len(devices)}"
        )

    return {
        "hardware_id": hardware_id,
        "status": "PASS",
        "full_inventory": full_inventory,
        "expected_logical_device_counts": expected_counts,
        "accepted_device_kind_substrings": accepted_kinds,
    }


def _devices(
    jax: Any, cfg: dict[str, Any], *, full_inventory: bool = False
) -> tuple[list[Any], dict[str, Any]]:
    devices = list(jax.devices())
    if not devices:
        raise RuntimeError("No JAX devices detected")
    evidence = " ".join(f"{d.platform} {d.device_kind}".lower() for d in devices)
    if "cpu" in evidence:
        raise RuntimeError("CPU fallback detected")
    return devices, _device_contract(cfg, devices, full_inventory=full_inventory)


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")


def _dtype(cfg: dict[str, Any], jnp: Any) -> Any:
    return jnp.float64 if cfg["numerics"]["precision"] == "float64" else jnp.float32


def _verify_precision_runtime(cfg: dict[str, Any], jax: Any) -> None:
    expected_x64 = bool(cfg["numerics"]["enable_x64"])
    observed_x64 = bool(jax.config.read("jax_enable_x64"))
    if observed_x64 != expected_x64:
        raise RuntimeError(
            f"JAX x64 state mismatch: expected {expected_x64}, observed {observed_x64}"
        )


def _slice_record(obj: Any) -> list[list[int]]:
    return [[int(lower), int(upper)] for lower, upper in obj.grid_slice_tuple]


def write_inventory(cfg: dict[str, Any], path: Path) -> None:
    import jax

    devices, contract = _devices(jax, cfg, full_inventory=True)
    _verify_precision_runtime(cfg, jax)
    payload = {
        "schema": 1,
        "status": "PASS",
        "backend_requested": cfg["backend"],
        "backend_observed": jax.default_backend(),
        "logical_device_count": len(devices),
        "device_kinds": sorted({d.device_kind for d in devices}),
        "devices": [
            {
                "id": int(device.id),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "process_index": int(device.process_index),
            }
            for device in devices
        ],
        "hardware_contract": contract,
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "software": _software(jax),
    }
    _save(path, payload)


def run_profile_case(cfg: dict[str, Any], shape: list[int], path: Path) -> None:
    import jax
    import jax.numpy as jnp
    import fdtdx

    devices, _ = _devices(jax, cfg)
    _verify_precision_runtime(cfg, jax)
    dtype = _dtype(cfg, jnp)
    spacing = 100e-9
    courant = 0.8
    dt = courant / math.sqrt(3) * spacing / 299_792_458.0
    config = fdtdx.SimulationConfig(
        time=int(cfg["profile"]["steps"]) * dt,
        grid=fdtdx.UniformGrid(spacing=spacing),
        backend="gpu",
        dtype=dtype,
        courant_factor=courant,
        gradient_config=None,
    )
    volume = fdtdx.SimulationVolume(partial_grid_shape=tuple(shape))
    boundaries, constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic", thickness=2), volume
    )
    source = fdtdx.PointDipoleSource(
        name="source",
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1.55e-6),
        polarization=2,
    )
    constraints.append(
        source.set_grid_coordinates(
            axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=tuple(v // 2 for v in shape)
        )
    )
    key = jax.random.PRNGKey(7)
    started = time.perf_counter()
    objects, arrays, params, config, _ = fdtdx.place_objects(
        [volume, *boundaries.values(), source], config, constraints, key
    )
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
    final_step, arrays = fdtdx.run_fdtd(arrays, objects, config, key=key, show_progress=False)
    field_norm2 = jnp.sum(arrays.fields.E**2) + jnp.sum(arrays.fields.H**2)
    field_norm2.block_until_ready()
    if not math.isfinite(float(field_norm2)) or float(field_norm2) <= 0:
        raise RuntimeError("Non-finite or zero field activity norm")
    cells = math.prod(shape)
    _save(
        path,
        {
            "schema": 1,
            "status": "PASS",
            "shape": shape,
            "cells_total": cells,
            "cells_per_device": cells // len(devices),
            "logical_devices": len(devices),
            "precision": cfg["numerics"]["precision"],
            "steps": int(final_step),
            "field_norm2": float(field_norm2),
            "field_norm_interpretation": "nonphysical_activity_sanity_check",
            "elapsed_seconds": time.perf_counter() - started,
            "field_shards": len(arrays.fields.E.addressable_shards),
            "software": _software(jax),
        },
    )


def run_science_case(
    cfg: dict[str, Any],
    resolution_nm: int,
    time_fs: float,
    path: Path,
    root: Path,
    sharding_multiple: int,
) -> None:
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    import fdtdx

    devices, _ = _devices(jax, cfg)
    _verify_precision_runtime(cfg, jax)
    dtype = _dtype(cfg, jnp)
    detector_dtype = jnp.complex128 if dtype == jnp.float64 else jnp.complex64
    s = cfg["science"]
    g = s["geometry"]
    um = 1e-6
    resolution = resolution_nm * 1e-9
    grid = science_grid(s, resolution_nm, sharding_multiple)
    total_shape = list(grid["total_shape"])
    total_size = tuple(float(v) * um for v in grid["total_size_um"])
    interior_size = tuple(float(v) * um for v in grid["interior_size_um"])
    interior_shape = [int(v) for v in grid["interior_shape"]]
    core_y = interior_size[1]
    pml = int(grid["pml_cells"])
    wave = fdtdx.WaveCharacter(wavelength=float(s["wavelength_um"]) * um)
    config = fdtdx.SimulationConfig(
        time=time_fs * 1e-15,
        grid=fdtdx.UniformGrid(spacing=resolution),
        backend="gpu",
        dtype=dtype,
        courant_factor=0.8,
        gradient_config=None,
    )
    volume = fdtdx.SimulationVolume(
        name="background",
        partial_real_shape=total_size,
        material=fdtdx.Material(permittivity=float(g["cladding_index"]) ** 2),
    )
    objects: list[Any] = [volume]
    boundaries, constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=pml), volume
    )
    objects.extend(boundaries.values())
    core_material = fdtdx.Material(permittivity=float(g["core_index"]) ** 2)

    dx_um = resolution_nm * 1e-3
    raster = grid["rasterized_cells"]
    lin_cells = int(raster["input_length"])
    body_cells = int(raster["body_length"])
    declared_output_cells = int(raster["declared_output_length"])
    output_cells = int(raster["effective_output_length"])
    declared_geometry_x_cells = int(raster["declared_geometry_x"])
    effective_geometry_x_cells = int(raster["effective_geometry_x"])
    access_cells = int(raster["access_width"])
    body_width_cells = int(raster["body_width"])
    offset_cells = int(raster["output_offset"])
    height_cells = int(raster["core_height"])
    if effective_geometry_x_cells != interior_shape[0]:
        raise RuntimeError("Canonical output extension does not fill the non-PML x domain")
    if 2 * offset_cells <= access_cells:
        raise RuntimeError("Rasterized output waveguides overlap")

    core_x = effective_geometry_x_cells * resolution
    lin, lb, lout_declared, lout_effective = (
        value * resolution
        for value in (lin_cells, body_cells, declared_output_cells, output_cells)
    )
    win, wb, off, h = (
        access_cells * resolution,
        body_width_cells * resolution,
        offset_cells * resolution,
        height_cells * resolution,
    )
    y0 = core_y / 2
    center_y_grid = pml + interior_shape[1] // 2
    z_lower = pml + (interior_shape[2] - height_cells) // 2
    x_input = pml
    x_body = x_input + lin_cells
    x_output = x_body + body_cells
    y_input = center_y_grid - access_cells // 2
    y_body = center_y_grid - body_width_cells // 2
    y_top = center_y_grid + offset_cells - access_cells // 2
    y_bottom = center_y_grid - offset_cells - access_cells // 2

    def grid_box(name: str, shape: tuple[int, int, int], lower: tuple[int, int, int]) -> Any:
        obj = fdtdx.UniformMaterialObject(
            name=name,
            partial_grid_shape=shape,
            material=core_material,
        )
        objects.append(obj)
        constraints.append(
            obj.set_grid_coordinates(
                axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=lower
            )
        )
        return obj

    pieces = [
        grid_box("input", (lin_cells, access_cells, height_cells), (x_input, y_input, z_lower)),
        grid_box("body", (body_cells, body_width_cells, height_cells), (x_body, y_body, z_lower)),
        grid_box("output_top", (output_cells, access_cells, height_cells), (x_output, y_top, z_lower)),
        grid_box("output_bottom", (output_cells, access_cells, height_cells), (x_output, y_bottom, z_lower)),
    ]
    profile = fdtdx.GaussianPulseProfile(
        center_wave=wave, spectral_width=fdtdx.WaveCharacter(wavelength=wave.get_wavelength() * 10)
    )
    port_y_cells = math.ceil(1.4 / dx_um - 1e-12)
    if port_y_cells % 2:
        port_y_cells += 1
    port_z_cells = min(interior_shape[2], math.ceil(1.4 / dx_um - 1e-12))
    port_shape = (1, port_y_cells, port_z_cells)
    source = fdtdx.ModePlaneSource(
        name="source", partial_grid_shape=port_shape, wave_character=wave, temporal_profile=profile,
        direction="+", mode_index=0, filter_pol="te"
    )
    input_norm = fdtdx.ModeOverlapDetector(
        name="input_norm", partial_grid_shape=port_shape, wave_characters=(wave,), direction="+",
        mode_index=0, filter_pol="te", scaling_mode="pulse", dtype=detector_dtype
    )
    out1 = fdtdx.ModeOverlapDetector(
        name="out1", partial_grid_shape=port_shape, wave_characters=(wave,), direction="+",
        mode_index=0, filter_pol="te", scaling_mode="pulse", dtype=detector_dtype
    )
    out2 = out1.aset("name", "out2")
    monitor_offset_cells = nearest_cells(0.55, dx_um, 2)
    source_x_grid = pml + monitor_offset_cells
    # Keep the output monitor tied to the declared device, before any straight
    # extension added solely to preserve waveguide contact with the +x PML.
    output_x_grid = pml + declared_geometry_x_cells - monitor_offset_cells
    port_z_lower = pml + (interior_shape[2] - port_z_cells) // 2
    port_locations = (
        (source, (source_x_grid, center_y_grid - port_y_cells // 2, port_z_lower)),
        (input_norm, (source_x_grid + 2, center_y_grid - port_y_cells // 2, port_z_lower)),
        (out1, (output_x_grid, center_y_grid + offset_cells - port_y_cells // 2, port_z_lower)),
        (out2, (output_x_grid, center_y_grid - offset_cells - port_y_cells // 2, port_z_lower)),
    )
    for obj, lower in port_locations:
        objects.append(obj)
        constraints.append(
            obj.set_grid_coordinates(
                axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=lower
            )
        )
    field = fdtdx.PhasorDetector(
        name="field", wave_characters=(wave,), components=(str(s["field_component"]),),
        partial_grid_shape=(effective_geometry_x_cells, interior_shape[1], 1),
        dtype=detector_dtype,
    )
    objects.append(field)
    constraints.append(
        field.set_grid_coordinates(
            axes=(0, 1, 2),
            sides=("-", "-", "-"),
            coordinates=(pml, pml, pml + interior_shape[2] // 2),
        )
    )

    rasterized_geometry = {
        "input_length_um": lin / um,
        "body_length_um": lb / um,
        "output_length_um": lout_declared / um,
        "output_pml_extension_um": float(grid["output_extension_um"]),
        "effective_output_length_um": lout_effective / um,
        "access_width_um": win / um,
        "body_width_um": wb / um,
        "output_offset_um": off / um,
        "core_height_um": h / um,
        "port_window_cells": list(port_shape),
    }

    key = jax.random.PRNGKey(17)
    started = time.perf_counter()
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    arrays = fdtdx.extend_material_to_pml(objects=objects, arrays=arrays)
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
    material_mirror_error = float(
        jnp.max(jnp.abs(arrays.inv_permittivities - jnp.flip(arrays.inv_permittivities, axis=2)))
    )
    material_scale = float(jnp.max(jnp.abs(arrays.inv_permittivities)))
    material_mirror_relative_error = material_mirror_error / max(material_scale, 1e-30)
    ex_mirror_error = float(
        jnp.max(
            jnp.abs(
                arrays.inv_permittivities[0]
                - jnp.flip(arrays.inv_permittivities[0], axis=1)
            )
        )
    )
    ex_scale = float(jnp.max(jnp.abs(arrays.inv_permittivities[0])))
    ex_mirror_relative_error = ex_mirror_error / max(ex_scale, 1e-30)
    port_slices = {
        name: _slice_record(objects[name]) for name in ("source", "input_norm", "out1", "out2")
    }
    top_y, bottom_y = port_slices["out1"][1], port_slices["out2"][1]
    port_placement_symmetry = (
        top_y[0] == total_shape[1] - bottom_y[1]
        and top_y[1] == total_shape[1] - bottom_y[0]
    )
    output_neff = {
        name: complex(jnp.asarray(objects[name]._mode_neff).squeeze()) for name in ("out1", "out2")
    }
    output_neff_relative_difference = abs(output_neff["out1"] - output_neff["out2"]) / max(
        abs(output_neff["out1"]), abs(output_neff["out2"]), 1.0
    )
    output_neff_match = output_neff_relative_difference <= float(
        s["validation"]["maximum_output_neff_relative_difference"]
    )
    material_grid_slices = {obj.name: _slice_record(objects[obj.name]) for obj in pieces}
    positive_x_pml_start = total_shape[0] - pml
    output_pml_contact = all(
        material_grid_slices[name][0][1] == positive_x_pml_start
        for name in ("output_top", "output_bottom")
    )
    final_step, arrays = fdtdx.run_fdtd(arrays, objects, config, key=key, show_progress=False)
    arrays.fields.E.block_until_ready()
    field_devices = list(arrays.fields.E.devices())
    if not field_devices or any(device.platform == "cpu" for device in field_devices):
        raise RuntimeError("Science field array fell back to CPU")
    detector_dtypes = {
        name: str(arrays.detector_states[name]["phasor"].dtype)
        for name in ("input_norm", "out1", "out2", "field")
    }
    expected_detector_dtype = "complex128" if dtype == jnp.float64 else "complex64"
    detector_precision_ok = all(value == expected_detector_dtype for value in detector_dtypes.values())
    field_state_dtypes = {
        "E": str(arrays.fields.E.dtype),
        "H": str(arrays.fields.H.dtype),
    }
    expected_field_dtype = "float64" if dtype == jnp.float64 else "float32"
    propagation_precision_ok = all(
        value == expected_field_dtype for value in field_state_dtypes.values()
    )
    norm = objects["input_norm"].compute_overlap(arrays.detector_states["input_norm"])
    norm_complex = complex(jnp.asarray(norm).squeeze())
    input_norm_magnitude = abs(norm_complex)
    minimum_input_norm = float(s["validation"]["minimum_input_overlap_magnitude"])
    if not math.isfinite(input_norm_magnitude) or input_norm_magnitude < minimum_input_norm:
        raise RuntimeError(
            f"Input overlap magnitude {input_norm_magnitude:.3e} is below {minimum_input_norm:.3e}"
        )
    a1 = objects["out1"].compute_overlap(arrays.detector_states["out1"]) / norm
    a2 = objects["out2"].compute_overlap(arrays.detector_states["out2"]) / norm
    t1, t2 = float(jnp.abs(a1).squeeze() ** 2), float(jnp.abs(a2).squeeze() ** 2)
    c1, c2 = complex(jnp.asarray(a1).squeeze()), complex(jnp.asarray(a2).squeeze())
    total_transmission = t1 + t2
    imbalance = 10 * math.log10(max(t1, 1e-30) / max(t2, 1e-30))
    relative_phase = math.degrees(math.atan2((c1 * c2.conjugate()).imag, (c1 * c2.conjugate()).real))
    validation_cfg = s["validation"]
    finite_ok = all(
        math.isfinite(value)
        for value in (c1.real, c1.imag, c2.real, c2.imag, t1, t2, total_transmission, imbalance)
    )
    transmission_floor_ok = total_transmission >= float(validation_cfg["minimum_total_transmission"])
    passivity_ok = total_transmission <= float(validation_cfg["maximum_total_transmission"])
    symmetry_ok = abs(imbalance) <= float(validation_cfg["maximum_imbalance_db"])
    checks = {
        "finite_complex_outputs": finite_ok,
        "input_normalization": input_norm_magnitude >= minimum_input_norm,
        "minimum_transmission": transmission_floor_ok,
        "passivity": passivity_ok,
        "symmetry": symmetry_ok,
        "material_ex_mirror_symmetry": ex_mirror_relative_error
        <= float(validation_cfg["maximum_material_mirror_relative_error"]),
        "output_port_placement_symmetry": port_placement_symmetry,
        "output_mode_neff_match": output_neff_match,
        "detector_precision": detector_precision_ok,
        "propagation_precision": propagation_precision_ok,
        "output_waveguide_positive_x_pml_contact": output_pml_contact,
    }
    measurement_valid = all(checks.values())
    phasor = np.asarray(jax.device_get(arrays.detector_states["field"]["phasor"]))
    image = np.real(phasor[0, 0, 0, :, :, 0]).T
    scale = max(float(np.max(np.abs(image))), 1e-30)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    extent = (0, core_x / um, 0, core_y / um)
    im = ax.imshow(image / scale, origin="lower", extent=extent, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for x, y, width, height in (
        (0, y0 / um - win / um / 2, lin / um, win / um),
        (lin / um, y0 / um - wb / um / 2, lb / um, wb / um),
        (
            (lin + lb) / um,
            y0 / um + off / um - win / um / 2,
            lout_effective / um,
            win / um,
        ),
        (
            (lin + lb) / um,
            y0 / um - off / um - win / um / 2,
            lout_effective / um,
            win / um,
        ),
    ):
        ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="black", linewidth=0.8, alpha=0.7))
    ax.set(xlabel="x (µm)", ylabel="y (µm)", title=f"Re({s['field_component']}) at 1.55 µm")
    fig.colorbar(im, ax=ax, label="normalized field")
    fig.tight_layout()
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / "field.png", dpi=180)
    plt.close(fig)
    report = {
        "schema": 1,
        "status": "PASS" if measurement_valid else "FAIL",
        "validation_level": str(s["validation_level"]),
        "measurement_valid": measurement_valid,
        "design": "generic_mmi_1x2",
        "design_origin": "analytic_synthetic",
        "imported_gds": False,
        "foundry_pcell": False,
        "geometry": dict(g),
        "rasterized_geometry": rasterized_geometry,
        "backend": cfg["backend"],
        "hardware_id": cfg["device_pool"]["hardware_id"],
        "logical_devices": len(devices),
        "field_shards": len(arrays.fields.E.addressable_shards),
        "precision": cfg["numerics"]["precision"],
        "field_state_dtypes": field_state_dtypes,
        "detector_dtype": expected_detector_dtype,
        "detector_state_dtypes": detector_dtypes,
        "canonical_sharding_multiple": sharding_multiple,
        "grid_contract_hash": grid["grid_contract_hash"],
        "output_extension_cells": grid["output_extension_cells"],
        "output_extension_um": grid["output_extension_um"],
        "geometry_audit": {
            "material_raw_all_component_y_mirror_relative_error": material_mirror_relative_error,
            "material_ex_y_mirror_relative_error": ex_mirror_relative_error,
            "port_grid_slices": port_slices,
            "material_grid_slices": material_grid_slices,
            "positive_x_pml_start_cell": positive_x_pml_start,
            "output_waveguide_positive_x_pml_contact": output_pml_contact,
            "output_monitor_cell": output_x_grid,
            "declared_geometry_end_cell": pml + declared_geometry_x_cells,
            "effective_geometry_end_cell": pml + effective_geometry_x_cells,
            "output_mode_neff_relative_difference": output_neff_relative_difference,
            "output_mode_neff": {
                name: {"real": value.real, "imag": value.imag} for name, value in output_neff.items()
            },
        },
        "resolution_nm": resolution_nm,
        "time_fs": time_fs,
        "pml_cells": pml,
        "pml_requested_um": float(s["pml_thickness_um"]),
        "pml_actual_um": float(grid["pml_actual_um"]),
        "grid_shape": total_shape,
        "cells_total": math.prod(total_shape),
        "steps": int(final_step),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": {
            "S21_real": c1.real,
            "S21_imag": c1.imag,
            "S31_real": c2.real,
            "S31_imag": c2.imag,
            "T1": t1,
            "T2": t2,
            "total_transmission": total_transmission,
            "relative_phase_deg": relative_phase,
            "relative_phase_interpretation": "diagnostic_mode_basis_gauge_dependent",
            "imbalance_db": imbalance,
            "input_overlap_magnitude": input_norm_magnitude,
        },
        "validation": {
            "checks": checks,
            "thresholds": dict(validation_cfg),
            "all_checks_pass": measurement_valid,
        },
        "software": _software(jax),
    }
    _save(path, report)
    _save(root / "report.json", report)
    (root / "report.html").write_text(science_html(report))
