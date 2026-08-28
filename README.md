# FDTDX Accelerator Check

A small FP64 test for a same-host JAX accelerator pool. It first measures the
largest FDTDX grid that can be allocated, then uses a conservative fraction of
that measured capacity for a generic 1x2 MMI field-propagation test. Maxwell
fields propagate in `float64`; mode-overlap and phasor detectors accumulate in
`complex128` and their realized state dtypes are checked after execution.

The model is generated analytically. It contains no imported GDS, foundry
PCell, measured device, customer geometry, host name, user name, IP address, or
credential.

## Quick start on four MI250X cards

Host prerequisites:

- Linux with a working ROCm driver/runtime
- four physical MI250X cards, 128 GiB HBM each (512 GiB declared total)
- Python 3.12 and Git for bare-metal setup, or Docker for the container path

Bare metal:

```bash
./setup.sh rocm
./run.sh inventory configs/rocm-mi250x-profile-fp64.yaml
./run.sh profile configs/rocm-mi250x-profile-fp64.yaml
./run.sh science configs/rocm-mi250x-science-fp64.yaml
```

The first two run stages are intentional:

1. `profile` launches fresh, short FDTDX jobs at increasing local grid sizes
   and at 1, 2, 4, and all visible logical devices. It writes the last
   successful allocation and a 70% safe capacity.
2. `science` reads that capacity, writes the exact selected parameters to
   `results/science-plan.yaml`, and runs the finest configured MMI grid that
   fits. Science retries use only the declared `1, 2, 4, all` milestones. If any
   time window fails with OOM, every time window is rerun in fresh processes at
   the next milestone; results from different device counts are never mixed in
   one convergence test. After exhausting the milestones, the failed attempt is
   recorded and the next coarser resolution is tried.

MI250X contains two GPU dies per physical card. Depending on the ROCm/JAX
topology, four cards may appear as four or eight logical devices. The scripts
discover that count at runtime and derive the planning memory as
`512 GiB / logical_device_count`; they do not mislabel eight logical dies as
eight physical cards.

## Docker

Build once while network access is available:

```bash
./docker.sh build
```

Then run with networking disabled inside the container:

```bash
./docker.sh inventory
./docker.sh profile
./docker.sh science
```

The runtime container receives only `/dev/kfd`, `/dev/dri`, and the local
`results/` directory. It drops Linux capabilities, uses
`no-new-privileges`, and has no embedded credential. A container cannot hide
its memory or files from a machine administrator with host root access; do not
put secrets in the image or runtime directory.

## FP64 and capacity semantics

Both MI250X YAML files use:

```yaml
numerics:
  precision: float64
  enable_x64: true
  bytes_per_cell: 640
```

`JAX_ENABLE_X64=true` is also set before JAX starts. Internal workers reject an
x64 mismatch rather than silently truncating to FP32. Every science detector is
constructed with `dtype=jnp.complex128`, and the report requires all four
realized detector states (`input_norm`, `out1`, `out2`, and `field`) to remain
`complex128`. The realized `E` and `H` arrays must independently report
`float64`; both checks are science PASS criteria.

`bytes_per_cell` is a conservative planning estimate; the profile result is the
empirical authority.
If the largest profile case passes, `capacity_interpretation` is
`tested_lower_bound_only`. Only a following OOM establishes a bounded last
success. The science stage never chooses a grid larger than the recorded 70%
safe-cell count.

The profile worker records `field_norm2 = sum(E**2) + sum(H**2)` only as a
finite, nonzero activity check. It is deliberately not named electromagnetic
energy; physical energy would require the material-weighted
`epsilon*|E|^2 + mu*|H|^2` expression.

## Physics regression contract

The science stage runs independent 320, 480, and 640 fs simulations. It keeps
the PML at a configured physical thickness of 0.8 um, so changing grid
resolution changes the PML cell count rather than the absorber thickness. The
last two windows must agree in both `|S21|` and `|S31|` within the configured
relative-drift limit. All windows must also have the same canonical grid hash
and use one fixed logical-device count.

The integer-cell geometry is the grid-sizing authority. One canonical x grid is
chosen to be divisible by every configured device milestone. If that requires
extra x cells, they extend the two straight output waveguides to the positive-x
PML instead of creating a cladding gap and an internal waveguide termination.
The output mode monitors remain tied to the declared device length, before this
purely straight PML-contact extension.

Each time window must also satisfy all of these gates:

- finite complex output overlaps and a nonzero input normalization
- minimum collected transmission and `T1 + T2 <= 1.05` coarse passivity
- output imbalance within 0.5 dB for the declared mirror-symmetric structure
- exact grid-material and output-port mirror placement
- exact output-waveguide contact with the positive-x PML
- matched output-mode effective indices within the declared tolerance
- runtime `float64` E/H propagation with `complex128` detector states

The analytic geometry is rasterized onto integer grid coordinates as explicit
mirror pairs. Both declared and grid-realized dimensions are written to the
reports. This avoids interpreting sub-cell continuous dimensions as if the Yee
grid represented them exactly.

## Outputs

All generated files are under `results/` and are ignored by Git:

- `capacity.yaml`: observed logical devices, every profile attempt, last pass,
  boundary status, and safe cell count
- `science-plan.yaml`: capacity-matched resolution, every resolution attempt,
  milestone attempts, canonical grid hash, output extension, time windows,
  physical PML, geometry, fixed device count, and retry policy
- `profile-*/report.html`: readable capacity report
- `science-*/report.html`: visual resolution, time-convergence, validation, and
  grid-realization report
- `science-*/resolution-*/time-*/field.png`: normalized real `Ey` field at 1.55 um
- `science-*/report.json`: complex normalized output overlaps, powers,
  gauge-invariant magnitude convergence, grid, precision, timing, and versions
- matching `.sha256` files for machine-readable primary records

The MMI result is an accelerator and end-to-end physics check, not a qualified
foundry component or a converged insertion-loss claim. Grid convergence,
material dispersion, port convergence, and a production pulse window would be
required before interpreting the overlap values as publishable S-parameters.
The reported relative output phase is diagnostic only: independently solved
port modes can carry arbitrary global phase, so relative phase is not a PASS
criterion without de-embedding or a shared phase reference.

## Local CUDA check

The CUDA YAML exercises the same FP64 path on one CUDA device:

```bash
./setup.sh cuda
./run.sh profile configs/cuda.yaml
./run.sh science configs/cuda.yaml
```

Do not reuse one `.venv` for both CUDA and ROCm plugins.

## Configuration

The normal tuning surface is limited to YAML:

- profile case shapes and device-count milestones
- science device-count milestones used to construct one canonical sharding grid
- safe fraction and FP64 planning bytes per cell
- candidate science resolutions, time windows, physical PML thickness, and gates
- generic MMI dimensions and refractive indices

The FDTDX dependency is pinned to public upstream commit
`77b4bc523a8e98cb7dd388e99481a0000f71dd4d`; no private fork is bundled.

Official references:

- [AMD ROCm JAX installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/jax-install.html)
- [JAX installation](https://docs.jax.dev/en/latest/installation.html)
- [JAX default dtypes and X64](https://docs.jax.dev/en/latest/default_dtypes.html)
- [AMD Instinct MI250X](https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html)

## QA

```bash
./qa.sh
```

QA compiles the Python package, parses every YAML file, checks shell syntax and
Git whitespace, and rejects local absolute paths, email-like strings, private
keys, and IP-like values from repository content. Unit regressions cover every
configured grid resolution, both four- and eight-logical-device MI250X views,
the 80 nm segment-rounding boundary, the 20 nm output extension, and full
time-window replay after an allocation OOM.
