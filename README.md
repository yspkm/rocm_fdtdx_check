# FDTDX accelerator check

This repository checks FDTDX FP64 execution on GPU accelerators. The one-device
CUDA path has already been exercised and is the reference. The ROCm path
targets two four-card AMD Instinct systems, MI250X and MI350P. Each path profiles
the largest grid that fits, keeps a 70% safety margin, and runs a generic 1x2 MMI
propagation test at that capacity. Maxwell fields use `float64`; mode-overlap
and phasor detectors use `complex128`.

The MMI is generated analytically. The repository contains no imported GDS,
foundry PCell, measured device, customer geometry, host details, or credentials.

## MI350P target: four cards

The MI350P configuration expects four unpartitioned PCIe cards on one Linux
host. JAX must report exactly four `MI350P` or `gfx950` devices. The declared
memory is 144 GiB per card and 576 GiB in total.

Bare metal:

```bash
./setup.sh rocm
./run.sh inventory configs/rocm-mi350p-profile-fp64.yaml
./run.sh profile configs/rocm-mi350p-profile-fp64.yaml
./run.sh science configs/rocm-mi350p-science-fp64.yaml
```

Container:

```bash
./docker.sh build
./docker.sh inventory configs/rocm-mi350p-profile-fp64.yaml
./docker.sh profile configs/rocm-mi350p-profile-fp64.yaml
./docker.sh science configs/rocm-mi350p-science-fp64.yaml
```

The pinned image, `rocm/jax:rocm7.2.4-jax0.8.2-py3.12`, is listed by AMD for
JAX 0.8.2 and `gfx950`. Check the host driver, firmware, OS, and partition mode
against the [ROCm 7.2.4 compatibility matrix](https://rocm.docs.amd.com/en/docs-7.2.4/compatibility/compatibility-matrix.html).
AMD publishes the card specifications on the [MI350P product page](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350p.html).

Before a long run, record `amd-smi list`, `amd-smi topology`, and `rocminfo`.
The four cards have separate local HBM rather than one physical 576 GiB memory
space, so peer communication depends on the server topology.

MI350P artifacts use the hardware ID `amd-instinct-mi350p-4x`. A science run
will reject a capacity file from the MI250X pool. The local QA checks the
configuration and scheduler; hardware support is established only when
inventory, profile, and science all pass on the server.

## MI250X target: four cards

This path expects four MI250X cards with 128 GiB each, plus Python 3.12 and Git
or Docker:

```bash
./setup.sh rocm
./run.sh inventory configs/rocm-mi250x-profile-fp64.yaml
./run.sh profile configs/rocm-mi250x-profile-fp64.yaml
./run.sh science configs/rocm-mi250x-science-fp64.yaml
```

An MI250X has two GPU dies. Four cards may therefore appear as four or eight
JAX devices. The scripts accept either view and calculate planning memory as
`512 GiB / logical_device_count` without treating dies as separate cards.

The Docker commands without an explicit configuration use the MI250X files:

```bash
./docker.sh build
./docker.sh inventory
./docker.sh profile
./docker.sh science
```

The container runs without networking and receives only `/dev/kfd`, `/dev/dri`,
and `results/`. It drops Linux capabilities and enables `no-new-privileges`.
Do not place secrets in the image or results directory; host root can still
read container memory and files.

## What the checks do

`profile` launches fresh jobs at increasing local grid sizes and at 1, 2, 4,
and all visible devices. It records the largest successful allocation. A
following OOM establishes a bounded capacity; otherwise the result is only a
tested lower bound. The science limit is 70% of the largest passing case.

`science` selects the finest configured MMI grid below that limit and runs
independent 320, 480, and 640 fs windows. An OOM moves the whole set of windows
to the next device milestone in fresh processes. Results from different device
counts are never mixed in one convergence check. Once the milestones are
exhausted, the runner tries the next coarser resolution.

Both ROCm paths set `JAX_ENABLE_X64=true`. The run fails if the E/H fields are
not `float64`, any detector state is not `complex128`, or execution falls back
to a CPU. `bytes_per_cell: 640` is a planning estimate; the allocation profile
is the capacity record. `field_norm2 = sum(E**2) + sum(H**2)` is only a finite,
nonzero activity check, not physical electromagnetic energy.

The MMI keeps a 0.8 um PML and one canonical grid across the time windows. Its
x dimension fits every permitted device count, with any extra cells extending
the straight output waveguides to the PML. The checks cover finite overlaps,
input normalization, transmission, coarse passivity, symmetry, port placement,
mode agreement, and runtime precision. Reports include the declared and
grid-realized geometry.

This is an accelerator and physics regression, not a foundry-qualified model
or publishable S-parameter result. Quantitative use still requires grid,
material, port, and pulse-window convergence. Relative output phase remains a
diagnostic because independently solved port modes may use different phase
gauges.

## Results

Generated files are written under `results/` and ignored by Git:

- `capacity.yaml`: inventory, profile attempts, boundary, and safe cell count
- `science-plan.yaml`: selected grid, device milestones, and retry history
- `profile-*/report.html`: capacity report
- `science-*/report.html` and `report.json`: physics and convergence evidence
- `science-*/resolution-*/time-*/field.png`: normalized real `Ey` at 1.55 um
- matching `.sha256` files for machine-readable records

## CUDA reference

The CUDA configuration runs the same FP64 path on one device. CUDA and ROCm
plugins must use separate virtual environments.

```bash
./setup.sh cuda
./run.sh profile configs/cuda.yaml
./run.sh science configs/cuda.yaml
```

## Configuration and QA

YAML controls profile shapes, device milestones, capacity margins, resolution,
time windows, PML thickness, validation gates, MMI dimensions, and refractive
indices. FDTDX is pinned to public upstream commit
`77b4bc523a8e98cb7dd388e99481a0000f71dd4d`; no private fork is included.

Run the local checks with:

```bash
./qa.sh
```

QA compiles the package, validates YAML and shell syntax, checks Git whitespace,
and runs the unit tests and privacy scan. Coverage includes grid divisibility,
MI250X and MI350P device contracts, capacity isolation, and OOM retries.

## References

- [AMD ROCm JAX installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/jax-install.html)
- [JAX installation](https://docs.jax.dev/en/latest/installation.html)
- [JAX default dtypes and X64](https://docs.jax.dev/en/latest/default_dtypes.html)
- [AMD Instinct MI250X](https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html)
