# Reproducibility

Mars OSR separates source-code reproducibility from archival result
reproducibility.

## What this repository guarantees

- The Python library, maintained command-line scripts, and tests are versioned.
- External NAIF kernels and the MRO120F field have explicit download locations
  and checksums in the source.
- `MARS_OSR_DATA_DIR` selects a shared, external scientific-data root for
  installed or containerized deployments.
- Command-line scripts record their configuration in generated summaries or
  manifests.
- Generated outputs are written to `simulation_outputs/` and are never committed.
- Fast tests are the default; long-baseline physical regressions are explicitly
  marked `slow`.

## What is not included

This source release does not include generated simulation products or
figure-generation inputs. It supports calculations from documented public
inputs but does not reproduce unversioned output files bit for bit.

## Suggested record for a scientific run

Record the Git commit, environment export, exact command, input artifact hashes,
stdout/stderr log, and hashes of generated summaries. Long runs should also note
hardware, worker count, and solver termination criteria.
