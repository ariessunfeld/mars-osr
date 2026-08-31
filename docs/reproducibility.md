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
- Exact manuscript reference images and machine-readable figure provenance are
  versioned under `figures/manuscript/`.
- The minimal simulation products used by the manuscript figures are distributed
  as a checksum-pinned release asset and remain outside Git history.
- Fast tests are the default; long-baseline physical regressions are explicitly
  marked `slow`.

## Manuscript figure inputs

Generated simulation products are not committed to the source repository. The
curated inputs for Figures 2–8 and S1–S4 can be installed from the separate
`mars-osr-manuscript-figure-data-v1.tar.gz` release asset with
`scripts/fetch_manuscript_figure_data.py`. Its archive checksum and every
extracted file hash are versioned with the figure registry.

Figure reproduction targets scientific content and layout rather than binary
identity. Font rendering, raster metadata, and the replacement of manual crops
or composites can change image hashes without changing the plotted result.

Figures 9 and 10 depend on a separately maintained atmospheric-transmission and
one-dimensional climate model. Their plotting layers and exact manuscript
reference images are included; the canonical generation package and inputs are
pending a coauthor handoff.

## Suggested record for a scientific run

Record the Git commit, environment export, exact command, input artifact hashes,
stdout/stderr log, and hashes of generated summaries. Long runs should also note
hardware, worker count, and solver termination criteria.
