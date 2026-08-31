# Manuscript figures

This directory maps every figure in *Doubling Sunlight at a Human Mars Base
Using Orbiting Solar Reflectors* to its source data and producing code.

The images under `reference/` are the exact files included by the TeX sources
in the Overleaf snapshot inspected on 2026-08-31. Reproduced images are written
to the ignored `generated/` directory so that a run never overwrites the
archival references. The generated images are expected to convey the same data
and visual structure, but they need not have identical hashes: several
manuscript images were manually cropped or assembled before inclusion.

## Setup and use

Install the plotting dependencies and the scientific data used by the core
models:

```bash
python -m pip install -e ".[figures]"
python scripts/fetch_kernels.py
```

The simulation products needed only for these figures are kept in the separate
`mars-osr-manuscript-figure-data-v1.tar.gz` release asset. After downloading
it, install and verify it with:

```bash
python scripts/fetch_manuscript_figure_data.py \
  --archive /path/to/mars-osr-manuscript-figure-data-v1.tar.gz
```

Then list or reproduce figures from the repository root:

```bash
python scripts/reproduce_manuscript_figures.py --list
python scripts/reproduce_manuscript_figures.py --figure 7
python scripts/reproduce_manuscript_figures.py --all-available
python scripts/reproduce_manuscript_figures.py --verify
```

Figures 2, 4, 5, and 6 use PyVista/VTK and require a working off-screen OpenGL
backend. On a headless Linux system, configure an OSMesa-enabled VTK build or a
virtual display before running those producers.

`manifest.json` is the machine-readable registry. `input_manifest.json`
records the SHA-256 and size of every file in the separate data archive, and
`data_archive.json` pins the archive itself.

## Coverage

| Figure | Subject | Reproduction status |
| --- | --- | --- |
| 1 | Concept of operations | Manual reference asset |
| 2 | Sail reference frames | Available |
| 3 | Earth escape, cruise, and Mars capture | Available |
| 4 | Eclipse-free LTAN families | Available |
| 5 | Station-keeping attitude | Available |
| 6 | Delivery-window attitude | Available |
| 7 | Fluence versus LTAN | Available; 1,000 m² reflector |
| 8 | Ring and shell scaling | Available; deterministic replacement for manual side-by-side assembly |
| 9 | Annual and diurnal insolation | Plot layer included; climate-model package and canonical inputs pending |
| 10 | Surface temperature | Plot layer included; climate-model package and canonical inputs pending |
| S1 | Escape and capture duration | Available |
| S2 | One-sol attitude angles | Available |
| S3 | Slew demand and fluence | Available from the archived sweep |
| S4 | Marginal ring fluence | Available from the archived per-LTAN series |

Figures 9 and 10 are intentionally separate from the vacuum-only optical model
in the Mars OSR core. Their reference images and strict NetCDF plotting and
validation code are included now; the external atmospheric-transmission and
one-dimensional climate-model generation package will be incorporated after
the canonical coauthor handoff is available.

## Third-party data

`data/manuscript_figures/mars_texture.jpg` in the separate figure-data archive
is the [Solar System Scope 8K Mars texture](https://www.solarsystemscope.com/textures/),
distributed under the [Creative Commons Attribution 4.0 International
license](https://creativecommons.org/licenses/by/4.0/). The texture is based
on NASA elevation and imagery data.
