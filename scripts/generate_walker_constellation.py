#!/usr/bin/env python3
"""Generate the Walker-style constellation or a gap-filled altitude design.

Default invocation (the frozen manuscript-v1 orbit model):

    python scripts/generate_walker_constellation.py

This writes a deterministic manifest plus shell, ring, and satellite CSVs under
``simulation_outputs/walker_constellation_v1``.  The manifest records every
input constant and constraint.  Existing products are never replaced unless
``--overwrite`` is supplied.

Use ``--orbit-model mro120f`` to recompute the hard-threshold packing with the
live PCK/MRO120F constants. That is a model
sensitivity calculation, not a bit-for-bit regeneration of the v1 table.

The continuous-altitude mode preserves all 84 manuscript-v1 shell centers,
inserts the maximum number of additional centers at the requested minimum
spacing, recomputes the whole-year eclipse-free LTAN band at every altitude,
and sizes every shell independently::

    python scripts/generate_walker_constellation.py --layout continuous-5km
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from reflectors.constellation import (  # noqa: E402
    DEFAULT_FAMILY_SPECS,
    ExplicitShellSpec,
    MANUSCRIPT_V1_ORBIT_MODEL,
    MANUSCRIPT_V1_PROFILE_NAME,
    MRO120F_PROFILE_NAME,
    WalkerSizingConstraints,
    family_stack_altitudes_km,
    generate_walker_constellation,
    generate_walker_constellation_from_shell_specs,
    maximally_fill_altitude_gaps_km,
    mro120f_orbit_model,
    nearest_reference_family_k,
    write_constellation_products,
)
from reflectors.eclipse_bands import (  # noqa: E402
    ECLIPSE_BAND_METHOD_NAME,
    REFERENCE_EPOCH_UTC,
    EclipseBandResult,
    EclipseBandSearchConfig,
    keplerian_period_s,
    prepare_eclipse_seasons,
    whole_year_eclipse_free_ltan_band,
)
from reflectors.ephemeris import utc_to_et  # noqa: E402
from reflectors.kernels import load_kernels, unload_kernels  # noqa: E402
from reflectors.parallel import (  # noqa: E402
    CloudpickleMap,
    configure_multiprocessing_for_spice,
)


logger = logging.getLogger(__name__)

MANUSCRIPT_LAYOUT = "manuscript-v1"
CONTINUOUS_LAYOUT = "continuous-5km"


@dataclass(frozen=True)
class ShellBandRecord:
    altitude_km: float
    reference_family_k: int
    result: EclipseBandResult


_BAND_WORKER_PID: int | None = None


def _compute_shell_band_worker(task) -> ShellBandRecord:
    """Worker entry point; reload SPICE once per PID to avoid stale DAF handles."""
    global _BAND_WORKER_PID
    (
        altitude_km,
        reference_family_k,
        semimajor_axis_km,
        inclination_deg,
        mu,
        epoch_et,
        seasons,
        config,
    ) = task
    if _BAND_WORKER_PID != os.getpid():
        load_kernels()
        _BAND_WORKER_PID = os.getpid()
    start = time.perf_counter()
    result = whole_year_eclipse_free_ltan_band(
        semimajor_axis_km,
        inclination_deg,
        epoch_et,
        orbital_period_s=keplerian_period_s(semimajor_axis_km, mu),
        seasons=seasons,
        config=config,
        mu_km3_s2=mu,
    )
    print(
        f"band altitude={altitude_km:.1f} km Kref={reference_family_k}: "
        f"[{result.ltan_start_h:.6f}, {result.ltan_end_h:.6f}] h "
        f"wall={time.perf_counter() - start:.1f}s",
        flush=True,
    )
    return ShellBandRecord(
        altitude_km=altitude_km,
        reference_family_k=reference_family_k,
        result=result,
    )


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Size and emit the circular equal-altitude Walker-style manuscript "
            "constellation."
        )
    )
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to walker_constellation_v1 for the "
            "manuscript layout and walker_constellation_continuous_5km for the "
            "continuous layout."
        ),
    )
    argument_parser.add_argument(
        "--layout",
        choices=(MANUSCRIPT_LAYOUT, CONTINUOUS_LAYOUT),
        default=MANUSCRIPT_LAYOUT,
        help=(
            "Frozen 84-shell manuscript stacks (default), or preserve those "
            "centers and maximally fill their gaps while recomputing each "
            "shell's whole-year eclipse-free band."
        ),
    )
    argument_parser.add_argument(
        "--orbit-model",
        choices=(MANUSCRIPT_V1_PROFILE_NAME, MRO120F_PROFILE_NAME),
        default=None,
        help=(
            "Frozen manuscript-v1 constants or live PCK/MRO120F constants. "
            "Defaults to manuscript-v1 for the manuscript layout and mro120f "
            "for the continuous physical recomputation."
        ),
    )
    argument_parser.add_argument(
        "--family",
        type=int,
        action="append",
        choices=tuple(spec.k_orbits_per_sol for spec in DEFAULT_FAMILY_SPECS),
        help="Generate only this K family; repeat to select multiple families.",
    )
    argument_parser.add_argument("--shell-count", type=int, default=21)
    argument_parser.add_argument("--shell-spacing-km", type=float, default=5.0)
    argument_parser.add_argument("--centered-ltan-h", type=float, default=18.0)
    argument_parser.add_argument("--min-raan-spacing-deg", type=float, default=5.0)
    argument_parser.add_argument(
        "--min-same-ring-spacing-km",
        type=float,
        default=300.0,
    )
    argument_parser.add_argument(
        "--min-inter-ring-spacing-km",
        type=float,
        default=50.0,
    )
    argument_parser.add_argument(
        "--u0-deg",
        type=float,
        default=0.0,
        help="Common initial argument-of-latitude offset in satellites.csv.",
    )
    argument_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "Process workers for continuous-mode eclipse bands (default: 4; "
            "ignored by manuscript-v1 mode)."
        ),
    )
    argument_parser.add_argument("--band-seasons", type=int, default=60)
    argument_parser.add_argument("--band-output-cadence-s", type=float, default=20.0)
    argument_parser.add_argument("--band-edge-tolerance-h", type=float, default=0.01)
    argument_parser.add_argument("--band-ltan-scan-start-h", type=float, default=13.0)
    argument_parser.add_argument("--band-ltan-scan-end-h", type=float, default=23.0)
    argument_parser.add_argument("--band-ltan-scan-step-h", type=float, default=0.5)
    argument_parser.add_argument(
        "--satellites",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the reflector-level satellites.csv product (default: yes).",
    )
    argument_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace this generator's existing product files in --output-dir.",
    )
    argument_parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return argument_parser


def _selected_family_specs(selected: list[int] | None):
    if not selected:
        return DEFAULT_FAMILY_SPECS
    if len(set(selected)) != len(selected):
        raise ValueError(f"duplicate --family selection: {selected}")
    selected_set = set(selected)
    return tuple(
        spec for spec in DEFAULT_FAMILY_SPECS if spec.k_orbits_per_sol in selected_set
    )


def _orbit_model(profile_name: str):
    if profile_name == MANUSCRIPT_V1_PROFILE_NAME:
        return MANUSCRIPT_V1_ORBIT_MODEL
    if profile_name == MRO120F_PROFILE_NAME:
        load_kernels()
        try:
            return mro120f_orbit_model()
        finally:
            unload_kernels()
    raise ValueError(f"unknown orbit-model profile {profile_name!r}")


def _resolved_output_directory(layout: str, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    if layout == MANUSCRIPT_LAYOUT:
        return Path("simulation_outputs/walker_constellation_v1")
    return Path("simulation_outputs/walker_constellation_continuous_5km")


def _preflight_output_paths(
    output_directory: Path,
    filenames: tuple[str, ...],
    *,
    overwrite: bool,
) -> None:
    existing = [output_directory / name for name in filenames if (output_directory / name).exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to overwrite existing constellation products: {joined}; "
            "pass --overwrite to replace them"
        )


def _band_search_config(args) -> EclipseBandSearchConfig:
    return EclipseBandSearchConfig(
        season_count=args.band_seasons,
        ltan_scan_start_h=args.band_ltan_scan_start_h,
        ltan_scan_end_h=args.band_ltan_scan_end_h,
        ltan_scan_step_h=args.band_ltan_scan_step_h,
        edge_tolerance_h=args.band_edge_tolerance_h,
        output_cadence_s=args.band_output_cadence_s,
    )


def _compute_continuous_shell_bands(
    altitudes_km: tuple[float, ...],
    *,
    workers: int,
    config: EclipseBandSearchConfig,
):
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    if workers > 1:
        configure_multiprocessing_for_spice()
    load_kernels()
    try:
        orbit_model = mro120f_orbit_model()
        perihelion_epoch_et = utc_to_et(REFERENCE_EPOCH_UTC)
        seasons = prepare_eclipse_seasons(perihelion_epoch_et, config)
        tasks = []
        for altitude_km in altitudes_km:
            semimajor_axis_km = orbit_model.equatorial_radius_km + altitude_km
            inclination_deg = math.degrees(
                orbit_model.sun_sync_inclination_rad(semimajor_axis_km)
            )
            tasks.append(
                (
                    altitude_km,
                    nearest_reference_family_k(altitude_km),
                    semimajor_axis_km,
                    inclination_deg,
                    orbit_model.mu_km3_s2,
                    perihelion_epoch_et,
                    seasons,
                    config,
                )
            )
        print(
            f"Computing {len(tasks)} whole-year eclipse bands at "
            f"{config.season_count} seasons with {workers} worker(s)...",
            flush=True,
        )
        if workers == 1:
            records = [_compute_shell_band_worker(task) for task in tasks]
        else:
            with CloudpickleMap(n_workers=workers) as process_map:
                records = process_map(_compute_shell_band_worker, tasks)
    finally:
        unload_kernels()
    return orbit_model, tuple(records)


def _continuous_constellation(args, reference_constraints: WalkerSizingConstraints):
    preserved_altitudes_km = family_stack_altitudes_km(
        DEFAULT_FAMILY_SPECS,
        reference_constraints,
    )
    altitudes_km = maximally_fill_altitude_gaps_km(
        preserved_altitudes_km,
        reference_constraints.shell_spacing_km,
    )
    config = _band_search_config(args)
    orbit_model, band_records = _compute_continuous_shell_bands(
        altitudes_km,
        workers=args.workers,
        config=config,
    )

    family_shell_indices = {
        spec.k_orbits_per_sol: 0 for spec in DEFAULT_FAMILY_SPECS
    }
    explicit_specs = []
    for record in band_records:
        family_k = record.reference_family_k
        shell_index = family_shell_indices[family_k]
        family_shell_indices[family_k] += 1
        result = record.result
        explicit_specs.append(
            ExplicitShellSpec(
                reference_family_k=family_k,
                shell_index=shell_index,
                altitude_km=record.altitude_km,
                eclipse_free_ltan_start_h=result.ltan_start_h,
                eclipse_free_ltan_end_h=result.ltan_end_h,
                band_method=ECLIPSE_BAND_METHOD_NAME,
                lower_binding_true_anomaly_deg=(
                    result.lower_binding_true_anomaly_deg
                ),
                upper_binding_true_anomaly_deg=(
                    result.upper_binding_true_anomaly_deg
                ),
            )
        )
    packing_constraints = replace(
        reference_constraints,
        shell_count=len(explicit_specs),
    )
    constellation = generate_walker_constellation_from_shell_specs(
        explicit_specs,
        constraints=packing_constraints,
        orbit_model=orbit_model,
        shell_layout_name=CONTINUOUS_LAYOUT,
    )
    adjacent_gaps_km = [
        upper - lower for lower, upper in zip(altitudes_km, altitudes_km[1:])
    ]
    metadata = {
        "layout": {
            "name": CONTINUOUS_LAYOUT,
            "construction": (
                "preserve every manuscript-v1 shell center, then advance from "
                "each lower center by the minimum spacing while leaving any "
                "indivisible remainder below the next preserved center"
            ),
            "preserved_shell_count": len(preserved_altitudes_km),
            "added_shell_count": len(altitudes_km) - len(preserved_altitudes_km),
            "total_shell_count": len(altitudes_km),
            "minimum_altitude_km": min(altitudes_km),
            "maximum_altitude_km": max(altitudes_km),
            "minimum_adjacent_spacing_km": min(adjacent_gaps_km),
            "maximum_adjacent_spacing_km": max(adjacent_gaps_km),
            "distinct_adjacent_spacings_km": sorted(set(adjacent_gaps_km)),
            "reference_family_label_note": (
                "K labels identify the nearest reference repeat-ground-track "
                "anchor; an off-anchor shell is not asserted to repeat K times per sol"
            ),
        },
        "eclipse_band_epoch_utc": REFERENCE_EPOCH_UTC,
        "eclipse_band_search": config.manifest_record(),
        "eclipse_band_period_choice": (
            "actual circular Kepler period at each altitude, rather than the "
            "reference-anchor approximation T_sol/K"
        ),
        "band_products": {
            "summary": "eclipse_bands.csv",
            "seasons": "eclipse_band_seasons.csv",
        },
    }
    return constellation, band_records, metadata


def _write_eclipse_band_products(
    band_records: tuple[ShellBandRecord, ...],
    output_directory: Path,
) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "eclipse_bands.csv"
    seasons_path = output_directory / "eclipse_band_seasons.csv"
    summary_fields = [
        "global_shell_index",
        "reference_family_k",
        "altitude_km",
        "semimajor_axis_km",
        "inclination_deg",
        "orbital_period_s",
        "ltan_start_h",
        "ltan_end_h",
        "width_h",
        "lower_binding_season_index",
        "upper_binding_season_index",
        "lower_binding_true_anomaly_deg",
        "upper_binding_true_anomaly_deg",
        "method",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for global_index, record in enumerate(band_records):
            result = record.result
            writer.writerow(
                {
                    "global_shell_index": global_index,
                    "reference_family_k": record.reference_family_k,
                    "altitude_km": record.altitude_km,
                    "semimajor_axis_km": result.semimajor_axis_km,
                    "inclination_deg": result.inclination_deg,
                    "orbital_period_s": result.orbital_period_s,
                    "ltan_start_h": result.ltan_start_h,
                    "ltan_end_h": result.ltan_end_h,
                    "width_h": result.width_h,
                    "lower_binding_season_index": result.lower_binding_season_index,
                    "upper_binding_season_index": result.upper_binding_season_index,
                    "lower_binding_true_anomaly_deg": (
                        result.lower_binding_true_anomaly_deg
                    ),
                    "upper_binding_true_anomaly_deg": (
                        result.upper_binding_true_anomaly_deg
                    ),
                    "method": ECLIPSE_BAND_METHOD_NAME,
                }
            )

    season_fields = [
        "global_shell_index",
        "reference_family_k",
        "altitude_km",
        "season_index",
        "epoch_et",
        "true_anomaly_deg",
        "equation_of_center_offset_h",
        "effective_ltan_start_h",
        "effective_ltan_end_h",
        "reference_ltan_start_h",
        "reference_ltan_end_h",
    ]
    with seasons_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=season_fields)
        writer.writeheader()
        for global_index, record in enumerate(band_records):
            for season in record.result.seasons:
                writer.writerow(
                    {
                        "global_shell_index": global_index,
                        "reference_family_k": record.reference_family_k,
                        "altitude_km": record.altitude_km,
                        "season_index": season.season_index,
                        "epoch_et": season.epoch_et,
                        "true_anomaly_deg": season.true_anomaly_deg,
                        "equation_of_center_offset_h": (
                            season.equation_of_center_offset_h
                        ),
                        "effective_ltan_start_h": season.effective_ltan_start_h,
                        "effective_ltan_end_h": season.effective_ltan_end_h,
                        "reference_ltan_start_h": season.reference_ltan_start_h,
                        "reference_ltan_end_h": season.reference_ltan_end_h,
                    }
                )
    return {"eclipse_bands": summary_path, "eclipse_band_seasons": seasons_path}


def _print_summary(constellation, products: dict[str, Path]) -> None:
    print(f"Orbit model: {constellation.orbit_model.name}")
    for spec in constellation.family_specs:
        family_shells = constellation.family_shells(spec.k_orbits_per_sol)
        ring_counts = [shell.rings for shell in family_shells]
        satellite_counts = [shell.satellites_per_ring for shell in family_shells]
        phasing_values = sorted({shell.phasing for shell in family_shells})
        print(
            f"K{spec.k_orbits_per_sol}: {len(family_shells)} shells, "
            f"{sum(ring_counts)} rings, "
            f"{sum(shell.satellite_count for shell in family_shells):,} OSRs; "
            f"P={min(ring_counts)}--{max(ring_counts)}, "
            f"S={min(satellite_counts)}--{max(satellite_counts)}, "
            f"F={phasing_values}"
        )
    print(
        f"Total: {len(constellation.shells)} shells, "
        f"{constellation.ring_count} rings, "
        f"{constellation.satellite_count:,} OSRs"
    )
    for product_name, path in products.items():
        print(f"{product_name}: {path}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not math.isfinite(args.u0_deg):
        raise ValueError("--u0-deg must be finite")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    reference_constraints = WalkerSizingConstraints(
        shell_count=args.shell_count,
        shell_spacing_km=args.shell_spacing_km,
        centered_ltan_h=args.centered_ltan_h,
        min_raan_spacing_deg=args.min_raan_spacing_deg,
        min_same_ring_spacing_km=args.min_same_ring_spacing_km,
        min_inter_ring_spacing_km=args.min_inter_ring_spacing_km,
    )
    output_directory = _resolved_output_directory(args.layout, args.output_dir)
    expected_filenames = ["manifest.json", "shells.csv", "rings.csv"]
    if args.satellites:
        expected_filenames.append("satellites.csv")
    if args.layout == CONTINUOUS_LAYOUT:
        expected_filenames.extend(("eclipse_bands.csv", "eclipse_band_seasons.csv"))
    _preflight_output_paths(
        output_directory,
        tuple(expected_filenames),
        overwrite=args.overwrite,
    )

    if args.layout == MANUSCRIPT_LAYOUT:
        profile_name = args.orbit_model or MANUSCRIPT_V1_PROFILE_NAME
        orbit_model = _orbit_model(profile_name)
        constellation = generate_walker_constellation(
            family_specs=_selected_family_specs(args.family),
            constraints=reference_constraints,
            orbit_model=orbit_model,
        )
        band_records = ()
        manifest_metadata = {
            "layout": {"name": MANUSCRIPT_LAYOUT},
            "eclipse_band_source": (
                "fixed family-specific eclipse-free LTAN bands reused across "
                "each 21-shell stack"
            ),
        }
    else:
        if args.family:
            raise ValueError(
                "--family cannot be combined with --layout continuous-5km; "
                "the continuous grid spans all four anchors"
            )
        profile_name = args.orbit_model or MRO120F_PROFILE_NAME
        if profile_name != MRO120F_PROFILE_NAME:
            raise ValueError(
                "--layout continuous-5km requires --orbit-model mro120f because "
                "its bands are recomputed with the current MRO120F J2 force model"
            )
        constellation, band_records, manifest_metadata = _continuous_constellation(
            args,
            reference_constraints,
        )

    products = write_constellation_products(
        constellation,
        output_directory,
        include_satellites=args.satellites,
        initial_argument_of_latitude_offset_deg=args.u0_deg,
        overwrite=args.overwrite,
        manifest_metadata=manifest_metadata,
    )
    if band_records:
        products.update(_write_eclipse_band_products(band_records, output_directory))
    _print_summary(constellation, products)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
