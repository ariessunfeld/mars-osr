"""Self-contained tests for cruise handoff phasing and interpolation."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from reflectors.cruise_phasing import PeriodicPchip
from reflectors.ephemeris import utc_to_et


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_periodic_pchip_wraps_in_range():
    period = 687.0
    phase = np.array([0.0, 172.0, 343.0, 515.0])
    values = np.array([1.0, 2.0, 1.0, 0.5])
    f = PeriodicPchip(phase, values, period)
    y = f(686.0)
    assert min(values) - 1e-9 <= y <= max(values) + 1e-9
    assert f(50.0) == pytest.approx(f(50.0 + period), abs=1e-9)


def test_escape_exit_phase_days_formula():
    """Exit phase is departure phase plus escape duration modulo the year."""
    from reflectors.cruise_phasing import SOL_SECONDS, escape_exit_phase_days

    period = 365.11
    assert escape_exit_phase_days(
        {"phase_days": "300.0", "T_escape": "528.0", "time_unit": "day"},
        period,
    ) == pytest.approx((300.0 + 528.0) % period, abs=1e-9)
    assert escape_exit_phase_days(
        {"phase_days": "10.0", "T_escape": "300.0", "time_unit": "day"},
        365.0,
    ) == pytest.approx(310.0)
    t_days = 400.0 * (SOL_SECONDS / 86400.0)
    assert escape_exit_phase_days(
        {"phase_days": "100.0", "T_escape": "400.0", "time_unit": "sols"},
        687.0,
    ) == pytest.approx((100.0 + t_days) % 687.0, abs=1e-9)


def test_catalog_handoff_rows_escape_epochs(tmp_path):
    from reflectors.cruise_phasing import catalog_handoff_rows

    grid = tmp_path / "grid.csv"
    _write_csv(
        grid,
        ["planet", "phase_index", "epoch_utc"],
        [
            {"planet": "earth", "phase_index": "0", "epoch_utc": "2028-01-04T00:00:00"},
            {"planet": "earth", "phase_index": "1", "epoch_utc": "2028-07-04T00:00:00"},
        ],
    )
    cols = [
        "run", "planet", "sigma_g", "phase_index", "mode", "escaped",
        "phase_days", "arrival_phase_days", "T_escape", "time_unit", "vinf_kmps",
        "vhelio_prograde_kmps", "vhelio_inplane_kmps", "vhelio_oop_kmps",
        "exitpos_prograde", "exitpos_inplane", "exitpos_oop",
    ]

    def row(idx, phase, duration, **overrides):
        values = {
            "run": f"r{idx}_{phase}", "planet": "earth", "sigma_g": "18.0",
            "phase_index": str(idx), "mode": "escape", "escaped": "True",
            "phase_days": str(phase), "arrival_phase_days": "",
            "T_escape": str(duration), "time_unit": "day", "vinf_kmps": "1.0",
            "vhelio_prograde_kmps": "29.0", "vhelio_inplane_kmps": "0.5",
            "vhelio_oop_kmps": "0.1", "exitpos_prograde": "0.3",
            "exitpos_inplane": "-0.6", "exitpos_oop": "-0.7",
        }
        values.update(overrides)
        return values

    period_days = 2 * (
        utc_to_et("2028-07-04T00:00:00") - utc_to_et("2028-01-04T00:00:00")
    ) / 86400.0
    characterization = tmp_path / "characterization.csv"
    _write_csv(
        characterization,
        cols,
        [row(0, 0.0, 300.0), row(1, period_days / 2.0, 100.0)],
    )
    rows = catalog_handoff_rows(
        str(characterization), 18.0, planet="earth", mode="escape", grid_csv=str(grid)
    )
    assert len(rows) == 2
    first = next(item for item in rows if item.phase_index == 0)
    assert first.handoff_et == pytest.approx(
        utc_to_et("2028-01-04T00:00:00") + 300.0 * 86400.0
    )
    assert first.handoff_phase_days == pytest.approx(300.0 % period_days, abs=1e-6)
    assert rows == sorted(rows, key=lambda item: item.handoff_et)

    bad = tmp_path / "bad.csv"
    _write_csv(
        bad,
        cols,
        [row(0, 37.0, 300.0), row(1, period_days / 2.0, 100.0)],
    )
    with pytest.raises(ValueError, match="bookkeeping inconsistent"):
        catalog_handoff_rows(
            str(bad), 18.0, planet="earth", mode="escape", grid_csv=str(grid)
        )


def test_build_escape_interp_indexes_by_exit_phase(tmp_path):
    from reflectors.cruise_phasing import build_escape_interp

    grid = tmp_path / "grid.csv"
    _write_csv(
        grid,
        ["planet", "phase_index", "epoch_utc"],
        [
            {"planet": "earth", "phase_index": "0", "epoch_utc": "2028-01-04T00:00:00"},
            {"planet": "earth", "phase_index": "1", "epoch_utc": "2028-07-04T00:00:00"},
        ],
    )
    cols = [
        "planet", "sigma_g", "phase_index", "mode", "escaped", "phase_days",
        "arrival_phase_days", "T_escape", "time_unit", "vhelio_prograde_kmps",
        "vhelio_inplane_kmps", "vhelio_oop_kmps", "exitpos_prograde",
        "exitpos_inplane", "exitpos_oop",
    ]

    def row(phase, duration):
        return {
            "planet": "earth", "sigma_g": "18.0", "phase_index": "0",
            "mode": "escape", "escaped": "True", "phase_days": str(phase),
            "arrival_phase_days": "", "T_escape": str(duration), "time_unit": "day",
            "vhelio_prograde_kmps": "29.0", "vhelio_inplane_kmps": "0.5",
            "vhelio_oop_kmps": "0.1", "exitpos_prograde": "0.3",
            "exitpos_inplane": "-0.6", "exitpos_oop": "-0.7",
        }

    characterization = tmp_path / "characterization.csv"
    _write_csv(
        characterization,
        cols,
        [row(10, 300), row(50, 100), row(100, 200), row(200, 160)],
    )
    interp = build_escape_interp(str(characterization), 18.0, grid_csv=str(grid))
    assert interp is not None and interp.n_phases == 4
    exit_phase = (100.0 + 200.0) % (interp.period_s / 86400.0)
    query_et = interp.perihelion_et + exit_phase * 86400.0
    assert interp.duration_days(query_et) == pytest.approx(200.0, abs=1.0)
