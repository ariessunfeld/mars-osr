"""Phasing interpolants for the interplanetary cruise: boundary states + durations
as smooth functions of orbital phase, reconstructed at any epoch.

The caller supplies a characterization CSV containing the cruise's START
(Earth forward-escape Hill handoff) and END (Mars backward-capture node). For a
given sail density sigma, this module builds per-phase PCHIP interpolants of:

  * the escape/capture DURATION (days/sols), and
  * the Hill handoff state, stored as components in the planet's CO-ROTATING
    heliocentric triad (``reflectors.ephemeris.corotating_heliocentric_triad``) so
    they vary SMOOTHLY with orbital phase: the mission HELIOCENTRIC velocity
    (``vhelio_*`` -- already INBOUND for captures / OUTBOUND for escapes) and the
    Hill-exit position DIRECTION (``exitpos_*``, a unit vector in the triad).

The boundary state at an arbitrary epoch is then reconstructed by building the
triad from the planet's live SPICE state at that epoch and composing the
phase-interpolated components, so the same physical handoff can be placed at any
departure or arrival epoch when transfer duration is a decision variable.

Reconstruction conventions:
  * velocity = ``vhelio_pro*p_hat + vhelio_in*e_hat + vhelio_oop*n_hat`` DIRECTLY
    (the ``vhelio_*`` columns are the FULL heliocentric velocity = ``v_planet +
    direction*vinf``; do NOT re-add the planet velocity -- the double-count trap);
  * position = ``planet_r + hill_radius * unit(exitpos_pro*p_hat + ... )`` (the node
    is at ``|r| = hill_radius``; ``exitpos_*`` carries only the direction).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from reflectors.central_body import earth_central_body, mars_central_body
from reflectors.ephemeris import (
    body_state,
    corotating_heliocentric_triad,
    utc_to_et,
)

SECONDS_PER_DAY = 86400.0
# Mars mean solar day (s); only used to convert a sol-valued escape duration to days
# in the exit-phase computation. Escape interps are Earth (days) in practice, but the
# conversion keeps the formula unit-safe for a Mars forward-escape interpolation.
SOL_SECONDS = 88775.244

# Planet -> (NAIF barycenter name for the heliocentric state, the central-body
# factory for the Hill radius). Barycenter (not center) gives the consistent orbital
# cycle used by the characterization phase coordinate.
_PLANET = {
    "mars": ("MARS BARYCENTER", mars_central_body),
    "earth": ("EARTH BARYCENTER", earth_central_body),
}


class PeriodicPchip:
    """PCHIP over a periodic phase axis ``[0, period)``.

    The samples are tiled one period below and one period above before fitting, so
    a query near the year boundary interpolates ACROSS the wrap instead of
    extrapolating off the end. This keeps both plotted curves and the
    duration-column Jacobian continuous in arrival epoch.
    Requires >= 2 samples.
    """

    def __init__(self, phase_days: np.ndarray, values: np.ndarray, period_days: float):
        from scipy.interpolate import PchipInterpolator

        x = np.asarray(phase_days, dtype=float)
        y = np.asarray(values, dtype=float)
        if x.size < 2 or x.size != y.size:
            raise ValueError(f"need >= 2 matched samples; got {x.size}, {y.size}")
        order = np.argsort(x)
        x, y = x[order], y[order]
        xt = np.concatenate([x - period_days, x, x + period_days])
        yt = np.concatenate([y, y, y])
        self._pchip = PchipInterpolator(xt, yt, extrapolate=True)
        self._period = float(period_days)

    def __call__(self, phase_days: float) -> float:
        return float(self._pchip(np.mod(float(phase_days), self._period)))


@dataclass(frozen=True)
class PhaseInterp:
    """Per-sigma phasing interpolant for one planet leg (escape or capture).

    Reconstructs the heliocentric boundary 6-state and the leg duration at any
    epoch via the planet's live co-rotating triad. ``direction`` (+1 escape /
    -1 capture) is metadata; the stored ``vhelio_*`` components already include
    this sign, so reconstruction does not re-apply it.
    """

    planet: str
    naif: str
    sigma_g: float
    perihelion_et: float
    period_s: float
    hill_radius_km: float
    direction: int
    n_phases: int
    _duration: PeriodicPchip
    _vh_pro: PeriodicPchip
    _vh_in: PeriodicPchip
    _vh_oop: PeriodicPchip
    _ep_pro: PeriodicPchip
    _ep_in: PeriodicPchip
    _ep_oop: PeriodicPchip

    def phase_days(self, et: float) -> float:
        return ((et - self.perihelion_et) % self.period_s) / SECONDS_PER_DAY

    def duration_days(self, et: float) -> float:
        """Interpolated leg duration (days) at the epoch's phase. NOTE the stored
        unit is sols for Mars / days for Earth; for Mars multiply by the sol-to-day
        factor at the call site if days are needed (the caller knows the unit)."""
        return self._duration(self.phase_days(et))

    def boundary_state(self, et: float) -> np.ndarray:
        """Heliocentric J2000 6-state of the handoff at ``et`` (km, km/s).

        velocity reconstructed DIRECTLY from ``vhelio_*`` (full heliocentric, with
        the escape/capture direction already baked in); position = planet position
        + ``hill_radius * unit(exitpos)``.
        """
        ph = self.phase_days(et)
        return _compose_handoff_state(
            self.naif, et, self.hill_radius_km,
            self._vh_pro(ph), self._vh_in(ph), self._vh_oop(ph),
            self._ep_pro(ph), self._ep_in(ph), self._ep_oop(ph),
        )


def _compose_handoff_state(
    naif: str,
    et: float,
    hill_radius_km: float,
    vh_pro: float, vh_in: float, vh_oop: float,
    ep_pro: float, ep_in: float, ep_oop: float,
) -> np.ndarray:
    """Compose a heliocentric Hill-handoff 6-state from co-rotating-triad components
    at ``et`` (the single-sourced reconstruction shared by the smooth
    :class:`PhaseInterp` and the exact per-row :class:`CatalogHandoff`).

    velocity = ``vh_pro*p_hat + vh_in*e_hat + vh_oop*n_hat`` DIRECTLY (the full
    heliocentric mission velocity -- no planet-velocity re-add); position = planet
    position + ``hill_radius * unit(exitpos)`` (module-docstring conventions)."""
    planet6, _ = body_state(naif, et, observer="SUN")
    p_hat, e_hat, n_hat = corotating_heliocentric_triad(planet6)
    v_helio = vh_pro * p_hat + vh_in * e_hat + vh_oop * n_hat
    ep = ep_pro * p_hat + ep_in * e_hat + ep_oop * n_hat
    ep = ep / float(np.linalg.norm(ep))  # unit Hill-exit direction
    r = planet6[:3] + hill_radius_km * ep
    return np.concatenate([r, v_helio])


def grid_perihelion_anchor(grid_csv: str) -> dict:
    """``{planet: (perihelion_et, period_s)}`` from the sweep grid CSV.

    Zero phase (phase_days==0) is perihelion; phases are evenly spaced over one
    anomalistic year, so ``period = spacing * (max_phase + 1)``. Single-sources the
    anchor from the same grid that defined the sweep (mirrors the aggregator's
    helper of the same name)."""
    if not grid_csv or not os.path.exists(grid_csv):
        return {}
    epochs: dict[str, dict[int, str]] = {}
    with open(grid_csv, newline="") as f:
        for row in csv.DictReader(f):
            epochs.setdefault(row["planet"], {})[int(row["phase_index"])] = row["epoch_utc"]
    out = {}
    for planet, ph in epochs.items():
        idxs = sorted(ph)
        if len(idxs) < 2 or idxs[0] != 0:
            continue
        et0 = utc_to_et(ph[0])
        spacing = (utc_to_et(ph[idxs[-1]]) - et0) / idxs[-1]
        out[planet] = (et0, spacing * (idxs[-1] + 1))
    return out


def _read_rows(csv_path: str, planet: str, mode: str, sigma_g: float):
    """Escaped, in-mode, this-sigma rows from the characterization CSV (masking
    non-escapes / blanks)."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["planet"] != planet or r["mode"] != mode:
                continue
            if int(round(float(r["sigma_g"]))) != int(round(sigma_g)):
                continue
            if str(r["escaped"]).strip().lower() != "true":
                continue
            if not r.get("arrival_phase_days") and mode == "capture":
                continue
            rows.append(r)
    return rows


def escape_exit_phase_days(row: dict, period_days: float) -> float:
    """Orbital-year phase (days) at which a forward escape EXITS the Hill sphere.

    ``exit_phase = (departure_phase + escape_duration) mod period``. The cruise START
    is the Hill-EXIT state, whose v_inf direction is set by the Sun-line over the final
    orbits (i.e. the Sun position at EXIT), so the exit-state components must be indexed
    by EXIT phase -- NOT the DEPARTURE phase the aggregator tags. Departure phase is the
    correct x-axis for an escape-time-vs-launch-date plot but is unsuitable for
    reconstructing the boundary state at the exit epoch. Captures
    are unaffected: they are indexed by ARRIVAL phase, and the node IS at the arrival
    epoch, so the convention is already consistent.

    ``T_escape`` is taken in days (``time_unit`` starting with 'sol' is converted via
    :data:`SOL_SECONDS`)."""
    t = float(row["T_escape"])
    unit = str(row.get("time_unit", "day")).strip().lower()
    t_days = t * (SOL_SECONDS / SECONDS_PER_DAY) if unit.startswith("sol") else t
    return (float(row["phase_days"]) + t_days) % period_days


def build_phase_interp(
    csv_path: str,
    sigma_g: float,
    *,
    planet: str,
    mode: str,
    grid_csv: str,
    hill_radius_km: Optional[float] = None,
) -> Optional[PhaseInterp]:
    """Build the per-sigma escape/capture phasing interpolant, or ``None`` if fewer
    than 2 escaping phases survive the mask (-> a masked (sigma, launch-date) gap).

    ``mode`` is ``"capture"`` (Mars, x = arrival_phase_days, direction -1) or
    ``"escape"`` (Earth, x = phase_days, direction +1).
    """
    if planet not in _PLANET:
        raise ValueError(f"planet must be one of {list(_PLANET)}, got {planet!r}")
    if mode not in ("capture", "escape"):
        raise ValueError(f"mode must be 'capture' or 'escape', got {mode!r}")
    naif, cb_factory = _PLANET[planet]
    anchor = grid_perihelion_anchor(grid_csv).get(planet)
    if anchor is None:
        raise ValueError(f"no perihelion anchor for {planet!r} in {grid_csv}")
    perihelion_et, period_s = anchor
    period_days = period_s / SECONDS_PER_DAY
    hill = hill_radius_km if hill_radius_km is not None else cb_factory().hill_radius_km

    rows = _read_rows(csv_path, planet, mode, sigma_g)
    if len(rows) < 2:
        return None
    # x-axis: ARRIVAL phase for captures (node IS at the arrival epoch) / EXIT phase for
    # escapes (the Hill-EXIT boundary state is reconstructed at the exit epoch). Using the
    # DEPARTURE phase_days for escapes mis-indexes the exit state by roughly the
    # escape duration modulo one year; escape_exit_phase_days corrects it.
    if mode == "capture":
        x = np.array([float(r["arrival_phase_days"]) for r in rows])
    else:
        x = np.array([escape_exit_phase_days(r, period_days) for r in rows])

    def pchip(col):
        return PeriodicPchip(x, np.array([float(r[col]) for r in rows]), period_days)

    return PhaseInterp(
        planet=planet, naif=naif, sigma_g=float(sigma_g),
        perihelion_et=perihelion_et, period_s=period_s, hill_radius_km=float(hill),
        direction=(-1 if mode == "capture" else 1), n_phases=len(rows),
        _duration=pchip("T_escape"),
        _vh_pro=pchip("vhelio_prograde_kmps"),
        _vh_in=pchip("vhelio_inplane_kmps"),
        _vh_oop=pchip("vhelio_oop_kmps"),
        _ep_pro=pchip("exitpos_prograde"),
        _ep_in=pchip("exitpos_inplane"),
        _ep_oop=pchip("exitpos_oop"),
    )


def build_capture_interp(csv_path, sigma_g, *, grid_csv, hill_radius_km=None):
    """Mars backward-capture interpolant (arrival phase -> node)."""
    return build_phase_interp(csv_path, sigma_g, planet="mars", mode="capture",
                              grid_csv=grid_csv, hill_radius_km=hill_radius_km)


def build_escape_interp(csv_path, sigma_g, *, grid_csv, hill_radius_km=None):
    """Earth forward-escape interpolant: EXIT phase -> Hill-exit START state.

    Indexed by EXIT phase (= departure phase + escape duration mod year), so
    ``boundary_state(exit_et)`` reconstructs the cruise START at its Hill-exit
    epoch consistently."""
    return build_phase_interp(csv_path, sigma_g, planet="earth", mode="escape",
                              grid_csv=grid_csv, hill_radius_km=hill_radius_km)


@dataclass(frozen=True)
class CatalogHandoff:
    """One EXACT escaped catalog row as a discrete transfer endpoint.

    The discrete complement of :class:`PhaseInterp`: an Earth escape row is an
    exact cruise START whose sensitive Hill-exit direction is never interpolated
    across, while a Mars capture row is an exact node target.

    Epochs: ``source_et`` is the leg's generation epoch (Earth LEO departure t0 /
    Mars LMO park t3); ``handoff_et`` is the Hill handoff epoch (Earth exit t1 =
    t0 + T / Mars forward-time capture node t2 = t3 - T, the backward-ephemeris
    capture run walking et DOWN from the park). ``duration_days`` is the leg
    duration with sol-valued rows already converted to days.
    """

    planet: str
    mode: str
    sigma_g: float
    phase_index: int
    run: str
    naif: str
    hill_radius_km: float
    source_et: float
    handoff_et: float
    duration_days: float
    handoff_phase_days: float
    vinf_kmps: float
    vh_pro: float
    vh_in: float
    vh_oop: float
    ep_pro: float
    ep_in: float
    ep_oop: float

    def boundary_state(self, et: Optional[float] = None) -> np.ndarray:
        """Heliocentric J2000 6-state of THIS row's Hill handoff (km, km/s).

        Default ``et=None`` reconstructs at the row's own ``handoff_et``, where the
        composition through the planet's live co-rotating triad reproduces the
        actual run-terminal state (exact, up to the ``|r| = hill_radius`` placement
        convention). Passing another ``et`` places the row at a periodic image of
        its orbital phase -- the documented year-periodicity surrogate; the caller
        owns keeping ``et`` near the same phase (e.g. ``handoff_et + k*period_s``).
        """
        t = self.handoff_et if et is None else float(et)
        return _compose_handoff_state(
            self.naif, t, self.hill_radius_km,
            self.vh_pro, self.vh_in, self.vh_oop,
            self.ep_pro, self.ep_in, self.ep_oop,
        )


def _grid_epochs_et(grid_csv: str, planet: str) -> dict:
    """``{phase_index: generation epoch (ET)}`` for ``planet`` from the sweep grid
    CSV. ``epoch_utc`` is a per-phase date (identical across the grid's sigma rows;
    last row wins, mirroring :func:`grid_perihelion_anchor`)."""
    out: dict[int, str] = {}
    with open(grid_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["planet"] == planet:
                out[int(row["phase_index"])] = row["epoch_utc"]
    return {k: utc_to_et(v) for k, v in out.items()}


def catalog_handoff_rows(
    csv_path: str,
    sigma_g: float,
    *,
    planet: str,
    mode: str,
    grid_csv: str,
    hill_radius_km: Optional[float] = None,
    phase_check_tol_days: float = 0.05,
) -> list:
    """The EXACT per-row handoffs of one (planet, mode, sigma) catalog, sorted by
    ``handoff_et`` -- each escaped characterization row as a :class:`CatalogHandoff`
    with absolute epochs, reconstructable at its own handoff epoch.

    Every row's recomputed handoff phase (from grid epoch + duration) is
    cross-checked against the aggregator's independently-stored phase
    (``arrival_phase_days`` for captures / :func:`escape_exit_phase_days` for
    escapes); a mismatch beyond ``phase_check_tol_days`` raises explicitly
    rather than masking epoch mis-bookkeeping. A phase_index missing from the
    grid also raises because the grid defines the sweep.
    """
    if planet not in _PLANET:
        raise ValueError(f"planet must be one of {list(_PLANET)}, got {planet!r}")
    if mode not in ("capture", "escape"):
        raise ValueError(f"mode must be 'capture' or 'escape', got {mode!r}")
    naif, cb_factory = _PLANET[planet]
    anchor = grid_perihelion_anchor(grid_csv).get(planet)
    if anchor is None:
        raise ValueError(f"no perihelion anchor for {planet!r} in {grid_csv}")
    perihelion_et, period_s = anchor
    period_days = period_s / SECONDS_PER_DAY
    hill = hill_radius_km if hill_radius_km is not None else cb_factory().hill_radius_km
    grid_et = _grid_epochs_et(grid_csv, planet)

    out = []
    for r in _read_rows(csv_path, planet, mode, sigma_g):
        idx = int(r["phase_index"])
        if idx not in grid_et:
            raise ValueError(
                f"{r['run']}: phase_index {idx} not in grid {grid_csv} for {planet!r}"
            )
        source_et = grid_et[idx]
        t = float(r["T_escape"])
        unit = str(r.get("time_unit", "day")).strip().lower()
        duration_days = t * (SOL_SECONDS / SECONDS_PER_DAY) if unit.startswith("sol") else t
        sign = 1.0 if mode == "escape" else -1.0
        handoff_et = source_et + sign * duration_days * SECONDS_PER_DAY
        handoff_phase = ((handoff_et - perihelion_et) % period_s) / SECONDS_PER_DAY
        stored_phase = (
            float(r["arrival_phase_days"]) if mode == "capture"
            else escape_exit_phase_days(r, period_days)
        )
        dphase = abs(handoff_phase - stored_phase)
        dphase = min(dphase, period_days - dphase)  # periodic distance
        if dphase > phase_check_tol_days:
            raise ValueError(
                f"{r['run']}: recomputed handoff phase {handoff_phase:.4f} d vs stored "
                f"{stored_phase:.4f} d (|d|={dphase:.4f} > {phase_check_tol_days}) -- "
                "grid epoch / duration bookkeeping inconsistent"
            )
        out.append(CatalogHandoff(
            planet=planet, mode=mode, sigma_g=float(sigma_g), phase_index=idx,
            run=r["run"], naif=naif, hill_radius_km=float(hill),
            source_et=float(source_et), handoff_et=float(handoff_et),
            duration_days=float(duration_days), handoff_phase_days=float(handoff_phase),
            vinf_kmps=float(r["vinf_kmps"]),
            vh_pro=float(r["vhelio_prograde_kmps"]),
            vh_in=float(r["vhelio_inplane_kmps"]),
            vh_oop=float(r["vhelio_oop_kmps"]),
            ep_pro=float(r["exitpos_prograde"]),
            ep_in=float(r["exitpos_inplane"]),
            ep_oop=float(r["exitpos_oop"]),
        ))
    out.sort(key=lambda h: h.handoff_et)
    return out


def make_piecewise_vartime_defect(
    z0_km_kmps: np.ndarray,
    dep_et: float,
    capture_interp: PhaseInterp,
    ref_normal: np.ndarray,
    sail,
    central_body,
    third_bodies,
    *,
    N: int,
    r_scale_km: float,
    v_scale_kmps: float,
    steps_per_orbit: int = 200,
):
    """Variable-time defect with a duration-dependent moving Mars target.

    Decision vector ``x = [phi_0..phi_{N-1}, theta_0..theta_{N-1}, D_days]``. Each
    call sets ``T_s = D * 86400``, ``t_arr = dep_et + T_s``, looks up the Mars
    capture-node target at that arrival epoch (``capture_interp.boundary_state``),
    propagates the piecewise-RTN cruise for ``T_s``, and returns the scaled 6-defect.
    The cruise START ``z0`` and clock reference ``ref_normal`` are FIXED (no v_inf
    freedom; the IC is the exact escaped handoff). Because the target is recomputed
    inside the closure, the parallel finite-difference Jacobian over ``x`` captures
    BOTH the trajectory's sensitivity to ``D`` (longer flight) AND the target's
    motion with ``D`` -- no analytic target derivative needed (the
    ``PeriodicPchip`` continuity in ``t_arr`` is what makes the D-column FD valid).

    Picklable + fork-safe (``_ensure_worker_kernels``) for the parallel-FD Jacobian.
    """
    from reflectors.cruise_cost import _ensure_worker_kernels
    from reflectors.cruise_piecewise import propagate_cruise_piecewise

    z0 = np.asarray(z0_km_kmps, dtype=float)
    ref = np.asarray(ref_normal, dtype=float)
    tb = tuple(third_bodies)
    creator_pid = os.getpid()

    def defect(x: np.ndarray) -> np.ndarray:
        _ensure_worker_kernels(creator_pid)
        x = np.asarray(x, dtype=float)
        phis, thetas = x[:N], x[N : 2 * N]
        D_days = float(x[2 * N])
        if not (D_days > 0.0):
            return np.full(6, 1.0e3)
        T_s = D_days * SECONDS_PER_DAY
        t_arr = dep_et + T_s
        z_tgt = capture_interp.boundary_state(t_arr)
        run = propagate_cruise_piecewise(
            phis, thetas, z0, dep_et, T_s, ref, sail, central_body, tb,
            steps_per_orbit=steps_per_orbit,
        )
        z_T = np.asarray(run.orbit_state_km_kmps[-1], dtype=float)
        g = np.empty(6)
        g[:3] = (z_T[:3] - z_tgt[:3]) / r_scale_km
        g[3:] = (z_T[3:6] - z_tgt[3:6]) / v_scale_kmps
        if not np.all(np.isfinite(g)):
            return np.full(6, 1.0e3)
        return g

    return defect
