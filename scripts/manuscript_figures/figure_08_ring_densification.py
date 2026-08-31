"""Render K12 irradiance profiles for rings of 1, 12, and 120 reflectors.

The calculation uses the closure-refined K12 orbit at 18 h LTAN and Mars
perihelion with 6x6 gravity, Sun/Phobos/Deimos third bodies, SRP, a 300 km
altitude floor, and the production visibility and slew gates. Delivered
irradiance follows Canady and Allen (1982), Eq. 9.

Reflector area is 10,000 m^2 at fixed areal density. Irradiance is linear in
area when finite-mirror correction is disabled, but the 1.0 J/m^2 useful-window
gate is applied after area scaling. The accepted window set can therefore
differ from the 1,000 m^2 case, so this case is evaluated directly.

Reflector j in a ring of N starts at M0_j = 360 deg j/N. The N=1 and N=12 phase
sets are subsets of the N=120 grid, allowing all panels to be assembled from
one cached set of 120 single-reflector propagations. Empty local-time spans are
compressed with axis breaks, and the illuminated fraction is the fraction of
the displayed sol for which combined irradiance is positive.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np
import spiceypy as spice

from reflectors.attitude import sun_pointing
from reflectors.dynamics import PropagationOptions, propagate
from reflectors.elements import state_from_classical_mme2000
from reflectors.ephemeris import utc_to_et
from reflectors.gravity import mars_gravity_model
from reflectors.kernels import load_kernels
from reflectors.mars_constants import SECONDS_PER_SOLAR_SOL_S
from reflectors.sail_designs import make_canonical_sail
from reflectors.sun_sync import raan_mme2000_from_ltan
from reflectors.termination import AltitudeFloor
from reflectors.third_body import (
    deimos_third_body,
    phobos_third_body,
    sun_third_body,
)
from reflectors.visibility import find_delivery_windows

logger = logging.getLogger(__name__)

# --- Base / collector / sail ---------------------------------------------------
TARGET_LAT_DEG = 40.0
TARGET_LON_DEG = 200.0
A_COLLECTOR_M2 = 1.0e6                 # 1 km^2 base solar array
SIGMA_KG_PER_M2 = 0.018000
A_SAIL_M2 = 10000.0

# --- Orbit / epoch / gates (production; identical to figH3 reference) ----------
LTAN_H = 18.0
PERIHELION_UTC = "2028-02-11T12:42:00"
N_SOLS = 2                             # propagate 2 sols (sol-1 representative); DISPLAY 1 sol
CADENCE_S = 30.0
ELEV_MIN_DEG = 10.0
BISECTOR_COS_ALPHA_MIN = 0.1
ALPHA_MAX_RAD_S2 = math.radians(0.003)
MIN_WINDOW_FLUENCE_J_PER_M2 = 1.0     # fixed useful-delivery threshold; not area-scaled
ATM_TRANSMISSION = 1.0
ALT_FLOOR_KM = 300.0

K = 12
DESIGN_POINT = (3903.924477, 93.420985)
R_MARS_KM = 3396.0

N_PANELS = [1, 12, 120]               # N=4 (the former second panel) omitted
N_PROP = max(N_PANELS)                # 120: its phase set (3 deg) is a superset of {1,12}

# Preserve every local-time sample, but compress the three inactive spans.  The
# resulting piecewise-linear coordinate is labelled in physical local-solar hours;
# slash marks make its nonuniform scale explicit.
LST_SEGMENTS = (
    (0.0, 4.0, 0.45),
    (4.0, 9.0, 5.0),
    (9.0, 15.0, 0.60),
    (15.0, 20.0, 5.0),
    (20.0, 24.0, 0.45),
)
LST_TICKS_H = (0.0, 4.0, 9.0, 15.0, 20.0, 24.0)
LST_MINOR_TICKS_H = (5.0, 6.0, 7.0, 8.0, 16.0, 17.0, 18.0, 19.0)
LST_BREAK_CENTRES_H = (2.0, 12.0, 22.0)
Y_LINTHRESH_W_PER_M2 = 0.1
Y_TICKS_W_PER_M2 = (0.0, 0.1, 1.0)

# Same mu the refined a* was tuned with + production producer uses.
_MU_KM3_S2 = float(mars_gravity_model(max_degree=2).mu_km3_s2)

OUT_DIR = Path("simulation_outputs")
FIG_DIR = Path("simulation_outputs/manuscript_figures")
OUT_PREFIX = "20260627_figH3_K12_10000m2"
FIG_OUT = FIG_DIR / "figure_08a_ring_densification.png"

# --- styling -------------------------------------------------------------------
AXIS_FS = 22       # axis titles
TICK_FS = 18       # tick labels
LABEL_FS = 20      # the frameless per-panel N / duty-cycle label
TITLE_FS = 20      # suptitle


# --- Local Apparent Solar Time at the base ------------------------------------
# This convention keeps local time consistent across the manuscript figures.
def subsolar_lon_deg(et: float) -> float:
    """Planetographic longitude (deg) of the sub-solar point = direction to the
    Sun in the IAU_MARS body frame."""
    pos, _ = spice.spkpos("SUN", et, "IAU_MARS", "NONE", "MARS")
    return math.degrees(math.atan2(float(pos[1]), float(pos[0])))


def hour_angle_signed_deg(et: float) -> float:
    """Base longitude minus sub-solar longitude, wrapped to [-180, 180].
    0 at local solar noon, +-180 at local solar midnight."""
    return (TARGET_LON_DEG - subsolar_lon_deg(et) + 180.0) % 360.0 - 180.0


def local_solar_time_h(et: float) -> float:
    """Local apparent solar time at the base, hours in [0, 24) (0 = midnight)."""
    return (hour_angle_signed_deg(et) / 15.0 + 12.0) % 24.0


def phase_set_deg(n: int) -> list[float]:
    return [round(j * 360.0 / n, 6) for j in range(n)]


def compressed_lstime_coordinate(lst_h):
    """Map local-solar hours to a monotonic broken-axis coordinate.

    per hour.  The three long, nearly empty intervals are compressed, not deleted,
    so no simulation samples are silently discarded.
    """
    values = np.asarray(lst_h, dtype=float)
    if np.any((values < 0.0) | (values > 24.0)):
        raise ValueError("local-solar hours must lie in [0, 24]")

    mapped = np.zeros_like(values)
    for start_h, stop_h, display_width in LST_SEGMENTS:
        fraction_h = np.clip(values - start_h, 0.0, stop_h - start_h)
        mapped += display_width * fraction_h / (stop_h - start_h)
    return mapped


def add_x_break_marks(ax) -> None:
    """Draw three ``//`` marks on the lower spine at compressed time spans."""
    break_x = compressed_lstime_coordinate(LST_BREAK_CENTRES_H)
    transform = ax.get_xaxis_transform()
    for centre in break_x:
        for offset in (-0.045, 0.045):
            ax.plot(
                [centre + offset - 0.035, centre + offset + 0.035],
                [-0.018, 0.018],
                color="black",
                lw=1.0,
                clip_on=False,
                transform=transform,
            )


def build_state(a_km, i_deg, m0_deg, epoch_et):
    raan_rad = raan_mme2000_from_ltan(LTAN_H, epoch_et)
    return state_from_classical_mme2000(
        a_km=a_km, e=0.0,
        inclination_rad=math.radians(i_deg),
        raan_rad=raan_rad, argp_rad=0.0,
        nu_rad=math.radians(m0_deg),
        mu_km3_s2=_MU_KM3_S2, epoch_et=epoch_et,
    )


def reflector_series(a_km, i_deg, m0_deg, epoch_et, duration_s, t_eval, sail):
    """One phased reflector: kept-irradiance series (W/m^2) over the run."""
    state0 = build_state(a_km, i_deg, m0_deg, epoch_et)
    result = propagate(
        state0_km_kmps=state0,
        t_span_s=(0.0, duration_s),
        epoch_et=epoch_et,
        gravity_degree=6, gravity_order=6,
        third_bodies=[sun_third_body(), phobos_third_body(), deimos_third_body()],
        solar_sail=sail,
        sail_normal=sun_pointing(),
        altitude_floor=AltitudeFloor.at_km(ALT_FLOOR_KM, label="altitude_floor"),
        options=PropagationOptions.fast(),
        t_eval_s=t_eval,
    )
    windows, samples = find_delivery_windows(
        result, TARGET_LAT_DEG, TARGET_LON_DEG,
        target_elevation_min_deg=ELEV_MIN_DEG,
        bisector_cos_alpha_min=BISECTOR_COS_ALPHA_MIN,
        require_sail_sunlit=True, require_sail_above_horizon=True,
        require_bisector_feasible=True,
        sail=sail, atmospheric_transmission=ATM_TRANSMISSION,
        alpha_max_rad_s2=ALPHA_MAX_RAD_S2,
        min_window_fluence_J_per_m2=MIN_WINDOW_FLUENCE_J_PER_M2,
        return_samples=True,
    )
    t_s = samples.t_s
    kept = np.zeros_like(t_s, dtype=bool)
    for w in windows:
        kept |= (t_s >= w.t_start_s) & (t_s <= w.t_end_s)
    return np.where(kept, samples.irradiance_W_per_m2, 0.0)


def propagate_phases(epoch_et, duration_s, t_eval):
    """Per-phase kept-irradiance series for the 120 evenly-phased reflectors at K=12,
    10,000 m^2. npz-cached so figure tweaks re-plot without re-propagating."""
    a_km, i_deg = DESIGN_POINT
    sail = make_canonical_sail(SIGMA_KG_PER_M2, area_m2=A_SAIL_M2)
    phases = phase_set_deg(N_PROP)
    nph, n_samp = len(phases), t_eval.shape[0]

    cache_npz = OUT_DIR / f"{OUT_PREFIX}_irr_cache.npz"
    if cache_npz.exists():
        z = np.load(cache_npz)
        if (z["irr"].shape == (nph, n_samp)
                and np.allclose(z["phases"], np.array(phases))
                and float(z["area_m2"]) == A_SAIL_M2):
            print(f"  loaded cache {cache_npz} (skip re-propagation)")
            return {round(phases[j], 6): z["irr"][j] for j in range(nph)}, t_eval

    irr_arr = np.empty((nph, n_samp), dtype=float)
    t0 = time.perf_counter()
    for idx, m0 in enumerate(phases):
        irr_arr[idx] = reflector_series(a_km, i_deg, m0, epoch_et, duration_s, t_eval, sail)
        if (idx + 1) % 10 == 0 or idx == nph - 1:
            print(f"  phase {idx + 1}/{nph} (M0={m0:.2f}) "
                  f"[{time.perf_counter() - t0:.0f}s]")
    np.savez_compressed(cache_npz, irr=irr_arr, phases=np.array(phases),
                        t_eval=t_eval, area_m2=A_SAIL_M2)
    print(f"  wrote cache {cache_npz}")
    return {round(phases[j], 6): irr_arr[j] for j in range(nph)}, t_eval


def assemble(irr_by_phase, t_eval):
    """I_N(t) for each N in N_PANELS + duty cycle over the displayed (first) sol."""
    show = t_eval <= SECONDS_PER_SOLAR_SOL_S            # one Mars sol
    n_show = int(np.count_nonzero(show))
    per_N = {}
    for N in N_PANELS:
        I_N = np.sum([irr_by_phase[round(m0, 6)] for m0 in phase_set_deg(N)], axis=0)
        duty_sol = float(np.count_nonzero(I_N[show] > 0.0)) / float(n_show)
        per_N[N] = dict(
            I_N=I_N,
            peak_I=float(I_N[show].max()),
            duty_sol=duty_sol,                          # fraction of the sol with any added irradiance
        )
    return per_N, show


def fig_densify(per_N, t_eval, show, epoch_et):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})

    a_km, i_deg = DESIGN_POINT
    alt_km = a_km - R_MARS_KM

    # x-axis = Local Apparent Solar Time at the base (0 = midnight -> 24 = midnight),
    # same convention as figO. One elapsed sol spans exactly one 0->24 LST sweep;
    # argsort by LST renders it in clean midnight->midnight order (figO does the same).
    ets = epoch_et + t_eval[show]
    lst = np.array([local_solar_time_h(float(e)) for e in ets])
    order = np.argsort(lst)
    x = lst[order]

    positive_values = np.concatenate(
        [per_N[N]["I_N"][show][per_N[N]["I_N"][show] > 0.0] for N in N_PANELS]
    )
    if positive_values.size == 0:
        raise ValueError("cannot scale the y-axis: no positive irradiance samples")
    ymax = float(positive_values.max()) * 1.08
    print(
        f"  shared symlog plot range: 0--{ymax:.3g} W/m^2 "
        f"(linear threshold {Y_LINTHRESH_W_PER_M2:g} W/m^2; "
        f"positive samples {positive_values.min():.3g}--{positive_values.max():.3g})"
    )

    x_plot = compressed_lstime_coordinate(x)
    tick_positions = compressed_lstime_coordinate(LST_TICKS_H)
    minor_tick_positions = compressed_lstime_coordinate(LST_MINOR_TICKS_H)
    y_minor_ticks = np.concatenate(
        (
            np.arange(0.2, min(1.0, ymax), 0.1),
            np.arange(2.0, ymax, 1.0),
        )
    )
    x_max = float(compressed_lstime_coordinate(24.0))

    fig, axes = plt.subplots(
        len(N_PANELS), 1, figsize=(12, 7.8), sharex=True, sharey=True
    )
    for panel_index, (ax, N) in enumerate(zip(axes, N_PANELS), start=1):
        y = per_N[N]["I_N"][show][order]
        y_curve = np.where(y > 0.0, y, np.nan)
        ax.plot(x_plot, y_curve, color="black", lw=1.1)
        ax.set_yscale(
            "symlog", linthresh=Y_LINTHRESH_W_PER_M2, linscale=1.0, base=10
        )
        ax.set_ylim(0.0, ymax)
        ax.set_yticks(Y_TICKS_W_PER_M2, ["0", "0.1", "1"])
        ax.set_yticks(y_minor_ticks, minor=True)
        ax.set_xlim(0.0, x_max)
        ax.set_xticks(tick_positions, [f"{hour:.0f}" for hour in LST_TICKS_H])
        ax.set_xticks(minor_tick_positions, minor=True)
        ax.set_ylabel(r"$\mathrm{W/m^{2}}$", fontsize=AXIS_FS)
        ax.tick_params(labelsize=TICK_FS)
        # Frameless per-panel label: identify the illuminated-time fraction.
        duty_label = rf"$T_{{\mathrm{{illum.}}}}={100 * per_N[N]['duty_sol']:.0f}\%$"
        ax.text(0.985, 0.92, f"N = {N}\n{duty_label}",
                transform=ax.transAxes, ha="right", va="top", fontsize=LABEL_FS)
        ax.text(
            0.02,
            0.92,
            f"A{panel_index}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=30,
            fontweight="bold",
        )
        ax.grid(which="major", axis="both", alpha=0.22, linewidth=0.8)
        ax.grid(which="minor", axis="x", alpha=0.18, linewidth=0.6)
        ax.grid(which="minor", axis="y", alpha=0.12, linewidth=0.5)
        add_x_break_marks(ax)
    axes[-1].set_xlabel("Local apparent solar time at base (h)", fontsize=AXIS_FS)
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {FIG_OUT}")


def main() -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_kernels()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t_wall0 = time.perf_counter()

    epoch_et = utc_to_et(PERIHELION_UTC)
    duration_s = float(N_SOLS) * SECONDS_PER_SOLAR_SOL_S
    t_eval = np.arange(0.0, duration_s, CADENCE_S)
    if t_eval[-1] < duration_s:
        t_eval = np.append(t_eval, duration_s)

    print("=" * 80)
    print("K=12, 10,000 m^2, ring sizes N in {1,12,120}")
    print("=" * 80)
    a_km, i_deg = DESIGN_POINT
    print(f"  target=({TARGET_LAT_DEG}N,{TARGET_LON_DEG}E)  collector=1 km^2  "
          f"sail sigma={SIGMA_KG_PER_M2} area={A_SAIL_M2:.0f} m^2")
    print(f"  K={K}: a*={a_km} km (alt {a_km - R_MARS_KM:.0f} km), i*={i_deg} deg")
    print(f"  LTAN={LTAN_H}h  epoch={PERIHELION_UTC}  N_SOLS={N_SOLS}  cadence={CADENCE_S}s")
    print(f"  propagating {N_PROP} phases (3 deg spacing); {{1,12}} are subsets")
    print("-" * 80)

    irr_by_phase, t_eval = propagate_phases(epoch_et, duration_s, t_eval)
    per_N, show = assemble(irr_by_phase, t_eval)

    print("\nN   peak_W/m2   duty_%   (over one displayed sol)")
    for N in N_PANELS:
        s = per_N[N]
        print(f"{N:>3d} {s['peak_I']:>10.3f} {100 * s['duty_sol']:>7.2f}")

    fig_densify(per_N, t_eval, show, epoch_et)
    print(f"\ntotal wall = {time.perf_counter() - t_wall0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
