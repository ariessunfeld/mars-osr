"""Mars orbital solar-reflector simulation and trajectory-design tools.

Sub-modules:
    kernels    -- NAIF SPICE kernel acquisition and load/unload helpers.
    ephemeris  -- Body state vectors from SPICE at arbitrary UTC epochs.
    surface    -- Mars surface-point tracking in the IAU_MARS body-fixed frame.
    dynamics   -- Point-mass Mars-centred translational dynamics (Cartesian,
                  J2000 axes) with a configurable propagator.
    elements   -- Classical orbital-element diagnostics computed from
                  Cartesian states; MME2000 reporting frame.
    gravity    -- Mars gravity field (MRO120F) loader and zonal acceleration.
    third_body -- Third-body gravitational perturbations (Sun, Phobos,
                  Deimos, Jupiter) on a Mars-centered sail.
    parallel   -- Multiprocessing configuration helpers for SPICE-using
                  parallel code (fork start-method on macOS).
"""

__version__ = "0.1.0"
