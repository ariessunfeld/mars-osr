# Model scope

## Vacuum optical boundary

Mars OSR 0.1 treats propagation of reflected sunlight from the sail to the
Martian surface as vacuum propagation. All maintained command-line workflows
use an atmospheric transmission factor of 1.0.

Consequently, an illumination value from this release does not include:

- wavelength-dependent gaseous absorption;
- dust, ice-cloud, or aerosol extinction;
- multiple scattering or diffuse irradiance;
- atmospheric or surface thermal response;
- surface bidirectional reflectance; or
- weather or climate feedback.

The results are geometrical and radiometric vacuum estimates. They should not be
reported as atmosphere-attenuated surface irradiance or as a temperature change.
The core visibility routine retains a generic dimensionless attenuation hook for
downstream callers, but this release supplies no Mars-atmosphere model and all
included workflows keep that factor at unity.

This boundary is intentional. Atmosphere and climate coupling belong to a
separate model and are not part of this software release.

## Included physical models

- Mars point-mass, zonal, and MRO120F gravity;
- Sun, Phobos, Deimos, and selected planetary third-body perturbations;
- eclipse and penumbra geometry;
- six-parameter flat-sail solar-radiation pressure;
- finite angular-acceleration and angular-rate attitude tracking;
- orbital elements, sun-synchronous design, ground visibility, and beam geometry;
- Earth point-mass/J2 gravity, lunar/solar third bodies, and Harris-Priester
  density for drag-aware escape; and
- Sun-centred solar-sail cruise dynamics and optimization.

Formulae, constants, units, and primary bibliographic citations are documented
in the relevant modules.

## Testing boundary

The test suite exercises the shipped library, analytical limits,
cross-implementation consistency, and selected end-to-end physical regressions.
