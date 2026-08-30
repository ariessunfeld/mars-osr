"""Session-wide fixtures for the reflectors test suite."""

from __future__ import annotations

import pytest

from reflectors.kernels import kernels_available, load_kernels, unload_kernels


@pytest.fixture(scope="session", autouse=True)
def furnsh_spice_kernels():
    """Furnish SPICE kernels once per session; clear on teardown.

    If the required kernels have not been downloaded yet, skip the whole
    suite with a pointer to the fetch script rather than blowing up inside
    individual tests.
    """
    if not kernels_available():
        pytest.skip(
            "SPICE kernels are not present in the configured Mars OSR data "
            "directory. Run `python scripts/fetch_kernels.py` first.",
            allow_module_level=True,
        )
    load_kernels()
    yield
    unload_kernels()
