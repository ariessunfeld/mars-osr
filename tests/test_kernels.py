"""Kernel-level checks: files present and furnsh/unload round-trip cleanly."""

from __future__ import annotations

import spiceypy as spice

from reflectors import kernels as k


def test_data_dir_environment_override(tmp_path):
    configured = tmp_path / "scientific-data"
    package_file = tmp_path / "site-packages" / "reflectors" / "kernels.py"
    actual = k._resolve_data_dir(
        package_file=package_file,
        environment={k.DATA_DIR_ENV: str(configured)},
    )
    assert actual == configured.resolve()


def test_data_dir_checkout_default(tmp_path):
    checkout = tmp_path / "checkout"
    package_file = checkout / "src" / "reflectors" / "kernels.py"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\nname='probe'\n")
    actual = k._resolve_data_dir(package_file=package_file, environment={})
    assert actual == checkout / "data"


def test_data_dir_installed_fallback_uses_xdg_cache(tmp_path):
    package_file = tmp_path / "site-packages" / "reflectors" / "kernels.py"
    cache = tmp_path / "cache"
    actual = k._resolve_data_dir(
        package_file=package_file,
        environment={"XDG_CACHE_HOME": str(cache)},
    )
    assert actual == cache.resolve() / "mars-osr"


def test_all_required_kernels_present():
    missing = [
        spec.filename for spec in k.REQUIRED_KERNELS if not k.kernel_path(spec.filename).exists()
    ]
    assert missing == [], f"missing kernels: {missing}"


def test_meta_kernel_exists():
    assert k.META_KERNEL_PATH.exists(), "meta-kernel was not generated"


def test_furnsh_roundtrip():
    # The session fixture loaded the kernels; assert the pool contains at least
    # three required files plus the meta-kernel entry itself.
    total = spice.ktotal("ALL")
    assert total >= len(k.REQUIRED_KERNELS), (
        f"expected >= {len(k.REQUIRED_KERNELS)} kernels loaded, got {total}"
    )

    k.unload_kernels()
    assert spice.ktotal("ALL") == 0

    # Restore state for remaining session tests.
    k.load_kernels()
    assert spice.ktotal("ALL") >= len(k.REQUIRED_KERNELS)
