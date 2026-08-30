"""NAIF SPICE kernel acquisition and load/unload helpers.

Kernels live under ``<repo>/data/spice/`` (gitignored) and are loaded as a set
via a generated meta-kernel (``meta.tm``). The module is intentionally thin:
download, write a meta-kernel, furnsh, unload. Everything is logged.

Kernel catalog (Sun + Mars position + Mars-centred dynamics):
    naif0012.tls  -- leapseconds kernel (LSK), required for UTC <-> ET.
    de440.bsp    -- JPL DE440 planetary ephemerides SPK (1550-2650),
                    covers barycenters including Mars barycenter (NAIF 4).
    mar099.bsp   -- NAIF Mars satellite ephemeris, provides Mars planet
                    centre (NAIF 499) relative to Mars barycenter, plus
                    Phobos (401) and Deimos (402).
    pck00011.tpc -- planetary constants (radii, pole orientation).
    gm_de440.tpc -- gravitational parameters (GM) for the Sun, the planets,
                    and the Moon, consistent with DE440. Provides BODY4_GM,
                    BODY499_GM, BODY10_GM, etc -- required for any dynamics
                    that involves Mars or Sun point-mass gravity.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
import spiceypy as spice


logger = logging.getLogger(__name__)

DATA_DIR_ENV = "MARS_OSR_DATA_DIR"


def _resolve_data_dir(
    package_file: str | Path = __file__,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the common data root for checkouts and installed packages.

    ``MARS_OSR_DATA_DIR`` is authoritative when set. An editable source checkout
    defaults to ``<repo>/data``. A wheel installation falls back to the local
    cache so downloads never land inside ``site-packages``.
    """
    env = os.environ if environment is None else environment
    override = env.get(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()

    checkout_root = Path(package_file).resolve().parents[2]
    if (checkout_root / "pyproject.toml").is_file():
        return checkout_root / "data"

    cache_root = env.get("XDG_CACHE_HOME")
    if cache_root:
        return Path(cache_root).expanduser().resolve() / "mars-osr"
    return Path.home() / ".cache" / "mars-osr"


DATA_DIR = _resolve_data_dir()
KERNEL_DIR = DATA_DIR / "spice"
META_KERNEL_PATH = KERNEL_DIR / "meta.tm"
SHA256_LOG_PATH = KERNEL_DIR / "SHA256SUMS.txt"

NAIF_BASE = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"


@dataclass(frozen=True)
class KernelSpec:
    filename: str
    url: str
    description: str


REQUIRED_KERNELS: tuple[KernelSpec, ...] = (
    KernelSpec(
        filename="naif0012.tls",
        url=f"{NAIF_BASE}/lsk/naif0012.tls",
        description="Leapseconds kernel (LSK) -- UTC/ET conversions.",
    ),
    KernelSpec(
        filename="de440.bsp",
        url=f"{NAIF_BASE}/spk/planets/de440.bsp",
        description="JPL DE440 planetary SPK, 1550-2650 (barycenters).",
    ),
    KernelSpec(
        filename="mar099.bsp",
        url=f"{NAIF_BASE}/spk/satellites/mar099.bsp",
        description="NAIF Mars satellite SPK: Mars (499), Phobos (401), Deimos (402).",
    ),
    KernelSpec(
        filename="pck00011.tpc",
        url=f"{NAIF_BASE}/pck/pck00011.tpc",
        description="Planetary constants (radii, pole orientation).",
    ),
    KernelSpec(
        filename="gm_de440.tpc",
        url=f"{NAIF_BASE}/pck/gm_de440.tpc",
        description="GM values consistent with DE440 (Sun, planets, Moon).",
    ),
)


def kernel_path(filename: str) -> Path:
    return KERNEL_DIR / filename


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download_one(spec: KernelSpec, force: bool) -> Path:
    target = kernel_path(spec.filename)
    if target.exists() and not force:
        logger.info("kernel already present: %s", target)
        return target

    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    logger.info("downloading %s -> %s", spec.url, target)
    with requests.get(spec.url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        written = 0
        with part.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
        if total and written != total:
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"short read for {spec.filename}: got {written} bytes, expected {total}"
            )
    shutil.move(str(part), str(target))
    logger.info("downloaded %s (%d bytes)", spec.filename, target.stat().st_size)
    return target


def _record_hashes(paths: list[Path]) -> None:
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [f"# recorded {stamp}"]
    for p in paths:
        digest = _sha256(p)
        lines.append(f"{digest}  {p.name}")
        logger.info("sha256 %s  %s", digest, p.name)
    with SHA256_LOG_PATH.open("a") as f:
        f.write("\n".join(lines) + "\n")


def download_kernels(force: bool = False) -> list[Path]:
    """Download every kernel in REQUIRED_KERNELS, record hashes, return paths."""
    paths = [_download_one(spec, force=force) for spec in REQUIRED_KERNELS]
    _record_hashes(paths)
    write_meta_kernel()
    return paths


def write_meta_kernel() -> Path:
    """(Re)generate the SPICE meta-kernel that lists every required kernel.

    SPICE meta-kernels use a small DSL: PATH_VALUES points at
    KERNEL_DIR, KERNELS_TO_LOAD names every file.
    """
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    kernel_lines = ",\n        ".join(
        f"'$KERNELS/{spec.filename}'" for spec in REQUIRED_KERNELS
    )
    body = (
        "KPL/MK\n\n"
        "Auto-generated meta-kernel for the reflectors package.\n"
        f"Generated at {datetime.now(timezone.utc).isoformat(timespec='seconds')}.\n\n"
        "\\begindata\n\n"
        f"    PATH_VALUES     = ( '{KERNEL_DIR}' )\n"
        "    PATH_SYMBOLS    = ( 'KERNELS' )\n"
        "    KERNELS_TO_LOAD = (\n"
        f"        {kernel_lines}\n"
        "    )\n\n"
        "\\begintext\n"
    )
    META_KERNEL_PATH.write_text(body, encoding="utf-8")
    logger.info("wrote meta-kernel %s", META_KERNEL_PATH)
    return META_KERNEL_PATH


def kernels_available() -> bool:
    """Every required kernel exists on disk."""
    return all(kernel_path(spec.filename).exists() for spec in REQUIRED_KERNELS)


def load_kernels() -> Path:
    """Furnsh the meta-kernel. Idempotent: a prior kclear is issued first.

    Returns the path to the meta-kernel that was loaded.
    """
    if not kernels_available():
        missing = [
            spec.filename
            for spec in REQUIRED_KERNELS
            if not kernel_path(spec.filename).exists()
        ]
        raise FileNotFoundError(
            "missing SPICE kernels: "
            + ", ".join(missing)
            + ". Run scripts/fetch_kernels.py to download them."
        )
    if not META_KERNEL_PATH.exists():
        write_meta_kernel()
    spice.kclear()
    spice.furnsh(str(META_KERNEL_PATH))
    logger.info("loaded %s (%d kernels total)", META_KERNEL_PATH, spice.ktotal("ALL"))
    return META_KERNEL_PATH


def unload_kernels() -> None:
    """Clear every kernel from the SPICE kernel pool."""
    spice.kclear()
    logger.info("kclear: %d kernels remain", spice.ktotal("ALL"))
