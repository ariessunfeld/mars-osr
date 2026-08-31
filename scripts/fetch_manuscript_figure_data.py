#!/usr/bin/env python3
"""Install and verify the curated manuscript-figure input archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import requests


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIRECTORY = REPOSITORY_ROOT / "figures" / "manuscript"
ARCHIVE_METADATA = FIGURE_DIRECTORY / "data_archive.json"
INPUT_MANIFEST = FIGURE_DIRECTORY / "input_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_archive(path: Path, metadata: dict[str, Any]) -> None:
    expected_size = int(metadata["size_bytes"])
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"archive size mismatch: {path.stat().st_size} bytes; expected {expected_size}"
        )
    actual_hash = sha256(path)
    if actual_hash != metadata["sha256"]:
        raise ValueError(f"archive SHA-256 mismatch: {actual_hash}")


def validate_members(members: list[tarfile.TarInfo]) -> None:
    for member in members:
        member_path = PurePosixPath(member.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"unsafe archive path: {member.name}")
        if not (member.isdir() or member.isfile()):
            raise ValueError(f"unsupported archive member type: {member.name}")
        if not member_path.parts or member_path.parts[0] not in {"data", "simulation_outputs"}:
            raise ValueError(f"unexpected top-level archive path: {member.name}")


def extract_archive(path: Path, destination: Path) -> None:
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        validate_members(members)
        archive.extractall(destination, members=members, filter="data")


def verify_inputs(destination: Path, manifest: dict[str, Any]) -> None:
    files = manifest["files"]
    if len(files) != manifest["file_count"]:
        raise ValueError("input manifest file count is inconsistent")
    verified_bytes = 0
    for entry in files:
        path = destination / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"input size mismatch: {path}")
        actual_hash = sha256(path)
        if actual_hash != entry["sha256"]:
            raise ValueError(f"input SHA-256 mismatch: {path}")
        verified_bytes += entry["size_bytes"]
    if verified_bytes != manifest["total_size_bytes"]:
        raise ValueError("input manifest byte count is inconsistent")
    print(f"verified {len(files)} input files ({verified_bytes} bytes)")


def download(url: str, output: Path) -> None:
    print(f"downloading {url}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with output.open("wb") as stream:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    stream.write(block)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path, help="path to a downloaded archive")
    source.add_argument("--url", help="download URL overriding data_archive.json")
    parser.add_argument(
        "--destination",
        type=Path,
        default=REPOSITORY_ROOT,
        help="repository root into which data/ and simulation_outputs/ are installed",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify already installed inputs without downloading or extracting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_json(ARCHIVE_METADATA)
    manifest = load_json(INPUT_MANIFEST)
    destination = args.destination.resolve()

    if args.verify_only:
        if args.archive or args.url:
            raise SystemExit("--verify-only cannot be combined with --archive or --url")
        verify_inputs(destination, manifest)
        return

    archive_path = args.archive.resolve() if args.archive else None
    url = args.url or metadata.get("download_url")
    if archive_path is None and not url:
        raise SystemExit(
            "No public archive URL is registered yet. Pass a local copy with "
            f"--archive {metadata['filename']}."
        )

    destination.mkdir(parents=True, exist_ok=True)
    if archive_path is not None:
        verify_archive(archive_path, metadata)
        extract_archive(archive_path, destination)
    else:
        with tempfile.TemporaryDirectory(prefix="mars-osr-figure-data-") as temporary:
            downloaded = Path(temporary) / metadata["filename"]
            download(url, downloaded)
            verify_archive(downloaded, metadata)
            extract_archive(downloaded, destination)
    verify_inputs(destination, manifest)


if __name__ == "__main__":
    main()
