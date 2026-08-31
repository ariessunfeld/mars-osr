#!/usr/bin/env python3
"""List, reproduce, and verify the figures used by the manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "figures" / "manuscript" / "manifest.json"
Image.MAX_IMAGE_PIXELS = None  # trusted, checksum-pinned 18,000 x 7,000 reference image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(f"unsupported manifest schema in {path}")
    figures = document.get("figures")
    if not isinstance(figures, list) or not figures:
        raise ValueError(f"manifest contains no figures: {path}")
    return document


def normalize_figure_id(value: str) -> str:
    normalized = value.strip().upper()
    if normalized.startswith("S"):
        suffix = normalized[1:]
        if suffix.isdigit():
            return f"S{int(suffix)}"
        return normalized
    if normalized.isdigit():
        return f"{int(normalized):02d}"
    return normalized


def figure_index(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {figure["id"].upper(): figure for figure in document["figures"]}


def list_figures(document: dict[str, Any]) -> None:
    print(f"{'ID':<4} {'status':<26} description")
    for figure in document["figures"]:
        print(f"{figure['id']:<4} {figure['status']:<26} {figure['description']}")


def _resolve_input_matches(pattern: str) -> list[Path]:
    if any(character in pattern for character in "*?["):
        return sorted(REPOSITORY_ROOT.glob(pattern))
    path = REPOSITORY_ROOT / pattern
    return [path] if path.exists() else []


def missing_inputs(figure: dict[str, Any]) -> list[str]:
    return [pattern for pattern in figure["inputs"] if not _resolve_input_matches(pattern)]


def run_figures(figures: list[dict[str, Any]]) -> None:
    commands_seen: set[tuple[str, ...]] = set()
    for figure in figures:
        missing = missing_inputs(figure)
        if missing:
            formatted = "\n  ".join(missing)
            raise FileNotFoundError(
                f"{figure['label']} is missing required inputs:\n  {formatted}\n"
                "Run scripts/fetch_manuscript_figure_data.py first."
            )
        for command in figure["commands"]:
            command_tuple = tuple(command)
            if command_tuple in commands_seen:
                continue
            commands_seen.add(command_tuple)
            argv = list(command)
            if argv and argv[0] == "python":
                argv[0] = sys.executable
            print(f"\n$ {shlex.join(argv)}", flush=True)
            subprocess.run(argv, cwd=REPOSITORY_ROOT, check=True)

    missing_outputs = [
        figure["output"]
        for figure in figures
        if figure["output"] and not (REPOSITORY_ROOT / figure["output"]).is_file()
    ]
    if missing_outputs:
        raise RuntimeError(f"figure commands completed without outputs: {missing_outputs}")


def _verify_png(path: Path, expected_width: int | None = None, expected_height: int | None = None) -> None:
    with Image.open(path) as image:
        image.verify()
    if expected_width is not None or expected_height is not None:
        with Image.open(path) as image:
            if image.size != (expected_width, expected_height):
                raise ValueError(
                    f"{path} is {image.width}x{image.height}, expected "
                    f"{expected_width}x{expected_height}"
                )


def verify(document: dict[str, Any], require_generated: bool = False) -> None:
    verified_references = 0
    verified_generated = 0
    missing_generated: list[str] = []
    missing_required: list[str] = []
    for figure in document["figures"]:
        reference = figure["reference"]
        reference_path = REPOSITORY_ROOT / reference["path"]
        if sha256(reference_path) != reference["sha256"]:
            raise ValueError(f"reference hash mismatch: {reference_path}")
        _verify_png(reference_path, reference["width_px"], reference["height_px"])
        verified_references += 1

        if not figure["output"]:
            continue
        output_path = REPOSITORY_ROOT / figure["output"]
        if not output_path.is_file():
            missing_generated.append(figure["id"])
            if figure["status"] == "reproducible":
                missing_required.append(figure["id"])
            continue
        _verify_png(output_path)
        verified_generated += 1

    if require_generated and missing_required:
        raise FileNotFoundError(
            "generated outputs are missing for reproducible figures "
            + ", ".join(missing_required)
        )
    print(
        f"verified {verified_references} reference images and "
        f"{verified_generated} generated images"
    )
    if missing_generated:
        print("generated outputs not present for: " + ", ".join(missing_generated))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list figures and their reproduction status")
    parser.add_argument(
        "--figure",
        action="append",
        default=[],
        metavar="ID",
        help="reproduce one figure (repeatable; accepts 2, 02, or S1)",
    )
    parser.add_argument(
        "--all-available",
        action="store_true",
        help="reproduce every figure marked reproducible",
    )
    parser.add_argument("--verify", action="store_true", help="verify reference hashes and readable outputs")
    parser.add_argument(
        "--require-generated",
        action="store_true",
        help="with --verify, require every non-manual generated output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.require_generated and not args.verify:
        raise SystemExit("--require-generated requires --verify")
    if not (args.list or args.figure or args.all_available or args.verify):
        raise SystemExit("choose --list, --figure, --all-available, or --verify")

    document = load_manifest()
    if args.list:
        list_figures(document)

    selected: list[dict[str, Any]] = []
    if args.all_available:
        selected.extend(
            figure for figure in document["figures"] if figure["status"] == "reproducible"
        )
    if args.figure:
        by_id = figure_index(document)
        for raw_id in args.figure:
            figure_id = normalize_figure_id(raw_id)
            if figure_id not in by_id:
                raise SystemExit(f"unknown figure ID: {raw_id}")
            figure = by_id[figure_id]
            if not figure["commands"]:
                raise SystemExit(f"{figure['label']} has no automated producer")
            selected.append(figure)

    deduplicated = {figure["id"]: figure for figure in selected}
    if deduplicated:
        run_figures(list(deduplicated.values()))
    if args.verify:
        verify(document, require_generated=args.require_generated)


if __name__ == "__main__":
    main()
