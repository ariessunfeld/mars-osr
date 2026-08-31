"""Tests for the manuscript-figure registry and reproduction utilities."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
import warnings
from pathlib import Path

import pytest
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    path = REPOSITORY_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def registry():
    return load_script(
        "reproduce_manuscript_figures",
        "scripts/reproduce_manuscript_figures.py",
    )


@pytest.fixture(scope="module")
def manifest():
    path = REPOSITORY_ROOT / "figures" / "manuscript" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_covers_all_manuscript_figures(manifest):
    assert manifest["schema_version"] == 1
    assert [figure["id"] for figure in manifest["figures"]] == [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "S1", "S2", "S3", "S4",
    ]
    assert len({figure["reference"]["path"] for figure in manifest["figures"]}) == 14


def test_reference_images_match_manifest(manifest):
    for figure in manifest["figures"]:
        reference = figure["reference"]
        path = REPOSITORY_ROOT / reference["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                assert image.size == (reference["width_px"], reference["height_px"])


def test_generated_outputs_do_not_overwrite_references(manifest):
    for figure in manifest["figures"]:
        output = figure["output"]
        if output is None:
            continue
        assert output.startswith("figures/manuscript/generated/")
        assert output != figure["reference"]["path"]


def test_input_manifest_is_complete_and_repository_relative():
    path = REPOSITORY_ROOT / "figures" / "manuscript" / "input_manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    files = document["files"]
    paths = [entry["path"] for entry in files]
    assert document["schema_version"] == 1
    assert len(files) == document["file_count"] == 913
    assert sum(entry["size_bytes"] for entry in files) == document["total_size_bytes"]
    assert len(paths) == len(set(paths))
    assert all(item.startswith(("data/", "simulation_outputs/")) for item in paths)
    assert all(len(entry["sha256"]) == 64 for entry in files)
    assert not any("manuscript_climate" in item for item in paths)


def test_reproducible_commands_reference_existing_scripts(manifest):
    for figure in manifest["figures"]:
        if figure["status"] != "reproducible":
            continue
        assert figure["commands"]
        for command in figure["commands"]:
            assert command[0] == "python"
            assert (REPOSITORY_ROOT / command[1]).is_file()


def test_figure_7_retains_manuscript_area(manifest):
    figure = next(item for item in manifest["figures"] if item["id"] == "07")
    assert "1,000 m^2" in figure["note"]
    source = (REPOSITORY_ROOT / figure["commands"][0][1]).read_text(encoding="utf-8")
    assert "10_000" not in source
    assert "10000" not in source


def test_s4_producer_emits_only_the_manuscript_figure():
    source = (
        REPOSITORY_ROOT
        / "scripts/manuscript_figures/figure_s4_ring_fluence.py"
    ).read_text(encoding="utf-8")
    assert "figH5_multishell_ring_fluence_vs_N_LTAN" not in source


def test_figure_4_preserves_full_canvas_for_sunlight_arrows():
    source = (
        REPOSITORY_ROOT
        / "scripts/manuscript_figures/figure_04_ltan_feasibility.py"
    ).read_text(encoding="utf-8")
    assert 'bbox_inches="tight"' not in source


def test_id_normalization(registry):
    assert registry.normalize_figure_id("2") == "02"
    assert registry.normalize_figure_id("02") == "02"
    assert registry.normalize_figure_id("s1") == "S1"


def test_composer_crops_and_joins_images(tmp_path):
    composer = load_script(
        "figure_08_constellation_scaling",
        "scripts/manuscript_figures/figure_08_constellation_scaling.py",
    )
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    output = tmp_path / "combined.png"

    left_image = Image.new("RGB", (30, 20), "white")
    left_image.paste("black", (5, 5, 25, 15))
    left_image.save(left)
    right_image = Image.new("RGB", (20, 30), "white")
    right_image.paste("red", (5, 5, 15, 25))
    right_image.save(right)

    composer.compose(left, right, output, gap_px=4, padding_px=0)
    with Image.open(output) as result:
        assert result.height == 20
        assert result.width == 54


def test_archive_validation_rejects_path_traversal():
    fetcher = load_script(
        "fetch_manuscript_figure_data",
        "scripts/fetch_manuscript_figure_data.py",
    )
    member = tarfile.TarInfo("../outside.txt")
    member.size = len(b"unsafe")
    with pytest.raises(ValueError, match="unsafe archive path"):
        fetcher.validate_members([member])


def test_input_verification_accepts_matching_file(tmp_path, capsys):
    fetcher = load_script(
        "fetch_manuscript_figure_data_verify",
        "scripts/fetch_manuscript_figure_data.py",
    )
    payload = b"figure input\n"
    path = tmp_path / "simulation_outputs" / "sample.csv"
    path.parent.mkdir()
    path.write_bytes(payload)
    manifest = {
        "file_count": 1,
        "total_size_bytes": len(payload),
        "files": [
            {
                "path": "simulation_outputs/sample.csv",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    fetcher.verify_inputs(tmp_path, manifest)
    assert "verified 1 input files" in capsys.readouterr().out
