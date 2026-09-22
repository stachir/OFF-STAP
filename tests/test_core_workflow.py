"""Public core execution contract, using synthetic inputs only."""

import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import zipfile

import pytest


def _config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "offstap" / "config" / "default.yaml"


def _options(tmp_path: Path):
    from offstap.paths import RunOptions

    ev3 = tmp_path / "raw EV3"
    vive = tmp_path / "raw Vive"
    ev3.mkdir()
    vive.mkdir()
    return RunOptions(ev3, vive, _config_path(), tmp_path / "run", frozenset({"core"}))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_core_stage_labels_cover_processing_contract():
    # Break caught: a missing/reordered stage silently skips a processing step.
    from offstap.workflow import STEP_LABELS, CORE_STAGES

    assert STEP_LABELS == {
        1: "Ingest, identify, pair, and tier",
        2: "Validate, normalize, and gate",
        3: "Partition calibration/evaluation",
        4: "Estimate temporal offset",
        5: "Select clock mapping",
        6: "Fit spatial transform and build reference",
        7: "Query held-out support and verify",
        8: "Compare, infer, and package",
    }
    assert [stage.module for stage in CORE_STAGES] == [
        "offstap.stages.s01_manifest",
        "offstap.stages.s02_validate",
        "offstap.stages.s03_quality",
        "offstap.stages.s04_partition",
        "offstap.stages.s05_parameters",
        "offstap.stages.s06_reference",
        "offstap.stages.s07_align",
        "offstap.stages.s08_metrics",
    ]
    assert [stage.step for stage in CORE_STAGES] == [1, 2, 2, 3, 4, 6, 7, 7]
    assert [stage.label for stage in CORE_STAGES] == [
        STEP_LABELS[step] for step in (1, 2, 2, 3, 4, 6, 7, 7)
    ]
    assert CORE_STAGES[4].substeps == (5, 6)


def test_combined_stage_log_records_exact_substep_labels(tmp_path, monkeypatch):
    # Break caught: one parameter process hides clock/spatial substeps or renames labels.
    from offstap.runner import run_stage
    from offstap.workflow import CORE_STAGES

    class Success:
        returncode = 0
        stderr = ""
        stdout = ""

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Success())
    run_stage(CORE_STAGES[4], tmp_path, {})

    row = _rows(tmp_path / "pipeline_stage_log.csv")[0]
    assert row["step"] == "4"
    assert row["label"] == "Estimate temporal offset"
    assert row["substeps"] == (
        "5: Select clock mapping; 6: Fit spatial transform and build reference"
    )


def test_stage_failure_stops_and_records_exit(tmp_path, monkeypatch):
    # Break caught: subprocess failure gets swallowed or lacks a durable status.
    from offstap.runner import run_stage
    from offstap.workflow import Stage

    class Failure:
        returncode = 7
        stderr = "bad schema"
        stdout = ""

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Failure())
    with pytest.raises(RuntimeError, match="bad schema"):
        run_stage(Stage(2, "Validate, normalize, and gate", "offstap.stages.s02_validate"), tmp_path, {})

    rows = _rows(tmp_path / "pipeline_stage_log.csv")
    assert len(rows) == 1
    assert rows[0]["status"] == "FAILED"
    assert rows[0]["error"] == "bad schema"
    assert rows[0]["module"] == "offstap.stages.s02_validate"


def test_valid_synthetic_pair_is_manifested_as_paired(tmp_path):
    # Break caught: module launch depends on source checkout CWD or loses a valid pair.
    from offstap.paths import prepare_run
    from offstap.runner import run_stage
    from offstap.workflow import CORE_STAGES

    options = _options(tmp_path)
    stem = "D1_T1_2026-01-01_12-00-00"
    (options.ev3_dir / f"{stem}_ev3.csv").write_text("time,motor_speed,gyro_heading\n0,0,0\n", encoding="utf-8")
    (options.vive_dir / f"{stem}_vrmt.csv").write_text("timestamp,x,y,z\n0,0,0,0\n", encoding="utf-8")
    run_dir = prepare_run(options)
    env = {
        "OFFSTAP_CONFIG": str(options.config.resolve()),
        "OFFSTAP_EV3_DIR": str(options.ev3_dir.resolve()),
        "OFFSTAP_VIVE_DIR": str(options.vive_dir.resolve()),
        "PIPELINE_RUN_DIR": str(run_dir),
    }
    run_stage(CORE_STAGES[0], run_dir, env)

    pairs = _rows(run_dir / "01_manifest" / "pair_manifest.csv")
    assert len(pairs) == 1
    assert pairs[0]["pair_status"] == "paired"
    assert len(_rows(run_dir / "pipeline_stage_log.csv")) == 1


def test_valid_synthetic_pair_reaches_quality_gate(tmp_path):
    # Break caught: valid paired source files never reach the accepted population.
    from offstap.paths import prepare_run
    from offstap.runner import run_stage
    from offstap.workflow import CORE_STAGES

    options = _options(tmp_path)
    stem = "D1_T1_2026-01-01_12-00-00"
    ev3_lines = ["time,motor_speed,gyro_heading,left_speed,right_speed"]
    vive_lines = ["timestamp,x,y,z,qx,qy,qz,qw"]
    for i in range(120):
        ev3_lines.append(f"{i * 100},180,0,180,180")
        vive_lines.append(f"{i / 10:.1f},{i / 100:.3f},0,0,0,0,0,1")
    (options.ev3_dir / f"{stem}_ev3.csv").write_text("\n".join(ev3_lines) + "\n", encoding="utf-8")
    (options.vive_dir / f"{stem}_vrmt.csv").write_text("\n".join(vive_lines) + "\n", encoding="utf-8")
    run_dir = prepare_run(options)
    env = {
        "OFFSTAP_CONFIG": str(options.config.resolve()),
        "OFFSTAP_EV3_DIR": str(options.ev3_dir.resolve()),
        "OFFSTAP_VIVE_DIR": str(options.vive_dir.resolve()),
        "PIPELINE_RUN_DIR": str(run_dir),
    }
    for stage in CORE_STAGES[:3]:
        run_stage(stage, run_dir, env)

    accepted = json.loads((run_dir / "04_quality" / "accepted_trials.json").read_text(encoding="utf-8"))
    assert accepted == [stem]
    assert len(_rows(run_dir / "pipeline_stage_log.csv")) == 3


def test_valid_synthetic_pair_reaches_actual_heldout_metrics_and_publication(tmp_path):
    """One deterministic pair traverses the real core stages without producer doubles."""
    # Break caught: individually importable stages still cannot complete one valid run.
    from offstap.runner import run_core

    options = _options(tmp_path)
    stem = "D1_T1_2026-01-01_12-00-00"
    rng = random.Random(2026)
    ev3_lines = ["time,motor_speed,gyro_heading,left_speed,right_speed"]
    vive_lines = ["timestamp,x,y,z,qx,qy,qz,qw"]
    x = 0.0
    speed = 0.0
    for index in range(401):
        time_s = index / 10.0
        if index < 20:
            speed = 0.0
        elif index % 2 == 0:
            speed = float(rng.choice((0, 900, 1300, 1800, 2400, 3000)))
        if index:
            x += float(speed) * 3.141592653589793 / 180.0 * 0.028 * 0.1
        ev3_lines.append(f"{index * 100},{speed:.1f},0,{speed:.1f},{speed:.1f}")
        vive_lines.append(f"{time_s:.1f},{x + 0.25:.9f},0,{-0.15:.9f},0,0,0,1")

    (options.ev3_dir / f"{stem}_ev3.csv").write_text("\n".join(ev3_lines) + "\n", encoding="utf-8")
    (options.vive_dir / f"{stem}_vrmt.csv").write_text("\n".join(vive_lines) + "\n", encoding="utf-8")

    run_dir = run_core(options)

    metrics = _rows(run_dir / "14_metrics" / "per_trial_metrics.csv")
    assert len(metrics) == 1
    assert metrics[0]["trial_id"] == stem
    assert metrics[0]["accepted_for_hypothesis_tests"] == "1"
    assert json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))["status"] == "PASS"
    index = json.loads((run_dir / "results" / "index.json").read_text(encoding="utf-8"))
    assert index["requested_results"] == ["core"]
    assert index["effective_results"] == ["core"]


def test_duplicate_and_malformed_sources_remain_visible(tmp_path):
    # Break caught: a duplicate is collapsed or a malformed source disappears.
    from offstap.core.config import Config
    from offstap.core.file_index import build_manifest, build_pair_manifest

    ev3 = tmp_path / "ev3"
    vive = tmp_path / "vive"
    ev3.mkdir()
    vive.mkdir()
    stem = "D1_T1_2026-01-01_12-00-00"
    for path in (
        ev3 / f"{stem}_ev3.csv",
        ev3 / f"{stem}_ev3..csv",
        vive / f"{stem}_vrmt.csv",
        vive / "malformed.csv",
    ):
        path.write_text("time,x\n0,0\n", encoding="utf-8")
    cfg = Config({
        "paths": {"input_ev3": str(ev3), "input_vrmt": str(vive)},
        "patterns": {"filename_regex": r"^(?P<drive_id>D\d+)_(?P<route_id>T\d+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_(?P<stream_type>ev3|vrmt)\.?\.csv$"},
    })
    manifest = build_manifest(cfg)
    pairs = build_pair_manifest(manifest)
    assert len(manifest) == 4
    assert any(row["parser_status"] == "malformed" for row in manifest)
    assert any("duplicate_ev3" in row["pair_status"] for row in pairs)
    assert not any(row["pair_status"] == "paired" for row in pairs)


def test_empty_eligible_population_fails_with_reasons_and_no_metrics(tmp_path):
    # Break caught: duplicate/malformed/unpaired files disappear or yield metrics.
    from offstap.runner import run_core

    options = _options(tmp_path)
    stem = "D1_T1_2026-01-01_12-00-00"
    for path in (
        options.ev3_dir / f"{stem}_ev3.csv",
        options.ev3_dir / f"{stem}_ev3..csv",
        options.vive_dir / f"{stem}_vrmt.csv",
        options.vive_dir / "malformed.csv",
        options.vive_dir / "D2_T1_2026-01-01_12-00-00_vrmt.csv",
    ):
        path.write_text("time,x\n0,0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="eligible|accepted"):
        run_core(options)

    assert (options.output_dir / "01_manifest" / "file_manifest.csv").exists()
    assert len(_rows(options.output_dir / "01_manifest" / "file_manifest.csv")) == 5
    rejections = _rows(options.output_dir / "04_quality" / "rejection_log.csv")
    assert any("duplicate_ev3" in row["reason"] for row in rejections)
    assert any("missing_ev3" in row["reason"] for row in rejections)
    assert any("malformed_filename" in row["reason"] for row in rejections)
    assert (options.output_dir / "04_quality" / "accepted_trials.json").read_text(encoding="utf-8").strip() == "[]"
    assert not (options.output_dir / "14_metrics" / "per_trial_metrics.csv").exists()
    rows = _rows(options.output_dir / "pipeline_stage_log.csv")
    assert [row["status"] for row in rows] == ["PASS", "PASS", "PASS"]
    assert (options.output_dir / "config_used.yaml").exists()
    assert (options.output_dir / "config_hash.txt").exists()
    assert (options.output_dir / "input_checksums.csv").exists()
    assert (options.output_dir / "software_environment.txt").exists()
    manifest = json.loads((options.output_dir / "software_manifest.json").read_text(encoding="utf-8"))
    assert manifest["python_version"]
    assert manifest["platform"]
    assert manifest["offstap_version"] == "0.1.0"


@pytest.mark.parametrize("failure_target", ["config_used.yaml", "source_ev3.csv"])
def test_preparation_read_failure_records_failed_reason(tmp_path, monkeypatch, failure_target):
    # Break caught: a config/input hash read error leaves an initialized run without status.
    from offstap.runner import run_core

    options = _options(tmp_path)
    (options.ev3_dir / "source_ev3.csv").write_text("time\n0\n", encoding="utf-8")
    original_open = Path.open

    def fail_selected_read(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if path.name == failure_target and mode == "rb":
            raise PermissionError(f"synthetic read denied: {failure_target}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_selected_read)
    with pytest.raises(PermissionError, match="synthetic read denied"):
        run_core(options)

    status = json.loads((options.output_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status == {
        "status": "FAILED",
        "reason": f"synthetic read denied: {failure_target}",
    }
    assert not (options.output_dir / "pipeline_stage_log.csv").exists()


def test_wheel_stages_import_and_manifest_runs_from_unrelated_directory(tmp_path):
    # Break caught: stage imports depend on the source checkout or invocation CWD.
    package_root = Path(__file__).resolve().parents[1]
    wheel_dir = tmp_path / "wheel"
    built = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "--wheel-dir", str(wheel_dir), str(package_root)],
        capture_output=True, text=True, check=False,
    )
    assert built.returncode == 0, built.stderr
    installed = tmp_path / "installed"
    with zipfile.ZipFile(next(wheel_dir.glob("offstap-*.whl"))) as wheel:
        wheel.extractall(installed)
    run_dir = tmp_path / "unrelated working directory"
    run_dir.mkdir()
    ev3 = tmp_path / "ev3"
    vive = tmp_path / "vive"
    ev3.mkdir()
    vive.mkdir()
    stem = "D1_T1_2026-01-01_12-00-00"
    (ev3 / f"{stem}_ev3.csv").write_text("time\n0\n", encoding="utf-8")
    (vive / f"{stem}_vrmt.csv").write_text("timestamp,x,y,z\n0,0,0,0\n", encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONPATH": str(installed),
        "OFFSTAP_CONFIG": str(_config_path()),
        "OFFSTAP_EV3_DIR": str(ev3),
        "OFFSTAP_VIVE_DIR": str(vive),
        "PIPELINE_RUN_DIR": str(run_dir),
        "MPLBACKEND": "Agg",
    }
    imported = subprocess.run(
        [sys.executable, "-c", "import importlib, pkgutil, offstap; "
         "modules=[item.name for item in pkgutil.walk_packages(offstap.__path__, offstap.__name__ + '.')]; "
         "[importlib.import_module(name) for name in modules]; "
         "print(offstap.__file__); print(len(modules))"],
        cwd=run_dir, env=env, capture_output=True, text=True, check=False,
    )
    assert imported.returncode == 0, imported.stderr
    assert str(installed).lower() in imported.stdout.lower()
    executed = subprocess.run(
        [sys.executable, "-m", "offstap.stages.s01_manifest"],
        cwd=run_dir, env=env, capture_output=True, text=True, check=False,
    )
    assert executed.returncode == 0, executed.stderr
    assert _rows(run_dir / "01_manifest" / "pair_manifest.csv")[0]["pair_status"] == "paired"
    for module in (
        "s02_validate", "s03_quality", "s04_partition", "s05_parameters",
        "s06_reference", "s07_align", "s08_metrics",
    ):
        probe = tmp_path / module
        probe.mkdir()
        probed = subprocess.run(
            [sys.executable, "-m", f"offstap.stages.{module}"],
            cwd=probe, env={**env, "PIPELINE_RUN_DIR": str(probe)},
            capture_output=True, text=True, check=False,
        )
        assert probed.returncode != 0, module
        assert "ModuleNotFoundError" not in probed.stderr, probed.stderr
        assert "FileNotFoundError" in probed.stderr, probed.stderr
