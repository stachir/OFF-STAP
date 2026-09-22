from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import yaml


def test_results_parser_rejects_unknown_name():
    """Reject a typo instead of silently omitting a requested product."""
    # Break caught: accepting an unrecognised result name would hide a user error.
    from offstap.selection import parse_results

    with pytest.raises(ValueError, match="unknown result"):
        parse_results("core,imaginary")


def test_results_parser_accepts_known_names_and_all():
    """Expose the documented result-selection vocabulary to the CLI."""
    # Break caught: a documented selection would be refused or parsed incorrectly.
    from offstap.selection import parse_results

    assert parse_results("h1,core") == frozenset({"h1", "core"})
    assert parse_results("all") == frozenset({"all"})


def test_run_paths_allow_unicode_and_reject_nonempty_destination(tmp_path):
    """Create a new Unicode-named run safely and preserve an existing run."""
    # Break caught: path handling fails for normal non-ASCII paths or overwrites output.
    from offstap.paths import RunOptions, prepare_run

    ev3 = tmp_path / "wejście EV3"
    vive = tmp_path / "Vive logs"
    ev3.mkdir()
    vive.mkdir()
    config = _write_valid_config(tmp_path / "config.yaml")
    out = tmp_path / "wyniki OFF-STAP"
    options = RunOptions(ev3, vive, config, out, frozenset({"core"}))

    assert prepare_run(options) == out.resolve()
    (out / "keep.txt").write_text("user file", encoding="utf-8")

    with pytest.raises(FileExistsError):
        prepare_run(options)


def test_prepare_run_creates_current_stage_layout(tmp_path):
    """Give active stages their output locations without legacy baseline paths."""
    # Break caught: a missing stage directory makes a valid future stage fail at startup.
    from offstap.paths import RunOptions, prepare_run

    ev3, vive, config = _valid_inputs(tmp_path)
    output = tmp_path / "run"

    run_dir = prepare_run(RunOptions(ev3, vive, config, output, frozenset({"core"})))

    assert all(
        (run_dir / path).is_dir()
        for path in (
            "01_manifest",
            "03_cleaned/ev3",
            "03_cleaned/vrmt",
            "13_integrated_logs/ct_reference",
            "14_metrics",
            "15_hypotheses/H2/h2_plots",
            "18_plots/results",
            "19_tables",
            "provenance",
        )
    )
    assert not (run_dir / "27_pipeline_baseline_comparison").exists()


def test_missing_input_directory_is_refused(tmp_path):
    """Reject a missing raw-input path before making an output directory."""
    # Break caught: a typo in an input path creates a partial run that cannot be processed.
    from offstap.paths import RunOptions, prepare_run

    _, vive, config = _valid_inputs(tmp_path)

    with pytest.raises(FileNotFoundError, match="EV3"):
        prepare_run(
            RunOptions(tmp_path / "missing EV3", vive, config, tmp_path / "output", frozenset({"core"}))
        )


def test_output_inside_raw_input_is_refused(tmp_path):
    """Never write generated artifacts beneath a raw-input directory."""
    # Break caught: output inside raw input contaminates user-provided data.
    from offstap.paths import RunOptions, prepare_run

    ev3, vive, config = _valid_inputs(tmp_path)

    with pytest.raises(ValueError, match="input"):
        prepare_run(RunOptions(ev3, vive, config, ev3 / "results", frozenset({"core"})))


def test_raw_input_inside_output_is_refused(tmp_path):
    """Never treat a directory containing raw inputs as a run destination."""
    # Break caught: an output parent could overwrite or mix user-provided input data.
    from offstap.paths import RunOptions, prepare_run

    raw_root = tmp_path / "raw input"
    ev3 = raw_root / "EV3"
    vive = raw_root / "Vive"
    ev3.mkdir(parents=True)
    vive.mkdir()
    config = _write_valid_config(tmp_path / "config.yaml")

    with pytest.raises(ValueError, match="input"):
        prepare_run(RunOptions(ev3, vive, config, raw_root, frozenset({"core"})))


def test_malformed_yaml_is_refused_before_run_creation(tmp_path):
    """Reject invalid configuration rather than creating an unusable run."""
    # Break caught: malformed YAML is accepted and leaves a partial output tree behind.
    from offstap.paths import RunOptions, prepare_run

    ev3, vive, config = _valid_inputs(tmp_path)
    config.write_text("paths: [unterminated", encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="YAML"):
        prepare_run(RunOptions(ev3, vive, config, output, frozenset({"core"})))

    assert not output.exists()


def test_same_input_directory_is_refused(tmp_path):
    """Keep EV3 and Vive source roles distinct before processing begins."""
    # Break caught: one directory can be misclassified as both input streams.
    from offstap.paths import RunOptions, prepare_run

    ev3, _, config = _valid_inputs(tmp_path)

    with pytest.raises(ValueError, match="distinct"):
        prepare_run(RunOptions(ev3, ev3, config, tmp_path / "output", frozenset({"core"})))


def test_cli_parses_paths_and_reports_empty_population(tmp_path):
    """Run the declared CLI and fail explicitly when its inputs contain no trials."""
    # Break caught: empty inputs are reported as a successful processing run.
    from offstap.cli import main

    ev3, vive, config = _valid_inputs(tmp_path)
    output = tmp_path / "CLI output"

    with pytest.raises(RuntimeError, match="eligible|accepted|pair_manifest"):
        main(
            [
                "--ev3-dir",
                str(ev3),
                "--vive-dir",
                str(vive),
                "--config",
                str(config),
                "--output-dir",
                str(output),
                "--results",
                "core,h1",
            ]
        )
    assert (output / "01_manifest").is_dir()
    assert (output / "run_status.json").exists()


def test_config_loader_injects_only_operational_paths(tmp_path, monkeypatch):
    """Allow the launcher to provide paths without changing scientific settings."""
    # Break caught: CLI paths are ignored or an environment flag enables development mode.
    from offstap.core.config import load_config

    config = _write_valid_config(tmp_path / "config.yaml")
    monkeypatch.setenv("OFFSTAP_CONFIG", str(config))
    monkeypatch.setenv("OFFSTAP_EV3_DIR", str(tmp_path / "EV3"))
    monkeypatch.setenv("OFFSTAP_VIVE_DIR", str(tmp_path / "Vive"))
    monkeypatch.setenv("PIPELINE_RUN_DIR", str(tmp_path / "output"))

    cfg = load_config("ignored.yaml")

    assert cfg.random_seed == 42
    assert cfg.paths["input_ev3"] == str(tmp_path / "EV3")
    assert cfg.paths["input_vrmt"] == str(tmp_path / "Vive")
    assert cfg.paths["output_aligned"] == str(tmp_path / "output" / "13_integrated_logs")
    assert cfg.dev_mode is False


def test_cli_rejects_changed_scientific_configuration(tmp_path):
    """Reject threshold changes before creating a run directory."""
    # Break caught: --config can alter frozen scientific thresholds for a public run.
    from offstap.cli import main

    ev3, vive, config = _valid_inputs(tmp_path)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["quality"]["min_rows"] = 999
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="scientific"):
        main(
            [
                "--ev3-dir",
                str(ev3),
                "--vive-dir",
                str(vive),
                "--config",
                str(config),
                "--output-dir",
                str(output),
            ]
        )

    assert not output.exists()


def test_config_loader_rejects_changed_scientific_configuration(tmp_path, monkeypatch):
    """Keep direct environment-selected loads under the same freeze contract."""
    # Break caught: OFFSTAP_CONFIG bypasses CLI validation and changes scientific behavior.
    from offstap.core.config import load_config

    config = _write_valid_config(tmp_path / "config.yaml")
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["temporal"]["grid_step_s"] = 0.1
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    monkeypatch.setenv("OFFSTAP_CONFIG", str(config))

    with pytest.raises(ValueError, match="scientific"):
        load_config()


def test_config_loader_accepts_path_only_configuration_override(tmp_path, monkeypatch):
    """Continue to allow portable locations in a custom approved configuration."""
    # Break caught: scientific validation rejects an allowed path-only customization.
    from offstap.core.config import load_config

    config = _write_valid_config(tmp_path / "config.yaml")
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["paths"]["input_ev3"] = "custom/EV3 logs"
    payload["config_id"] = "portable_identity"
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    monkeypatch.setenv("OFFSTAP_CONFIG", str(config))

    assert load_config().paths["input_ev3"] == "custom/EV3 logs"


def test_built_wheel_includes_default_configuration(tmp_path):
    """Ship the default configuration with the declared console entry point."""
    # Break caught: an installed CLI cannot find its documented packaged default config.
    package_root = Path(__file__).resolve().parents[1]
    wheel_dir = tmp_path / "wheel"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(package_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    wheel = next(wheel_dir.glob("offstap-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert "offstap/config/default.yaml" in archive.namelist()


def _valid_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    ev3 = tmp_path / "EV3 files"
    vive = tmp_path / "Vive files"
    ev3.mkdir()
    vive.mkdir()
    config = _write_valid_config(tmp_path / "valid.yaml")
    return ev3, vive, config


def _write_valid_config(path: Path) -> Path:
    default = Path(__file__).resolve().parents[1] / "offstap" / "config" / "default.yaml"
    path.write_text(default.read_text(encoding="utf-8"), encoding="utf-8")
    return path
