"""Portable, failure-explicit execution of the public core workflow."""

import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import yaml

from offstap.paths import RunOptions, prepare_run, validate_config
from offstap.publication import publish_run_results, write_result_index
from offstap.selection import expand_results
from offstap.workflow import STEP_LABELS, Stage, build_plan, callable_producers_after


STAGE_LOG_FIELDS = ("step", "label", "substeps", "module", "started_utc", "ended_utc", "status", "error")
EXECUTED_RESULT_PRODUCTS = frozenset({"core"})

STAGE_SOURCE_REQUIREMENTS = {
    "offstap.stages.s09_comparisons": ("14_metrics/per_trial_metrics.csv",),
    "offstap.stages.s10_plots": ("14_metrics/per_trial_metrics.csv",),
    "offstap.stages.s11_dataset_design": (
        "01_manifest/pair_manifest.csv", "14_metrics/per_trial_metrics.csv",
    ),
    "offstap.stages.s12_repeatability": (
        "14_metrics/per_trial_metrics.csv", "alignment_parameters.json",
    ),
    "offstap.stages.s13_scenario_robustness": ("14_metrics/per_trial_metrics.csv",),
    "offstap.stages.s14b_population_gate": (
        "01_manifest/pair_manifest.csv", "14_metrics/per_trial_metrics.csv",
    ),
    "offstap.stages.s14a_cascade": ("19_tables/authoritative_population_gate_ledger.csv",),
    "offstap.stages.s14_cascade": ("14_metrics/cascade_per_trial_metrics.csv",),
    "offstap.stages.s15_common_sampling": ("14_metrics/per_trial_metrics.csv",),
    "offstap.figures.generate_figures": (
        "15_hypotheses/H1/h1_trial_pairs.csv",
        "15_hypotheses/H2/h2_trial_comparison.csv",
        "15_hypotheses/H3/h3_calibration_mode_comparison.csv",
        "baseline/baseline_trial_metrics.csv",
        "00_experiment_design/design_balance_report.csv",
        "17_statistics/repeatability_per_scenario.csv",
        "17_statistics/scenario_level_metrics.csv",
        "14_metrics/cascade_per_stage_summary.csv",
    ),
    "offstap.stages.s18_tables": (
        "17_statistics/hypothesis_test_summary.csv",
        "19_tables/authoritative_population_gate_ledger.csv",
        "baseline/baseline_trial_metrics.csv",
        "baseline/baseline_summary.json",
    ),
    "offstap.stages.s19_report": (
        "19_tables/table_9_baseline_vs_proposed_pipeline.csv",
        "19_tables/table_10_paired_pipeline_difference_summary.csv",
    ),
}

STAGE_OUTPUT_REQUIREMENTS = {
    "offstap.stages.s10_plots": ("18_plots/plot_status.csv",),
    "offstap.stages.s11_dataset_design": ("00_experiment_design/design_balance_report.csv",),
    "offstap.stages.s12_repeatability": ("17_statistics/repeatability_per_scenario.csv",),
    "offstap.stages.s13_scenario_robustness": ("17_statistics/scenario_level_metrics.csv",),
    "offstap.stages.s14b_population_gate": ("19_tables/authoritative_population_gate_ledger.csv",),
    "offstap.stages.s14a_cascade": ("14_metrics/cascade_per_trial_metrics.csv",),
    "offstap.stages.s14_cascade": ("14_metrics/cascade_per_stage_summary.csv",),
    "offstap.stages.s15_common_sampling": ("13_integrated_logs/sampling_strategy_registry.csv",),
    "offstap.stages.s16b_statistics": (
        "17_statistics/hypothesis_test_summary.csv",
        "17_statistics/hypothesis_decision_summary.csv",
    ),
    "offstap.figures.generate_figures": ("paper_figures/figure_manifest.csv",),
    "offstap.stages.s18_tables": (
        "19_tables/table_1_dataset_summary.csv",
        "19_tables/table_10_paired_pipeline_difference_summary.csv",
    ),
    "offstap.stages.s19_report": ("report.md", "report.html"),
}


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def run_stage(stage: Stage, run_dir: Path, env: dict[str, str]) -> None:
    """Run one importable stage and append exactly one durable status row."""
    run_dir = Path(run_dir).resolve()
    package_root = str(Path(__file__).resolve().parent.parent)
    existing_pythonpath = env.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    pythonpath = os.pathsep.join(part for part in (package_root, existing_pythonpath) if part)
    child_env = {
        **os.environ,
        **env,
        "PYTHONPATH": pythonpath,
        "MPLBACKEND": "Agg",
        "OFFSTAP_RUNNER_MANAGED_LOG": "1",
    }
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(
        [sys.executable, "-m", stage.module],
        cwd=run_dir,
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )
    ended = datetime.now(timezone.utc).isoformat()
    error = completed.stderr or (f"exit code {completed.returncode}" if completed.returncode else "")
    log_path = run_dir / "pipeline_stage_log.csv"
    with log_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=STAGE_LOG_FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow({
            "step": stage.step,
            "label": stage.label,
            "substeps": "; ".join(f"{step}: {STEP_LABELS[step]}" for step in stage.substeps),
            "module": stage.module,
            "started_utc": started,
            "ended_utc": ended,
            "status": "FAILED" if completed.returncode else "PASS",
            "error": error if completed.returncode else "",
        })
    if completed.returncode:
        raise RuntimeError(f"{stage.label}: {error}")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_initial_evidence(options: RunOptions, run_dir: Path) -> Path:
    payload = validate_config(Path(options.config).resolve())
    payload["paths"] = {
        **payload["paths"],
        "input_ev3": str(Path(options.ev3_dir).resolve()),
        "input_vrmt": str(Path(options.vive_dir).resolve()),
        "output_base": str(run_dir),
        "output_cleaned": str(run_dir / "03_cleaned"),
        "output_aligned": str(run_dir / "13_integrated_logs"),
        "output_schema": str(run_dir / "02_schema"),
        "output_plots": str(run_dir / "18_plots"),
        "output_provenance": str(run_dir / "provenance"),
    }
    effective_config = run_dir / "config_used.yaml"
    effective_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (run_dir / "config_hash.txt").write_text(_hash_file(effective_config) + "\n", encoding="utf-8")
    with (run_dir / "input_checksums.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("file_path", "file_size", "checksum"))
        for directory in (Path(options.ev3_dir).resolve(), Path(options.vive_dir).resolve()):
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix.lower() == ".csv":
                    writer.writerow((str(path), path.stat().st_size, _hash_file(path)))
    (run_dir / "software_environment.txt").write_text(
        f"Python: {sys.version}\nOS: {platform.platform()}\n",
        encoding="utf-8",
    )
    (run_dir / "software_manifest.json").write_text(
        json.dumps({
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "offstap_version": _version("offstap"),
            "dependencies": {
                package: _version(package)
                for package in ("numpy", "pandas", "scipy", "pyarrow", "PyYAML", "matplotlib", "seaborn")
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return effective_config


def _write_run_status(run_dir: Path, status: str, reason: str = "") -> None:
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": status, "reason": reason}, indent=2) + "\n",
        encoding="utf-8",
    )


def _pending_result_reason(
    selected: frozenset[str],
    executed: frozenset[str] = EXECUTED_RESULT_PRODUCTS,
) -> str:
    """Describe requested products whose producers land in later tasks."""
    pending = sorted(selected - executed)
    if not pending:
        return ""
    if len(pending) == 1:
        return f"{pending[0]} requires a dependency producer that is not integrated"
    return f"required dependency producers are not integrated: {', '.join(pending)}"


def _write_not_applicable_results(run_dir: Path, selected: frozenset[str], reason: str) -> None:
    """Record selected optional products blocked by a failed eligibility gate."""
    products = sorted(selected - {"core"})
    if not products:
        return
    payload = {
        product: {"status": "NOT_APPLICABLE", "reason": reason}
        for product in products
    }
    (run_dir / "result_status.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _require_paths(run_dir: Path, paths: tuple[str, ...], context: str) -> None:
    missing = [relative for relative in paths if not (run_dir / relative).is_file()]
    if missing:
        raise RuntimeError(f"{context}: missing current-run artifacts: {', '.join(missing)}")


def _require_selected_h_sources(run_dir: Path, selected: frozenset[str]) -> None:
    paths = {
        "h1": "15_hypotheses/H1/h1_trial_pairs.csv",
        "h2": "15_hypotheses/H2/h2_trial_comparison.csv",
        "h3": "15_hypotheses/H3/h3_calibration_mode_comparison.csv",
    }
    missing = [paths[family] for family in sorted(selected & paths.keys()) if not (run_dir / paths[family]).is_file()]
    if missing:
        raise RuntimeError(f"selected H source artifacts are missing: {', '.join(missing)}")


def run_core(options: RunOptions) -> Path:
    """Prepare and execute the selected public plan, failing closed on empty populations."""
    run_dir = prepare_run(options)
    _write_run_status(run_dir, "RUNNING")
    try:
        effective_config = _write_initial_evidence(options, run_dir)
        env = {
            "OFFSTAP_CONFIG": str(effective_config),
            "OFFSTAP_EV3_DIR": str(Path(options.ev3_dir).resolve()),
            "OFFSTAP_VIVE_DIR": str(Path(options.vive_dir).resolve()),
            "PIPELINE_RUN_DIR": str(run_dir),
        }
        selected = expand_results(options.results)
        executed_products = set(EXECUTED_RESULT_PRODUCTS)
        for stage in build_plan(options.results):
            _require_paths(
                run_dir,
                STAGE_SOURCE_REQUIREMENTS.get(stage.module, ()),
                f"{stage.module} source check",
            )
            run_stage(stage, run_dir, env)
            if stage.module == "offstap.stages.s03_quality":
                accepted_path = run_dir / "04_quality" / "accepted_trials.json"
                accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
                if not accepted:
                    reason = "No eligible trials accepted by quality gate"
                    _write_not_applicable_results(run_dir, selected, reason)
                    raise RuntimeError(reason)
            if stage.module == "offstap.stages.s05_parameters":
                params = json.loads((run_dir / "alignment_parameters.json").read_text(encoding="utf-8"))
                if not params:
                    reason = "No eligible calibration parameter sets"
                    _write_not_applicable_results(run_dir, selected, reason)
                    raise RuntimeError(reason)
            if stage.module == "offstap.stages.s08_metrics":
                metric_path = run_dir / "14_metrics" / "per_trial_metrics.csv"
                if not metric_path.exists():
                    reason = "No eligible held-out metrics"
                    _write_not_applicable_results(run_dir, selected, reason)
                    raise RuntimeError(reason)
                with metric_path.open(newline="", encoding="utf-8") as stream:
                    if not any(row.get("accepted_for_hypothesis_tests") == "1" for row in csv.DictReader(stream)):
                        reason = "No eligible held-out metrics"
                        _write_not_applicable_results(run_dir, selected, reason)
                        raise RuntimeError(reason)
            if stage.module == "offstap.stages.s09_comparisons":
                _require_selected_h_sources(run_dir, selected)
            _require_paths(
                run_dir,
                STAGE_OUTPUT_REQUIREMENTS.get(stage.module, ()),
                f"{stage.module} output check",
            )
            for producer in callable_producers_after(stage.module, selected):
                producer.run(run_dir)
                executed_products.add(producer.product)
            if stage.module == "offstap.stages.s16b_statistics":
                executed_products.update(selected & {"h1", "h2", "h3"})
            elif stage.module == "offstap.stages.s15_common_sampling":
                executed_products.add("diagnostics")
            elif stage.module == "offstap.figures.generate_figures":
                executed_products.add("figures")
            elif stage.module == "offstap.stages.s18_tables":
                executed_products.add("tables")
            elif stage.module == "offstap.stages.s19_report":
                executed_products.add("report")
        pending_reason = _pending_result_reason(selected, frozenset(executed_products))
        if pending_reason:
            raise RuntimeError(pending_reason)
        published = publish_run_results(run_dir, options.results)
        write_result_index(run_dir, options.results, published)
    except Exception as exc:
        _write_run_status(run_dir, "FAILED", str(exc))
        raise
    _write_run_status(run_dir, "PASS")
    return run_dir
