"""Publication, presentation, and current-run dependency contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _write(path: Path, content: str = "value\n1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _options(tmp_path: Path, results: frozenset[str]):
    from offstap.paths import RunOptions

    ev3 = tmp_path / "ev3"
    vive = tmp_path / "vive"
    ev3.mkdir()
    vive.mkdir()
    config = Path(__file__).resolve().parents[1] / "offstap" / "config" / "default.yaml"
    return RunOptions(ev3, vive, config, tmp_path / "run", results)


def test_publication_index_records_requested_effective_and_matching_bytes(tmp_path):
    """The public index distinguishes intent from dependency closure and hashes bytes."""
    # Break caught: dependency-expanded selectors overwrite the user's request or a stale
    # digest is recorded instead of the bytes that actually shipped.
    from offstap.publication import write_result_index

    artifact = _write(
        tmp_path / "results" / "h1" / "15_hypotheses" / "H1" / "h1_trial_pairs.csv",
        "trial_id,delta_rmse_mm\nA,1.0\n",
    )
    index = write_result_index(tmp_path, frozenset({"figures"}), [artifact])
    payload = json.loads(index.read_text(encoding="utf-8"))

    assert payload["requested_results"] == ["figures"]
    assert set(payload["effective_results"]) >= {
        "core", "h1", "h2", "h3", "baseline", "figures",
    }
    assert payload["files"] == [{
        "path": "results/h1/15_hypotheses/H1/h1_trial_pairs.csv",
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }]


def test_h1_selection_excludes_h2_and_h3_catalog_entries():
    """Selecting one family cannot leak sibling-family artifacts."""
    # Break caught: publication flattens the entire shared comparison output.
    from offstap.publication import publish_selected

    published = publish_selected({"h1"}, {
        "h1": ["h1_trial_pairs.csv"],
        "h2": ["h2_trial_comparison.csv"],
        "h3": ["h3_calibration_mode_comparison.csv"],
    })

    assert published == ["h1_trial_pairs.csv"]


def test_publication_keeps_nested_diagnostic_results_directory(tmp_path):
    """Only the publication root is excluded; stage folders named results remain eligible."""
    # Break caught: a broad path-part filter silently drops 18_plots/results artifacts.
    from offstap.publication import _sources_for_pattern

    nested = _write(tmp_path / "18_plots" / "results" / "plot_status.csv")
    _write(tmp_path / "results" / "stale-publication.csv")

    assert _sources_for_pattern(tmp_path, "18_plots/**/*") == [nested]


def test_current_run_figure_uses_declared_metric_not_first_numeric_column(tmp_path, monkeypatch):
    """Current-run plots bind their scientific metric explicitly."""
    # Break caught: H1 plots repetition_id because it precedes delta_rmse_mm numerically.
    from offstap.figures import generate_figures

    spec = dict(
        next(item for item in generate_figures.FIGURE_SPECS if item["figure_id"] == "fig_h1_current_run")
    )
    monkeypatch.setattr(generate_figures, "FIGURE_SPECS", [spec])
    monkeypatch.setattr(
        generate_figures,
        "load_config",
        lambda *_: SimpleNamespace(paths={"output_base": str(tmp_path)}),
    )
    observed = {}

    def capture_metric(frame, metric, output_path):
        observed["metric"] = metric
        Path(output_path).write_bytes(b"rendered")

    monkeypatch.setattr(
        generate_figures,
        "plot_helpers",
        SimpleNamespace(plot_metrics_boxplot=capture_metric),
    )
    monkeypatch.setattr(generate_figures, "persist_figure_source_table", lambda *_: None)
    _write(
        tmp_path / "15_hypotheses" / "H1" / "h1_trial_pairs.csv",
        "trial_id,repetition_id,delta_rmse_mm\nA,1,-2.0\nB,1,-1.0\n",
    )

    generate_figures.generate_paper_figures()

    assert observed["metric"] == "delta_rmse_mm"


def test_publication_copies_selected_family_layout_without_changing_bytes(tmp_path):
    """Publication copies current-run sources beneath their family without flattening."""
    # Break caught: files are renamed/flattened, recomputed, or optional siblings leak.
    from offstap.publication import publish_run_results

    source = _write(
        tmp_path / "15_hypotheses" / "H1" / "h1_trial_pairs.csv",
        "trial_id,delta_rmse_mm\nA,1.0\n",
    )
    _write(
        tmp_path / "15_hypotheses" / "H2" / "h2_trial_comparison.csv",
        "trial_id,method,rmse_mm\nA,zoh,1.0\n",
    )
    published = publish_run_results(
        tmp_path,
        frozenset({"h1"}),
        catalog={
            "h1": ["15_hypotheses/H1/h1_trial_pairs.csv"],
            "h2": ["15_hypotheses/H2/h2_trial_comparison.csv"],
        },
    )

    target = tmp_path / "results" / "h1" / "15_hypotheses" / "H1" / source.name
    assert published == [tmp_path / "results" / "core" / "OUTPUT_GUIDE.md", target]
    assert target.read_bytes() == source.read_bytes()
    assert not (tmp_path / "results" / "h2").exists()


def test_optional_plan_uses_true_dependency_order():
    """Diagnostics precede inference outputs consumed by presentation stages."""
    # Break caught: a presentation stage is scheduled before its current-run sources.
    from offstap.workflow import build_plan

    modules = [stage.module for stage in build_plan(frozenset({"all"}))]
    comparison = modules.index("offstap.stages.s09_comparisons")
    diagnostic_positions = [
        modules.index(f"offstap.stages.{name}")
        for name in (
            "s10_plots", "s11_dataset_design", "s12_repeatability",
            "s13_scenario_robustness", "s14b_population_gate",
            "s14a_cascade", "s14_cascade", "s15_common_sampling",
        )
    ]
    statistics = modules.index("offstap.stages.s16b_statistics")
    figures = modules.index("offstap.figures.generate_figures")
    tables = modules.index("offstap.stages.s18_tables")
    report = modules.index("offstap.stages.s19_report")

    assert comparison < min(diagnostic_positions)
    assert diagnostic_positions == sorted(diagnostic_positions)
    assert max(diagnostic_positions) < statistics < figures < tables < report


def _seed_table_inputs(run_dir: Path) -> None:
    _write(
        run_dir / "19_tables" / "authoritative_population_gate_ledger.csv",
        "trial_id,gate,gate_status,evidence_path\nA,primary,eligible,current-run\n",
    )
    _write(run_dir / "17_statistics" / "accepted_trials_summary.csv", "n\n1\n")
    _write(run_dir / "17_statistics" / "rejected_trials_summary.csv", "reason,count\nnone,0\n")
    _write(
        run_dir / "17_statistics" / "hypothesis_decision_summary.csv",
        "analysis_set,hypothesis_id,decision,n,p_value\nprimary,H1,supported,1,0.01\n"
        "primary,H2,not_supported,1,0.2\nprimary,H3,inconclusive,1,\n",
    )
    _write(
        run_dir / "14_metrics" / "per_trial_metrics.csv",
        "trial_id,spatial_fit_rmse_mm,temporal_offset_s,peak_confidence_ratio\nA,1.0,0.1,3.0\n",
    )
    _write(run_dir / "15_hypotheses" / "H1" / "h1_summary.csv", "mean_delta_rmse_mm\n1.0\n")
    _write(run_dir / "15_hypotheses" / "H1" / "h1_trial_pairs.csv", "trial_id,delta_rmse_mm\nA,1.0\n")
    _write(run_dir / "15_hypotheses" / "H2" / "h2_summary_by_method.csv", "method,mean_rmse_mm\nzoh,2.0\n")
    _write(
        run_dir / "15_hypotheses" / "H3" / "h3_calibration_mode_comparison.csv",
        "trial_id,calibration_mode,heldout_rmse_mm\nA,split,2.0\nA,full,3.0\n",
    )
    _write(
        run_dir / "baseline" / "baseline_trial_metrics.csv",
        "trial_id,baseline_rmse_mm,offstap_rmse_mm,delta_rmse_mm\nA,4.0,2.0,2.0\n",
    )
    _write(
        run_dir / "baseline" / "baseline_summary.json",
        json.dumps([{
            "population": "planned_population",
            "n": 1,
            "mean_baseline_rmse_mm": 4.0,
            "mean_offstap_rmse_mm": 2.0,
            "mean_delta_mm": 2.0,
        }]),
    )


def test_tables_retarget_to_corrected_current_run_baseline(tmp_path, monkeypatch):
    """Baseline tables preserve the corrected adapter's rows and summaries."""
    # Break caught: tables read the defective legacy comparison directory or emit empties.
    from offstap.stages import s18_tables

    _seed_table_inputs(tmp_path)
    monkeypatch.setattr(
        s18_tables,
        "load_config",
        lambda *_: SimpleNamespace(paths={"output_base": str(tmp_path)}, config_id="test"),
    )
    s18_tables.main()

    trial_table = pd.read_csv(tmp_path / "19_tables" / "table_9_baseline_vs_proposed_pipeline.csv")
    summary_table = pd.read_csv(tmp_path / "19_tables" / "table_10_paired_pipeline_difference_summary.csv")
    assert trial_table.loc[0, "trial_id"] == "A"
    assert trial_table.loc[0, "baseline_rmse_mm"] == 4.0
    assert summary_table.loc[0, "population"] == "planned_population"
    decisions = pd.read_csv(tmp_path / "19_tables" / "table_7_hypothesis_results.csv")
    assert decisions[["hypothesis_id", "decision"]].to_dict("records") == [
        {"hypothesis_id": "H1", "decision": "supported"},
        {"hypothesis_id": "H2", "decision": "not_supported"},
        {"hypothesis_id": "H3", "decision": "inconclusive"},
    ]


def test_figure_manifest_consumes_current_h_and_corrected_baseline(tmp_path, monkeypatch):
    """The figure wrapper names only newly generated H/baseline sources."""
    # Break caught: figure production is disconnected from numerical closure or points at
    # a historical/defective baseline location.
    from offstap.figures import generate_figures

    _seed_table_inputs(tmp_path)
    monkeypatch.setattr(
        generate_figures,
        "load_config",
        lambda *_: SimpleNamespace(paths={"output_base": str(tmp_path)}),
    )
    monkeypatch.setattr(generate_figures, "persist_figure_source_table", lambda *_: None)
    monkeypatch.setattr(generate_figures, "plot_helpers", None)
    generate_figures.generate_paper_figures()

    manifest = pd.read_csv(tmp_path / "paper_figures" / "figure_manifest.csv")
    sources = ";".join(manifest["source_files"].astype(str))
    assert "15_hypotheses/H1/h1_trial_pairs.csv" in sources
    assert "15_hypotheses/H2/h2_trial_comparison.csv" in sources
    assert "15_hypotheses/H3/h3_calibration_mode_comparison.csv" in sources
    assert "baseline/baseline_trial_metrics.csv" in sources
    assert "27_pipeline_baseline_comparison" not in sources


def test_report_contains_exact_pipeline_conditional_limitation(tmp_path, monkeypatch):
    """A generated report states the non-metrological limitation verbatim."""
    # Break caught: report prose implies absolute robot accuracy or universal superiority.
    from offstap.publication import REPORT_LIMITATION
    from offstap.stages import s19_report

    _write(tmp_path / "14_metrics" / "evaluation_metrics.csv", "trial_id,ate_rmse\nA,1.0\n")
    _write(tmp_path / "14_metrics" / "aggregated_metrics.json", "{}\n")
    _seed_table_inputs(tmp_path)
    monkeypatch.setattr(
        s19_report,
        "load_config",
        lambda *_: SimpleNamespace(
            paths={"output_base": str(tmp_path), "output_provenance": str(tmp_path / "provenance")},
            config_id="test",
        ),
    )
    from offstap.core import reporting

    original_read_csv = reporting._read_csv

    def reject_legacy_baseline(path):
        if "27_pipeline_baseline_comparison" in str(path):
            raise AssertionError("report attempted to read defective baseline output")
        return original_read_csv(path)

    monkeypatch.setattr(reporting, "_read_csv", reject_legacy_baseline)
    monkeypatch.setattr(
        s19_report,
        "ProvenanceTracker",
        lambda *_: SimpleNamespace(record_stage=lambda **_: None),
    )
    s19_report.main()

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert REPORT_LIMITATION in report
    assert "15_hypotheses/H1/h1_trial_pairs.csv" in report
    assert "15_hypotheses/H2/h2_summary_by_method.csv" in report
    assert "15_hypotheses/H3/h3_calibration_mode_comparison.csv" in report
    assert "baseline/baseline_summary.json" in report
    assert "planned_population" in report


def test_population_gate_uses_corrected_current_run_baseline_membership(tmp_path, monkeypatch):
    """The diagnostic gate projects accepted corrected-baseline rows without old evidence."""
    # Break caught: s14b reads the defective baseline directory or marks a valid corrected
    # current-run baseline trial unavailable.
    from offstap.stages import s14b_population_gate

    _write(
        tmp_path / "01_manifest" / "pair_manifest.csv",
        "trial_id,dataset_tier,analysis_set,scenario_id,pair_status\n"
        "A,planned_405,primary,D1_T1,paired\n",
    )
    _write(tmp_path / "04_quality" / "rejection_log.csv", "trial_id,reason\n")
    _write(
        tmp_path / "baseline" / "baseline_trial_metrics.csv",
        "trial_id,baseline_rmse_mm,offstap_rmse_mm,delta_rmse_mm\nA,2.0,1.0,1.0\n",
    )
    monkeypatch.setattr(
        s14b_population_gate,
        "load_config",
        lambda *_: SimpleNamespace(paths={"output_base": str(tmp_path)}),
    )

    s14b_population_gate.main()

    ledger = pd.read_csv(tmp_path / "19_tables" / "authoritative_population_gate_ledger.csv")
    baseline = ledger.loc[(ledger["trial_id"] == "A") & (ledger["gate"] == "baseline_valid")].iloc[0]
    assert baseline["gate_status"] == "passed"
    assert baseline["evidence_path"].replace("\\", "/").endswith(
        "baseline/population_gate_membership.csv"
    )
    assert "27_pipeline_baseline_comparison" not in baseline["evidence_path"]


def test_core_only_publication_has_no_optional_family(tmp_path, monkeypatch):
    """A successful core-only run publishes core and no optional product."""
    # Break caught: dependency/publication logic creates optional output by default.
    from offstap.runner import run_core

    def fake_run_stage(stage, run_dir, env):
        if stage.module == "offstap.stages.s03_quality":
            _write(run_dir / "04_quality" / "accepted_trials.json", '["A"]')
        elif stage.module == "offstap.stages.s05_parameters":
            _write(run_dir / "alignment_parameters.json", '{"A": {}}')
        elif stage.module == "offstap.stages.s08_metrics":
            _write(
                run_dir / "14_metrics" / "per_trial_metrics.csv",
                "trial_id,accepted_for_hypothesis_tests\nA,1\n",
            )

    monkeypatch.setattr("offstap.runner.run_stage", fake_run_stage)
    run_dir = run_core(_options(tmp_path, frozenset({"core"})))
    payload = json.loads((run_dir / "results" / "index.json").read_text(encoding="utf-8"))

    assert payload["requested_results"] == ["core"]
    assert payload["effective_results"] == ["core"]
    assert {Path(row["path"]).parts[1] for row in payload["files"]} == {"core"}
    assert not any(
        (run_dir / "results" / family).exists()
        for family in ("h1", "h2", "h3", "baseline", "diagnostics", "figures", "tables", "report")
    )
    assert not (run_dir / "27_pipeline_baseline_comparison").exists()


def _fake_all_stage(stage, run_dir: Path, _env) -> None:
    module = stage.module
    if module == "offstap.stages.s01_manifest":
        _write(run_dir / "01_manifest" / "pair_manifest.csv", "trial_id,dataset_tier\nA,planned_405\n")
    elif module == "offstap.stages.s03_quality":
        _write(run_dir / "04_quality" / "accepted_trials.json", '["A"]')
    elif module == "offstap.stages.s05_parameters":
        _write(run_dir / "alignment_parameters.json", '{"A": {}}')
    elif module == "offstap.stages.s08_metrics":
        _write(
            run_dir / "14_metrics" / "per_trial_metrics.csv",
            "trial_id,accepted_for_hypothesis_tests\nA,1\n",
        )
    elif module == "offstap.stages.s09_comparisons":
        _write(run_dir / "15_hypotheses" / "H1" / "h1_trial_pairs.csv", "trial_id,delta_rmse_mm\nA,1.0\n")
        _write(run_dir / "15_hypotheses" / "H2" / "h2_trial_comparison.csv", "trial_id,method,rmse_mm\nA,zoh,1.0\n")
        _write(run_dir / "15_hypotheses" / "H3" / "h3_calibration_mode_comparison.csv", "trial_id,calibration_mode,heldout_rmse_mm\nA,split,1.0\n")
    elif module == "offstap.stages.s10_plots":
        _write(run_dir / "18_plots" / "plot_status.csv", "figure_id,status\ncore_plot,PASS\n")
    elif module == "offstap.stages.s11_dataset_design":
        _write(run_dir / "00_experiment_design" / "design_balance_report.csv", "scenario_id,n\nD1_T1,1\n")
    elif module == "offstap.stages.s12_repeatability":
        _write(run_dir / "17_statistics" / "repeatability_per_scenario.csv", "scenario_id,mean_rmse_mm\nD1_T1,1.0\n")
    elif module == "offstap.stages.s13_scenario_robustness":
        _write(run_dir / "17_statistics" / "scenario_level_metrics.csv", "scenario_id,residual_mean_mm\nD1_T1,1.0\n")
    elif module == "offstap.stages.s14b_population_gate":
        _write(run_dir / "19_tables" / "authoritative_population_gate_ledger.csv", "trial_id,gate,gate_status,evidence_path\nA,primary,eligible,current\n")
    elif module == "offstap.stages.s14a_cascade":
        _write(run_dir / "14_metrics" / "cascade_per_trial_metrics.csv", "trial_id,stage_id,rmse_mm\nA,S0,1.0\n")
    elif module == "offstap.stages.s14_cascade":
        _write(run_dir / "14_metrics" / "cascade_per_stage_summary.csv", "stage_id,mean_rmse_mm\nS0,1.0\n")
    elif module == "offstap.stages.s15_common_sampling":
        _write(run_dir / "13_integrated_logs" / "sampling_strategy_registry.csv", "strategy_id\nct_query\n")
    elif module == "offstap.stages.s16b_statistics":
        _write(run_dir / "17_statistics" / "hypothesis_test_summary.csv", "test,p_value\nH1,0.1\nH2,0.2\nH3,0.3\n")
        _write(run_dir / "17_statistics" / "hypothesis_decision_summary.csv", "hypothesis_id,decision\nH1,not_supported\nH2,not_supported\nH3,not_supported\n")
    elif module == "offstap.figures.generate_figures":
        rows = ["figure_id,file_name,valid_for_paper"]
        (run_dir / "paper_figures").mkdir(parents=True, exist_ok=True)
        for family in ("h1", "h2", "h3", "baseline"):
            name = f"fig_{family}_current_run.png"
            rows.append(f"fig_{family}_current_run,{name},True")
            (run_dir / "paper_figures" / name).write_bytes(f"current-run-{family}".encode())
        _write(run_dir / "paper_figures" / "figure_manifest.csv", "\n".join(rows) + "\n")
    elif module == "offstap.stages.s18_tables":
        names = (
            "table_1_dataset_summary.csv",
            "table_2_quality_filtering.csv",
            "table_3_spatial_diagnostics.csv",
            "table_4_temporal_alignment.csv",
            "table_5_representation_methods.csv",
            "table_6_calibration_modes.csv",
            "table_7_hypothesis_results.csv",
            "table_8_selected_pipeline_configuration.csv",
            "table_9_baseline_vs_proposed_pipeline.csv",
            "table_10_paired_pipeline_difference_summary.csv",
        )
        for name in names:
            _write(run_dir / "19_tables" / name, "metric,value\ncurrent,1\n")
    elif module == "offstap.stages.s19_report":
        from offstap.publication import REPORT_LIMITATION

        _write(run_dir / "report.md", f"> **Limitation:** {REPORT_LIMITATION}\n")
        _write(run_dir / "report.html", f"<p>{REPORT_LIMITATION}</p>\n")


def _fake_baseline(_source_run: Path, output_dir: Path) -> dict[str, Path]:
    trial = _write(
        output_dir / "baseline_trial_metrics.csv",
        "trial_id,baseline_rmse_mm,offstap_rmse_mm,delta_rmse_mm\nA,2.0,1.0,1.0\n",
    )
    rejection = _write(output_dir / "baseline_rejections.csv", "trial_id,reason\n")
    summary = _write(output_dir / "baseline_summary.json", '[{"population":"planned_population","n":1}]\n')
    manifest = _write(output_dir / "baseline_manifest.json", '{"identity":"baseline"}\n')
    return {"trial_metrics": trial, "rejections": rejection, "summary": summary, "manifest": manifest}


def test_all_run_publishes_every_family_and_index_hashes_match(tmp_path, monkeypatch):
    """A complete synthetic all-run publishes every family with byte-true hashes."""
    # Break caught: an effective family is absent, an internal path is flattened, or the
    # index records bytes other than the published artifact.
    from offstap.runner import run_core
    from offstap.selection import ALL

    monkeypatch.setattr("offstap.runner.run_stage", _fake_all_stage)
    monkeypatch.setattr("offstap.baseline.run_baseline", _fake_baseline)
    run_dir = run_core(_options(tmp_path, frozenset({"all"})))
    payload = json.loads((run_dir / "results" / "index.json").read_text(encoding="utf-8"))

    assert payload["requested_results"] == ["all"]
    assert set(payload["effective_results"]) == set(ALL)
    assert {Path(row["path"]).parts[1] for row in payload["files"]} == set(ALL)
    for row in payload["files"]:
        artifact = run_dir / row["path"]
        assert artifact.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == row["sha256"]
    assert json.loads((run_dir / "run_status.json").read_text(encoding="utf-8")) == {
        "status": "PASS", "reason": "",
    }


def test_missing_selected_figure_fails_instead_of_marking_pass(tmp_path, monkeypatch):
    """A selected but empty figure family cannot end in PASS."""
    # Break caught: stage scheduling alone is mistaken for a produced result.
    from offstap.runner import run_core

    def no_figure_stage(stage, run_dir, env):
        if stage.module != "offstap.figures.generate_figures":
            _fake_all_stage(stage, run_dir, env)

    monkeypatch.setattr("offstap.runner.run_stage", no_figure_stage)
    monkeypatch.setattr("offstap.baseline.run_baseline", _fake_baseline)

    with pytest.raises(RuntimeError, match="figures"):
        run_core(_options(tmp_path, frozenset({"figures"})))

    status = json.loads((tmp_path / "run" / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "FAILED"
    assert "figures" in status["reason"]


def test_manifest_only_figure_family_is_blocked(tmp_path):
    """A figure manifest without a rendered current-run image is not a product."""
    # Break caught: metadata-only or pending figure rows are called PASS.
    from offstap.publication import publish_run_results

    _write(tmp_path / "14_metrics" / "per_trial_metrics.csv", "trial_id\nA\n")
    _write(
        tmp_path / "paper_figures" / "figure_manifest.csv",
        "figure_id,file_name,valid_for_paper,reason_if_not_valid\n"
        "current,current.png,False,missing source\n",
    )

    with pytest.raises(RuntimeError, match="rendered current-run image"):
        publish_run_results(tmp_path, frozenset({"figures"}))


def test_missing_h_sources_stop_before_diagnostics_execute(tmp_path, monkeypatch):
    """Optional stages execute only after their declared current-run sources exist."""
    # Break caught: diagnostics run after an H producer returned without required outputs.
    from offstap.runner import run_core

    called: list[str] = []

    def incomplete_comparison(stage, run_dir, env):
        called.append(stage.module)
        if stage.module in {
            "offstap.stages.s03_quality",
            "offstap.stages.s05_parameters",
            "offstap.stages.s08_metrics",
        }:
            _fake_all_stage(stage, run_dir, env)
        # s09 intentionally returns without selected H artifacts.

    monkeypatch.setattr("offstap.runner.run_stage", incomplete_comparison)
    monkeypatch.setattr("offstap.baseline.run_baseline", _fake_baseline)

    with pytest.raises(RuntimeError, match="selected H source"):
        run_core(_options(tmp_path, frozenset({"figures"})))

    assert called[-1] == "offstap.stages.s09_comparisons"
    assert "offstap.stages.s10_plots" not in called


def test_h1_publication_includes_only_h1_inferential_rows(tmp_path):
    """Shared statistics are packaged into a selected family without sibling leakage."""
    # Break caught: H1 lacks its inferential decision or publishes H2/H3 rows with it.
    from offstap.publication import publish_run_results

    _write(tmp_path / "14_metrics" / "per_trial_metrics.csv", "trial_id\nA\n")
    _write(tmp_path / "15_hypotheses" / "H1" / "h1_trial_pairs.csv", "trial_id,delta_rmse_mm\nA,1.0\n")
    _write(
        tmp_path / "17_statistics" / "hypothesis_test_summary.csv",
        "test,analysis_set,p_value\nH1,primary,0.01\nH2,primary,0.2\nH3,primary,0.3\n",
    )
    _write(
        tmp_path / "17_statistics" / "hypothesis_decision_summary.csv",
        "hypothesis_id,analysis_set,decision\nH1,primary,supported\nH2,primary,not_supported\nH3,primary,inconclusive\n",
    )

    published = publish_run_results(tmp_path, frozenset({"h1"}))
    summary = next(path for path in published if path.name == "hypothesis_test_summary.csv")
    decisions = next(path for path in published if path.name == "hypothesis_decision_summary.csv")

    assert pd.read_csv(summary)["test"].tolist() == ["H1"]
    assert pd.read_csv(decisions)["hypothesis_id"].tolist() == ["H1"]


def test_h2_publication_accepts_canonical_friedman_identifier_and_pairwise_files(tmp_path):
    """The H2 family routes the producer's canonical identifier and flat pairwise files."""
    # Break caught: publication expects H2 while Stage 16b persists H2_Friedman.
    from offstap.publication import publish_run_results

    _write(tmp_path / "14_metrics" / "per_trial_metrics.csv", "trial_id\nA\n")
    _write(
        tmp_path / "15_hypotheses" / "H2" / "h2_trial_comparison.csv",
        "trial_id,method,rmse_mm\nA,zoh,1.0\nA,linear,1.1\nA,pchip,1.2\nA,spline,1.3\n",
    )
    _write(
        tmp_path / "17_statistics" / "hypothesis_test_summary.csv",
        "test,analysis_set,p_value\nH1,primary,0.01\nH2_Friedman,primary,0.2\nH3,primary,0.3\n",
    )
    _write(
        tmp_path / "17_statistics" / "hypothesis_decision_summary.csv",
        "hypothesis_id,analysis_set,decision\nH1,primary,supported\n"
        "H2_Friedman,primary,not_supported\nH3,primary,inconclusive\n",
    )
    pairwise = _write(
        tmp_path / "17_statistics" / "h2_pairwise_primary_planned_405.csv",
        "method_a,method_b,p_value\nzoh,linear,0.5\n",
    )

    published = publish_run_results(tmp_path, frozenset({"h2"}))

    summary = next(path for path in published if path.name == "hypothesis_test_summary.csv")
    decisions = next(path for path in published if path.name == "hypothesis_decision_summary.csv")
    published_pairwise = tmp_path / "results" / "h2" / pairwise.relative_to(tmp_path)
    assert pd.read_csv(summary)["test"].tolist() == ["H2_Friedman"]
    assert pd.read_csv(decisions)["hypothesis_id"].tolist() == ["H2_Friedman"]
    assert published_pairwise in published
    assert published_pairwise.read_bytes() == pairwise.read_bytes()


def test_h1_publication_includes_canonical_signed_rank_audit_path(tmp_path):
    """H1 publishes the signed-rank rows from the producer's actual directory."""
    from offstap.publication import publish_run_results

    _write(tmp_path / "14_metrics" / "per_trial_metrics.csv", "trial_id\nA\n")
    _write(tmp_path / "15_hypotheses" / "H1" / "h1_trial_pairs.csv", "trial_id,delta_rmse_mm\nA,1.0\n")
    _write(tmp_path / "17_statistics" / "hypothesis_test_summary.csv", "test,analysis_set,p_value\nH1,primary,0.01\n")
    audit = _write(
        tmp_path / "17_statistics" / "signed_rank_audits" / "primary_H1_rows.csv",
        "trial_id,delta_rmse_mm\nA,1.0\n",
    )

    published = publish_run_results(tmp_path, frozenset({"h1"}))

    target = tmp_path / "results" / "h1" / audit.relative_to(tmp_path)
    assert target in published
    assert target.read_bytes() == audit.read_bytes()


def test_method_sensitivity_is_explicitly_unavailable_and_never_published(tmp_path, monkeypatch):
    """Placeholder or mislabeled method-sensitivity products cannot enter public results."""
    from offstap.publication import publish_run_results
    from offstap.stages import s13_scenario_robustness

    _write(
        tmp_path / "14_metrics" / "per_trial_metrics.csv",
        "trial_id,rmse_mm,ate_rmse_mm,rpe_rmse_mm\nD1_T1_A,10.0,11.0,12.0\n",
    )
    monkeypatch.setattr(
        s13_scenario_robustness,
        "load_config",
        lambda *_: SimpleNamespace(paths={"output_base": str(tmp_path)}),
    )

    s13_scenario_robustness.main()

    status_path = tmp_path / "17_statistics" / "method_sensitivity_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "NOT_AVAILABLE"
    assert not (tmp_path / "17_statistics" / "scenario_level_method_sensitivity.csv").exists()
    assert not (tmp_path / "18_plots" / "scenario_grid" / "method_sensitivity_9x9_heatmap.png").exists()

    published = publish_run_results(tmp_path, frozenset({"diagnostics"}))
    published_rel = {path.relative_to(tmp_path).as_posix() for path in published}
    assert "results/diagnostics/17_statistics/method_sensitivity_status.json" in published_rel
    assert not any("method_sensitivity_9x9_heatmap" in path for path in published_rel)
    assert not any("scenario_level_method_sensitivity.csv" in path for path in published_rel)


def test_public_figure_catalog_has_no_method_sensitivity_placeholder():
    from offstap.figures.generate_figures import FIGURE_SPECS

    assert "fig_method_sensitivity_heatmap" not in {spec["figure_id"] for spec in FIGURE_SPECS}


def test_manifest_only_baseline_family_is_blocked(tmp_path):
    """Baseline provenance without trial metrics and summary is not a result product."""
    # Break caught: a callable that wrote metadata only is marked PASS.
    from offstap.publication import publish_run_results

    _write(tmp_path / "14_metrics" / "per_trial_metrics.csv", "trial_id\nA\n")
    _write(tmp_path / "baseline" / "baseline_manifest.json", '{"identity":"baseline"}\n')

    with pytest.raises(RuntimeError, match="baseline trial metrics"):
        publish_run_results(tmp_path, frozenset({"baseline"}))


def test_figures_require_rendered_h_and_baseline_products(tmp_path):
    """An unrelated valid image cannot mask missing numerical-family figures."""
    # Break caught: figures PASS while H1/H2/H3/baseline rows remain pending.
    from offstap.publication import publish_run_results

    _write(tmp_path / "14_metrics" / "per_trial_metrics.csv", "trial_id\nA\n")
    _write(
        tmp_path / "paper_figures" / "figure_manifest.csv",
        "figure_id,file_name,valid_for_paper,reason_if_not_valid\n"
        "fig_dataset_design_9x9_grid,design.png,True,\n",
    )
    (tmp_path / "paper_figures" / "design.png").write_bytes(b"current-run-diagnostic")

    with pytest.raises(RuntimeError, match="H1/H2/H3/baseline"):
        publish_run_results(tmp_path, frozenset({"figures"}))


def test_tables_require_all_ten_nonempty_current_run_tables(tmp_path):
    """A partial table set cannot be published as a completed tables family."""
    # Break caught: one generated table masks missing or header-only presentation tables.
    from offstap.publication import publish_run_results

    _write(tmp_path / "14_metrics" / "per_trial_metrics.csv", "trial_id\nA\n")
    _write(tmp_path / "19_tables" / "table_1_dataset_summary.csv", "metric,count\naccepted,1\n")

    with pytest.raises(RuntimeError, match="ten nonempty"):
        publish_run_results(tmp_path, frozenset({"tables"}))
