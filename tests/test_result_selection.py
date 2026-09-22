"""Result-selection dependencies and optional-stage execution contracts."""

import json
import hashlib
from pathlib import Path

import pandas as pd
import pytest


def _options(tmp_path: Path, results: frozenset[str]):
    from offstap.paths import RunOptions

    ev3 = tmp_path / "ev3"
    vive = tmp_path / "vive"
    ev3.mkdir()
    vive.mkdir()
    config = Path(__file__).resolve().parents[1] / "offstap" / "config" / "default.yaml"
    return RunOptions(ev3, vive, config, tmp_path / "run", results)


def test_figure_request_includes_scientific_sources():
    """Presentation cannot omit the numerical evidence it needs."""
    # Break caught: figures can run without their H-family or baseline sources.
    from offstap.selection import expand_results

    assert expand_results(frozenset({"figures"})) >= {
        "core", "h1", "h2", "h3", "baseline", "figures",
    }


def test_all_expands_to_each_public_result_name():
    """The all selector denotes every supported result product, not itself."""
    # Break caught: an all request omits a documented public product.
    from offstap.selection import ALL, expand_results

    assert expand_results(frozenset({"all"})) == ALL


def test_core_only_does_not_schedule_hypothesis_statistics():
    """The default run remains limited to the mandatory core stages."""
    # Break caught: a core-only request unexpectedly executes H computation.
    from offstap.workflow import build_plan

    modules = [stage.module for stage in build_plan(frozenset({"core"}))]
    assert "offstap.stages.s09_comparisons" not in modules
    assert "offstap.stages.s16b_statistics" not in modules


def test_h1_plan_runs_shared_comparison_and_statistics_once():
    """An H-family request schedules each shared producer one time."""
    # Break caught: H1 duplicates or omits either producer in the DAG.
    from offstap.workflow import CORE_STAGES, build_plan

    modules = [stage.module for stage in build_plan(frozenset({"h1"}))]
    assert modules[:len(CORE_STAGES)] == [stage.module for stage in CORE_STAGES]
    assert modules.count("offstap.stages.s09_comparisons") == 1
    assert modules.count("offstap.stages.s16b_statistics") == 1
    assert modules[-2:] == [
        "offstap.stages.s09_comparisons",
        "offstap.stages.s16b_statistics",
    ]


def _support_hash() -> str:
    return hashlib.sha256(b"10\n11\n12\n13").hexdigest()


def _h2_inputs() -> tuple[dict, dict]:
    ev3 = pd.DataFrame({
        "trial_id": ["trial"] * 4,
        "partition": ["evaluation"] * 4,
        "source_row_index": [10, 11, 12, 13],
        "local_time_s": [0.0, 1.0, 2.0, 3.0],
        "odometry_x": [0.0, 1.0, 2.0, 3.0],
        "odometry_y": [0.0, 0.0, 0.0, 0.0],
    })
    vive = pd.DataFrame({
        "relative_time_s": [0.0, 1.0, 2.0, 3.0],
        "planar_x": [0.0, 1.0, 2.0, 3.0],
        "planar_y": [0.0, 0.0, 0.0, 0.0],
    })
    params = {
        "vive_samples": vive,
        "spatial": {"theta_rad": 0.0, "vx0": 0.0, "vy0": 0.0, "ex0": 0.0, "ey0": 0.0},
    }
    return params, {"evaluation": {"ev3": ev3}}


def test_h2_producer_emits_exact_arms_on_one_stable_common_support():
    """The executed H2 producer emits only its four inferential arms."""
    # Break caught: the producer changes an arm or hashes arm support differently.
    from offstap.stages.s09_comparisons import build_h2_comparison

    produced = build_h2_comparison(*_h2_inputs())

    assert produced["method"].tolist() == ["zoh", "linear", "pchip", "spline"]
    assert produced["vive_operator"].tolist() == ["zoh", "linear", "pchip", "spline"]
    assert produced["common_support_count"].tolist() == [4, 4, 4, 4]
    assert set(produced["common_support_hash"]) == {_support_hash()}
    assert set(produced["common_support_hash_sha256"]) == {_support_hash()}
    assert set(produced["status"]) == {"valid"}


def test_h2_producer_rejects_continuous_time_query_arm():
    """The executed H2 producer rejects the descriptive continuous-time row."""
    # Break caught: a non-inferential arm is accepted into H2 computation.
    from offstap.stages.s09_comparisons import build_h2_comparison

    with pytest.raises(ValueError, match="exactly the four Vive operators"):
        build_h2_comparison(*_h2_inputs(), methods=("zoh", "linear", "pchip", "continuous_time_query"))


def test_h1_and_h3_comparators_preserve_their_common_support_hashes():
    """The real H1/H3 consumers retain one exact finite support identity."""
    # Break caught: a comparison drops or reserializes persisted support keys.
    from offstap.stages.s09_comparisons import Arm, compare_h1_arms, compare_h3_arms

    def arm(name: str, residuals: list[float]):
        evaluation = pd.DataFrame({
            "source_row_index": [10, 11, 12, 13],
            "residual_norm": residuals,
        })
        return Arm(name, {}, {}, None, evaluation, {10, 11, 12, 13}, {"status": "ok"})

    h1 = compare_h1_arms({"onset": arm("onset", [2.0, 2.0, 2.0, 2.0]), "refined": arm("refined", [1.0, 1.0, 1.0, 1.0])})
    h3 = compare_h3_arms(arm("split", [2.0, 2.0, 2.0, 2.0]), arm("full", [1.0, 1.0, 1.0, 1.0]))

    assert h1.diagnostics["common_support_count"] == 4
    assert h1.diagnostics["common_support_hash"] == _support_hash()
    assert h3.diagnostics["common_support_count"] == 4
    assert h3.common_support_hash == _support_hash()


def test_optional_stage_failure_marks_run_failed(tmp_path, monkeypatch):
    """A requested H producer failure cannot leave the run marked PASS."""
    # Break caught: runner marks PASS after an optional-stage subprocess fails.
    from offstap.runner import run_core

    called: list[str] = []

    def fake_run_stage(stage, run_dir, env):
        called.append(stage.module)
        if stage.module == "offstap.stages.s03_quality":
            (run_dir / "04_quality" / "accepted_trials.json").write_text('["trial"]', encoding="utf-8")
        elif stage.module == "offstap.stages.s05_parameters":
            (run_dir / "alignment_parameters.json").write_text('{"trial": {}}', encoding="utf-8")
        elif stage.module == "offstap.stages.s08_metrics":
            (run_dir / "14_metrics" / "per_trial_metrics.csv").write_text(
                "accepted_for_hypothesis_tests\n1\n", encoding="utf-8"
            )
        elif stage.module == "offstap.stages.s09_comparisons":
            path = run_dir / "15_hypotheses" / "H1" / "h1_trial_pairs.csv"
            path.write_text("trial_id,delta_rmse_mm\ntrial,1.0\n", encoding="utf-8")
        elif stage.module == "offstap.stages.s16b_statistics":
            raise RuntimeError("synthetic statistics failure")

    monkeypatch.setattr("offstap.runner.run_stage", fake_run_stage)

    with pytest.raises(RuntimeError, match="synthetic statistics failure"):
        run_core(_options(tmp_path, frozenset({"h1"})))

    status = json.loads((tmp_path / "run" / "run_status.json").read_text(encoding="utf-8"))
    assert status == {"status": "FAILED", "reason": "synthetic statistics failure"}
    assert called[-1] == "offstap.stages.s16b_statistics"


def test_baseline_only_run_executes_callable_producer_and_marks_pass(tmp_path, monkeypatch):
    """A selected baseline executes after core and clears the pending status."""
    # Break caught: baseline remains PARTIAL or its callable producer is skipped.
    from offstap.runner import run_core

    def fake_run_stage(stage, run_dir, env):
        if stage.module == "offstap.stages.s03_quality":
            (run_dir / "04_quality" / "accepted_trials.json").write_text('["trial"]', encoding="utf-8")
        elif stage.module == "offstap.stages.s05_parameters":
            (run_dir / "alignment_parameters.json").write_text('{"trial": {}}', encoding="utf-8")
        elif stage.module == "offstap.stages.s08_metrics":
            (run_dir / "14_metrics" / "per_trial_metrics.csv").write_text(
                "accepted_for_hypothesis_tests\n1\n", encoding="utf-8"
            )

    monkeypatch.setattr("offstap.runner.run_stage", fake_run_stage)

    def fake_run_baseline(source_run, output_dir):
        output_dir.mkdir()
        artifact = output_dir / "baseline_manifest.json"
        artifact.write_text('{"identity": "baseline"}', encoding="utf-8")
        trial = output_dir / "baseline_trial_metrics.csv"
        trial.write_text(
            "trial_id,baseline_rmse_mm,offstap_rmse_mm,delta_rmse_mm\ntrial,2.0,1.0,1.0\n",
            encoding="utf-8",
        )
        summary = output_dir / "baseline_summary.json"
        summary.write_text('[{"population":"planned_population","n":1}]', encoding="utf-8")
        return {"manifest": artifact, "trial_metrics": trial, "summary": summary}

    monkeypatch.setattr("offstap.baseline.run_baseline", fake_run_baseline)

    run_core(_options(tmp_path, frozenset({"baseline"})))

    status = json.loads((tmp_path / "run" / "run_status.json").read_text(encoding="utf-8"))
    assert status == {"status": "PASS", "reason": ""}
    assert json.loads(
        (tmp_path / "run" / "baseline" / "baseline_manifest.json").read_text(encoding="utf-8")
    ) == {"identity": "baseline"}


def test_presentation_without_generated_figure_fails_closed(tmp_path, monkeypatch):
    """A scheduled presentation product is not equivalent to a generated product."""
    # Break caught: a figures request reaches PASS/PARTIAL without a figure artifact.
    from offstap.runner import run_core

    def fake_run_stage(stage, run_dir, env):
        if stage.module == "offstap.stages.s03_quality":
            (run_dir / "04_quality" / "accepted_trials.json").write_text('["trial"]', encoding="utf-8")
        elif stage.module == "offstap.stages.s05_parameters":
            (run_dir / "alignment_parameters.json").write_text('{"trial": {}}', encoding="utf-8")
        elif stage.module == "offstap.stages.s08_metrics":
            (run_dir / "14_metrics" / "per_trial_metrics.csv").write_text(
                "accepted_for_hypothesis_tests\n1\n", encoding="utf-8"
            )
        elif stage.module == "offstap.stages.s09_comparisons":
            outputs = {
                "H1/h1_trial_pairs.csv": "trial_id,delta_rmse_mm\ntrial,1.0\n",
                "H2/h2_trial_comparison.csv": "trial_id,method,rmse_mm\ntrial,zoh,1.0\n",
                "H3/h3_calibration_mode_comparison.csv": "trial_id,calibration_mode,heldout_rmse_mm\ntrial,split,1.0\n",
            }
            for relative, content in outputs.items():
                (run_dir / "15_hypotheses" / relative).write_text(content, encoding="utf-8")

    monkeypatch.setattr("offstap.runner.run_stage", fake_run_stage)

    def fake_run_baseline(source_run, output_dir):
        output_dir.mkdir()
        artifact = output_dir / "baseline_manifest.json"
        artifact.write_text('{"identity": "baseline"}', encoding="utf-8")
        return {"manifest": artifact}

    monkeypatch.setattr("offstap.baseline.run_baseline", fake_run_baseline)
    monkeypatch.setattr("offstap.runner.STAGE_SOURCE_REQUIREMENTS", {})
    monkeypatch.setattr("offstap.runner.STAGE_OUTPUT_REQUIREMENTS", {})

    with pytest.raises(RuntimeError, match="figures"):
        run_core(_options(tmp_path, frozenset({"figures"})))

    status = json.loads((tmp_path / "run" / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "FAILED"
    assert status["reason"] == "selected result family has no generated artifacts: figures"


def test_core_only_run_marks_pass_after_core_stages_succeed(tmp_path, monkeypatch):
    """Core-only has no pending product producer after its stages finish."""
    # Break caught: an empty pending set is serialized as a partial-result reason.
    from offstap.runner import run_core

    def fake_run_stage(stage, run_dir, env):
        if stage.module == "offstap.stages.s03_quality":
            (run_dir / "04_quality" / "accepted_trials.json").write_text('["trial"]', encoding="utf-8")
        elif stage.module == "offstap.stages.s05_parameters":
            (run_dir / "alignment_parameters.json").write_text('{"trial": {}}', encoding="utf-8")
        elif stage.module == "offstap.stages.s08_metrics":
            (run_dir / "14_metrics" / "per_trial_metrics.csv").write_text(
                "accepted_for_hypothesis_tests\n1\n", encoding="utf-8"
            )

    monkeypatch.setattr("offstap.runner.run_stage", fake_run_stage)

    run_core(_options(tmp_path, frozenset({"core"})))

    status = json.loads((tmp_path / "run" / "run_status.json").read_text(encoding="utf-8"))
    assert status == {"status": "PASS", "reason": ""}


def test_no_eligible_population_stops_before_hypothesis_stages(tmp_path, monkeypatch):
    """An ineligible core population cannot invoke H statistics."""
    # Break caught: optional H stages run after the core quality gate rejected every trial.
    from offstap.runner import run_core
    from offstap.workflow import CORE_STAGES

    called: list[str] = []

    def fake_run_stage(stage, run_dir, env):
        called.append(stage.module)
        if stage.module == "offstap.stages.s03_quality":
            (run_dir / "04_quality" / "accepted_trials.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr("offstap.runner.run_stage", fake_run_stage)

    with pytest.raises(RuntimeError, match="No eligible trials"):
        run_core(_options(tmp_path, frozenset({"h1"})))

    assert called == [stage.module for stage in CORE_STAGES[:3]]
    assert "offstap.stages.s09_comparisons" not in called
    assert "offstap.stages.s16b_statistics" not in called
    status = json.loads((tmp_path / "run" / "run_status.json").read_text(encoding="utf-8"))
    assert status == {"status": "FAILED", "reason": "No eligible trials accepted by quality gate"}
    result_status = json.loads((tmp_path / "run" / "result_status.json").read_text(encoding="utf-8"))
    assert result_status == {
        "h1": {
            "status": "NOT_APPLICABLE",
            "reason": "No eligible trials accepted by quality gate",
        }
    }
