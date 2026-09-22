"""Corrected current-run baseline contract tests."""

import hashlib
import json
from pathlib import Path
import pickle
import shutil
import warnings

import pandas as pd
import pytest


@pytest.fixture
def two_arm_frames():
    proposed = pd.DataFrame({
        "source_row_index": [0, 1, 2],
        "partition": ["evaluation"] * 3,
        "odometry_x": [0.0, 1.0, 2.0],
        "odometry_y": [0.0] * 3,
        "vrmt_aligned_x": [0.0, 1.1, 2.1],
        "vrmt_aligned_y": [0.0] * 3,
    })
    return proposed.iloc[:2].copy(), proposed


def test_baseline_rejects_incomplete_canonical_support(two_arm_frames):
    """A paired intersection smaller than canonical support is excluded."""
    # Break caught: the adapter silently rescored OFF-STAP on baseline support.
    from offstap.core.baseline_comparator import score_paired_frames
    from offstap.baseline import canonical_support_complete

    baseline, proposed = two_arm_frames
    score = score_paired_frames(baseline, proposed)
    assert score.common_support_count == 2
    assert not canonical_support_complete(score, expected_count=3)


def test_baseline_accepts_complete_canonical_support(two_arm_frames):
    """A complete exact finite intersection remains eligible."""
    # Break caught: equality at the canonical support boundary is rejected.
    from offstap.core.baseline_comparator import score_paired_frames
    from offstap.baseline import canonical_support_complete

    baseline, proposed = two_arm_frames
    score = score_paired_frames(baseline, proposed)
    assert canonical_support_complete(score, expected_count=2)


def test_duplicate_baseline_key_is_not_valid_support(two_arm_frames):
    """Duplicate finite source identities fail closed before scoring."""
    # Break caught: duplicate rows inflate or distort the paired baseline score.
    from offstap.core.baseline_comparator import score_paired_frames

    baseline, proposed = two_arm_frames
    duplicated = pd.concat([baseline, baseline.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="support"):
        score_paired_frames(duplicated, proposed)


def test_complete_score_has_exact_public_support_hash_and_metric(two_arm_frames):
    """The copied scorer retains exact support identity and canonical RMSE equality."""
    # Break caught: support serialization or canonical metric units drift.
    from offstap.core.baseline_comparator import (
        canonical_offstap_metric_matches,
        score_paired_frames,
    )

    _, proposed = two_arm_frames
    score = score_paired_frames(proposed, proposed)
    expected_hash = hashlib.sha256(b"0\n1\n2").hexdigest()
    expected_rmse_mm = 81.64965809277268

    assert score.common_support_hash == expected_hash
    assert score.common_support_keys == (0, 1, 2)
    assert canonical_offstap_metric_matches(proposed, expected_rmse_mm)
    assert not canonical_offstap_metric_matches(proposed, expected_rmse_mm + 1e-6)


def _write_current_run(
    source_run: Path,
    proposed: pd.DataFrame,
    *,
    declared_count: int = 3,
    canonical_rmse_m: float = 0.08164965809277268,
) -> None:
    trial_id = "trial_public"
    package_root = Path(__file__).resolve().parents[1]
    shutil.copyfile(package_root / "offstap" / "config" / "default.yaml", source_run / "config_used.yaml")

    csvs = {
        "01_manifest/pair_manifest.csv": pd.DataFrame({
            "trial_id": [trial_id], "dataset_tier": ["planned_405"],
        }),
        "09_temporal/crosscorr_report.csv": pd.DataFrame({
            "trial_id": [trial_id], "b_onset": [0.0], "b_final": [0.1],
        }),
        "10_clock_model/fitted_parameters.csv": pd.DataFrame({
            "trial_id": [trial_id], "selected_model": ["constant_offset"], "scale_a": [1.0],
        }),
        "07_spatial/spatial_calibration_parameters.csv": pd.DataFrame({
            "trial_id": [trial_id], "theta_rad": [0.0], "tx": [0.0], "ty": [0.0],
        }),
        "14_metrics/evaluation_metrics.csv": pd.DataFrame({
            "trial_id": [trial_id],
            "ate_rmse": [canonical_rmse_m],
            "num_samples": [declared_count],
            "status": ["success"],
        }),
    }
    for relative, frame in csvs.items():
        path = source_run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    canonical = source_run / "13_integrated_logs" / "ct_reference" / f"{trial_id}_public.parquet"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    proposed.to_parquet(canonical, index=False)
    for relative in (
        f"03_cleaned/ev3/{trial_id}_ev3_cleaned.parquet",
        f"03_cleaned/vrmt/{trial_id}_vrmt_cleaned.parquet",
    ):
        path = source_run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"placeholder": [1]}).to_parquet(path, index=False)
    spline = source_run / "11_continuous_time_reference" / "models" / f"{trial_id}_spline.pkl"
    spline.parent.mkdir(parents=True, exist_ok=True)
    with spline.open("wb") as stream:
        pickle.dump({"valid": True, "t": [0.0], "x": [0.0], "y": [0.0]}, stream)


def test_run_baseline_writes_release_neutral_artifacts_and_exact_contract(
    tmp_path, monkeypatch, two_arm_frames,
):
    """A valid current run produces public artifacts with approved arm provenance."""
    # Break caught: historical labels/policies leak or a valid trial is omitted.
    from offstap import baseline as baseline_module

    _, proposed = two_arm_frames
    source_run = tmp_path / "current-run"
    source_run.mkdir()
    _write_current_run(source_run, proposed)

    monkeypatch.setattr(
        baseline_module,
        "baseline_frame",
        lambda source_run, trial_id, offstap, b_onset, cfg: (
            proposed.copy(), {"theta_rad": 0.2, "tx": 1.0, "ty": 2.0},
        ),
    )

    paths = baseline_module.run_baseline(source_run, source_run / "baseline")

    assert set(paths) == {"trial_metrics", "rejections", "summary", "manifest"}
    assert {path.name for path in paths.values()} == {
        "baseline_trial_metrics.csv",
        "baseline_rejections.csv",
        "baseline_summary.json",
        "baseline_manifest.json",
    }
    trials = pd.read_csv(paths["trial_metrics"])
    assert trials[["trial_id", "common_support_count", "common_support_hash"]].to_dict("records") == [{
        "trial_id": "trial_public",
        "common_support_count": 3,
        "common_support_hash": hashlib.sha256(b"0\n1\n2").hexdigest(),
    }]
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["identity"] == "baseline"
    assert manifest["source_run"] == "."
    assert manifest["contract"]["baseline"] == {
        "arm": "baseline",
        "spatial_policy": "full_trajectory_rigid_se2",
        "temporal_policy": "onset_only",
        "clock_policy": "constant_offset",
        "representation_policy": "zoh",
    }
    assert manifest["contract"]["offstap"] == {
        "arm": "offstap",
        "spatial_policy": "calibration_evaluation_split",
        "temporal_policy": "crosscorr_refined",
        "clock_policy": "affine_or_constant_fallback",
        "representation_policy": "natural_cubic",
    }
    assert "git" not in manifest
    assert all("CORRECTION" not in path.name for path in paths.values())


def test_run_baseline_reason_codes_incomplete_support_without_plausible_result(
    tmp_path, monkeypatch, two_arm_frames,
):
    """Incomplete support yields an explicit rejection and zero paired rows."""
    # Break caught: an incomplete arm emits a normal-looking paired estimate.
    from offstap import baseline as baseline_module

    incomplete, proposed = two_arm_frames
    source_run = tmp_path / "current-run"
    source_run.mkdir()
    _write_current_run(source_run, proposed)
    monkeypatch.setattr(
        baseline_module,
        "baseline_frame",
        lambda source_run, trial_id, offstap, b_onset, cfg: (
            incomplete.copy(), {"theta_rad": 0.2, "tx": 1.0, "ty": 2.0},
        ),
    )

    paths = baseline_module.run_baseline(source_run, source_run / "baseline")

    assert pd.read_csv(paths["trial_metrics"]).empty
    assert pd.read_csv(paths["rejections"]).to_dict("records") == [{
        "trial_id": "trial_public",
        "dataset_tier": "planned_405",
        "reason": "baseline_common_support_does_not_cover_canonical_offstap_support",
        "baseline_common_support_count": 2,
        "canonical_offstap_support_count": 3,
    }]
    summaries = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert [row["paired_n"] for row in summaries] == [0, 0, 0]
    assert all(row["mean_delta_mm"] is None for row in summaries)


def test_run_baseline_rejects_declared_count_smaller_than_actual_canonical_support(
    tmp_path, monkeypatch,
):
    """A stale metric count cannot redefine the canonical evaluation support."""
    # Break caught: declared count 3 lets three-arm coverage pass on a four-key frame.
    from offstap import baseline as baseline_module

    proposed = pd.DataFrame({
        "source_row_index": [0, 1, 2, 3],
        "partition": ["evaluation"] * 4,
        "odometry_x": [0.0, 1.0, 2.0, 3.0],
        "odometry_y": [0.0] * 4,
        "vrmt_aligned_x": [0.0, 1.1, 2.1, 3.1],
        "vrmt_aligned_y": [0.0] * 4,
    })
    source_run = tmp_path / "current-run"
    source_run.mkdir()
    _write_current_run(
        source_run,
        proposed,
        declared_count=3,
        canonical_rmse_m=0.08660254037844388,
    )
    monkeypatch.setattr(
        baseline_module,
        "baseline_frame",
        lambda source_run, trial_id, offstap, b_onset, cfg: (
            proposed.iloc[:3].copy(), {"theta_rad": 0.2, "tx": 1.0, "ty": 2.0},
        ),
    )

    paths = baseline_module.run_baseline(source_run, source_run / "baseline")

    assert pd.read_csv(paths["trial_metrics"]).empty
    rejection = pd.read_csv(paths["rejections"]).iloc[0]
    assert rejection["reason"] == "canonical_offstap_support_count_mismatch"
    assert pd.isna(rejection["baseline_common_support_count"])
    assert rejection["canonical_offstap_support_count"] == 4


def test_run_baseline_rejects_nonfinite_score_from_canonicalized_string_keys(
    tmp_path, monkeypatch, two_arm_frames,
):
    """Canonicalized key support cannot authorize non-finite paired metrics."""
    # Break caught: valid canonical key count/hash masks empty original-key scoring rows.
    from offstap import baseline as baseline_module

    _, proposed = two_arm_frames
    proposed = proposed.assign(source_row_index=["0", "1", "2"])
    source_run = tmp_path / "current-run"
    source_run.mkdir()
    _write_current_run(source_run, proposed)
    monkeypatch.setattr(
        baseline_module,
        "baseline_frame",
        lambda source_run, trial_id, offstap, b_onset, cfg: (
            proposed.copy(), {"theta_rad": 0.2, "tx": 1.0, "ty": 2.0},
        ),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        paths = baseline_module.run_baseline(source_run, source_run / "baseline")

    assert pd.read_csv(paths["trial_metrics"]).empty
    assert not [warning for warning in caught if warning.category is RuntimeWarning]
    rejection = pd.read_csv(paths["rejections"]).iloc[0]
    assert rejection["reason"] == "paired_score_metrics_non_finite"
    assert rejection["baseline_common_support_count"] == 3
    assert rejection["canonical_offstap_support_count"] == 3


def test_run_baseline_refuses_nonempty_destination(tmp_path):
    """A prior baseline result is never silently overwritten."""
    # Break caught: rerunning baseline replaces existing evidence in place.
    from offstap.baseline import run_baseline

    destination = tmp_path / "baseline"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="nonempty"):
        run_baseline(tmp_path / "source", destination)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_baseline_is_callable_after_core_but_not_a_subprocess_stage():
    """The callable producer is registered at the core/output boundary."""
    # Break caught: baseline is omitted, ordered early, or added to build_plan.
    from offstap.workflow import (
        BASELINE_PRODUCER,
        CORE_STAGES,
        build_plan,
        callable_producers_after,
    )

    assert BASELINE_PRODUCER.after_module == CORE_STAGES[-1].module
    assert callable_producers_after(CORE_STAGES[-1].module, frozenset({"baseline"})) == (
        BASELINE_PRODUCER,
    )
    assert build_plan(frozenset({"baseline"})) == CORE_STAGES
