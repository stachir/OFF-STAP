"""Isolated execution primitives for the reopened baseline comparator.

This module deliberately does not participate in the frozen H1/H2/H3 paths.
It validates the composite-arm contract and scores the two resulting aligned
frames only after exact finite evaluation support is established.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from offstap.core.support import intersect_support


@dataclass(frozen=True)
class ArmProvenance:
    arm: str
    spatial_policy: str
    temporal_policy: str
    clock_policy: str
    representation_policy: str


@dataclass(frozen=True)
class PairedScore:
    baseline_rmse_mm: float
    offstap_rmse_mm: float
    delta_rmse_mm: float
    common_support_count: int
    common_support_hash: str
    common_support_keys: tuple[int | str, ...]


def validate_arm_provenance(baseline: ArmProvenance, offstap: ArmProvenance) -> None:
    """Reject aliases and policy substitutions before numerical scoring."""
    expected_baseline = (
        "baseline", "full_trajectory_rigid_se2", "onset_only", "constant_offset", "zoh"
    )
    expected_offstap = (
        "offstap", "calibration_evaluation_split", "crosscorr_refined",
        "affine_or_constant_fallback", "natural_cubic",
    )
    actual_baseline = (
        baseline.arm, baseline.spatial_policy, baseline.temporal_policy,
        baseline.clock_policy, baseline.representation_policy,
    )
    actual_offstap = (
        offstap.arm, offstap.spatial_policy, offstap.temporal_policy,
        offstap.clock_policy, offstap.representation_policy,
    )
    if actual_baseline != expected_baseline:
        raise ValueError(f"invalid baseline comparator policy: {actual_baseline!r}")
    if actual_offstap != expected_offstap:
        raise ValueError(f"invalid OFF-STAP comparator policy: {actual_offstap!r}")
    if actual_baseline[1:] == actual_offstap[1:]:
        raise ValueError("baseline and OFF-STAP comparator policies are aliased")


def _finite_evaluation(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "source_row_index", "partition", "odometry_x", "odometry_y",
        "vrmt_aligned_x", "vrmt_aligned_y",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"aligned frame lacks required columns: {missing}")
    selected = frame.loc[
        frame["partition"].astype(str).str.lower().isin({"evaluation", "eval", "heldout", "held_out", "test"})
    ].copy()
    values = selected[["odometry_x", "odometry_y", "vrmt_aligned_x", "vrmt_aligned_y"]]
    finite = np.isfinite(values.apply(pd.to_numeric, errors="coerce").to_numpy(float)).all(axis=1)
    return selected.loc[finite].copy()


def _ordered_common_keys(common: pd.DataFrame) -> tuple[int | str, ...]:
    values: Iterable[object] = common["source_row_index"].tolist()
    return tuple(int(value) if isinstance(value, (int, np.integer)) else value for value in values)


def _rmse_mm(frame: pd.DataFrame) -> float:
    dx = pd.to_numeric(frame["odometry_x"], errors="coerce").to_numpy(float) - pd.to_numeric(frame["vrmt_aligned_x"], errors="coerce").to_numpy(float)
    dy = pd.to_numeric(frame["odometry_y"], errors="coerce").to_numpy(float) - pd.to_numeric(frame["vrmt_aligned_y"], errors="coerce").to_numpy(float)
    return float(np.sqrt(np.mean(dx * dx + dy * dy)) * 1000.0)


def canonical_offstap_metric_matches(offstap: pd.DataFrame, canonical_rmse_mm: float) -> bool:
    """Check a persisted frozen OFF-STAP frame against its canonical metric."""
    expected = float(canonical_rmse_mm)
    observed = _rmse_mm(_finite_evaluation(offstap))
    return bool(np.isclose(observed, expected, rtol=1e-12, atol=1e-12))


def score_paired_frames(baseline: pd.DataFrame, offstap: pd.DataFrame) -> PairedScore:
    """Compute both arm RMSEs over one exact finite persisted-key intersection."""
    filtered = {"baseline": _finite_evaluation(baseline), "offstap": _finite_evaluation(offstap)}
    support = intersect_support(filtered, ("source_row_index",))
    if not support.valid or support.common_count == 0:
        raise ValueError(f"invalid paired support: {support.status}")
    common_keys = set(_ordered_common_keys(support.common_keys))
    scored = {}
    for arm, frame in filtered.items():
        key = frame["source_row_index"].map(lambda value: int(value) if isinstance(value, (int, np.integer)) else value)
        scored[arm] = frame.loc[key.isin(common_keys)].copy()
    baseline_rmse = _rmse_mm(scored["baseline"])
    offstap_rmse = _rmse_mm(scored["offstap"])
    return PairedScore(
        baseline_rmse_mm=baseline_rmse,
        offstap_rmse_mm=offstap_rmse,
        delta_rmse_mm=baseline_rmse - offstap_rmse,
        common_support_count=support.common_count,
        common_support_hash=support.hash_sha256,
        common_support_keys=_ordered_common_keys(support.common_keys),
    )
