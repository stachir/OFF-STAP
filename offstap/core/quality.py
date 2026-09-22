"""
src/quality.py

Computes stream diagnostics and applies explicit quality gating.
"""

import os
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Tuple
from offstap.core.config import Config

def compute_stream_diagnostics(df: pd.DataFrame, time_col: str) -> Dict[str, Any]:
    if df.empty:
        return {}

    t = df[time_col].values
    dt = np.diff(t)

    # Handle single row case
    if len(dt) == 0:
        return {
            "valid_rows": len(df),
            "duration": 0.0,
            "median_dt": 0.0,
            "mean_dt": 0.0,
            "std_dt": 0.0,
            "eff_rate": 0.0,
            "irregularity_index": 0.0,
            "max_gap": 0.0
        }

    median_dt = np.median(dt)
    mean_dt = np.mean(dt)
    std_dt = np.std(dt)
    eff_rate = 1.0 / mean_dt if mean_dt > 0 else 0.0
    irregularity = std_dt / mean_dt if mean_dt > 0 else 0.0

    return {
        "valid_rows": len(df),
        "duration": t[-1] - t[0],
        "median_dt": float(median_dt),
        "mean_dt": float(mean_dt),
        "std_dt": float(std_dt),
        "eff_rate": float(eff_rate),
        "irregularity_index": float(irregularity),
        "max_gap": float(np.max(dt))
    }

def apply_quality_gates(cfg: Config, trial_id: str, ev3_diag: dict, vrmt_diag: dict) -> List[Dict[str, Any]]:
    """
    Applies rules from config and returns a list of rejection reasons.
    Empty list means accepted.
    """
    rejections = []
    q_cfg = cfg.quality

    # Check minimum rows
    min_rows = q_cfg.get("min_rows", 10)
    if ev3_diag.get("valid_rows", 0) < min_rows:
        rejections.append({
            "trial_id": trial_id, "stage": "quality_gate",
            "reason": "too_few_ev3_rows", "value": ev3_diag.get("valid_rows", 0),
            "threshold": min_rows, "config_id": cfg.config_id
        })
    if vrmt_diag.get("valid_rows", 0) < min_rows:
        rejections.append({
            "trial_id": trial_id, "stage": "quality_gate",
            "reason": "too_few_vrmt_rows", "value": vrmt_diag.get("valid_rows", 0),
            "threshold": min_rows, "config_id": cfg.config_id
        })

    # Check minimum duration
    min_duration = q_cfg.get("min_duration_s", 5.0)
    if ev3_diag.get("duration", 0) < min_duration:
        rejections.append({
            "trial_id": trial_id, "stage": "quality_gate",
            "reason": "insufficient_ev3_duration", "value": ev3_diag.get("duration", 0),
            "threshold": min_duration, "config_id": cfg.config_id
        })

    # Check gaps
    max_gap = q_cfg.get("max_timestamp_gap_s", 2.0)
    if ev3_diag.get("max_gap", 0) > max_gap:
        rejections.append({
            "trial_id": trial_id, "stage": "quality_gate",
            "reason": "excessive_ev3_gap", "value": ev3_diag.get("max_gap", 0),
            "threshold": max_gap, "config_id": cfg.config_id
        })
    if vrmt_diag.get("max_gap", 0) > max_gap:
        rejections.append({
            "trial_id": trial_id, "stage": "quality_gate",
            "reason": "excessive_vrmt_gap", "value": vrmt_diag.get("max_gap", 0),
            "threshold": max_gap, "config_id": cfg.config_id
        })

    # Check irregularity
    max_irreg = q_cfg.get("max_irregularity_index", 2.0)
    if ev3_diag.get("irregularity_index", 0) > max_irreg:
        rejections.append({
            "trial_id": trial_id, "stage": "quality_gate",
            "reason": "excessive_ev3_irregularity", "value": ev3_diag.get("irregularity_index", 0),
            "threshold": max_irreg, "config_id": cfg.config_id
        })

    return rejections
