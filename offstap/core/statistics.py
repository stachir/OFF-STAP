"""
src/statistics.py

Statistical comparisons between different configurations.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from scipy.stats import friedmanchisquare
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Any, List, Sequence


@dataclass
class FriedmanResult:
    """Complete-block Friedman result and auditable pairwise contrasts."""

    n: int
    methods: tuple[str, ...]
    statistic: float
    p_value: float
    kendall_w: float
    complete_block_frame: pd.DataFrame
    pairwise: pd.DataFrame
    support_column: str | None = None


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return Holm step-down adjusted p-values in input order."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    order = idx[np.argsort(p[finite], kind="mergesort")]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * float(p[i]))
        out[i] = min(1.0, running)
    return out


def signed_rank_audit(
    values_a: np.ndarray,
    values_b: np.ndarray,
    trial_ids: List[str] | np.ndarray,
    *,
    zero_method: str = "zsplit",
    alternative: str = "two-sided",
    method: str = "auto",
    correction: bool = False,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute a fully reconcilable paired Wilcoxon audit.

    Differences are defined as ``values_a - values_b``. With ``zsplit``, zero
    ranks are divided equally between positive and negative rank sums, so every
    pair participates in the same rank total used by the rank-biserial effect.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    ids = np.asarray(trial_ids, dtype=str)
    if not (len(a) == len(b) == len(ids)):
        raise ValueError("values_a, values_b, and trial_ids must have equal length")
    # These arguments are part of the frozen correction-cycle estimand.  Do
    # not silently accept caller changes: a different SciPy configuration is
    # a different test and would invalidate the persisted provenance string.
    if zero_method != "zsplit":
        raise ValueError("the correction-cycle statistical contract freezes zero_method='zsplit'")
    if alternative != "two-sided":
        raise ValueError("the correction-cycle statistical contract freezes alternative='two-sided'")
    if method != "auto":
        raise ValueError("the correction-cycle statistical contract freezes method='auto'")
    if bool(correction):
        raise ValueError("the correction-cycle statistical contract freezes correction=False")
    finite = np.isfinite(a) & np.isfinite(b)
    a, b, ids = a[finite], b[finite], ids[finite]
    diff = a - b
    ranks = rankdata(np.abs(diff), method="average") if len(diff) else np.array([])
    positive = diff > 0
    negative = diff < 0
    zero = diff == 0
    plus = np.where(positive, ranks, np.where(zero, ranks / 2.0 if zero_method == "zsplit" else 0.0, 0.0))
    minus = np.where(negative, ranks, np.where(zero, ranks / 2.0 if zero_method == "zsplit" else 0.0, 0.0))
    w_plus = float(plus.sum())
    w_minus = float(minus.sum())
    denom = w_plus + w_minus
    rank_biserial = float((w_plus - w_minus) / denom) if denom else 0.0

    all_zero = bool(len(diff) and np.all(zero))
    if len(diff) < 2:
        # SciPy raises for n=1 and returns NaN for n=0.  Keep the audit row
        # and all rank accounting, but mark the inferential result honestly.
        statistic, p_value = (float(min(w_plus, w_minus)) if len(diff) else np.nan), np.nan
        status = "no_finite_pairs" if len(diff) == 0 else ("insufficient_pairs_all_zero" if all_zero else "insufficient_pairs")
    else:
        try:
            result = wilcoxon(
                diff,
                zero_method=zero_method,
                correction=False,
                alternative="two-sided",
                method="auto",
            )
            statistic, p_value = float(result.statistic), float(result.pvalue)
            status = "all_zero_differences" if all_zero else "ok"
        except ValueError:
            statistic, p_value = np.nan, np.nan
            status = "test_error"

    if len(diff):
        i, j = np.triu_indices(len(diff))
        pseudomedian = float(np.median((diff[i] + diff[j]) / 2.0))
    else:
        pseudomedian = np.nan

    rows = pd.DataFrame({
        "trial_id": ids,
        "arm_a": a,
        "arm_b": b,
        "signed_difference_a_minus_b": diff,
        "zero_status": np.where(zero, "zero", "nonzero"),
        "sign": np.where(positive, "positive", np.where(negative, "negative", "zero")),
        "rank": ranks,
        "w_plus_contribution": plus,
        "w_minus_contribution": minus,
        "audit_status": status,
    })
    summary = {
        "n_total": int(len(diff)),
        "n_positive": int(positive.sum()),
        "n_negative": int(negative.sum()),
        "n_zero": int(zero.sum()),
        "W_plus": w_plus,
        "W_minus": w_minus,
        "reported_W": statistic,
        "p_value": p_value,
        "p_raw": p_value,
        "p_adjusted": np.nan,
        "p_family_adjusted": np.nan,
        "zero_method": zero_method,
        "alternative": alternative,
        "method": method,
        "continuity_correction": bool(correction),
        "rank_biserial": rank_biserial,
        "mean_difference": float(np.mean(diff)) if len(diff) else np.nan,
        "median_difference": float(np.median(diff)) if len(diff) else np.nan,
        "hodges_lehmann_pseudomedian": pseudomedian,
        "exact_function_call": (
            "scipy.stats.wilcoxon(diff, zero_method='zsplit', correction=False, "
            "alternative='two-sided', method='auto')"
        ),
        "scipy_version": __import__("scipy").__version__,
        "location_estimands": "mean_difference, median_difference, Hodges-Lehmann pseudomedian",
        "status": status,
    }
    return rows, summary


def signed_rank_delta_audit(
    deltas: np.ndarray,
    trial_ids: List[str] | np.ndarray,
    **kwargs,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Audit a persisted paired-difference vector without inventing arm values.

    ``signed_rank_audit`` remains the frozen implementation; this adapter
    supplies a zero reference solely for its algebra and removes synthetic arm
    columns from the persisted audit so Stage16b remains a pure delta consumer.
    """
    values = np.asarray(deltas, dtype=float)
    rows, summary = signed_rank_audit(values, np.zeros(len(values), dtype=float), trial_ids, **kwargs)
    return rows.drop(columns=["arm_a", "arm_b"], errors="ignore"), summary


def friedman_complete_blocks(
    frame: pd.DataFrame,
    methods: Sequence[str],
    *,
    trial_column: str = "trial_id",
    method_column: str = "method",
    value_column: str = "rmse_mm",
) -> FriedmanResult:
    """Run Friedman and all pairwise tests on complete repeated-measure blocks.

    A trial is eligible only when every requested method occurs exactly once,
    all values are finite, and (when present) all methods share one support hash.
    This prevents incomplete or mismatched-support rows from entering inference.
    """
    methods = tuple(str(m) for m in methods)
    if len(methods) < 2 or len(set(methods)) != len(methods):
        raise ValueError("methods must contain at least two unique method names")
    required = {trial_column, method_column, value_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    support_column = next(
        (c for c in ("common_support_hash", "support_hash", "common_support_hash_sha256") if c in frame.columns),
        None,
    )
    data = frame[frame[method_column].astype(str).isin(methods)].copy()
    data[method_column] = data[method_column].astype(str)
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    if "fit_or_resampling_status" in data.columns:
        data = data[data["fit_or_resampling_status"].astype(str).eq("success")]
    elif "status" in data.columns:
        data = data[data["status"].astype(str).isin(("success", "ok", "valid"))]

    eligible: list[str] = []
    for trial_id, group in data.groupby(trial_column, sort=True):
        if set(group[method_column]) != set(methods) or len(group) != len(methods):
            continue
        if not np.isfinite(group[value_column].to_numpy(float)).all():
            continue
        # A complete block is inferentially admissible only when persisted
        # support evidence is present under one of the recognized names.
        # Never infer common support from row counts alone.
        if support_column is None:
            continue
        if support_column is not None:
            hashes = group[support_column].astype(str)
            if hashes.isin(("", "nan", "None", "NaN")).any() or hashes.nunique(dropna=False) != 1:
                continue
        eligible.append(trial_id)

    complete = data[data[trial_column].isin(eligible)].copy()
    if complete.empty:
        wide = pd.DataFrame(columns=[trial_column, *methods])
        n = 0
        statistic, p_value, kendall_w = np.nan, np.nan, np.nan
    else:
        wide = complete.pivot(index=trial_column, columns=method_column, values=value_column)
        wide = wide.reindex(columns=list(methods)).dropna().reset_index()
        n = int(len(wide))
        if n:
            statistic, p_value = friedmanchisquare(*(wide[m].to_numpy(float) for m in methods))
            statistic, p_value = float(statistic), float(p_value)
            kendall_w = float(statistic / (n * (len(methods) - 1)))
        else:
            statistic, p_value, kendall_w = np.nan, np.nan, np.nan

    pair_rows: list[dict[str, Any]] = []
    pair_audits: list[tuple[pd.DataFrame, Dict[str, Any]]] = []
    for method_a, method_b in combinations(methods, 2):
        if n:
            audit, summary = signed_rank_audit(
                wide[method_a].to_numpy(float), wide[method_b].to_numpy(float), wide[trial_column].astype(str).to_numpy()
            )
            diff = audit["signed_difference_a_minus_b"].to_numpy(float)
            pair_audits.append((audit, summary))
            pair_rows.append({
                "contrast": f"{method_a} - {method_b}",
                "method_a": method_a,
                "method_b": method_b,
                "n": summary["n_total"],
                "mean_difference_mm": float(np.mean(diff)),
                "median_difference_mm": float(np.median(diff)),
                "IQR_or_MAD_mm": float(np.quantile(diff, .75) - np.quantile(diff, .25)),
                "n_positive": summary["n_positive"],
                "n_negative": summary["n_negative"],
                "n_zero": summary["n_zero"],
                "W_plus": summary["W_plus"],
                "W_minus": summary["W_minus"],
                "reported_W": summary["reported_W"],
                "rank_biserial": summary["rank_biserial"],
                "p_raw": summary["p_value"],
                "p_adjusted": np.nan,
                "p_family_adjusted": np.nan,
                "P95_abs_difference_mm": float(np.quantile(np.abs(diff), .95)),
                "zero_method": summary["zero_method"],
                "alternative": summary["alternative"],
                "method": summary["method"],
                "correction": summary["continuity_correction"],
                "status": summary["status"],
                "scipy_version": summary["scipy_version"],
                "exact_function_call": summary["exact_function_call"],
            })
        else:
            pair_rows.append({"contrast": f"{method_a} - {method_b}", "method_a": method_a,
                              "method_b": method_b, "n": 0, "p_raw": np.nan,
                              "p_adjusted": np.nan, "p_family_adjusted": np.nan,
                              "p_value": np.nan, "p_value_holm": np.nan,
                              "status": "insufficient_complete_blocks",
                              "zero_method": "zsplit", "alternative": "two-sided",
                              "method": "auto", "correction": False})
    pairwise = pd.DataFrame(pair_rows)
    if not pairwise.empty and "p_raw" in pairwise:
        pairwise["p_adjusted"] = _holm_adjust(pairwise["p_raw"].to_numpy(float))
        pairwise["p_family_adjusted"] = pairwise["p_adjusted"]
        pairwise["p_value"] = pairwise["p_raw"]
        pairwise["p_value_holm"] = pairwise["p_adjusted"]
        if "status" not in pairwise:
            pairwise["status"] = "ok"

    # Preserve the long complete-block representation for support auditing.
    if n:
        keep = [trial_column, method_column, value_column]
        if support_column:
            keep.append(support_column)
        complete = complete[keep].sort_values([trial_column, method_column]).reset_index(drop=True)
    return FriedmanResult(n=n, methods=methods, statistic=statistic, p_value=p_value,
                          kendall_w=kendall_w, complete_block_frame=complete,
                          pairwise=pairwise, support_column=support_column)

def compare_configurations(metrics_df: pd.DataFrame, base_config: str, test_config: str, metric: str = "ate_rmse") -> Dict[str, Any]:
    """
    Performs a Wilcoxon signed-rank test between two configurations
    on a paired dataset of trials.
    """
    base_data = metrics_df[metrics_df["config_id"] == base_config].set_index("trial_id")[metric]
    test_data = metrics_df[metrics_df["config_id"] == test_config].set_index("trial_id")[metric]

    # Inner join to get paired data
    common = pd.concat([base_data, test_data], axis=1, join="inner")
    common.columns = ["base", "test"]

    if len(common) < 3:
        return {"status": "insufficient_data"}

    diff = common["base"] - common["test"]

    # Test
    try:
        stat, p = wilcoxon(common["base"], common["test"])
    except ValueError:
        # E.g. zero differences
        stat, p = float('nan'), float('nan')

    median_diff = float(np.median(diff))

    return {
        "status": "success",
        "n_pairs": len(common),
        "median_diff": median_diff,
        "test_statistic": float(stat),
        "p_value": float(p),
        "significant_05": p < 0.05 if not np.isnan(p) else False
    }
