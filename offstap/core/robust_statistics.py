"""
src/robust_statistics.py

Computes robust aggregated statistics (Median, MAD, IQR) for metrics.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List

def compute_robust_statistics(metrics_list: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    """
    Computes Median, MAD, and IQR for a list of dictionaries given a specific key.
    """
    values = [m[key] for m in metrics_list if key in m and m["status"] == "success"]

    if not values:
        return {"median": 0.0, "mad": 0.0, "iqr": 0.0, "min": 0.0, "max": 0.0}

    arr = np.array(values)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    q75, q25 = np.percentile(arr, [75 ,25])
    iqr = q75 - q25

    return {
        "median": float(median),
        "mad": float(mad),
        "iqr": float(iqr),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }
