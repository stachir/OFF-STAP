"""
src/schema.py

Functions for mapping and standardizing column names based on the configuration.
"""
from typing import Dict, List
import pandas as pd
from offstap.core.config import Config

def standardize_columns(df: pd.DataFrame, aliases: Dict[str, str]) -> pd.DataFrame:
    """
    Renames columns in the DataFrame based on the provided aliases map.
    Converts remaining columns to lowercase snake_case.
    """
    # 1. Apply aliases
    df = df.rename(columns=aliases)

    # 2. Normalize to snake_case for anything unmapped
    def to_snake(name):
        return name.strip().lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')

    df.columns = [to_snake(c) for c in df.columns]

    return df

def generate_schema_report(df: pd.DataFrame, trial_id: str, stream_type: str) -> dict:
    """
    Generates a schema summary for the DataFrame.
    """
    return {
        "trial_id": trial_id,
        "stream_type": stream_type,
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "num_rows": len(df),
        "missing_values": df.isnull().sum().to_dict()
    }
