"""Exact finite-support validation for paired comparisons.

Support is defined by stable source identities, not by positional row numbers
after filtering or by the timestamps produced by an interpolator.  This
module keeps validation evidence with the result so an invalid arm cannot be
silently dropped before an intersection is computed.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _as_columns(key_columns: Sequence[str]) -> tuple[str, ...]:
    columns = tuple(str(column) for column in key_columns)
    if not columns:
        raise ValueError("key_columns must contain at least one column")
    if any(not column for column in columns):
        raise ValueError("key_columns must contain non-empty column names")
    if len(set(columns)) != len(columns):
        raise ValueError("key_columns must not contain duplicates")
    return columns


def _canonical_value(value: Any) -> tuple[Any, bool, str | None]:
    """Return an exact, hashable key value and whether it is finite.

    Source identities are normally integer nanoseconds or integer row
    indexes. Integer-valued numerics are canonicalized deterministically;
    non-integral and precision-unsafe numerics are rejected rather than
    rounded into a potentially colliding source key.
    """
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None, False, "non_finite"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value), True, None
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return None, False, "non_finite"
        if not number.is_integer():
            return None, False, "non_integral"
        # IEEE-754 doubles cannot preserve adjacent integer identities above
        # 2**53, which is common for nanosecond epoch keys.
        if abs(number) > 2**53:
            return None, False, "unsafe_integer"
        return int(number), True, None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
            return None, False, "non_finite"
        try:
            number = Decimal(text)
        except ValueError:
            return text, True, None
        except InvalidOperation:
            return text, True, None
        if not number.is_finite():
            return None, False, "non_finite"
        if number != number.to_integral_value():
            return None, False, "non_integral"
        integer = int(number)
        if abs(integer) > 2**63 - 1:
            return None, False, "unsafe_integer"
        return integer, True, None
    try:
        if pd.isna(value):
            return None, False, "non_finite"
    except (TypeError, ValueError):
        pass
    return str(value), True, None


def _canonical_frame(df: pd.DataFrame, columns: tuple[str, ...]) -> tuple[pd.DataFrame, list[int], dict[str, list[int]]]:
    if any(column not in df.columns for column in columns):
        return pd.DataFrame(columns=list(columns)), list(range(len(df))), {"missing": list(range(len(df)))}
    rows: list[tuple[Any, ...]] = []
    non_finite_rows: list[int] = []
    invalid_reasons: dict[str, list[int]] = {}
    for row_position, values in enumerate(df.loc[:, list(columns)].itertuples(index=False, name=None)):
        canonical: list[Any] = []
        finite = True
        for value in values:
            converted, is_finite, issue = _canonical_value(value)
            canonical.append(converted)
            finite = finite and is_finite
            if issue:
                invalid_reasons.setdefault(issue, []).append(row_position)
        if finite:
            rows.append(tuple(canonical))
        elif row_position in invalid_reasons.get("non_finite", []):
            non_finite_rows.append(row_position)
    return pd.DataFrame(rows, columns=list(columns)), non_finite_rows, invalid_reasons


def _sort_key(values: tuple[Any, ...]) -> tuple[tuple[int, Any], ...]:
    # Numeric identities sort numerically.  The fallback keeps mixed/string
    # keys deterministic without relying on Python's incomparable types.
    result = []
    for value in values:
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            result.append((0, int(value)))
        else:
            result.append((1, str(value)))
    return tuple(result)


def _row_token(values: tuple[Any, ...]) -> str:
    if len(values) == 1:
        return str(values[0])
    # Length-prefix fields so tabs/newlines/control characters in textual
    # identities cannot make distinct multi-column rows hash alike.
    return "".join(
        f"{len(text)}:{text}" for text in (str(value) for value in values)
    )


@dataclass(frozen=True)
class SupportValidation:
    """Validation and preserved evidence for one comparison arm."""

    key_columns: tuple[str, ...]
    status: str
    valid: bool
    total_count: int
    finite_count: int
    non_finite_count: int
    duplicate_key_count: int
    duplicate_row_count: int
    canonical_keys: pd.DataFrame = field(compare=False, repr=False)
    non_finite_rows: tuple[int, ...] = ()
    duplicate_keys: tuple[tuple[Any, ...], ...] = ()
    invalid_rows: tuple[int, ...] = ()
    invalid_reason_counts: dict[str, int] = field(default_factory=dict, compare=False)
    invalid_reason_rows: dict[str, tuple[int, ...]] = field(default_factory=dict, compare=False)

    @property
    def is_valid(self) -> bool:
        return self.valid

    @property
    def arm_valid_count(self) -> int:
        return self.unique_count

    @property
    def valid_count(self) -> int:
        return self.unique_count

    @property
    def finite_row_count(self) -> int:
        return self.finite_count

    @property
    def unique_count(self) -> int:
        return int(len(set(self.canonical_keys.itertuples(index=False, name=None))))

    @property
    def duplicate_count(self) -> int:
        return self.duplicate_key_count

    @property
    def support_keys(self) -> pd.DataFrame:
        return self.canonical_keys.copy()


@dataclass(frozen=True)
class CommonSupport:
    """Exact intersection and evidence for a set of comparison arms."""

    key_columns: tuple[str, ...]
    common_keys: pd.DataFrame
    arm_counts: dict[str, int]
    common_count: int
    status: str
    hash_sha256: str
    validations: dict[str, SupportValidation]
    arm_row_counts: dict[str, int] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.status == "valid"

    @property
    def arm_valid_count(self) -> dict[str, int]:
        return dict(self.arm_counts)

    @property
    def arm_valid_counts(self) -> dict[str, int]:
        return dict(self.arm_counts)

    @property
    def common_support_count(self) -> int:
        return self.common_count

    @property
    def common_support_hash_sha256(self) -> str:
        return self.hash_sha256

    @property
    def common_support_hash(self) -> str:
        return self.hash_sha256

    @property
    def n_common(self) -> int:
        return self.common_count

    @property
    def duplicate_key_counts(self) -> dict[str, int]:
        return {
            name: validation.duplicate_key_count
            for name, validation in self.validations.items()
        }

    @property
    def support_status(self) -> str:
        return self.status

    @property
    def counts(self) -> dict[str, int]:
        return dict(self.arm_counts)


def validate_arm_support(df: pd.DataFrame, key_columns: Sequence[str]) -> SupportValidation:
    """Validate finite, unique source keys for one arm.

    Duplicate detection is performed on canonical keys and reports both the
    number of duplicated key values and affected rows.  No rows are removed;
    ``canonical_keys`` contains only finite keys while the row evidence is
    retained in ``non_finite_rows`` and duplicate metadata.
    """
    columns = _as_columns(key_columns)
    total = len(df)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        rows = tuple(range(total))
        return SupportValidation(
            columns, "missing_key_column", False, total, 0, total, 0, 0,
            pd.DataFrame(columns=list(columns)), rows, (), rows,
            {"missing": total}, {"missing": rows}
        )
    canonical, non_finite_rows, invalid_reasons = _canonical_frame(df, columns)
    if canonical.empty:
        duplicate_keys: tuple[tuple[Any, ...], ...] = ()
        duplicate_key_count = duplicate_row_count = 0
    else:
        tuples = list(canonical.itertuples(index=False, name=None))
        counts = pd.Series(tuples, dtype=object).value_counts()
        duplicate_keys = tuple(sorted((tuple(key) if isinstance(key, tuple) else (key,) for key in counts[counts > 1].index), key=_sort_key))
        duplicate_key_count = len(duplicate_keys)
        duplicate_row_count = int(sum(int(counts[key]) for key in counts[counts > 1].index))
    if duplicate_key_count:
        status = "duplicate_key"
    elif invalid_reasons.get("unsafe_integer"):
        status = "unsafe_integer_key"
    elif invalid_reasons.get("non_integral"):
        status = "non_integral_key"
    elif non_finite_rows:
        status = "non_finite_key"
    else:
        status = "empty" if total == 0 else "valid"
    valid = status == "valid"
    invalid_rows = tuple(sorted({row for rows in invalid_reasons.values() for row in rows}))
    reason_counts = {reason: len(rows) for reason, rows in invalid_reasons.items()}
    reason_rows = {reason: tuple(rows) for reason, rows in invalid_reasons.items()}
    return SupportValidation(
        columns, status, valid, total, len(canonical), len(non_finite_rows),
        duplicate_key_count, duplicate_row_count, canonical,
        tuple(non_finite_rows), duplicate_keys, invalid_rows, reason_counts,
        reason_rows,
    )


def _empty_common(columns: tuple[str, ...], validations: dict[str, SupportValidation], status: str, arm_rows: dict[str, int]) -> CommonSupport:
    return CommonSupport(columns, pd.DataFrame(columns=list(columns)), {name: value.unique_count for name, value in validations.items()}, 0, status, hashlib.sha256(b"").hexdigest(), validations, arm_rows)


def intersect_support(arms: Mapping[str, pd.DataFrame], key_columns: Sequence[str]) -> CommonSupport:
    """Return exact finite intersection, rejecting invalid arms first."""
    columns = _as_columns(key_columns)
    ordered_arms = {str(name): frame for name, frame in sorted(arms.items(), key=lambda item: str(item[0]))}
    if not ordered_arms:
        return _empty_common(columns, {}, "no_arms", {})
    validations = {name: validate_arm_support(frame, columns) for name, frame in ordered_arms.items()}
    row_counts = {name: len(frame) for name, frame in ordered_arms.items()}
    invalid = next((validation.status for validation in validations.values() if not validation.valid), None)
    if invalid:
        return _empty_common(columns, validations, invalid, row_counts)
    sets = [set(validation.canonical_keys.itertuples(index=False, name=None)) for validation in validations.values()]
    common = set.intersection(*sets) if sets else set()
    sorted_common = sorted(common, key=_sort_key)
    common_keys = pd.DataFrame(sorted_common, columns=list(columns))
    payload = "\n".join(_row_token(row) for row in sorted_common).encode("utf-8")
    status = "valid" if sorted_common else "no_common_support"
    return CommonSupport(
        columns,
        common_keys,
        {name: validation.unique_count for name, validation in validations.items()},
        len(sorted_common),
        status,
        hashlib.sha256(payload).hexdigest(),
        validations,
        row_counts,
    )


__all__ = ["SupportValidation", "CommonSupport", "validate_arm_support", "intersect_support"]
