"""Input validation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pandas.api.types import is_bool_dtype, is_numeric_dtype

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


def normalize_columns(
    value: str | Sequence[str],
    *,
    parameter: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Return a validated tuple of column names."""
    columns = (value,) if isinstance(value, str) else tuple(value)
    if not columns and not allow_empty:
        raise ValueError(f"{parameter} must name at least one column.")
    if any(not isinstance(column, str) or not column for column in columns):
        raise ValueError(f"{parameter} must contain non-empty column names.")
    if len(columns) != len(set(columns)):
        raise ValueError(f"{parameter} cannot contain duplicate column names.")
    return columns


def require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    frame_name: str,
) -> None:
    """Require a frame to contain every named column."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        formatted = ", ".join(repr(column) for column in missing)
        raise ValueError(f"{frame_name} is missing required columns: {formatted}.")


def require_complete_keys(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    frame_name: str,
) -> None:
    """Reject null values in join or grouping keys."""
    if frame.loc[:, list(columns)].isna().any(axis=None):
        formatted = ", ".join(repr(column) for column in columns)
        raise ValueError(f"{frame_name} has null values in key columns: {formatted}.")


def require_unique_keys(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    frame_name: str,
) -> None:
    """Reject duplicate key combinations."""
    if frame.duplicated(subset=list(columns), keep=False).any():
        formatted = ", ".join(repr(column) for column in columns)
        raise ValueError(
            f"{frame_name} has duplicate rows for key columns: {formatted}."
        )


def numeric_values(
    frame: pd.DataFrame,
    column: str,
    *,
    frame_name: str,
) -> np.ndarray:
    """Return a numeric column as float values."""
    series = frame[column]
    if series.empty:
        return np.array([], dtype=float)
    if is_bool_dtype(series.dtype) or not is_numeric_dtype(series.dtype):
        raise ValueError(f"{frame_name} column {column!r} must be numeric.")
    return series.to_numpy(dtype=float, na_value=np.nan)
