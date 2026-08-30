"""Pure calculations for the China 2019 heatwave validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class HeatwaveAnomaly:
    """Heatwave-day results relative to the 1986-2005 baseline.

    Attributes:
        threshold: Per-cell daily maximum temperature threshold.
        baseline_days_by_year: Per-cell heatwave days for each baseline year.
        baseline_days: Per-cell mean annual heatwave days during the baseline.
        target_days: Per-cell heatwave days during the target year.
        additional_days: Target days minus baseline mean days for each cell.

    Examples:
        >>> result = heatwave_anomaly(
        ...     np.array([[[[1.0]], [[2.0]], [[3.0]]]]),
        ...     np.array([[[4.0]], [[4.0]], [[4.0]]]),
        ...     percentile=50,
        ... )
        >>> result.target_days.item()
        3
    """

    threshold: NDArray[np.float64]
    baseline_days_by_year: NDArray[np.int16]
    baseline_days: NDArray[np.float64]
    target_days: NDArray[np.int16]
    additional_days: NDArray[np.float64]


def count_heatwave_days(
    threshold_exceeded: NDArray[np.bool_],
    *,
    minimum_run_days: int = 3,
) -> NDArray[np.int16]:
    """Count every day in runs that meet the minimum duration.

    Args:
        threshold_exceeded: Boolean array whose first dimension is consecutive
            days and whose remaining dimensions identify grid cells.
        minimum_run_days: Smallest consecutive run counted as a heatwave.

    Returns:
        An array over the grid dimensions containing all days in qualifying
        runs.

    Raises:
        TypeError: If ``threshold_exceeded`` is not boolean.
        ValueError: If the input has no time dimension or the duration is less
            than one.

    Examples:
        >>> values = np.array([[True], [True], [True], [False], [True]])
        >>> count_heatwave_days(values).tolist()
        [3]
    """
    if threshold_exceeded.dtype != np.bool_:
        raise TypeError("threshold_exceeded must be a boolean array.")
    if threshold_exceeded.ndim < 1 or threshold_exceeded.shape[0] == 0:
        raise ValueError("threshold_exceeded must have a non-empty time dimension.")
    if (
        isinstance(minimum_run_days, bool)
        or not isinstance(minimum_run_days, int)
        or minimum_run_days < 1
    ):
        raise ValueError("minimum_run_days must be a positive integer.")

    shape = threshold_exceeded.shape[1:]
    run_lengths = np.zeros(shape, dtype=np.int16)
    heatwave_days = np.zeros(shape, dtype=np.int16)
    for current in threshold_exceeded:
        ended = np.logical_and(~current, run_lengths >= minimum_run_days)
        heatwave_days[ended] += run_lengths[ended]
        run_lengths = np.where(current, run_lengths + 1, 0).astype(
            np.int16,
            copy=False,
        )
    completed = run_lengths >= minimum_run_days
    heatwave_days[completed] += run_lengths[completed]
    return heatwave_days


def heatwave_anomaly(
    baseline_daily_maximum: NDArray[np.floating],
    target_daily_maximum: NDArray[np.floating],
    *,
    percentile: float = 92.5,
    minimum_run_days: int = 3,
) -> HeatwaveAnomaly:
    """Calculate target heatwave days relative to mean baseline heatwave days.

    Args:
        baseline_daily_maximum: Temperature array ordered by baseline year,
            day, and then one or more grid dimensions.
        target_daily_maximum: Temperature array ordered by day and the same
            grid dimensions.
        percentile: Per-cell percentile computed across all baseline warm-season
            days with NumPy's linear interpolation.
        minimum_run_days: Smallest consecutive run counted as a heatwave.

    Returns:
        Per-cell threshold, baseline mean, target count, and anomaly arrays.

    Raises:
        ValueError: If array shapes are incompatible, contain non-finite values,
            or the percentile is outside zero through 100.

    Examples:
        >>> baseline = np.array(
        ...     [
        ...         [[[1.0]], [[2.0]], [[3.0]]],
        ...         [[[1.0]], [[2.0]], [[3.0]]],
        ...     ]
        ... )
        >>> target = np.array([[[4.0]], [[4.0]], [[4.0]]])
        >>> heatwave_anomaly(baseline, target, percentile=50).target_days.item()
        3
    """
    if baseline_daily_maximum.ndim < 3:
        raise ValueError(
            "baseline_daily_maximum must have year, day, and grid dimensions."
        )
    if target_daily_maximum.shape != baseline_daily_maximum.shape[1:]:
        raise ValueError(
            "target_daily_maximum must match one baseline year's day and grid shape."
        )
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between zero and 100.")
    if not np.isfinite(baseline_daily_maximum).all():
        raise ValueError("baseline_daily_maximum must contain only finite values.")
    if not np.isfinite(target_daily_maximum).all():
        raise ValueError("target_daily_maximum must contain only finite values.")

    threshold = np.percentile(
        baseline_daily_maximum,
        percentile,
        axis=(0, 1),
        method="linear",
    )
    baseline_days_by_year = np.stack(
        [
            count_heatwave_days(
                year > threshold,
                minimum_run_days=minimum_run_days,
            )
            for year in baseline_daily_maximum
        ]
    )
    baseline_days = baseline_days_by_year.mean(axis=0, dtype=np.float64)
    target_days = count_heatwave_days(
        target_daily_maximum > threshold,
        minimum_run_days=minimum_run_days,
    )
    return HeatwaveAnomaly(
        threshold=np.asarray(threshold, dtype=np.float64),
        baseline_days_by_year=baseline_days_by_year,
        baseline_days=np.asarray(baseline_days, dtype=np.float64),
        target_days=target_days,
        additional_days=np.asarray(target_days - baseline_days, dtype=np.float64),
    )
