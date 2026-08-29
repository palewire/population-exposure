"""Shared configuration for opt-in real population-provider tests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rasterio.windows import Window

from population_exposure import populations

if TYPE_CHECKING:
    from rasterio.io import DatasetReader

WORLDPOP = "worldpop-global-1km"
GHSL = "ghsl-r2023a-mollweide-1km"
GPW = "gpwv4-r11-count"
CHAMBERS = "chambers-hybrid"
LANDSCAN = "landscan-global"

SCHEDULED_PROVIDERS = (WORLDPOP, GHSL)
DOWNLOADABLE_PROVIDERS = (WORLDPOP, GHSL, GPW, CHAMBERS)
MANUAL_CHOICES = ("scheduled", *DOWNLOADABLE_PROVIDERS)
_LIVE_PROVIDERS_ENV = "POPULATION_EXPOSURE_LIVE_PROVIDERS"


@dataclass(frozen=True, slots=True)
class GpwCoarseOracle:
    """The official one-degree GPW count raster paired with a fine selection."""

    selection: str
    official_url: str
    archive_member: str


@dataclass(frozen=True, slots=True)
class GpwParityResult:
    """The result of comparing GPW's fine cells against its coarse count cells."""

    compared_cells: int
    maximum_absolute_difference: float
    maximum_tolerance: float
    maximum_tolerance_normalized_difference: float
    maximum_ulp_normalized_difference: float
    aggregate_difference: float
    aggregate_tolerance: float


def providers_for_run(value: str | None = None) -> tuple[str, ...]:
    """Return the approved source IDs selected for one live test run."""
    choice = (
        value if value is not None else os.environ.get(_LIVE_PROVIDERS_ENV, "scheduled")
    )
    normalized = choice.strip().lower()
    if normalized == "scheduled":
        return SCHEDULED_PROVIDERS
    if normalized in DOWNLOADABLE_PROVIDERS:
        return (normalized,)
    choices = ", ".join(MANUAL_CHOICES)
    raise ValueError(
        f"{_LIVE_PROVIDERS_ENV} must be one of: {choices}; got {choice!r}."
    )


def selection_for_provider(source_id: str) -> str:
    """Choose the latest catalog-supported year without copying provider URLs."""
    sources = {source.source_id: source for source in populations.list()}
    source = sources.get(source_id)
    if source is None or source_id not in DOWNLOADABLE_PROVIDERS:
        raise ValueError(f"{source_id!r} is not an approved live-download provider.")
    return f"{source.source_id}:{max(source.supported_years)}"


def gpw_coarse_oracle(selection: str) -> GpwCoarseOracle:
    """Derive GPW's official one-degree count archive from catalog metadata."""
    selected = populations.info(selection)
    if selected.source_id != GPW:
        raise ValueError(f"{selection!r} is not a GPW population-count selection.")
    fine_archive = f"_{selected.year}_30_sec_tif.zip"
    fine_member = "_30_sec.tif"
    if not selected.official_url.endswith(fine_archive):
        raise ValueError(
            f"{selection!r} does not use the expected official GPW 30-arc-second archive."
        )
    if not selected.expected_filename.endswith(fine_member):
        raise ValueError(
            f"{selection!r} does not use the expected GPW 30-arc-second GeoTIFF name."
        )
    return GpwCoarseOracle(
        selection=selection,
        official_url=(
            selected.official_url.removesuffix(fine_archive)
            + f"_{selected.year}_1_deg_tif.zip"
        ),
        archive_member=(
            selected.expected_filename.removesuffix(fine_member) + "_1_deg.tif"
        ),
    )


def compare_gpw_fine_to_coarse(
    fine: DatasetReader,
    coarse: DatasetReader,
) -> GpwParityResult:
    """Compare one-degree sums from GPW's 30-arc-second count grid.

    The official one-degree population-count raster stores values as float32.
    Each fine-cell sum therefore may differ by half an adjacent coarse float32
    value, plus the bounded error from a float64 summation. For values where
    that bound is below one person, a one-person allowance preserves the
    existing practical guardrail against low-value count differences.

    Args:
        fine: Open 30-arc-second official GPW population-count raster.
        coarse: Open one-degree official GPW population-count raster.

    Returns:
        Parity diagnostics, including per-cell and global precision-aware
        tolerances.

    Raises:
        ValueError: If the grids do not align or the official coarse grid is
            not float32.

    Examples:
        >>> # Compare two aligned Rasterio readers.
        >>> parity = compare_gpw_fine_to_coarse(fine, coarse)
        >>> parity.maximum_tolerance_normalized_difference <= 1
        True
    """
    rows_per_cell, columns_per_cell = _aligned_aggregation_shape(fine, coarse)
    fine_sums = np.empty((coarse.height, coarse.width), dtype=np.float64)
    fine_summation_errors = np.empty_like(fine_sums)
    fine_terms_per_sum = rows_per_cell * columns_per_cell
    for row in range(coarse.height):
        fine_row = fine.read(
            1,
            window=Window(
                col_off=0,
                row_off=row * rows_per_cell,
                width=fine.width,
                height=rows_per_cell,
            ),
            masked=True,
        )
        values = np.asarray(fine_row.filled(0), dtype=np.float64)
        blocks = values.reshape(
            rows_per_cell,
            coarse.width,
            columns_per_cell,
        )
        fine_sums[row] = blocks.sum(axis=(0, 2), dtype=np.float64)
        fine_summation_errors[row] = _float64_sum_error_bound(
            np.abs(blocks).sum(axis=(0, 2), dtype=np.float64),
            fine_terms_per_sum,
        )

    coarse_values = coarse.read(1, masked=True)
    if coarse_values.dtype != np.dtype(np.float32):
        raise ValueError(
            "Official GPW coarse oracle must use float32 population-count values."
        )
    valid = ~np.ma.getmaskarray(coarse_values)
    if not np.any(valid):
        raise ValueError("Official GPW coarse oracle has no valid population cells.")
    coarse_data = np.asarray(coarse_values.data, dtype=np.float32)
    coarse_float64 = np.asarray(coarse_data, dtype=np.float64)
    differences = np.abs(fine_sums[valid] - coarse_float64[valid])
    coarse_ulps = np.abs(np.spacing(coarse_data[valid])).astype(np.float64)
    tolerances = np.maximum(
        1.0,
        0.5 * coarse_ulps + fine_summation_errors[valid],
    )
    aggregate_difference = abs(
        np.sum(fine_sums[valid], dtype=np.float64)
        - np.sum(coarse_float64[valid], dtype=np.float64)
    )
    aggregate_tolerance = float(
        np.sum(tolerances, dtype=np.float64)
        + _float64_sum_error_bound(
            np.sum(np.abs(fine_sums[valid]), dtype=np.float64),
            int(np.count_nonzero(valid)),
        )
        + _float64_sum_error_bound(
            np.sum(np.abs(coarse_float64[valid]), dtype=np.float64),
            int(np.count_nonzero(valid)),
        )
    )
    return GpwParityResult(
        compared_cells=int(np.count_nonzero(valid)),
        maximum_absolute_difference=float(np.max(differences)),
        maximum_tolerance=float(np.max(tolerances)),
        maximum_tolerance_normalized_difference=float(np.max(differences / tolerances)),
        maximum_ulp_normalized_difference=float(np.max(differences / coarse_ulps)),
        aggregate_difference=float(aggregate_difference),
        aggregate_tolerance=aggregate_tolerance,
    )


def _float64_sum_error_bound(
    absolute_sums: float | np.ndarray,
    terms: int,
) -> float | np.ndarray:
    """Return a conservative error bound for summing float64 values.

    Args:
        absolute_sums: Sum of the absolute values being added.
        terms: Number of values in each sum.

    Returns:
        An upper bound for standard float64 summation error.

    Examples:
        >>> _float64_sum_error_bound(100.0, 4) > 0
        True
    """
    operations = max(terms - 1, 0)
    unit_roundoff = np.finfo(np.float64).eps
    gamma = operations * unit_roundoff / (1 - operations * unit_roundoff)
    return np.nextafter(np.asarray(absolute_sums) * gamma * (1 + gamma), np.inf)


def _aligned_aggregation_shape(
    fine: DatasetReader,
    coarse: DatasetReader,
) -> tuple[int, int]:
    """Return cells per side only when both published grids align exactly."""
    if fine.crs != coarse.crs:
        raise ValueError("Fine and coarse GPW rasters must use the same CRS.")
    if fine.width % coarse.width or fine.height % coarse.height:
        raise ValueError(
            "Fine GPW dimensions must be whole multiples of the coarse grid."
        )
    columns_per_cell = fine.width // coarse.width
    rows_per_cell = fine.height // coarse.height
    fine_transform = fine.transform
    coarse_transform = coarse.transform
    if not np.allclose(
        (fine_transform.b, fine_transform.d, coarse_transform.b, coarse_transform.d),
        (0.0, 0.0, 0.0, 0.0),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Fine and coarse GPW rasters must be north-up grids.")
    if not np.allclose(
        (fine_transform.c, fine_transform.f),
        (coarse_transform.c, coarse_transform.f),
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError("Fine and coarse GPW rasters must have the same origin.")
    if not np.allclose(
        (
            fine_transform.a * columns_per_cell,
            fine_transform.e * rows_per_cell,
        ),
        (coarse_transform.a, coarse_transform.e),
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError("Fine GPW cells do not align exactly with the coarse grid.")
    return rows_per_cell, columns_per_cell


def download_failure_phase(error: Exception) -> str:
    """Classify package errors so an external-provider failure is actionable."""
    message = str(error).lower()
    if any(
        marker in message
        for marker in (
            "checksum",
            "content-length",
            "download size",
            "byte safety limit",
            "verified size",
            "expected exactly",
        )
    ):
        return "checksum/size verification"
    if any(marker in message for marker in ("archive", "zip", "extraction")):
        return "archive extraction"
    if any(
        marker in message
        for marker in (
            "population file is not a readable geotiff",
            "requires crs",
            "requires width",
            "requires height",
            "requires pixel size",
            "requires bounds",
            "requires nodata",
            "population total",
        )
    ):
        return "raster validation"
    return "acquisition"
