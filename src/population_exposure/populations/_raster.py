"""Catalog-specific population raster checks and observed metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

from population_exposure.raster import validate_population_raster

if TYPE_CHECKING:
    from pathlib import Path

    from rasterio.io import DatasetReader

    from population_exposure.populations._sources import SourceSpec


def validate_catalog_raster(
    path: Path,
    source: SourceSpec,
    year: int,
    *,
    require_year_marker: bool,
) -> dict[str, object]:
    """Validate one catalog raster and return receipt-ready observed facts."""
    try:
        with rasterio.open(path) as dataset:
            total = validate_population_raster(dataset)
            _validate_source_structure(dataset, source)
            if require_year_marker:
                _validate_registered_year(dataset, path, year)
            if source.plausible_total is not None:
                minimum, maximum = source.plausible_total
                if not minimum <= total <= maximum:
                    raise ValueError(
                        f"Population total {total:.12g} is outside the documented "
                        f"catalog safety range {minimum:.12g}-{maximum:.12g}."
                    )
            return observed_raster_facts(dataset, total=total)
    except RasterioIOError as error:
        raise ValueError(
            f"Population file is not a readable GeoTIFF: {path}."
        ) from error


def observed_raster_facts(
    dataset: DatasetReader,
    *,
    total: float,
) -> dict[str, object]:
    """Return stable facts observed directly from an open raster."""
    units = dataset.units[0] if dataset.units else None
    return {
        "width": dataset.width,
        "height": dataset.height,
        "band_count": dataset.count,
        "dtype": dataset.dtypes[0],
        "crs": dataset.crs.to_string() if dataset.crs is not None else None,
        "transform": list(dataset.transform)[:6],
        "bounds": [
            dataset.bounds.left,
            dataset.bounds.bottom,
            dataset.bounds.right,
            dataset.bounds.top,
        ],
        "nodata": dataset.nodata,
        "units": units,
        "population_total": total,
    }


def _validate_source_structure(dataset: DatasetReader, source: SourceSpec) -> None:
    """Require the source's stable published grid properties."""
    expected_crs = rasterio.CRS.from_user_input(source.crs)
    if dataset.crs != expected_crs:
        observed = dataset.crs.to_string() if dataset.crs is not None else None
        raise ValueError(
            f"{source.source_id!r} requires CRS {source.crs}; found {observed!r}."
        )
    if source.expected_width is not None and dataset.width != source.expected_width:
        raise ValueError(
            f"{source.source_id!r} requires width {source.expected_width}; "
            f"found {dataset.width}."
        )
    if source.expected_height is not None and dataset.height != source.expected_height:
        raise ValueError(
            f"{source.source_id!r} requires height {source.expected_height}; "
            f"found {dataset.height}."
        )
    if source.expected_resolution is not None:
        observed_resolution = (
            abs(float(dataset.transform.a)),
            abs(float(dataset.transform.e)),
        )
        if not np.allclose(
            observed_resolution,
            source.expected_resolution,
            rtol=0,
            atol=max(source.expected_resolution) * 1e-9,
        ):
            raise ValueError(
                f"{source.source_id!r} requires pixel size "
                f"{source.expected_resolution}; found {observed_resolution}."
            )
        if not np.allclose(
            (dataset.transform.b, dataset.transform.d),
            (0.0, 0.0),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(f"{source.source_id!r} requires a north-up grid.")
    if source.expected_bounds is not None:
        observed_bounds = tuple(float(value) for value in dataset.bounds)
        tolerance = (
            max(source.expected_resolution) * 1e-6
            if source.expected_resolution is not None
            else 1e-9
        )
        if not np.allclose(
            observed_bounds,
            source.expected_bounds,
            rtol=0,
            atol=tolerance,
        ):
            raise ValueError(
                f"{source.source_id!r} requires bounds {source.expected_bounds}; "
                f"found {observed_bounds}."
            )
    if source.expected_nodata is not None and not _nodata_matches(
        dataset.nodata,
        source.expected_nodata,
    ):
        raise ValueError(
            f"{source.source_id!r} requires nodata {source.expected_nodata}; "
            f"found {dataset.nodata!r}."
        )


def _nodata_matches(
    observed: float | None,
    expected_values: tuple[float, ...],
) -> bool:
    """Return whether nodata matches one documented source value."""
    if observed is None:
        return False
    for expected in expected_values:
        if np.isnan(expected) and np.isnan(observed):
            return True
        if np.isclose(observed, expected, rtol=1e-6, atol=0):
            return True
    return False


def _validate_registered_year(
    dataset: DatasetReader,
    path: Path,
    year: int,
) -> None:
    """Require a requested year in source metadata or the local filename."""
    tags = {
        str(key).lower(): str(value)
        for mapping in (dataset.tags(), dataset.tags(1))
        for key, value in mapping.items()
    }
    declared = next(
        (tags[key] for key in ("year", "population_year", "time") if key in tags),
        None,
    )
    if declared is not None:
        if str(year) not in declared:
            raise ValueError(
                f"Population raster declares year {declared!r}, not requested year "
                f"{year}."
            )
        return
    if str(year) not in path.name:
        raise ValueError(
            "Registered population filename or raster metadata must identify the "
            f"requested year {year}."
        )
