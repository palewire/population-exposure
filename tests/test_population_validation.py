"""Tests for source-specific raster and receipt validation."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from rasterio import Affine
from rasterio.transform import from_origin

from population_exposure.populations import _cache, _raster, _sources
from population_exposure.populations._http import sha256_file

if TYPE_CHECKING:
    from pathlib import Path


def write_raster(
    path: Path,
    *,
    crs: str | None = "EPSG:4326",
    transform: Affine | None = None,
    nodata: float | None = -9999,
    year: int | None = 2020,
    unit: str | None = None,
) -> Path:
    """Write a tiny source-validation fixture."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float64",
        crs=crs,
        transform=transform or from_origin(0, 2, 1, 1),
        nodata=nodata,
    ) as dataset:
        dataset.write(np.array([[1.0, 2.0], [3.0, 4.0]]), 1)
        dataset.update_tags(1, population_semantics="count")
        if year is not None:
            dataset.update_tags(1, year=str(year))
        if unit is not None:
            dataset.set_band_unit(1, unit)
    return path


def tiny_source(**changes):
    """Return WorldPop with checks sized for the tiny raster."""
    defaults = {
        "url_template": "https://example.test/{year}",
        "filename_template": "population-{year}.tif",
        "expected_width": 2,
        "expected_height": 2,
        "expected_resolution": (1.0, 1.0),
        "expected_bounds": (0.0, 0.0, 2.0, 2.0),
        "expected_nodata": (-9999.0,),
        "expected_bounds_by_year": MappingProxyType({}),
        "expected_nodata_by_year": MappingProxyType({}),
        "plausible_total": (0.0, 100.0),
    }
    defaults.update(changes)
    return replace(_sources.WORLDPOP, **defaults)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (tiny_source(crs="EPSG:3857"), "requires CRS"),
        (tiny_source(expected_width=3), "requires width"),
        (tiny_source(expected_height=3), "requires height"),
        (tiny_source(expected_resolution=(0.5, 1.0)), "requires pixel size"),
        (tiny_source(expected_bounds=(1.0, 0.0, 3.0, 2.0)), "requires bounds"),
        (tiny_source(expected_nodata=(-200.0,)), "requires nodata"),
        (tiny_source(plausible_total=(20.0, 30.0)), "safety range"),
    ],
)
def test_source_structure_mismatches_fail(
    tmp_path: Path,
    source,
    message: str,
) -> None:
    path = write_raster(tmp_path / "population-2020.tif")

    with pytest.raises(ValueError, match=message):
        _raster.validate_catalog_raster(
            path,
            source,
            2020,
            require_year_marker=True,
        )


@pytest.mark.filterwarnings(
    "ignore:Use `@` matmul instead of `\\*` mul operator:PendingDeprecationWarning"
)
def test_rotated_source_grid_fails(tmp_path: Path) -> None:
    path = write_raster(
        tmp_path / "population-2020.tif",
        transform=Affine(1, 0.1, 0, 0, -1, 2),
    )

    with pytest.raises(ValueError, match="north-up grid"):
        _raster.validate_catalog_raster(
            path,
            tiny_source(expected_bounds=None),
            2020,
            require_year_marker=True,
        )


def test_optional_source_checks_can_be_unspecified(tmp_path: Path) -> None:
    path = write_raster(
        tmp_path / "population-2020.tif",
        unit="people",
    )
    source = tiny_source(
        expected_width=None,
        expected_height=None,
        expected_resolution=None,
        expected_bounds=None,
        expected_nodata=None,
        plausible_total=None,
    )

    observed = _raster.validate_catalog_raster(
        path,
        source,
        2020,
        require_year_marker=False,
    )

    assert observed["units"] == "people"
    assert observed["population_total"] == 10.0


def test_bad_geotiff_and_year_markers_fail_clearly(tmp_path: Path) -> None:
    corrupt = tmp_path / "population-2020.tif"
    corrupt.write_text("not a raster")
    with pytest.raises(ValueError, match="not a readable GeoTIFF"):
        _raster.validate_catalog_raster(
            corrupt,
            tiny_source(),
            2020,
            require_year_marker=False,
        )

    wrong = write_raster(tmp_path / "population.tif", year=2019)
    with pytest.raises(ValueError, match="not requested year"):
        _raster.validate_catalog_raster(
            wrong,
            tiny_source(),
            2020,
            require_year_marker=True,
        )

    unmarked = write_raster(tmp_path / "population.tif", year=None)
    with pytest.raises(ValueError, match="must identify"):
        _raster.validate_catalog_raster(
            unmarked,
            tiny_source(),
            2020,
            require_year_marker=True,
        )

    named = write_raster(tmp_path / "population-2020.tif", year=None)
    _raster.validate_catalog_raster(
        named,
        tiny_source(),
        2020,
        require_year_marker=True,
    )


def test_nodata_matching_handles_none_nan_and_numeric_values() -> None:
    assert not _raster._nodata_matches(None, (-9999.0,))
    assert _raster._nodata_matches(float("nan"), (float("nan"),))
    assert _raster._nodata_matches(-9999.0, (-9999.0,))
    assert not _raster._nodata_matches(-1.0, (-9999.0,))


@pytest.mark.parametrize(
    ("year", "expected_bounds", "expected_nodata"),
    [
        (
            2000,
            (
                -180.001249265,
                -71.99208284398998,
                179.99874929500004,
                84.00791653201003,
            ),
            (3.4028234663852886e38,),
        ),
        (
            2010,
            (
                -180.001249265,
                -71.99208284398998,
                179.99874929500004,
                84.00791653201003,
            ),
            (-3.4028234663852886e38,),
        ),
        (
            2020,
            (
                -180.001249265,
                -72.00041617728999,
                179.99874929500004,
                83.99958319871001,
            ),
            (-3.4028234663852886e38,),
        ),
    ],
)
def test_worldpop_uses_documented_year_specific_grid_values(
    year: int,
    expected_bounds: tuple[float, float, float, float],
    expected_nodata: tuple[float, ...],
) -> None:
    assert _sources.WORLDPOP.expected_bounds_for(year) == expected_bounds
    assert _sources.WORLDPOP.expected_nodata_for(year) == expected_nodata


@pytest.mark.parametrize(
    ("year", "bounds", "nodata"),
    [
        (2000, (0.0, 0.0, 2.0, 2.0), -100.0),
        (2010, (3.0, 3.0, 5.0, 5.0), -200.0),
        (2020, (6.0, 6.0, 8.0, 8.0), -300.0),
    ],
)
def test_year_specific_grid_values_are_selected_for_validation(
    tmp_path: Path,
    year: int,
    bounds: tuple[float, float, float, float],
    nodata: float,
) -> None:
    path = write_raster(
        tmp_path / f"population-{year}.tif",
        transform=Affine(1, 0, bounds[0], 0, -1, bounds[3]),
        nodata=nodata,
        year=year,
    )
    source = tiny_source(
        expected_bounds=None,
        expected_nodata=None,
        expected_bounds_by_year=MappingProxyType({year: bounds}),
        expected_nodata_by_year=MappingProxyType({year: (nodata,)}),
    )

    _raster.validate_catalog_raster(
        path,
        source,
        year,
        require_year_marker=True,
    )


def test_receipt_validation_rejects_every_stale_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raster = write_raster(tmp_path / "population-2020.tif")
    receipt = raster.with_suffix(".tif.json")
    info = tiny_source().selection_info(2020)
    original = _cache.write_receipt(
        raster,
        receipt,
        info=info,
        sha256=sha256_file(raster),
        observed={"nodata": float("nan")},
        processing_note="test",
    )
    assert _cache.verified_receipt(
        raster,
        receipt,
        selection=info.selection,
    )
    assert json.loads(receipt.read_text())["observed"]["nodata"] == "NaN"
    assert original["_verified_ctime_ns"] == raster.stat().st_ctime_ns
    assert original["_verified_device"] == raster.stat().st_dev
    assert original["_verified_inode"] == raster.stat().st_ino

    for changed in (
        [],
        {**original, "selection": "other:2020"},
        {**original, "local_sha256": "bad"},
        {**original, "local_size_bytes": -1},
        {**original, "_verified_mtime_ns": -1},
        {**original, "_verified_ctime_ns": -1},
        {**original, "_verified_device": -1},
        {**original, "_verified_inode": -1},
    ):
        receipt.write_text(json.dumps(changed))
        assert (
            _cache.verified_receipt(
                raster,
                receipt,
                selection=info.selection,
            )
            is None
        )

    receipt.write_text("{")
    assert _cache.verified_receipt(raster, receipt, selection=None) is None
    receipt.unlink()
    assert _cache.verified_receipt(raster, receipt, selection=None) is None

    _cache.write_receipt(
        raster,
        receipt,
        info=info,
        sha256=sha256_file(raster),
        observed={},
        processing_note="test",
    )
    stat = raster.stat()
    tampered = bytearray(raster.read_bytes())
    tampered[-1] ^= 1
    raster.write_bytes(tampered)
    os.utime(raster, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert _cache.verified_receipt(raster, receipt, selection=None) is None

    monkeypatch.delenv(_cache.CACHE_ENV, raising=False)
    monkeypatch.setattr(_cache, "user_cache_path", lambda *args, **kwargs: tmp_path)
    assert _cache.cache_root(None) == tmp_path


def test_json_safe_handles_non_finite_and_nested_values() -> None:
    assert _cache._json_safe(float("nan")) == "NaN"
    assert _cache._json_safe(float("inf")) == "Infinity"
    assert _cache._json_safe(float("-inf")) == "-Infinity"
    assert _cache._json_safe({1: (float("nan"),)}) == {"1": ["NaN"]}
    assert _cache._json_safe("plain") == "plain"


def test_archive_member_is_absent_for_direct_sources() -> None:
    assert _sources.WORLDPOP.archive_member(2020) is None
