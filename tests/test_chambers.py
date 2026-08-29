"""Tests for bounded Chambers annual extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from population_exposure.populations import _chambers

if TYPE_CHECKING:
    from pathlib import Path


class FakeChambersReader:
    """NetCDF-like reader that creates only requested output windows."""

    width = 1_440
    height = 721
    shape = (721, 1_440)
    count = 21
    crs = rasterio.CRS.from_epsg(4326)
    transform = from_origin(-180, 90.125, 0.25, 0.25)

    def __init__(self, *, year: int = 2019) -> None:
        self.year = year
        self.read_shapes: list[tuple[int, int]] = []

    def tags(self, band: int) -> dict[str, str]:
        return {
            "NETCDF_DIM_year": f"{{{self.year}}}",
            "NETCDF_DIM_age_band_lower_bound": str((band - 1) * 5),
        }

    def read(self, band: int, *, window, masked: bool):
        assert masked
        shape = (int(window.height), int(window.width))
        self.read_shapes.append(shape)
        return np.ma.array(np.full(shape, float(band)), mask=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_derives_one_year_by_summing_age_bands_in_windows(tmp_path: Path) -> None:
    reader = FakeChambersReader()
    output = tmp_path / "chambers-hybrid-2019.tif"

    _chambers._derive_from_reader(reader, output, 2019)

    with rasterio.open(output) as dataset:
        assert dataset.shape == (721, 1_440)
        assert dataset.count == 1
        assert dataset.tags(1)["year"] == "2019"
        assert dataset.tags(1)["processing"] == "sum of 21 five-year age bands"
        assert float(dataset.read(1, window=Window(0, 0, 1, 1))[0, 0]) == 231.0
    assert max(rows * columns for rows, columns in reader.read_shapes) <= 256 * 256
    assert len(reader.read_shapes) > 21


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"width": 10}, "1440 by 721"),
        ({"crs": rasterio.CRS.from_epsg(3857)}, "WGS 84"),
        (
            {"transform": from_origin(-180, 90.125, 0.5, 0.25)},
            "0.25 degree",
        ),
        ({"year": 2018}, "all 21 age bands"),
    ],
)
def test_rejects_unexpected_chambers_structure(
    tmp_path: Path,
    change: dict[str, object],
    message: str,
) -> None:
    reader = FakeChambersReader()
    for key, value in change.items():
        setattr(reader, key, value)

    with pytest.raises(ValueError, match=message):
        _chambers._derive_from_reader(reader, tmp_path / "output.tif", 2019)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("{2020}", 2020),
        ("5.0", 5),
        ("5.5", None),
        ("bad", None),
        (None, None),
    ],
)
def test_integer_dimension_tag_parsing(value: str | None, expected: int | None) -> None:
    assert _chambers._integer_tag(value) == expected


def test_invalid_source_values_fail_and_partial_output_is_caller_cleaned(
    tmp_path: Path,
) -> None:
    reader = FakeChambersReader()

    def invalid_read(band: int, *, window, masked: bool):
        values = np.ones((int(window.height), int(window.width)))
        values[0, 0] = -1
        return np.ma.array(values, mask=False)

    reader.read = invalid_read  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="invalid population values"):
        _chambers._derive_from_reader(reader, tmp_path / "output.tif", 2019)


def test_public_derivation_opens_expected_source_and_wraps_open_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = FakeChambersReader()
    opened: list[Path] = []

    def open_source(path: Path):
        opened.append(path)
        return reader

    monkeypatch.setattr(_chambers, "_open_source", open_source)
    source = tmp_path / "source.nc"
    output = tmp_path / "output.tif"
    _chambers.derive_chambers_year(source, output, 2019)
    assert opened == [source]
    assert output.is_file()

    def fail_open(path: Path):
        raise rasterio.errors.RasterioIOError("bad NetCDF")

    monkeypatch.setattr(_chambers, "_open_source", fail_open)
    with pytest.raises(ValueError, match="expected NetCDF-4"):
        _chambers.derive_chambers_year(source, output, 2019)


def test_open_source_uses_the_published_netcdf_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    sentinel = object()

    def fake_open(path: str):
        opened.append(path)
        return sentinel

    monkeypatch.setattr(_chambers.rasterio, "open", fake_open)
    result = _chambers._open_source(tmp_path / "source.nc")

    assert result is sentinel
    assert opened == [
        f'NETCDF:"{tmp_path / "source.nc"}":demographic_totals',
    ]
