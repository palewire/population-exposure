"""Tests for bounded Chambers annual extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest
import rasterio
from rasterio.windows import Window

from population_exposure.populations import _chambers

if TYPE_CHECKING:
    from pathlib import Path


def _write_source(
    path: Path,
    *,
    shape: tuple[int, int, int, int] = (720, 1_440, 14, 71),
    latitudes: np.ndarray | None = None,
    values: float = 1,
) -> None:
    """Write a sparse source file with the published Chambers structure.

    Args:
        path: Destination path for the temporary NetCDF-4 file.
        shape: Dimensions for the demographic totals dataset.
        latitudes: Optional latitude coordinate values.
        values: Fill value for demographic totals.

    Returns:
        None. The source file is written to ``path``.

    Examples:
        >>> _write_source(Path("source.nc"))
    """
    with h5py.File(path, "w") as source:
        source.create_dataset(
            "demographic_totals",
            shape=shape,
            chunks=tuple(
                min(chunk_size, dimension)
                for chunk_size, dimension in zip((64, 256, 14, 1), shape, strict=True)
            ),
            dtype=np.float32,
            fillvalue=values,
        )
        source.create_dataset(
            "latitude",
            data=(_chambers._LATITUDES if latitudes is None else latitudes),
        )
        source.create_dataset("longitude", data=_chambers._LONGITUDES)
        source.create_dataset("age_band_lower_bound", data=_chambers._AGE_BANDS)
        source.create_dataset("year", data=_chambers._YEARS)


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_derives_one_year_from_the_published_hdf5_layout(tmp_path: Path) -> None:
    source = tmp_path / "source.nc"
    output = tmp_path / "chambers-hybrid-2019.tif"
    _write_source(source)
    with h5py.File(source, "r+") as file:
        file["demographic_totals"][0, 720, :, 69] = 2

    _chambers.derive_chambers_year(source, output, 2019)

    with rasterio.open(output) as dataset:
        assert dataset.shape == (720, 1_440)
        assert dataset.bounds == (-180.0, -90.0, 180.0, 90.0)
        assert dataset.tags(1)["year"] == "2019"
        assert dataset.tags(1)["processing"] == "sum of 14 five-year age bands"
        assert float(dataset.read(1, window=Window(0, 0, 1, 1))[0, 0]) == 28.0
        assert float(dataset.read(1, window=Window(720, 0, 1, 1))[0, 0]) == 14.0


@pytest.mark.parametrize(
    ("shape", "latitudes", "message"),
    [
        ((720, 10, 14, 71), None, "720 by 1440 by 14 by 71"),
        ((720, 1_440, 21, 71), None, "720 by 1440 by 14 by 71"),
        (
            (720, 1_440, 14, 71),
            np.arange(720, dtype=np.float64),
            "latitude coordinates",
        ),
    ],
)
def test_rejects_unexpected_chambers_structure(
    tmp_path: Path,
    shape: tuple[int, int, int, int],
    latitudes: np.ndarray | None,
    message: str,
) -> None:
    source = tmp_path / "source.nc"
    _write_source(source, shape=shape, latitudes=latitudes)

    with pytest.raises(ValueError, match=message):
        _chambers.derive_chambers_year(source, tmp_path / "output.tif", 2019)


def test_rejects_invalid_source_values(tmp_path: Path) -> None:
    source = tmp_path / "source.nc"
    _write_source(source)
    with h5py.File(source, "r+") as file:
        file["demographic_totals"][0, 0, 0, 69] = -1

    with pytest.raises(ValueError, match="invalid population values"):
        _chambers.derive_chambers_year(source, tmp_path / "output.tif", 2019)


def test_rejects_non_hdf5_source(tmp_path: Path) -> None:
    source = tmp_path / "source.nc"
    source.write_text("not a NetCDF file")

    with pytest.raises(ValueError, match="expected NetCDF-4"):
        _chambers.derive_chambers_year(source, tmp_path / "output.tif", 2019)
