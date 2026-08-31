"""Regenerate the offline GPWv4 2020 cross-resolution tabular golden fixture.

Example:
    EARTHDATA_TOKEN=... uv run python scripts/regenerate_gpwv4_tabular_golden.py \
        --accept-download tests/data/gpwv4_r11_iceland_tabular
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import rasterio
from platformdirs import user_cache_dir
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from population_exposure.populations._api import _authentication_headers
from population_exposure.populations._archives import extract_members
from population_exposure.populations._http import download_file, sha256_file
from population_exposure.populations._sources import GPW

if TYPE_CHECKING:
    from rasterio.io import DatasetReader
    from rasterio.transform import Affine

FIXTURE_NAME = "GPWv4 Revision 11 Iceland 2020 tabular cross-resolution"
FINE_URL = (
    "https://data.earthdata.nasa.gov/nasa-earth/human-dimensions/sedac-root/"
    "downloads/data/gpw-v4/gpw-v4-population-count-rev11/"
    "gpw-v4-population-count-rev11_2020_30_sec_tif.zip"
)
COARSE_URL = FINE_URL.replace("_30_sec_tif.zip", "_1_deg_tif.zip")
FINE_ARCHIVE = "gpw-v4-population-count-rev11_2020_30_sec_tif.zip"
COARSE_ARCHIVE = "gpw-v4-population-count-rev11_2020_1_deg_tif.zip"
FINE_MEMBER = "gpw_v4_population_count_rev11_2020_30_sec.tif"
COARSE_MEMBER = "gpw_v4_population_count_rev11_2020_1_deg.tif"
FINE_ARCHIVE_BYTES = 405_209_680
COARSE_ARCHIVE_BYTES = 89_273
FINE_ARCHIVE_SHA256 = "0ffe8501213d00b98707d89884212c3bb18a1917f9b304e39fe91cc3db40720f"  # pragma: allowlist secret
COARSE_ARCHIVE_SHA256 = "2d797c57abcb09bf6b0e917acec7737961cc017c1bcefe9df477da6cb0c21f6c"  # pragma: allowlist secret
FINE_MAX_DOWNLOAD_BYTES = 450_000_000
COARSE_MAX_DOWNLOAD_BYTES = 1_000_000
CELLS_PER_COARSE_SIDE = 120
COARSE_WINDOW = Window(col_off=157, row_off=25, width=2, height=2)
COORDINATE_DECIMAL_PLACES = 12
DOWNLOAD_TIMEOUT_NOTE = "The shared downloader uses a 60-second response timeout."


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file.

    Args:
        path: Existing file whose bytes will be hashed.

    Returns:
        Lowercase hexadecimal SHA-256 digest.

    Examples:
        >>> _sha256(Path("source.bin"))
        '...'
    """
    return sha256_file(path)


def _download_archive(
    *,
    url: str,
    cache_path: Path,
    expected_bytes: int,
    expected_sha256: str,
    maximum_bytes: int,
    headers: dict[str, str],
) -> Path:
    """Return one source archive after strict cache and download verification.

    Args:
        url: Official HTTPS source URL.
        cache_path: Verified-archive cache location.
        expected_bytes: Exact expected byte count.
        expected_sha256: Exact expected SHA-256 digest.
        maximum_bytes: Upper safety limit for a transfer.
        headers: Transient Earthdata authorization headers.

    Returns:
        Path to the verified local archive.

    Raises:
        ValueError: If cached or downloaded bytes do not match the pinned source.

    Examples:
        >>> _download_archive(
        ...     url="https://example.test/source.zip",
        ...     cache_path=Path("source.zip"),
        ...     expected_bytes=1,
        ...     expected_sha256="...",
        ...     maximum_bytes=1,
        ...     headers={"Authorization": "Bearer ..."},
        ... )
        PosixPath('source.zip')
    """
    if cache_path.is_file():
        if (
            cache_path.stat().st_size == expected_bytes
            and _sha256(cache_path) == expected_sha256
        ):
            return cache_path
        cache_path.unlink()

    partial_path = cache_path.with_suffix(cache_path.suffix + ".partial")
    result = download_file(
        url,
        partial_path,
        headers=headers,
        max_bytes=maximum_bytes,
        exact_bytes=expected_bytes,
        publisher_checksum=None,
    )
    if result.sha256 != expected_sha256:
        partial_path.unlink()
        raise ValueError(
            f"Official source checksum changed for {url}: {result.sha256}."
        )
    partial_path.replace(cache_path)
    return cache_path


def _extract_raster(archive: Path, member: str, destination: Path) -> Path:
    """Extract one known raster member through the shared path-safe extractor.

    Args:
        archive: Verified official ZIP archive.
        member: Exact GeoTIFF archive member name.
        destination: Empty directory receiving the selected member.

    Returns:
        Extracted GeoTIFF path.

    Raises:
        ValueError: If the archive member is missing or has an unsafe path.

    Examples:
        >>> _extract_raster(Path("source.zip"), "source.tif", Path("source"))
        PosixPath('source/source.tif')
    """
    prefix = member.removesuffix(".tif")
    extract_members(archive, destination, (prefix,))
    path = destination / member
    if not path.is_file():
        raise ValueError(f"Archive did not extract the expected raster: {member}.")
    return path


def _validate_source_grids(fine: DatasetReader, coarse: DatasetReader) -> None:
    """Require the pinned GPW population-count grids to align exactly.

    Args:
        fine: Open official 30-arc-second Population Count raster.
        coarse: Open official 1-degree Population Count raster.

    Returns:
        None.

    Raises:
        ValueError: If grid dimensions, CRS, transforms, or dtypes differ.

    Examples:
        >>> # Open aligned GPW readers before calling this validator.
        >>> _validate_source_grids(fine, coarse)
    """
    if fine.dtypes != ("float32",) or coarse.dtypes != ("float32",):
        raise ValueError("GPW source rasters must be single-band float32.")
    if fine.crs != coarse.crs or fine.crs.to_string() != "EPSG:4326":
        raise ValueError("GPW source rasters must share EPSG:4326.")
    if fine.width != coarse.width * CELLS_PER_COARSE_SIDE or fine.height != (
        coarse.height * CELLS_PER_COARSE_SIDE
    ):
        raise ValueError("GPW source raster dimensions are not 120-to-1 aligned.")
    fine_transform = fine.transform
    coarse_transform = coarse.transform
    if not np.allclose(
        (fine_transform.b, fine_transform.d, coarse_transform.b, coarse_transform.d),
        (0.0, 0.0, 0.0, 0.0),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("GPW source grids must be north-up.")
    if not np.allclose(
        (fine_transform.c, fine_transform.f),
        (coarse_transform.c, coarse_transform.f),
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError("GPW source grids have different origins.")
    if not np.allclose(
        (
            fine_transform.a * CELLS_PER_COARSE_SIDE,
            fine_transform.e * CELLS_PER_COARSE_SIDE,
        ),
        (coarse_transform.a, coarse_transform.e),
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError("GPW source grids have different cell sizes.")


def _fine_window(coarse_window: Window) -> Window:
    """Return the matching native-resolution window for whole coarse cells.

    Args:
        coarse_window: Whole-pixel one-degree window.

    Returns:
        Aligned 30-arc-second window with 120 cells per coarse side.

    Raises:
        ValueError: If the coarse window is not aligned to whole source pixels.

    Examples:
        >>> _fine_window(Window(157, 25, 2, 2))
        Window(col_off=18840, row_off=3000, width=240, height=240)
    """
    values = (
        coarse_window.col_off,
        coarse_window.row_off,
        coarse_window.width,
        coarse_window.height,
    )
    if any(value != int(value) for value in values):
        raise ValueError("Selected coarse window must use whole cells.")
    return Window(
        col_off=coarse_window.col_off * CELLS_PER_COARSE_SIDE,
        row_off=coarse_window.row_off * CELLS_PER_COARSE_SIDE,
        width=coarse_window.width * CELLS_PER_COARSE_SIDE,
        height=coarse_window.height * CELLS_PER_COARSE_SIDE,
    )


def _sum_tolerance(values: np.ndarray, expected: np.float32) -> float:
    """Return the bounded absolute error for one published float32 coarse cell.

    Args:
        values: Fine float32 values summed in float64.
        expected: Published float32 one-degree Population Count value.

    Returns:
        Half an expected-value float32 ULP plus float64 summation roundoff.

    Examples:
        >>> _sum_tolerance(np.array([1, 2], dtype=np.float32), np.float32(3)) > 0
        True
    """
    operations = max(values.size - 1, 0)
    unit_roundoff = np.finfo(np.float64).eps
    gamma = operations * unit_roundoff / (1 - operations * unit_roundoff)
    summation_error = float(
        np.nextafter(
            np.abs(values, dtype=np.float64).sum(dtype=np.float64) * gamma,
            np.inf,
        )
    )
    return float(0.5 * abs(np.spacing(expected)) + summation_error)


def _write_crop(
    source: DatasetReader,
    values: np.ndarray,
    window: Window,
    destination: Path,
) -> None:
    """Write a deterministic lossless crop with the original grid placement.

    Args:
        source: Open source raster supplying CRS, dtype, and nodata facts.
        values: Two-dimensional source values for the requested window.
        window: Source-pixel window corresponding to ``values``.
        destination: New GeoTIFF path to write.

    Returns:
        None.

    Examples:
        >>> # ``source`` is an open Rasterio dataset and ``values`` is its crop.
        >>> _write_crop(source, values, Window(0, 0, 1, 1), Path("crop.tif"))
    """
    profile = source.profile.copy()
    profile.update(
        height=values.shape[0],
        width=values.shape[1],
        transform=window_transform(window, source.transform),
        compress="deflate",
        predictor=3,
        zlevel=9,
    )
    with rasterio.open(destination, "w", **profile) as output:
        output.write(values, 1)


def _center_coordinates(
    transform: Affine, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return pixel-center coordinates in north-to-south, west-to-east order.

    Args:
        transform: Pixel-corner affine transform.
        shape: Raster height and width.

    Returns:
        Pair of rounded longitude and latitude arrays.

    Examples:
        >>> longitudes, latitudes = _center_coordinates(Affine.identity(), (1, 1))
        >>> (longitudes.tolist(), latitudes.tolist())
        ([0.5], [0.5])
    """
    rows, columns = np.indices(shape)
    longitudes = transform.c + (columns.ravel() + 0.5) * transform.a
    latitudes = transform.f + (rows.ravel() + 0.5) * transform.e
    return (
        np.round(longitudes, COORDINATE_DECIMAL_PLACES),
        np.round(latitudes, COORDINATE_DECIMAL_PLACES),
    )


def _parent_cells() -> np.ndarray:
    """Return one stable parent category per native cell in the selected window.

    Args:
        None.

    Returns:
        String array identifying each global one-degree parent row and column.

    Examples:
        >>> _parent_cells().shape
        (57600,)
    """
    local_rows, local_columns = np.indices(
        (
            int(COARSE_WINDOW.height) * CELLS_PER_COARSE_SIDE,
            int(COARSE_WINDOW.width) * CELLS_PER_COARSE_SIDE,
        )
    )
    rows = local_rows // CELLS_PER_COARSE_SIDE + int(COARSE_WINDOW.row_off)
    columns = local_columns // CELLS_PER_COARSE_SIDE + int(COARSE_WINDOW.col_off)
    return np.char.add(
        np.char.add("row-", rows.astype(str)),
        np.char.add("-column-", columns.astype(str)),
    ).ravel()


def _write_readme(destination: Path) -> None:
    """Write plain-language source and purpose notes for the fixture.

    Args:
        destination: Fixture README path.

    Returns:
        None.

    Examples:
        >>> _write_readme(Path("README.md"))
    """
    destination.write_text(
        f"""# {FIXTURE_NAME}

This is a network-free, real-data **same-product cross-resolution consistency**
fixture. It is not a hazard or exposure result, and its two grids are not
independent population sources. It verifies that
`population_exposure.assign_population` matches native GPW Population Count
values to exact coordinate rows, preserves hazard-row ordering, and reproduces
separately published one-degree CIESIN values after grouping by
`parent_1_degree_cell`.

The fixture contains a 2 by 2 block of complete one-degree cells around
Iceland (global rows 25--26, columns 157--158) and its matching 240 by 240
30-arc-second crop. Its coordinate table contains the crop's finite cells;
the source nodata cells remain in the raster. Both are lossless float32
GeoTIFF derivatives of GPWv4 Revision 11 Population Count 2020. The source
product is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
reuse requires the attribution in `metadata.json`. That license applies to the
two rasters and the coordinate table derived from them, not because this
repository is MIT licensed.

The [30-arc-second input]({FINE_URL})
and [one-degree grid]({COARSE_URL})
are separately published outputs of the cited GPW product. Both archives need
an Earthdata token only during regeneration. GPW documentation describes the
coarse Population Count grids as aggregations of native cells. The check
therefore tests agreement within one product, not agreement with an outside
estimate. `{DOWNLOAD_TIMEOUT_NOTE}`

Regenerate only after deliberately accepting the authenticated 405 MB fine
archive:

```sh
EARTHDATA_TOKEN=... uv run python scripts/regenerate_gpwv4_tabular_golden.py \\
  --accept-download tests/data/gpwv4_r11_iceland_tabular
```

The script validates cache size and SHA-256 before reuse, deletes a corrupted
cached archive before replacing it, bounds transfers, uses the package's
credential-safe downloader, and uses the shared archive extractor that rejects
unsafe member paths. `metadata.json` pins the source files, crop checksums,
grid facts, published coarse values, and the float32 precision bounds.
""",
        encoding="utf-8",
    )


def build_fixture(output_directory: Path, cache_directory: Path) -> None:
    """Build the Iceland tabular fixture from pinned official GPW archives.

    Args:
        output_directory: New directory that will receive offline test files.
        cache_directory: Directory holding verified, reusable source archives.

    Returns:
        None.

    Raises:
        ValueError: If source identity, alignment, values, or float precision
            differ from the pinned GPW product.

    Examples:
        >>> build_fixture(Path("fixture"), Path("cache"))
    """
    if output_directory.exists():
        raise ValueError(f"Output directory already exists: {output_directory}.")
    headers = _authentication_headers(GPW, None)
    if headers is None:
        raise ValueError("GPW regeneration requires Earthdata authorization headers.")
    cache_directory.mkdir(parents=True, exist_ok=True)
    fine_archive = _download_archive(
        url=FINE_URL,
        cache_path=cache_directory / FINE_ARCHIVE,
        expected_bytes=FINE_ARCHIVE_BYTES,
        expected_sha256=FINE_ARCHIVE_SHA256,
        maximum_bytes=FINE_MAX_DOWNLOAD_BYTES,
        headers=headers,
    )
    coarse_archive = _download_archive(
        url=COARSE_URL,
        cache_path=cache_directory / COARSE_ARCHIVE,
        expected_bytes=COARSE_ARCHIVE_BYTES,
        expected_sha256=COARSE_ARCHIVE_SHA256,
        maximum_bytes=COARSE_MAX_DOWNLOAD_BYTES,
        headers=headers,
    )
    output_directory.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="population-exposure-gpwv4-") as temporary:
        extracted = Path(temporary)
        fine_path = _extract_raster(fine_archive, FINE_MEMBER, extracted / "fine")
        coarse_path = _extract_raster(
            coarse_archive, COARSE_MEMBER, extracted / "coarse"
        )
        with rasterio.open(fine_path) as fine, rasterio.open(coarse_path) as coarse:
            _validate_source_grids(fine, coarse)
            fine_window = _fine_window(COARSE_WINDOW)
            fine_values = fine.read(1, window=fine_window, masked=True)
            coarse_values = coarse.read(1, window=COARSE_WINDOW, masked=True)
            fine_mask = np.ma.getmaskarray(fine_values)
            coarse_mask = np.ma.getmaskarray(coarse_values)
            if coarse_mask.any():
                raise ValueError(
                    "Selected published GPW coarse cells must not contain nodata."
                )
            fine_data = np.asarray(fine_values.data, dtype=np.float32)
            coarse_array = np.asarray(coarse_values.data, dtype=np.float32)
            if (
                not np.isfinite(fine_data[~fine_mask]).all()
                or not np.isfinite(coarse_array).all()
            ):
                raise ValueError("Selected valid GPW cells must be finite.")
            if (fine_data[~fine_mask] < 0).any() or (coarse_array < 0).any():
                raise ValueError("Selected GPW cells must be non-negative.")

            fine_array = fine_values.filled(fine.nodata).astype(np.float32)
            measured_cells: list[dict[str, float | int | str]] = []
            for local_row in range(coarse_array.shape[0]):
                for local_column in range(coarse_array.shape[1]):
                    values = fine_values[
                        local_row * CELLS_PER_COARSE_SIDE : (local_row + 1)
                        * CELLS_PER_COARSE_SIDE,
                        local_column * CELLS_PER_COARSE_SIDE : (local_column + 1)
                        * CELLS_PER_COARSE_SIDE,
                    ].compressed()
                    published = coarse_array[local_row, local_column]
                    reproduced = float(values.sum(dtype=np.float64))
                    tolerance = _sum_tolerance(values, published)
                    difference = reproduced - float(published)
                    if abs(difference) > tolerance:
                        raise ValueError(
                            "Fine-cell sum exceeds published coarse float32 precision: "
                            f"row={local_row}, column={local_column}, "
                            f"difference={difference}, tolerance={tolerance}."
                        )
                    measured_cells.append(
                        {
                            "parent_1_degree_cell": (
                                f"row-{int(COARSE_WINDOW.row_off) + local_row}"
                                f"-column-{int(COARSE_WINDOW.col_off) + local_column}"
                            ),
                            "published_population": float(published),
                            "reproduced_population": reproduced,
                            "difference": difference,
                            "absolute_tolerance": tolerance,
                        }
                    )

            fine_fixture = output_directory / "population_30_sec.tif"
            coarse_fixture = output_directory / "population_1_deg.tif"
            _write_crop(fine, fine_array, fine_window, fine_fixture)
            _write_crop(coarse, coarse_array, COARSE_WINDOW, coarse_fixture)
            fine_fixture_transform = window_transform(fine_window, fine.transform)
            longitudes, latitudes = _center_coordinates(
                fine_fixture_transform,
                fine_array.shape,
            )
            valid = ~fine_mask.ravel()
            cells = pd.DataFrame(
                {
                    "longitude": longitudes[valid],
                    "latitude": latitudes[valid],
                    "parent_1_degree_cell": _parent_cells()[valid],
                }
            )
            cells.to_csv(
                output_directory / "hazard_cells.csv",
                index=False,
                float_format=f"%.{COORDINATE_DECIMAL_PLACES}f",
                lineterminator="\n",
            )
            metadata = {
                "fixture": FIXTURE_NAME,
                "purpose": (
                    "Exact-coordinate tabular population assignment validation; "
                    "not a hazard or exposure result."
                ),
                "evidence": {
                    "category": "same-product cross-resolution consistency",
                    "proves": (
                        "Exact-coordinate tabular assignment preserves row order and "
                        "matches native GPW cells, while the grouped native values "
                        "agree with separately published GPW one-degree cells within "
                        "the recorded source-precision bound."
                    ),
                    "does_not_prove": (
                        "The two GPW grids are separately published outputs of one "
                        "product, not independent population sources. This fixture "
                        "does not validate a hazard or exposure result."
                    ),
                },
                "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
                "reuse": {
                    "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "source_record": "https://doi.org/10.7927/H4JW8BX5",
                    "statement": (
                        "The source product permits redistribution under CC BY 4.0. "
                        "Retain the citation when reusing the derived rasters or "
                        "coordinate table; this repository's MIT license does not "
                        "replace the source attribution."
                    ),
                },
                "citation": (
                    "Center for International Earth Science Information Network "
                    "(CIESIN), Columbia University. (2018). Gridded Population of "
                    "the World, Version 4 (GPWv4): Population Count, Revision 11. "
                    "NASA SEDAC. https://doi.org/10.7927/H4JW8BX5"
                ),
                "source_archives": {
                    "population_30_sec": {
                        "url": FINE_URL,
                        "archive": FINE_ARCHIVE,
                        "archive_member": FINE_MEMBER,
                        "bytes": FINE_ARCHIVE_BYTES,
                        "sha256": FINE_ARCHIVE_SHA256,
                    },
                    "population_1_deg": {
                        "url": COARSE_URL,
                        "archive": COARSE_ARCHIVE,
                        "archive_member": COARSE_MEMBER,
                        "bytes": COARSE_ARCHIVE_BYTES,
                        "sha256": COARSE_ARCHIVE_SHA256,
                    },
                },
                "source_grid": {
                    "crs": fine.crs.to_string(),
                    "dtype": fine.dtypes[0],
                    "nodata": fine.nodata,
                    "fine_shape": list(fine.shape),
                    "coarse_shape": list(coarse.shape),
                    "fine_transform": list(fine.transform)[:6],
                    "coarse_transform": list(coarse.transform)[:6],
                    "fine_bounds": list(fine.bounds),
                    "coarse_bounds": list(coarse.bounds),
                    "cells_per_coarse_side": CELLS_PER_COARSE_SIDE,
                },
                "fixture_grid": {
                    "coarse_window": [
                        int(COARSE_WINDOW.col_off),
                        int(COARSE_WINDOW.row_off),
                        int(COARSE_WINDOW.width),
                        int(COARSE_WINDOW.height),
                    ],
                    "fine_shape": list(fine_array.shape),
                    "fine_transform": list(fine_fixture_transform)[:6],
                    "fine_bounds": list(
                        rasterio.windows.bounds(fine_window, fine.transform)
                    ),
                    "fine_nodata": fine.nodata,
                    "coarse_shape": list(coarse_array.shape),
                    "coarse_transform": list(
                        window_transform(COARSE_WINDOW, coarse.transform)
                    )[:6],
                    "coarse_bounds": list(
                        rasterio.windows.bounds(COARSE_WINDOW, coarse.transform)
                    ),
                    "coarse_nodata": coarse.nodata,
                    "coordinate_decimal_places": COORDINATE_DECIMAL_PLACES,
                    "hazard_rows": len(cells),
                    "fine_masked_cells": int(fine_mask.sum()),
                },
                "derivation": {
                    "population_30_sec.tif": (
                        "Lossless GeoTIFF window of the aligned 240 by 240 native "
                        "source cells; compression changes storage, not raster values."
                    ),
                    "population_1_deg.tif": (
                        "Lossless GeoTIFF window of four complete one-degree source "
                        "cells; compression changes storage, not raster values."
                    ),
                    "hazard_cells.csv": (
                        "Pixel-center coordinates and parent one-degree labels for "
                        "each finite cell in the 30-arc-second crop; it is not a "
                        "hazard dataset."
                    ),
                },
                "published_oracle": {
                    "resolution": "1 degree",
                    "cells": measured_cells,
                    "tolerance_basis": (
                        "Half of each published float32 cell's ULP plus the "
                        "standard float64 summation roundoff bound for up to "
                        "14,400 valid fine values; relative tolerance is zero."
                    ),
                },
            }

    fixture_files = {
        path.name: _sha256(path)
        for path in (
            output_directory / "population_30_sec.tif",
            output_directory / "population_1_deg.tif",
            output_directory / "hazard_cells.csv",
        )
    }
    metadata["fixture_files"] = fixture_files
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_readme(output_directory / "README.md")


def main() -> None:
    """Regenerate the fixture after explicit download acknowledgement.

    Args:
        None.

    Returns:
        None.

    Examples:
        >>> main()
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-download",
        action="store_true",
        help="Confirm the roughly 405 MB authenticated official fine-raster download.",
    )
    parser.add_argument(
        "--cache-directory",
        type=Path,
        default=Path(user_cache_dir("population-exposure")) / "golden",
        help="Verified source-archive cache directory.",
    )
    parser.add_argument(
        "output_directory",
        type=Path,
        help="New directory that will receive the offline fixture.",
    )
    arguments = parser.parse_args()
    if not arguments.accept_download:
        parser.error("--accept-download is required before downloading GPW sources.")
    build_fixture(arguments.output_directory, arguments.cache_directory)
    print(f"Wrote {FIXTURE_NAME} to {arguments.output_directory}")


if __name__ == "__main__":
    main()
