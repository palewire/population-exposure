"""Regenerate the offline UNOSAT FL20221125COD vector golden fixture.

This maintenance-only script downloads about 910 MB. It is intentionally not
part of the package API or normal test suite.

Example:
    uv run python scripts/regenerate_unosat_vector_golden.py \
        --accept-download tests/data/unosat_fl20221125cod_basankusu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import geopandas as gpd
import rasterio
import shapely
from rasterio.features import geometry_mask, geometry_window
from rasterio.windows import transform as window_transform

from population_exposure import assign_population, populations

UNOSAT_WORKBOOK_URL = (
    "https://unosat.org/static/unosat_filesystem/3456/"
    "UNOSAT_Population_Exposure_FL20221125COD_November_2019_2020_2021_2022_"
    "Equateur_NordUbangi_SudUbangi_RDC_FR.xlsx"
)
UNOSAT_VECTOR_URL = (
    "https://unosat.org/static/unosat_filesystem/3456/FL20221125COD_SHP.zip"
)
OCHA_BOUNDARIES_URL = (
    "https://data.humdata.org/dataset/f42132b9-8cc6-4201-b020-9259c56e8868/"
    "resource/7514482b-f7af-4654-8ea2-e8a34a6acb6a/download/"
    "cod_admin_boundaries.shp.zip"
)
WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/"
    "0_Mosaicked/ppp_2020_1km_Aggregated.tif"
)
UNOSAT_WORKBOOK_SHA256 = "e6a68d2b0f1ad8d78c00d126cc04e9fca2ca96c5493727d54219f76b894970c6"  # pragma: allowlist secret
UNOSAT_VECTOR_SHA256 = "1003786f22094ca038b2ab39576614e4ac5d38606f18880257be83d84d819f33"  # pragma: allowlist secret
OCHA_BOUNDARIES_SHA256 = "a819758bebacb042a167701c84e0a227c96ad9ee769b39dfa9d3ec70977161fd"  # pragma: allowlist secret
WORLDPOP_SHA256 = "d98f9efd911ed4afed696dff9024b20a4e3ec5bf6c0e21f241a26725b7df0866"  # pragma: allowlist secret
DOWNLOAD_TIMEOUT_SECONDS = 120
VECTOR_MEMBER = (
    "FL20221125COD_SHP/"
    "VIIRS_20221101_20221128_MaximumFloodWaterExtent_Equateur_NordUbangi_"  # pragma: allowlist secret
    "SudUbangiProvinces_DRC"
)
PUBLISHED_POPULATION = 9570.69032327
PUBLISHED_AREA_SQKM = 393.735


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for one file.

    Args:
        path: File whose bytes will be hashed.

    Returns:
        Lowercase hexadecimal SHA-256 digest.

    Examples:
        >>> sha256(Path("data.bin"))
        '...'
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: str) -> None:
    """Download one source file and require its known content digest.

    Args:
        url: Stable official source URL.
        destination: Local path where the source is written.
        expected_sha256: Expected lowercase SHA-256 digest.

    Returns:
        None. Raises ValueError when the downloaded bytes differ from the
        recorded source digest.

    Examples:
        >>> download("https://example.test/source", Path("source"), "...")
    """
    if urlsplit(url).scheme != "https":
        raise ValueError(f"Source URL must use HTTPS: {url}.")
    with (
        urllib.request.urlopen(  # noqa: S310 -- HTTPS required above.
            url,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        ) as response,
        destination.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)
    observed = sha256(destination)
    if observed != expected_sha256:
        raise ValueError(
            f"Source checksum mismatch for {url}: expected {expected_sha256}, "
            f"got {observed}."
        )


def extract_members(
    archive: Path, destination: Path, prefixes: tuple[str, ...]
) -> None:
    """Extract only named shapefile groups from an archive.

    Args:
        archive: ZIP archive containing shapefile sidecar files.
        destination: Directory receiving selected archive members.
        prefixes: Member path prefixes to retain, without file extensions.

    Returns:
        None. Raises ValueError when no requested members are available.

    Examples:
        >>> extract_members(Path("source.zip"), Path("extract"), ("layer",))
    """
    with zipfile.ZipFile(archive) as source:
        members = [
            name
            for name in source.namelist()
            if any(name.startswith(prefix) for prefix in prefixes)
        ]
        if not members:
            raise ValueError(f"Archive does not contain requested members: {archive}.")
        for member in members:
            member_path = PurePosixPath(member)
            if (
                "\\" in member
                or member_path.is_absolute()
                or ".." in member_path.parts
                or any(":" in part for part in member_path.parts)
            ):
                raise ValueError(f"Archive member has an unsafe path: {member}.")
            target = destination.joinpath(*member_path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with (
                source.open(member) as input_stream,
                target.open("wb") as output_stream,
            ):
                shutil.copyfileobj(input_stream, output_stream)


def crop_population(
    population_path: Path,
    hazard: gpd.GeoDataFrame,
    destination: Path,
) -> tuple[float, float]:
    """Write a minimal WorldPop crop and calculate two coverage totals.

    Args:
        population_path: Full WorldPop 2020 count raster.
        hazard: Valid, non-empty Basankusu water-extent polygon in raster CRS.
        destination: GeoTIFF location for the cropped population cells.

    Returns:
        Pair containing the full crop total and a cell-center inclusion total.

    Examples:
        >>> crop_population(Path("worldpop.tif"), hazard, Path("crop.tif"))
        (..., ...)
    """
    with rasterio.open(population_path) as source:
        window = geometry_window(source, hazard.geometry, pad_x=1, pad_y=1)
        values = source.read(1, window=window, masked=True)
        profile = source.profile.copy()
        profile.update(
            height=window.height,
            width=window.width,
            transform=window_transform(window, source.transform),
            compress="deflate",
            predictor=3,
            zlevel=9,
        )
        with rasterio.open(destination, "w", **profile) as output:
            output.write(values.filled(source.nodata), 1)

        included = geometry_mask(
            hazard.geometry,
            out_shape=values.shape,
            transform=window_transform(window, source.transform),
            invert=True,
        )
        cell_center_total = float(values[included].sum(dtype="float64"))
        crop_total = float(values.sum(dtype="float64"))
    return crop_total, cell_center_total


def build_fixture(output_directory: Path, sources_directory: Path) -> None:
    """Build the Basankusu water-extent fixture from checked official sources.

    Args:
        output_directory: Empty directory that will receive offline test data.
        sources_directory: Directory holding the verified source downloads.

    Returns:
        None. Raises ValueError when source layers, dates, CRS, or population
        catalog identity differ from the recorded source facts.

    Examples:
        >>> build_fixture(Path("fixture"), Path("downloads"))
    """
    vector_archive = sources_directory / "unosat-vectors.zip"
    boundary_archive = sources_directory / "ocha-boundaries.zip"
    extracted = sources_directory / "extracted"
    extract_members(vector_archive, extracted, (VECTOR_MEMBER,))
    extract_members(
        boundary_archive,
        extracted,
        ("cod_admin2",),
    )

    source_water = gpd.read_file(
        extracted / f"{VECTOR_MEMBER}.shp",
        engine="pyogrio",
    )
    boundary = gpd.read_file(extracted / "cod_admin2.shp", engine="pyogrio")
    if (
        len(source_water) != 1
        or source_water.crs.to_string() != "EPSG:4326"
        or source_water["EventCode"].item() != "FL20221125COD"
        or source_water["Sensor_ID"].item() != "VIIRS-NOAA"
        or source_water["Water_Clas"].item()
        != "Maximum Satellite Observed Water (Cumulative)"
    ):
        raise ValueError("UNOSAT source is not the expected single VIIRS layer.")
    if str(source_water["Sensor_Dat"].item().date()) != "2022-11-28":
        raise ValueError("UNOSAT source date is not 2022-11-28.")
    source_geometry = source_water.geometry.iloc[0]
    invalid_reason = shapely.is_valid_reason(source_geometry)
    if source_water.geometry.is_valid.all() or invalid_reason is None:
        raise ValueError("Expected the recorded invalid UNOSAT source geometry.")

    district = boundary.loc[boundary["adm2_pcode"].eq("CD4107")].copy()
    if len(district) != 1 or district["adm2_name"].item() != "Basankusu":
        raise ValueError("OCHA boundary package does not identify Basankusu as CD4107.")

    repaired_water = shapely.make_valid(source_geometry)
    hazard = gpd.GeoDataFrame(
        {"district": ["Basankusu"], "pcode": ["CD4107"]},
        geometry=[repaired_water.intersection(district.geometry.iloc[0])],
        crs=source_water.crs,
    )
    if not hazard.geometry.is_valid.all() or hazard.geometry.is_empty.any():
        raise ValueError("Repaired Basankusu hazard geometry is not valid.")
    area_sqkm = float(hazard.to_crs(6933).area.item() / 1_000_000)
    if abs(area_sqkm - PUBLISHED_AREA_SQKM) > 0.001:
        raise ValueError(
            f"Basankusu area differs from published value: {area_sqkm} km2."
        )

    selected = populations.info("worldpop-global-1km:2020")
    if selected.official_url != WORLDPOP_URL:
        raise ValueError("WorldPop catalog URL no longer matches this fixture source.")

    output_directory.mkdir(parents=True, exist_ok=False)
    hazard_path = output_directory / "hazard.geojson"
    population_path = output_directory / "population.tif"
    metadata_path = output_directory / "metadata.json"
    hazard.to_file(hazard_path, driver="GeoJSON", engine="pyogrio")
    full_source_assigned = assign_population(
        hazard,
        sources_directory / "worldpop-2020.tif",
    )
    full_source_exactextract_total = float(full_source_assigned["population"].item())
    crop_total, cell_center_total = crop_population(
        sources_directory / "worldpop-2020.tif",
        hazard,
        population_path,
    )
    assigned = assign_population(hazard, population_path)
    exactextract_total = float(assigned["population"].item())
    with rasterio.open(population_path) as crop:
        crop_shape = list(crop.shape)
        crop_bounds = list(crop.bounds)

    metadata = {
        "fixture": "UNOSAT FL20221125COD Basankusu 2022 maximum surface water",
        "sources": {
            "unosat_workbook": {
                "sha256": UNOSAT_WORKBOOK_SHA256,
                "url": UNOSAT_WORKBOOK_URL,
            },
            "unosat_vector": {
                "sha256": UNOSAT_VECTOR_SHA256,
                "url": UNOSAT_VECTOR_URL,
            },
            "ocha_boundaries": {
                "release": "COD-AB v01, valid 2019-09-11",
                "sha256": OCHA_BOUNDARIES_SHA256,
                "url": OCHA_BOUNDARIES_URL,
            },
            "worldpop": {
                "catalog_selection": "worldpop-global-1km:2020",
                "sha256": WORLDPOP_SHA256,
                "url": WORLDPOP_URL,
            },
        },
        "source_layer": {
            "area_sqkm": 6760.57,
            "crs": source_water.crs.to_string(),
            "geometry_valid_before_repair": False,
            "invalid_reason": invalid_reason,
            "layer": f"{VECTOR_MEMBER}.shp",
            "repair": "shapely.make_valid",
            "sensor": "VIIRS-NOAA",
            "sensor_date": "2022-11-28",
            "source_feature_count": 1,
            "source_geometry_sha256": hashlib.sha256(source_geometry.wkb).hexdigest(),
            "source_overlap_pairs": 0,
            "water_class": "Maximum Satellite Observed Water (Cumulative)",
        },
        "district": {"name": "Basankusu", "pcode": "CD4107"},
        "published": {
            "area_sqkm": PUBLISHED_AREA_SQKM,
            "population": PUBLISHED_POPULATION,
        },
        "measured": {
            "area_sqkm": area_sqkm,
            "cell_center_population": cell_center_total,
            "crop_population": crop_total,
            "crop_raster_bounds": crop_bounds,
            "crop_raster_shape": crop_shape,
            "exactextract_population": exactextract_total,
            "exactextract_difference_from_published": (
                exactextract_total - PUBLISHED_POPULATION
            ),
            "cell_center_difference_from_published": (
                cell_center_total - PUBLISHED_POPULATION
            ),
            "full_source_exactextract_population": full_source_exactextract_total,
            "full_source_exactextract_difference_from_published": (
                full_source_exactextract_total - PUBLISHED_POPULATION
            ),
            "fixture_difference_from_full_source": (
                exactextract_total - full_source_exactextract_total
            ),
        },
        "methodology": {
            "exactextract": "Coverage-weighted population-count sum.",
            "cell_center_reference": (
                "Raster cell centers included by geometry_mask; a transparent "
                "reference for ArcGIS-style edge handling, not an ArcGIS result."
            ),
            "published_method_limit": (
                "The workbook names its sources but does not publish its ArcGIS "
                "zonal-statistics edge rule."
            ),
        },
        "fixture_files": {
            "hazard.geojson": sha256(hazard_path),
            "population.tif": sha256(population_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main() -> None:
    """Regenerate a fixture only after an explicit large-download acknowledgement.

    Args:
        None.

    Returns:
        None. Writes the fixture to the supplied output directory and prints its
        location. Raises ValueError when a source changes.

    Examples:
        >>> main()
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-download",
        action="store_true",
        help="Confirm the approximately 910 MB official-source download.",
    )
    parser.add_argument(
        "output_directory",
        type=Path,
        help="New directory that will receive the generated fixture.",
    )
    arguments = parser.parse_args()
    if not arguments.accept_download:
        parser.error(
            "--accept-download is required because this downloads about 910 MB."
        )
    if arguments.output_directory.exists():
        parser.error(f"Output directory already exists: {arguments.output_directory}")
    if not WORLDPOP_SHA256:
        raise ValueError("Set WORLDPOP_SHA256 before running this regeneration script.")

    with tempfile.TemporaryDirectory(prefix="population-exposure-unosat-") as temporary:
        downloads = Path(temporary)
        download(
            UNOSAT_WORKBOOK_URL,
            downloads / "unosat-workbook.xlsx",
            UNOSAT_WORKBOOK_SHA256,
        )
        download(
            UNOSAT_VECTOR_URL, downloads / "unosat-vectors.zip", UNOSAT_VECTOR_SHA256
        )
        download(
            OCHA_BOUNDARIES_URL,
            downloads / "ocha-boundaries.zip",
            OCHA_BOUNDARIES_SHA256,
        )
        download(WORLDPOP_URL, downloads / "worldpop-2020.tif", WORLDPOP_SHA256)
        build_fixture(arguments.output_directory, downloads)
    print(f"Wrote UNOSAT vector golden fixture to {arguments.output_directory}")


if __name__ == "__main__":
    main()
