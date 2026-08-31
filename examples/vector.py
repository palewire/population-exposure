"""Assign population to tiny invented hazard polygons."""

from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import population_exposure as pe

hazard = gpd.GeoDataFrame(
    {"risk": ["high", "low"]},
    geometry=[box(0, 0, 1.5, 2), box(1.5, 0, 2, 2)],
    crs="EPSG:3857",
)

with TemporaryDirectory() as directory:
    population_path = Path(directory) / "population.tif"
    with rasterio.open(
        population_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float64",
        crs="EPSG:3857",
        transform=from_origin(0, 2, 1, 1),
        nodata=-9999,
    ) as population:
        population.write(np.array([[100, 200], [300, 400]], dtype=float), 1)

    print("illustrative population raster (reference year: not applicable):")
    exposed = pe.assign_population(hazard, population_path)
    print(exposed)

    # A different coordinate system stops the work and explains both routes.
    other_system = hazard.to_crs("EPSG:4326")
    try:
        pe.assign_population(other_system, population_path)
    except pe.CrsMismatchError as error:
        print(f"\nStopped: {error}")

    # Ask for it and the package moves the boundaries carefully.
    reprojected = pe.assign_population(
        other_system,
        population_path,
        allow_reprojection=True,
    )
    print(f"\nReprojected totals: {reprojected['population'].tolist()}")

    # A polygon reaching past the raster also stops, for the same reason.
    reaching = gpd.GeoDataFrame(geometry=[box(1, 0, 4, 2)], crs="EPSG:3857")
    try:
        pe.assign_population(reaching, population_path)
    except pe.PartialCoverageError as error:
        print(f"\nStopped: {error}")

    partial = pe.assign_population(
        reaching,
        population_path,
        allow_partial_coverage=True,
    )
    print("\nPartial result:")
    print(partial[["population", "population_coverage_fraction"]])

    # Every result also says how much real population data it found.
    print("\nData support:")
    print(exposed[["population_data_fraction", "population_data_complete"]])

    # A polygon sitting only on no-data cells stops too. No-data is not zero.
    empty_population_path = Path(directory) / "empty-population.tif"
    with rasterio.open(
        empty_population_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float64",
        crs="EPSG:3857",
        transform=from_origin(0, 2, 1, 1),
        nodata=-9999,
    ) as population:
        population.write(np.array([[-9999, -9999], [300, 400]], dtype=float), 1)

    top_row = gpd.GeoDataFrame(geometry=[box(0, 1, 2, 2)], crs="EPSG:3857")
    try:
        pe.assign_population(top_row, empty_population_path)
    except pe.MissingPopulationDataError as error:
        print(f"\nStopped: {error}")

    unknown = pe.assign_population(
        top_row,
        empty_population_path,
        allow_missing_population_data=True,
    )
    print("\nUnknown rather than zero:")
    print(unknown[["population", "population_data_fraction"]])
