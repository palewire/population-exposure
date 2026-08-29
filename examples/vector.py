"""Assign population to tiny invented hazard polygons."""

from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from population_exposure import assign_population

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

    exposed = assign_population(hazard, population_path)
    print(exposed)
