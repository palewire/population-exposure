"""Align population counts to a tiny invented hazard raster."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio.transform import from_origin

import population_exposure as pe

with TemporaryDirectory() as directory:
    directory_path = Path(directory)
    hazard_path = directory_path / "hazard.tif"
    population_path = directory_path / "population.tif"
    profile = {
        "driver": "GTiff",
        "width": 2,
        "height": 2,
        "count": 1,
        "crs": "EPSG:3857",
        "transform": from_origin(0, 2, 1, 1),
    }

    with rasterio.open(
        hazard_path,
        "w",
        **profile,
        dtype="int16",
        nodata=-32768,
    ) as hazard:
        hazard.write(np.array([[1, 2], [3, 4]], dtype=np.int16), 1)

    with rasterio.open(
        population_path,
        "w",
        **profile,
        dtype="float64",
        nodata=-9999,
    ) as population:
        population.write(np.array([[100, 200], [300, 400]], dtype=float), 1)

    exposed = pe.assign_population(hazard_path, population_path)
    hazard_values, population_values = exposed.read()
    print(hazard_values)
    print(population_values)
