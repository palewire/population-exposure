# population-exposure

`population-exposure` assigns population counts to tabular hazard rows, polygon
features, or raster cells. It joins exact table keys, calculates the population
inside polygons, or aligns a population raster with a hazard raster. It does
not group or sum results by hazard category: callers make those choices later
with pandas, NumPy, or their usual analysis tools.

## Installation

```console
pip install population-exposure
```

Python 3.11 or newer is required. GeoPandas, Pyogrio, Rasterio, Shapely, and
Exactextract install with the package.

## Quick starts

The documentation uses the conventional `pe` alias.

### Table

Give both tables the same cell key. Keys match exactly as written.

```python
import pandas as pd

import population_exposure as pe

hazard = pd.DataFrame({"cell": ["A", "B"], "risk": ["high", "low"]})
population = pd.DataFrame({"cell": ["A", "B"], "population": [100.0, 200.0]})

exposed = pe.assign_population(hazard, population, cell_columns="cell")
```

`exposed` keeps the hazard columns, index, and row order, with a new
`population` column. Every hazard row must have one matching population cell.

### Polygon map

Pass a GeoPandas frame of non-overlapping polygons and a population-count
GeoTIFF, an open Rasterio reader, or a catalog selection.

```python
import geopandas as gpd

import population_exposure as pe

hazard = gpd.read_file("flood-zones.geojson")
exposed = pe.assign_population(hazard, "population-counts.tif")
```

The result keeps the original geometry and coordinate system. Population cells
crossing a polygon boundary are counted by their covered share, so totals can
be fractional.

### Raster hazard

Raster assignment stays lazy. It returns a `pe.RasterAssignment` that reads
matching hazard and population arrays only when requested.

```python
import population_exposure as pe

assignment = pe.assign_population("hazard.tif", "worldpop-global-1km:2020")

for window, hazard_values, population_values in assignment.iter_blocks():
    # Analyze this bounded pair of masked NumPy arrays.
    pass
```

## `assign_population()` options

`assign_population(hazard, population, *, cell_columns=("longitude",
"latitude"), population_column="population", allow_overlaps=False,
hazard_band=None, conservation_tolerance=1e-6)` selects the matching behavior
from the input types.

| Option | Meaning |
| --- | --- |
| `hazard` | A pandas table, GeoPandas frame, supported vector file, GeoTIFF path, or open Rasterio reader. |
| `population` | A pandas table for table hazards. For maps and rasters, use a one-band population-count GeoTIFF, open Rasterio reader, or exact catalog selection. |
| `cell_columns` | One table key column or a sequence of table key columns. It is used only for table assignment. The default is `longitude` and `latitude`. |
| `population_column` | Name of the new output population column. It defaults to `population` and cannot overwrite an existing hazard column. |
| `allow_overlaps` | Allows overlapping vector polygons. It is `False` by default because adding independent overlapping totals would count shared areas more than once. It applies only to vector hazards. |
| `hazard_band` | A 1-based hazard band number. It is used only for multiband hazard rasters; a one-band raster selects band 1 automatically. |
| `conservation_tolerance` | The allowed relative difference when a population raster is reprojected onto a hazard raster. It applies only to raster hazards and defaults to `1e-6`. |

Table keys must be complete and unique in both inputs. They are never rounded,
trimmed, or otherwise normalized. An unmatched hazard row raises an error
rather than receiving a guessed or missing population value.

Vector inputs must have a coordinate system and valid, non-empty `Polygon` or
`MultiPolygon` geometry. Each polygon must cover valid population data.

## Raster results

`pe.RasterAssignment` records the hazard grid through `shape`, `crs`, `transform`,
`bounds`, and `hazard_band`. Its `attrs` mapping records the assignment method,
population totals, tolerance, and catalog facts when relevant.

- `read(window=None)` returns `(hazard, population)`: two masked NumPy arrays
  for the same cells, optionally limited to a Rasterio window.
- `iter_blocks()` yields `(window, hazard, population)` in bounded blocks, so a
  large raster need not be loaded all at once.

Population is aligned to the hazard coordinate system and grid with
coverage-weighted sum resampling, not interpolation. The package compares the
aligned total with the population covered by the hazard footprint and raises an
error when the difference is above:

```text
conservation_tolerance * max(1, covered_population)
```

Population outside the hazard footprint and population no-data cells are not
part of that covered total. Hazard no-data and areas without population remain
masked. Population inputs must be finite, non-negative **counts per cell**;
rasters explicitly marked as density are rejected instead of being silently
treated as counts.

Paths are opened and closed for each operation. An open Rasterio reader belongs
to the caller and is never closed by this package. Keep caller-owned hazard and
population readers open for as long as a `RasterAssignment` that refers to
them is used.

## Population catalog

Catalog selections must use the exact form `source:year`, where `source` is a
source ID, such as `worldpop-global-1km:2020`. Bare source names, `latest`,
malformed selections, and unsupported years raise errors. Ask the catalog about
a selection before downloading it:

```python
import population_exposure as pe

selection = pe.populations.info("worldpop-global-1km:2020")
print(selection.license)
print(selection.citation)
```

| Source ID | Years | Population meaning | How to obtain it |
| --- | --- | --- | --- |
| `worldpop-global-1km` | 2000-2020, yearly | Residential | Automatic official WorldPop download |
| `ghsl-r2023a-mollweide-1km` | 1975-2020, every 5 years | Residential | Automatic official JRC download |
| `gpwv4-r11-count` | 2000, 2005, 2010, 2015, 2020 | Residential | Official Earthdata download with a user-owned token, or manual registration |
| `chambers-hybrid` | 1950-2020, yearly | Residential | Automatic Zenodo download; the selected year is derived from its 21 age bands |
| `landscan-global` | 2000-2024, yearly | Ambient | Manual ORNL download and registration |

WorldPop, GHSL, and Chambers download automatically. GPW uses a user-owned
Earthdata token supplied as `earthdata_token=` to `populations.download()`, or
as `EARTHDATA_TOKEN` for direct catalog assignment. The token is only used for
the official request; it is not stored. GPW is a population-count product, not
density.

LandScan requires registration and license acceptance at the
[ORNL portal](https://landscan.ornl.gov/). Download one selected year yourself,
then validate and copy it into the local cache:

```python
import population_exposure as pe

path = pe.populations.register(
    "landscan-global:2024",
    "/downloads/landscan-global-2024.tif",
)
```

`populations.register()` also accepts a locally acquired GPW file. It checks
the source, year, grid, and population counts without changing the original
file. `populations.download()` caches verified catalog data; use `offline=True`
or `POPULATION_EXPOSURE_OFFLINE=1` to require an already cached or registered
file.

## API reference

The public API is intentionally small.

```{eval-rst}
.. autofunction:: population_exposure.assign_population

.. autoclass:: population_exposure.RasterAssignment
   :members: read, iter_blocks

.. automodule:: population_exposure.populations
   :members: list, info, download, register, SourceInfo, SelectionInfo
```

## Links

- [Source code](https://github.com/palewire/population-exposure)
- [Issue tracker](https://github.com/palewire/population-exposure/issues)
- [Changelog](https://github.com/palewire/population-exposure/blob/main/CHANGELOG.md)
- [PyPI package](https://pypi.org/project/population-exposure/)

## About

Ben Welsh released `population-exposure` in 2026 for assigning population
counts to hazard data. GitHub Copilot, an AI-powered text generator, helped
draft this documentation.
