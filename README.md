# population-exposure

`population-exposure` assigns population counts to hazard rows, polygon
features, or raster cells. You provide the hazard data and a local population
dataset. The package handles exact table joins, vector reprojection and
coverage, or raster alignment. Grouping and analysis remain ordinary pandas or
NumPy work.

## Install

```sh
pip install population-exposure
```

Python 3.11 or newer is required. GeoPandas, Pyogrio, Rasterio, Shapely, and
Exactextract are installed as normal dependencies.

## Public API

The package has one operation:

```python
def assign_population(
    hazard,
    population,
    *,
    cell_columns: str | Sequence[str] = ("longitude", "latitude"),
    population_column: str = "population",
    allow_overlaps: bool = False,
    hazard_band: int | None = None,
    conservation_tolerance: float = 1e-6,
) -> pd.DataFrame | gpd.GeoDataFrame | RasterAssignment: ...
```

| Hazard input | Population input | Return type |
|---|---|---|
| `pandas.DataFrame` | `pandas.DataFrame` | `pandas.DataFrame` |
| `geopandas.GeoDataFrame`, GeoJSON, Shapefile, or GeoPackage | local population-count GeoTIFF or open Rasterio reader | `geopandas.GeoDataFrame` |
| local hazard GeoTIFF or open Rasterio reader | local population-count GeoTIFF or open Rasterio reader | `RasterAssignment` |

`RasterAssignment` is the documented raster result type. It stores grid
metadata and source references, not a long DataFrame or eagerly loaded global
array.

## Tables

```python
import pandas as pd

from population_exposure import assign_population

hazard = pd.DataFrame(
    {
        "cell": ["A", "B", "C", "D"],
        "county": ["North", "North", "South", "South"],
        "severity": ["warning", "watch", "warning", pd.NA],
    }
)
population = pd.DataFrame(
    {
        "cell": ["D", "B", "A", "C"],
        "population": [400.5, 200.0, 100.0, 300.25],
    }
)

exposed = assign_population(hazard, population, cell_columns="cell")
```

The output preserves every hazard row and column, its index and order, missing
hazard values, and fractional population values. Cell keys are matched exactly;
they are not rounded or normalized.

| cell | county | severity | population |
|---|---|---|---:|
| A | North | warning | 100.0 |
| B | North | watch | 200.0 |
| C | South | warning | 300.25 |
| D | South | `<NA>` | 400.5 |

Use ordinary pandas operations for analysis:

```python
total = exposed["population"].sum()
by_county = exposed.groupby("county")["population"].sum()
by_severity = exposed.groupby("severity", dropna=False)["population"].sum()
county_severity = exposed.groupby(["county", "severity"], dropna=False)[
    "population"
].sum()
```

## Vector maps

```python
import geopandas as gpd
from shapely.geometry import box

from population_exposure import assign_population

hazard = gpd.GeoDataFrame(
    {"risk": ["high", "low"]},
    geometry=[box(0, 0, 1.5, 2), box(1.5, 0, 2, 2)],
    crs="EPSG:3857",
)

exposed = assign_population(hazard, "population-counts.tif")
```

The vector result preserves the original columns, geometry, CRS, index, and row
order, then appends `population`. Geometries are copied and reprojected to the
population raster CRS only for calculation. Exactextract sums each population
cell in proportion to polygon coverage, so boundary allocations remain
fractional and are never rounded. Population nodata is excluded.

Missing, empty, invalid, or non-polygon geometry raises an error. Every feature
must cover at least one valid population cell. Polygon overlaps raise by
default, because independently calculated totals could then be summed twice.
Use `allow_overlaps=True` only when independent, non-additive feature totals are
intentional.

Vector files are read with Pyogrio. A GeoPackage with multiple layers should be
opened by the caller as a `GeoDataFrame` so the layer choice is explicit.

## Raster maps

```python
from population_exposure import assign_population

exposed = assign_population("hazard.tif", "population-counts.tif")

hazard_values, population_values = exposed.read()

for window, hazard_block, population_block in exposed.iter_blocks():
    # Analyze one bounded pair of NumPy masked arrays at a time.
    pass
```

`RasterAssignment` exposes:

- `shape`, `crs`, `transform`, `bounds`, and the selected `hazard_band`;
- `attrs`, including source, covered, and aligned population totals;
- `read(window=None)`, returning hazard and aligned-population NumPy masked
  arrays for the same cells; and
- `iter_blocks()`, yielding `(window, hazard, population)` without loading the
  entire grid.

Population is aligned to the hazard grid and CRS with Rasterio/GDAL's
coverage-weighted `sum` resampling. Bilinear interpolation is never used. The
package compares the aligned total with Exactextract's coverage-aware
population total inside the hazard footprint. The allowed absolute difference
is:

```text
conservation_tolerance * max(1, covered_population)
```

The default tolerance is `1e-6`. Population outside the hazard extent and
population nodata are excluded from the covered total. Hazard nodata and areas
outside population coverage remain masked in reads.

A one-band hazard is selected automatically. Multiband hazards require an
explicit one-based `hazard_band`. Population rasters must contain exactly one
band of finite, non-negative counts. Metadata that explicitly describes density
is rejected; the package does not convert density to counts.

Path inputs are opened and closed for each operation. Caller-owned Rasterio
readers are never closed and must remain open while a `RasterAssignment` uses
them.

## Scope

The package does not define bands or categories, group results, calculate
shares, download population sources, convert density rasters, or resolve
overlapping polygons. Callers can use normal pandas, NumPy, or xarray operations
after assignment.

See [the data model](docs/data-model.md) and the executable
[table](examples/basic.py), [vector](examples/vector.py), and
[raster](examples/raster.py) examples.

## Development

```sh
make bootstrap
make verify
```

The project uses uv, Ruff, ty, pytest, Hypothesis, and pre-commit.
