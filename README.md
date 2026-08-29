# population-exposure

`population-exposure` is a small pandas-first library that assigns a population
value to each hazard row. It handles the exact-key join and input checks, then
leaves categories, totals, shares, pivots, and charts to pandas.

## Install

```sh
pip install population-exposure
```

Python 3.11 or newer is required.

## Public API

The package exports one function:

```python
def assign_population(
    hazard: pd.DataFrame,
    population: pd.DataFrame,
    *,
    cell_columns: Sequence[str] = ("longitude", "latitude"),
    population_column: str = "population",
) -> pd.DataFrame: ...
```

The named population column is both the source column in `population` and the
new output column.

## Example

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

`exposed` contains the original rows and columns, in their original order, plus
the assigned population:

| cell | county | severity | population |
|---|---|---|---:|
| A | North | warning | 100.0 |
| B | North | watch | 200.0 |
| C | South | warning | 300.25 |
| D | South | missing | 400.5 |

Use ordinary pandas operations for analysis:

```python
total = exposed["population"].sum()
by_county = exposed.groupby("county")["population"].sum()
by_severity = exposed.groupby("severity", dropna=False)["population"].sum()
county_severity = exposed.groupby(["county", "severity"], dropna=False)[
    "population"
].sum()
```

## Behavior

Cell keys are matched exactly as supplied; they are never rounded or
normalized. Every hazard row must match one population row. Extra population
rows are allowed, which lets callers assign population to a subset of a larger
grid.

The function preserves every hazard column, row order, index, missing hazard
values, and fractional population values. It does not mutate either input.
Missing columns, null keys, duplicate keys, unmatched hazard rows, and
non-numeric, non-finite, or negative population values raise clear errors.

The package does not define hazard categories or calculate grouped summaries.
It also does not load files, process spatial data, convert coordinates, or
check that a grid is complete. See [the data model](docs/data-model.md) for
complete tabular semantics. Vector and raster assignment can build on the same
row-level result in future work.

## Development

```sh
make bootstrap
make verify
```

The project uses uv, Ruff, ty, pytest, Hypothesis, and pre-commit.
