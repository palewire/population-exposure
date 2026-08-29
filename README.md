# population-exposure

`population-exposure` is a small pandas-first library for totaling population
weights across ordered hazard bands. It joins tables on exact cell keys and can
calculate totals globally or for groups supplied by the caller.

## Install

```sh
pip install population-exposure
```

Python 3.11 or newer is required.

## Public API

The package exports three symbols:

```text
ExposureBand(
    id: str,
    lower_bound: float | None,
    upper_bound: float | None,
    label: str | None = None,
)

ExposureBands(bands: tuple[ExposureBand, ...])

calculate_exposure(
    hazard: pd.DataFrame,
    population: pd.DataFrame,
    *,
    bands: ExposureBands,
    hazard_column: str,
    population_column: str = "population",
    cell_columns: Sequence[str] = ("longitude", "latitude"),
    group_by: str | Sequence[str] | None = None,
) -> pd.DataFrame
```

Use `ExposureBands.from_breaks(...)` for the common case, or construct
`ExposureBand` objects directly when the bounds already exist.

## Global example

```python
import pandas as pd

from population_exposure import ExposureBands, calculate_exposure

hazard = pd.DataFrame(
    {
        "longitude": [10.0, 11.0, 12.0],
        "latitude": [20.0, 20.0, 20.0],
        "temperature": [-3.0, 0.0, 4.0],
    }
)
population = pd.DataFrame(
    {
        "longitude": [10.0, 11.0, 12.0],
        "latitude": [20.0, 20.0, 20.0],
        "population": [100.0, 200.0, 50.0],
    }
)
bands = ExposureBands.from_breaks(
    [-2.0, 2.0],
    ids=("below", "near", "above"),
)

result = calculate_exposure(
    hazard,
    population,
    bands=bands,
    hazard_column="temperature",
)
```

Values exactly on a break enter the higher band. Every band appears in the
result, including bands with zero population.

## Grouped example

Add a column to the population table and name it with `group_by`:

```python
population["region"] = ["west", "central", "east"]

regional = calculate_exposure(
    hazard,
    population,
    bands=bands,
    hazard_column="temperature",
    group_by="region",
)
```

Groups may overlap: the same cell can appear in several groups, but not twice
within one group. Each group is valid on its own; overlapping group totals are
not additive.

## Failure behavior

Cell and group keys are matched exactly. Missing keys, null keys, duplicate
cells, unmatched population cells, and invalid population weights raise clear
errors. Missing or infinite hazard values are excluded from both band totals
and represented population. The input data frames are never changed.

The package does not perform spatial assignment, coordinate conversion,
longitude normalization, population splitting, full-grid checks, or file
loading. See [the data model](docs/data-model.md) for complete semantics.

## Development

```sh
make bootstrap
make verify
```

The project uses uv, Ruff, ty, pytest, Hypothesis, and pre-commit.
