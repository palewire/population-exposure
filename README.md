# population-exposure

[![CI](https://github.com/palewire/population-exposure/actions/workflows/ci.yml/badge.svg)](https://github.com/palewire/population-exposure/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/population-exposure.svg)](https://pypi.org/project/population-exposure/)
[![Python](https://img.shields.io/pypi/pyversions/population-exposure.svg)](https://pypi.org/project/population-exposure/)
[![License](https://img.shields.io/pypi/l/population-exposure.svg)](https://github.com/palewire/population-exposure/blob/main/LICENSE)

`population-exposure` adds population estimates to hazard tables, polygons, and
raster cells.

## Install

```sh
pip install population-exposure
```

Python 3.11 or newer is required.

## Quick start

```python
import pandas as pd

import population_exposure as pe

population_source = "illustrative counts"
population_year = "not applicable"
hazard = pd.DataFrame({"cell": ["A", "B"], "risk": ["high", "low"]})
population = pd.DataFrame({"cell": ["A", "B"], "population": [100, 200]})

exposed = pe.assign_population(hazard, population, cell_columns="cell")
print(f"{population_source} ({population_year}):")
print(exposed)
```

This toy table has no external population source or reference year. Replace it
with a documented source and year for a real analysis.

## Documentation

Read the [full documentation](https://palewi.re/docs/population-exposure/) for
vector and raster workflows, population sources, and the API reference.

## Project

[Source code](https://github.com/palewire/population-exposure) ·
[Issue tracker](https://github.com/palewire/population-exposure/issues) ·
[Contributing](https://github.com/palewire/population-exposure/blob/main/CONTRIBUTING.md) ·
[Code of conduct](https://github.com/palewire/population-exposure/blob/main/CODE_OF_CONDUCT.md) ·
[Changelog](https://github.com/palewire/population-exposure/blob/main/CHANGELOG.md) ·
[MIT License](https://github.com/palewire/population-exposure/blob/main/LICENSE)
