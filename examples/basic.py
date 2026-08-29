"""Calculate exposure totals for a tiny invented grid."""

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
    labels=("Below -2", "-2 to 2", "At least 2"),
)

print(
    calculate_exposure(
        hazard,
        population,
        bands=bands,
        hazard_column="temperature",
    )
)
