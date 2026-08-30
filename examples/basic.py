"""Assign population to a tiny invented hazard table."""

import pandas as pd

import population_exposure as pe

hazard = pd.DataFrame(
    {
        "cell": ["A", "B", "C", "D"],
        "county": ["North", "North", "South", "South"],
        "severity": ["warning", "watch", "warning", "advisory"],
    }
)
population = pd.DataFrame(
    {
        "cell": ["D", "B", "A", "C"],
        "population": [400.5, 200.0, 100.0, 300.25],
    }
)

exposed = pe.assign_population(hazard, population, cell_columns="cell")
print(exposed)
print(exposed.groupby("severity")["population"].sum())
