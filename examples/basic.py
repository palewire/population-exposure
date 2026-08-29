"""Assign population to a tiny invented hazard table."""

import pandas as pd

from population_exposure import assign_population

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

exposed = assign_population(hazard, population, cell_columns="cell")
print(exposed)
print(exposed.groupby("severity")["population"].sum())
