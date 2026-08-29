"""Inspect the built-in catalog without downloading provider data."""

from population_exposure import populations

for source in populations.list():
    print(source.source_id, source.supported_years)

selected = populations.info("worldpop-global-1km:2020")
print(selected.license)
print(selected.citation)
print(selected.download_size)
