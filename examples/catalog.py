"""Inspect the built-in catalog without downloading provider data."""

import population_exposure as pe

for source in pe.populations.list():
    print(source.source_id, source.supported_years)

selected = pe.populations.info("worldpop-global-1km:2020")
print(selected.license)
print(selected.citation)
print(selected.download_size)
