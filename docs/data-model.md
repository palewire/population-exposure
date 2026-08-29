# Data model and behavior

`population-exposure` combines two tables:

- A **hazard table** has one row per cell and one numeric hazard value.
- A **population table** has one non-negative population weight per cell, or
  per cell and group when grouped output is requested.

The join uses the cell columns exactly as supplied. The package does not round,
normalize, reshape, load, or map spatial data. A population row without one
matching hazard row raises an error. Hazard cells that have no population row
are ignored.

## Exposure bands

Bands are ordered from low to high and must cover every finite number without
gaps or overlaps. Their lower bounds are included and upper bounds are
excluded. A value exactly equal to a break therefore enters the higher band.

Hazard values that are missing or infinite are omitted. Their population is
not included in `represented_population`.

## Output

The result always has one row for every band and observed group combination.
An ungrouped calculation always returns every band, even when no population is
represented. Empty bands have a `population_total` of `0.0`.

`population_fraction` is each band's share of `represented_population`. It is
missing when represented population is zero, because no share can be
calculated. Totals are floating-point values and are not rounded.

Group columns come from the population table. A cell can appear in several
groups, provided it appears only once within each group. This supports
overlapping regions, but totals from overlapping groups should not be added
together.

## Scope

The package does not assign cells to polygons, split population between cells,
convert coordinate systems, normalize longitude, check that a grid is
complete, or load files. Callers prepare pandas data frames before calculation.
