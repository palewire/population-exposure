# Data model and behavior

`population-exposure` combines two pandas data frames:

- A **hazard table** contains the input rows to enrich. All of its columns are
  caller-defined.
- A **population table** contains one finite, non-negative population value per
  cell.

`assign_population()` joins the population value onto each hazard row using one
or more caller-selected cell-key columns. The default keys are `longitude` and
`latitude`; the default population column is `population`.

## Matching

Keys are matched exactly as supplied. The package does not round coordinates,
normalize values, reshape data, or infer cells. Null keys and duplicate key
combinations are rejected in both tables.

Every hazard row must match exactly one population row. Population rows that
are not needed by the hazard input are allowed, so a caller may submit a
sparse hazard subset against a larger population grid.

## Output

The result is a new data frame with:

- every hazard row and column unchanged;
- the original row order and index; and
- one appended population column containing floating-point values.

Population values are never rounded, so fractional allocations remain intact.
Missing values in hazard or category columns are preserved because the package
does not interpret those columns. Neither input data frame is mutated.

Callers can use pandas operations such as `.sum()`, `.groupby()`, and
`.pivot_table()` to define categories and summaries after assignment.

## Scope

This layer does not define hazard bands or categories, aggregate totals, create
shares, load files, process vector or raster data, split population between
cells, convert coordinate systems, or check that a grid is complete. Future
vector and raster support can perform its spatial work and return the same
kind of enriched, row-level data.
