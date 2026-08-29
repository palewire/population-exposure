# Agent guide

## Scope

This repository provides a small pandas-first library for calculating
population totals across ordered hazard bands. Keep the runtime dependencies
limited to NumPy and pandas.

Do not add command-line tools, file loading, spatial processing, coordinate
conversion, caching, parallelism, plugins, or compatibility layers without a
clear project decision.

## Development

Use the `src/population_exposure/` package layout. Keep the public API limited
to the symbols exported from `population_exposure.__init__`.

```sh
make bootstrap
make check
make test
make verify
```

Add tests for behavior changes, including property tests when they express a
general rule more clearly than examples. Keep documentation and
`CHANGELOG.md` aligned with public behavior.

Immediately after opening a pull request, request a GitHub Copilot review.
Before declaring the work complete, reply to and resolve every review comment
and fix every failing CI check.

Use Ruff for linting and formatting, ty for static type checks, and uv for
dependency management. Do not commit build output, virtual environments,
environment files, credentials, or agent scratch files.

Never publish a package, create a tag, or create a GitHub Release without
explicit human approval.
