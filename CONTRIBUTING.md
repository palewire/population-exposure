# Contributing

Clone the repository, enter the checkout, and install the locked development
environment:

```sh
make bootstrap
```

Run the complete local checks before opening a pull request:

```sh
make verify
```

The suite runs Ruff, ty, dependency and workflow checks, tests with coverage,
distribution checks, and an example against an installed wheel. You can run
`make test` for a faster test-only pass.

Before a release candidate, check the declared lower bounds with:

```sh
make minimum-dependency-check
```

This uses Python 3.11, installs the lowest compatible versions of the direct
runtime dependencies, and lets uv resolve compatible transitive dependencies.
Test-only tools are installed separately at their newest Python 3.11-compatible
versions so the check tests this project rather than obsolete test tooling.

Install the included Git hooks with:

```sh
uv run pre-commit install
```

Public behavior belongs in the README and `docs/`. Add a concise entry under
`Unreleased` in `CHANGELOG.md` for user-facing changes.

Immediately after opening a pull request, request a GitHub Copilot code review.
Address every review comment in its thread, resolve each answered thread, and
fix all CI failures before considering the pull request complete.
