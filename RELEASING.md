# Releasing

This project follows Semantic Versioning and Keep a Changelog.

## Checklist

1. Run `make verify`.
2. Review `CHANGELOG.md` and move the relevant `Unreleased` entries into a
   dated version section.
3. Update the version in `pyproject.toml`.
4. Open and merge a reviewed release pull request.
5. Obtain explicit human approval before creating a version tag, publishing
   the package, or creating a GitHub Release.

Agents may prepare release changes and run checks. They must not create tags,
GitHub Releases, or package publications without explicit human approval.
