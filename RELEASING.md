# Releasing

This project follows Semantic Versioning and Keep a Changelog.

## Candidate preparation

These steps are safe to prepare in a branch or Git worktree:

1. Install the locked development dependencies before running checks:

   ```sh
   make bootstrap
   make verify
   ```

   A fresh clone or worktree does not include the test dependencies that
   `make verify` needs, including `pytest-cov`.
2. Review `CHANGELOG.md` and move the relevant `Unreleased` entries into a
   dated version section.
3. Set the intended candidate version in `pyproject.toml`, such as
   `0.1.0rc1`.
4. Open and merge a reviewed release-candidate pull request.

## Actions requiring explicit human approval

Do not perform any of the following without explicit human approval:

1. Create or push a version tag.
2. Run the **Publish package** workflow to publish to TestPyPI.
3. Publish to PyPI or create a GitHub Release.

### 0.1.0rc1 TestPyPI flow

After approval, tag and push the merged candidate as `v0.1.0rc1`, then use
**Actions > Publish package > Run workflow** with the `testpypi` target and
the pushed `v0.1.0rc1` ref. The workflow builds and publishes that exact ref
to TestPyPI.

Install the candidate by pinning its full version, rather than requesting an
unqualified package name:

```sh
uv venv /tmp/population-exposure-0.1.0rc1
uv pip install \
  --python /tmp/population-exposure-0.1.0rc1/bin/python \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "population-exposure==0.1.0rc1"
```

TestPyPI already has `population-exposure==0.1.0`. The `==0.1.0rc1` pin
ensures the candidate is installed instead of that existing release.
