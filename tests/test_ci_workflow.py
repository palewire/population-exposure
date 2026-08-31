"""Tests for the continuous-integration workflow configuration."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


@pytest.mark.unit
def test_ci_checks_minimum_dependencies_on_python_311() -> None:
    """Keep the release-candidate lower-bound check in CI.

    Args:
        None.

    Returns:
        None.

    Examples:
        This test reads the CI workflow from the repository checkout.
    """
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")

    minimum_job = re.search(
        r"(?ms)^  test-minimum-dependencies:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:|\Z)",
        workflow,
    )
    assert minimum_job
    minimum_job_body = minimum_job.group("body")
    assert re.search(r"(?m)^\s+name:\s+Test minimum dependencies\s*$", minimum_job_body)
    assert re.search(r"(?m)^\s+run:\s+uv python install\s+3\.11\s*$", minimum_job_body)
    assert re.search(
        r"(?m)^\s+run:\s+make minimum-dependency-check\s+UV_PYTHON=3\.11\s*$",
        minimum_job_body,
    )

    build_job = re.search(r"(?ms)^  build:\n(?:(?!^  [a-zA-Z0-9_-]+:).)*", workflow)
    assert build_job
    assert re.search(r"(?m)^\s+needs:.*test-minimum-dependencies", build_job.group())
