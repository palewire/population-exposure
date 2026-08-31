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

    assert re.search(r"(?m)^  test-minimum-dependencies:\s*$", workflow)
    assert re.search(
        r"(?ms)^  test-minimum-dependencies:\n.*?^\s+name:\s+Test minimum dependencies\s*$",
        workflow,
    )
    assert re.search(r"(?m)^\s+run:\s+uv python install\s+3\.11\s*$", workflow)
    assert re.search(
        r"(?m)^\s+run:\s+make minimum-dependency-check\s+UV_PYTHON=3\.11\s*$",
        workflow,
    )

    build_job = re.search(r"(?ms)^  build:\n(?:(?!^  \w).)*", workflow)
    assert build_job
    assert re.search(r"(?m)^\s+needs:.*test-minimum-dependencies", build_job.group())
