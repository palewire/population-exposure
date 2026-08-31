"""Tests for the continuous-integration workflow configuration."""

from __future__ import annotations

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

    assert "test-minimum-dependencies:" in workflow
    assert "name: Test minimum dependencies" in workflow
    assert "run: uv python install 3.11" in workflow
    assert "run: make minimum-dependency-check UV_PYTHON=3.11" in workflow
    assert "needs: [check, test-python, test-minimum-dependencies]" in workflow
