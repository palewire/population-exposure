"""Tests for the trusted-publishing workflow configuration."""

from __future__ import annotations

import re
from pathlib import Path

PUBLISH_ACTION_SHA = "dc37677b2e1c63e2034f94d8a5b11f265b73ba33"


def test_publish_jobs_share_a_pinned_action_with_metadata_checks() -> None:
    """Keep both package indexes on the approved publisher action.

    Args:
        None.

    Returns:
        None.

    Examples:
        This test reads the publish workflow from the repository checkout.
    """
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "publish.yml"
    ).read_text(encoding="utf-8")

    action_pins = re.findall(
        r"^\s*uses: pypa/gh-action-pypi-publish@([0-9a-f]{40})\s*$",
        workflow,
        flags=re.MULTILINE,
    )
    metadata_verification_disabled = re.compile(
        r"""(?im)^\s*verify[-_]metadata:\s*(?:["']?false["']?)(?:\s+#.*)?\s*$"""
    )

    assert action_pins == [PUBLISH_ACTION_SHA, PUBLISH_ACTION_SHA]
    assert not metadata_verification_disabled.search(workflow)
