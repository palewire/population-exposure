"""Safe, limited extraction helpers for catalog source archives."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path, PurePosixPath


def extract_members(
    archive: Path, destination: Path, prefixes: tuple[str, ...]
) -> None:
    """Extract selected archive members while rejecting unsafe paths.

    Args:
        archive: ZIP archive containing source files.
        destination: Directory receiving the extracted members.
        prefixes: Member path prefixes to retain, without file extensions.

    Returns:
        None. Raises ValueError if no requested members exist or any requested
        path is absolute, has parent traversal, Windows separators, or a drive
        prefix.

    Examples:
        >>> extract_members(Path("source.zip"), Path("extract"), ("layer",))
    """
    with zipfile.ZipFile(archive) as source:
        members = [
            name
            for name in source.namelist()
            if any(
                name == prefix
                or name.startswith((f"{prefix}.", f"{prefix}/", f"{prefix}\\"))
                for prefix in prefixes
            )
        ]
        if not members:
            raise ValueError(f"Archive does not contain requested members: {archive}.")
        for member in members:
            member_path = PurePosixPath(member)
            if (
                "\\" in member
                or member_path.is_absolute()
                or ".." in member_path.parts
                or any(":" in part for part in member_path.parts)
            ):
                raise ValueError(f"Archive member has an unsafe path: {member}.")
            target = destination.joinpath(*member_path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with (
                source.open(member) as input_stream,
                target.open("wb") as output_stream,
            ):
                shutil.copyfileobj(input_stream, output_stream)
