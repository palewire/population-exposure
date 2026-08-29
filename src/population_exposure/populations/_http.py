"""Small streaming HTTP downloader with safe partial-file resume."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from http.client import IncompleteRead
from typing import TYPE_CHECKING, Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)

if TYPE_CHECKING:
    from email.message import Message
    from http.client import HTTPMessage
    from pathlib import Path
    from typing import IO

_CHUNK_SIZE = 1024 * 1024
_CONTENT_RANGE = re.compile(r"bytes (?P<start>[0-9]+)-[0-9]+/(?P<total>[0-9]+)")
_USER_AGENT = "population-exposure/0.1"


class _Response(Protocol):
    """The response surface used by the downloader and its tests."""

    headers: Message
    status: int

    def read(self, size: int = -1) -> bytes: ...

    def close(self) -> None: ...

    def __enter__(self) -> _Response: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool | None: ...


class _SafeRedirectHandler(HTTPRedirectHandler):
    """Never forward an authorization header to a different host."""

    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        """Return a redirected request with cross-host credentials removed."""
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:  # pragma: no cover - downloader uses GET and HEAD
            return None
        if urlsplit(req.full_url).netloc != urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


_OPENER = build_opener(_SafeRedirectHandler())


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Observed facts for a completed local download."""

    size: int
    sha256: str


def download_file(
    url: str,
    partial_path: Path,
    *,
    headers: dict[str, str] | None,
    max_bytes: int,
    exact_bytes: int | None,
    publisher_checksum: str | None,
) -> DownloadResult:
    """Stream one URL to a partial path and verify its final bytes."""
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    request_headers = {"User-Agent": _USER_AGENT, **(headers or {})}
    offset = partial_path.stat().st_size if partial_path.is_file() else 0
    if offset > max_bytes:
        partial_path.unlink()
        raise ValueError(
            f"Partial download exceeds the {max_bytes}-byte safety limit; "
            "the partial file was removed."
        )

    can_resume = offset > 0 and _server_supports_resume(url, request_headers)
    if can_resume:
        completed = _resume_download(
            url,
            partial_path,
            request_headers=request_headers,
            offset=offset,
            max_bytes=max_bytes,
            exact_bytes=exact_bytes,
        )
    else:
        completed = _fresh_download(
            url,
            partial_path,
            request_headers=request_headers,
            max_bytes=max_bytes,
            exact_bytes=exact_bytes,
        )

    size = partial_path.stat().st_size
    if exact_bytes is not None and size != exact_bytes:
        if size > exact_bytes:
            partial_path.unlink()
        raise ValueError(
            f"Download size is {size} bytes; expected exactly {exact_bytes} bytes."
        )
    if completed is not None and size != completed:
        raise ValueError(
            f"Download ended at {size} bytes; the server advertised {completed} bytes. "
            "The partial file was retained for resume."
        )

    digests = _file_digests(
        partial_path,
        include_md5=publisher_checksum is not None,
    )
    if publisher_checksum is not None:
        algorithm, expected = publisher_checksum.split(":", maxsplit=1)
        observed = digests.get(algorithm)
        if observed != expected:
            partial_path.unlink()
            raise ValueError(
                f"Publisher {algorithm} checksum verification failed; "
                "the partial file was removed."
            )
    return DownloadResult(size=size, sha256=digests["sha256"])


def sha256_file(path: Path) -> str:
    """Return a local file's SHA-256 digest."""
    return _file_digests(path, include_md5=False)["sha256"]


def _server_supports_resume(url: str, headers: dict[str, str]) -> bool:
    """Return whether a HEAD response explicitly advertises byte ranges."""
    request = Request(url, headers=headers, method="HEAD")  # noqa: S310
    try:
        with _open_url(request) as response:
            return response.headers.get("Accept-Ranges", "").lower() == "bytes"
    except (HTTPError, URLError, OSError):
        return False


def _resume_download(
    url: str,
    partial_path: Path,
    *,
    request_headers: dict[str, str],
    offset: int,
    max_bytes: int,
    exact_bytes: int | None,
) -> int | None:
    """Resume only when the server accepts and correctly describes the range."""
    headers = {**request_headers, "Range": f"bytes={offset}-"}
    request = Request(url, headers=headers, method="GET")  # noqa: S310
    try:
        response = _open_url(request)
    except (HTTPError, URLError, OSError) as error:
        raise ValueError(f"Could not download the official file from {url}.") from error

    content_range = response.headers.get("Content-Range", "")
    match = _CONTENT_RANGE.fullmatch(content_range)
    if response.status != 206 or match is None or int(match.group("start")) != offset:
        response.close()
        return _fresh_download(
            url,
            partial_path,
            request_headers=request_headers,
            max_bytes=max_bytes,
            exact_bytes=exact_bytes,
        )

    total = int(match.group("total"))
    if total > max_bytes or (exact_bytes is not None and total != exact_bytes):
        response.close()
        partial_path.unlink()
        raise ValueError(
            "The server advertised a file size outside the allowed verified size."
        )
    with response:
        _write_response(
            response,
            partial_path,
            mode="ab",
            starting_size=offset,
            max_bytes=max_bytes,
        )
    return total


def _fresh_download(
    url: str,
    partial_path: Path,
    *,
    request_headers: dict[str, str],
    max_bytes: int,
    exact_bytes: int | None,
) -> int | None:
    """Start a fresh download, replacing stale partial bytes only after connect."""
    request = Request(url, headers=request_headers, method="GET")  # noqa: S310
    try:
        response = _open_url(request)
    except (HTTPError, URLError, OSError) as error:
        raise ValueError(f"Could not download the official file from {url}.") from error
    if response.status != 200:
        response.close()
        raise ValueError(
            f"Official download returned unexpected HTTP status {response.status}."
        )

    content_length = _content_length(response.headers)
    if content_length is not None and (
        content_length > max_bytes
        or (exact_bytes is not None and content_length != exact_bytes)
    ):
        response.close()
        if partial_path.exists():
            partial_path.unlink()
        raise ValueError(
            "The server advertised a file size outside the allowed verified size."
        )
    with response:
        _write_response(
            response,
            partial_path,
            mode="wb",
            starting_size=0,
            max_bytes=max_bytes,
        )
    return content_length


def _write_response(
    response: _Response,
    path: Path,
    *,
    mode: str,
    starting_size: int,
    max_bytes: int,
) -> None:
    """Write bounded response chunks and preserve an interrupted partial file."""
    size = starting_size
    try:
        with path.open(mode) as output:
            while chunk := response.read(_CHUNK_SIZE):
                size += len(chunk)
                if size > max_bytes:
                    output.close()
                    path.unlink()
                    raise ValueError(
                        f"Download exceeds the {max_bytes}-byte safety limit; "
                        "the partial file was removed."
                    )
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    except (IncompleteRead, OSError) as error:
        raise ValueError(
            "Download was interrupted; the partial file was retained for resume."
        ) from error


def _content_length(headers: Message) -> int | None:
    """Parse a non-negative Content-Length header."""
    value = headers.get("Content-Length")
    if value is None:
        return None
    try:
        length = int(value)
    except ValueError as error:
        raise ValueError("Server returned an invalid Content-Length header.") from error
    if length < 0:
        raise ValueError("Server returned an invalid Content-Length header.")
    return length


def _file_digests(path: Path, *, include_md5: bool) -> dict[str, str]:
    """Calculate required local digests in one streaming pass."""
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False) if include_md5 else None
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_SIZE):
            sha256.update(chunk)
            if md5 is not None:
                md5.update(chunk)
    digests = {"sha256": sha256.hexdigest()}
    if md5 is not None:
        digests["md5"] = md5.hexdigest()
    return digests


def _open_url(request: Request) -> _Response:  # pragma: no cover - mocked in tests
    """Open one request through the credential-safe redirect handler."""
    return cast("_Response", _OPENER.open(request, timeout=60))
