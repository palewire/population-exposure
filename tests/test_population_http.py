"""Tests for bounded, resumable catalog downloads."""

from __future__ import annotations

import hashlib
from email.message import Message
from http.client import IncompleteRead
from io import BytesIO
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import Request

import pytest

from population_exposure.populations import _http

if TYPE_CHECKING:
    from pathlib import Path


class FakeResponse:
    """Small context-managed HTTP response used without network access."""

    def __init__(
        self,
        data: bytes = b"",
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        fail_on_read: int | None = None,
    ) -> None:
        self.status = status
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value
        self._body = BytesIO(data)
        self._reads = 0
        self._fail_on_read = fail_on_read
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        self._reads += 1
        if self._reads == self._fail_on_read:
            raise IncompleteRead(b"", 1)
        return self._body.read(size)

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def response_sequence(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[FakeResponse],
    requests: list[object],
) -> None:
    """Install an ordered fake opener."""

    def open_url(request):
        requests.append(request)
        return responses.pop(0)

    monkeypatch.setattr(_http, "_open_url", open_url)


def test_interrupted_download_resumes_only_after_range_advertisement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = b"a" * (1024 * 1024) + b"b" * 25
    partial = tmp_path / "file.partial"
    requests: list[object] = []
    first = FakeResponse(
        data,
        headers={"Content-Length": str(len(data)), "Accept-Ranges": "bytes"},
        fail_on_read=2,
    )
    response_sequence(monkeypatch, [first], requests)

    with pytest.raises(ValueError, match="retained for resume"):
        _http.download_file(
            "https://example.test/file",
            partial,
            headers=None,
            max_bytes=len(data) + 1,
            exact_bytes=None,
            publisher_checksum=None,
        )

    assert partial.stat().st_size == 1024 * 1024
    remaining = data[partial.stat().st_size :]
    responses = [
        FakeResponse(headers={"Accept-Ranges": "bytes"}),
        FakeResponse(
            remaining,
            status=206,
            headers={
                "Content-Range": (f"bytes {1024 * 1024}-{len(data) - 1}/{len(data)}")
            },
        ),
    ]
    response_sequence(monkeypatch, responses, requests)

    result = _http.download_file(
        "https://example.test/file",
        partial,
        headers=None,
        max_bytes=len(data) + 1,
        exact_bytes=None,
        publisher_checksum=None,
    )

    assert partial.read_bytes() == data
    assert result.size == len(data)
    assert requests[-1].get_header("Range") == f"bytes={1024 * 1024}-"


def test_bad_range_response_restarts_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = tmp_path / "file.partial"
    partial.write_bytes(b"old")
    replacement = b"complete replacement"
    requests: list[object] = []
    bad_range = FakeResponse(b"ignored", status=200)
    responses = [
        FakeResponse(headers={"Accept-Ranges": "bytes"}),
        bad_range,
        FakeResponse(
            replacement,
            headers={"Content-Length": str(len(replacement))},
        ),
    ]
    response_sequence(monkeypatch, responses, requests)

    result = _http.download_file(
        "https://example.test/file",
        partial,
        headers=None,
        max_bytes=100,
        exact_bytes=None,
        publisher_checksum=None,
    )

    assert bad_range.closed
    assert partial.read_bytes() == replacement
    assert result.size == len(replacement)


def test_server_without_range_support_restarts_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = tmp_path / "file.partial"
    partial.write_bytes(b"old")
    requests: list[object] = []
    response_sequence(
        monkeypatch,
        [
            FakeResponse(headers={"Accept-Ranges": "none"}),
            FakeResponse(b"new", headers={"Content-Length": "3"}),
        ],
        requests,
    )

    _http.download_file(
        "https://example.test/file",
        partial,
        headers=None,
        max_bytes=10,
        exact_bytes=None,
        publisher_checksum=None,
    )

    assert partial.read_bytes() == b"new"
    assert requests[-1].get_header("Range") is None


@pytest.mark.parametrize(
    ("headers", "message"),
    [
        ({"Content-Length": "101"}, "outside the allowed"),
        ({"Content-Length": "wrong"}, "invalid Content-Length"),
        ({"Content-Length": "-1"}, "invalid Content-Length"),
    ],
)
def test_invalid_advertised_sizes_fail_without_installing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    headers: dict[str, str],
    message: str,
) -> None:
    partial = tmp_path / "file.partial"
    response_sequence(monkeypatch, [FakeResponse(b"x", headers=headers)], [])

    with pytest.raises(ValueError, match=message):
        _http.download_file(
            "https://example.test/file",
            partial,
            headers=None,
            max_bytes=100,
            exact_bytes=None,
            publisher_checksum=None,
        )

    assert not partial.exists()


def test_streamed_oversize_and_exact_size_mismatch_remove_bad_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = tmp_path / "file.partial"
    response_sequence(monkeypatch, [FakeResponse(b"too large")], [])
    with pytest.raises(ValueError, match="safety limit"):
        _http.download_file(
            "https://example.test/file",
            partial,
            headers=None,
            max_bytes=3,
            exact_bytes=None,
            publisher_checksum=None,
        )
    assert not partial.exists()

    response_sequence(
        monkeypatch,
        [FakeResponse(b"short", headers={"Content-Length": "5"})],
        [],
    )
    with pytest.raises(ValueError, match="outside the allowed verified size"):
        _http.download_file(
            "https://example.test/file",
            partial,
            headers=None,
            max_bytes=10,
            exact_bytes=6,
            publisher_checksum=None,
        )
    assert not partial.exists()


def test_publisher_checksum_is_verified_and_bad_bytes_are_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = b"publisher bytes"
    partial = tmp_path / "file.partial"
    checksum = hashlib.md5(data, usedforsecurity=False).hexdigest()
    response_sequence(monkeypatch, [FakeResponse(data)], [])

    result = _http.download_file(
        "https://example.test/file",
        partial,
        headers={"Authorization": "Bearer private"},
        max_bytes=100,
        exact_bytes=None,
        publisher_checksum=f"md5:{checksum}",
    )
    assert result.sha256 == hashlib.sha256(data).hexdigest()

    partial.unlink()
    response_sequence(monkeypatch, [FakeResponse(data)], [])
    with pytest.raises(ValueError, match="checksum verification failed"):
        _http.download_file(
            "https://example.test/file",
            partial,
            headers=None,
            max_bytes=100,
            exact_bytes=None,
            publisher_checksum="md5:00000000000000000000000000000000",
        )
    assert not partial.exists()


def test_oversized_stale_partial_is_removed_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = tmp_path / "file.partial"
    partial.write_bytes(b"1234")
    monkeypatch.setattr(
        _http,
        "_open_url",
        lambda request: pytest.fail("network called"),
    )

    with pytest.raises(ValueError, match="Partial download exceeds"):
        _http.download_file(
            "https://example.test/file",
            partial,
            headers=None,
            max_bytes=3,
            exact_bytes=None,
            publisher_checksum=None,
        )
    assert not partial.exists()


def test_short_body_keeps_partial_for_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = tmp_path / "file.partial"
    response_sequence(
        monkeypatch,
        [FakeResponse(b"short", headers={"Content-Length": "10"})],
        [],
    )

    with pytest.raises(ValueError, match="retained for resume"):
        _http.download_file(
            "https://example.test/file",
            partial,
            headers=None,
            max_bytes=20,
            exact_bytes=None,
            publisher_checksum=None,
        )
    assert partial.read_bytes() == b"short"


def test_body_larger_than_exact_size_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = tmp_path / "file.partial"
    response_sequence(monkeypatch, [FakeResponse(b"seven!!")], [])

    with pytest.raises(ValueError, match="expected exactly 6"):
        _http.download_file(
            "https://example.test/file",
            partial,
            headers=None,
            max_bytes=20,
            exact_bytes=6,
            publisher_checksum=None,
        )
    assert not partial.exists()


def test_resume_rejects_wrong_total_and_download_connection_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = tmp_path / "file.partial"
    partial.write_bytes(b"abc")
    requests: list[object] = []
    response_sequence(
        monkeypatch,
        [
            FakeResponse(headers={"Accept-Ranges": "bytes"}),
            FakeResponse(
                b"rest",
                status=206,
                headers={"Content-Range": "bytes 3-6/100"},
            ),
        ],
        requests,
    )
    with pytest.raises(ValueError, match="outside the allowed"):
        _http.download_file(
            "https://example.test/file",
            partial,
            headers=None,
            max_bytes=10,
            exact_bytes=None,
            publisher_checksum=None,
        )
    assert not partial.exists()

    monkeypatch.setattr(
        _http,
        "_open_url",
        lambda request: (_ for _ in ()).throw(URLError("offline")),
    )
    with pytest.raises(ValueError, match="Could not download"):
        _http.download_file(
            "https://example.test/file",
            partial,
            headers=None,
            max_bytes=10,
            exact_bytes=None,
            publisher_checksum=None,
        )

    partial.write_bytes(b"abc")
    with pytest.raises(ValueError, match="Could not download"):
        _http._resume_download(
            "https://example.test/file",
            partial,
            request_headers={},
            offset=3,
            max_bytes=10,
            exact_bytes=None,
        )


def test_unexpected_success_status_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = FakeResponse(b"body", status=206)
    response_sequence(monkeypatch, [response], [])

    with pytest.raises(ValueError, match="unexpected HTTP status 206"):
        _http.download_file(
            "https://example.test/file",
            tmp_path / "file.partial",
            headers=None,
            max_bytes=10,
            exact_bytes=None,
            publisher_checksum=None,
        )
    assert response.closed


def test_head_failure_disables_resume_and_restarts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = tmp_path / "file.partial"
    partial.write_bytes(b"old")
    calls = 0

    def open_url(request):
        nonlocal calls
        calls += 1
        if request.get_method() == "HEAD":
            raise URLError("HEAD unavailable")
        return FakeResponse(b"new")

    monkeypatch.setattr(_http, "_open_url", open_url)

    _http.download_file(
        "https://example.test/file",
        partial,
        headers=None,
        max_bytes=10,
        exact_bytes=None,
        publisher_checksum=None,
    )
    assert calls == 2
    assert partial.read_bytes() == b"new"


def test_redirect_handler_removes_cross_host_authorization() -> None:
    handler = _http._SafeRedirectHandler()
    request = Request(
        "https://data.earthdata.nasa.gov/file",
        headers={"Authorization": "Bearer private"},
    )
    headers = Message()

    redirected = handler.redirect_request(
        request,
        BytesIO(),
        302,
        "Found",
        headers,
        "https://urs.earthdata.nasa.gov/login",
    )
    assert redirected is not None
    assert redirected.get_header("Authorization") is None

    same_host = handler.redirect_request(
        request,
        BytesIO(),
        302,
        "Found",
        headers,
        "https://data.earthdata.nasa.gov/other",
    )
    assert same_host is not None
    assert same_host.get_header("Authorization") == "Bearer private"


def test_short_exact_download_is_retained_for_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = tmp_path / "file.partial"
    response_sequence(monkeypatch, [FakeResponse(b"short")], [])

    with pytest.raises(ValueError, match="expected exactly 6"):
        _http.download_file(
            "https://example.test/file",
            partial,
            headers=None,
            max_bytes=10,
            exact_bytes=6,
            publisher_checksum=None,
        )
    assert partial.read_bytes() == b"short"


def test_bad_advertised_size_removes_an_existing_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = tmp_path / "file.partial"
    partial.write_bytes(b"old")
    response_sequence(
        monkeypatch,
        [
            FakeResponse(headers={"Accept-Ranges": "none"}),
            FakeResponse(b"", headers={"Content-Length": "100"}),
        ],
        [],
    )

    with pytest.raises(ValueError, match="outside the allowed"):
        _http.download_file(
            "https://example.test/file",
            partial,
            headers=None,
            max_bytes=10,
            exact_bytes=None,
            publisher_checksum=None,
        )
    assert not partial.exists()
