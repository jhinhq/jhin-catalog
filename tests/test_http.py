"""Bounded, redirect-refusing fetches: caps, retries, and backoff."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Protocol

import httpx
import pytest

from jhin_catalog.http import (
    DEFAULT_USER_AGENT,
    MAX_ATTEMPTS,
    FetchError,
    RedirectRefused,
    ResponseTooLarge,
    backoff_delay,
    build_client,
    fetch,
    fetch_json,
)

type ClientFactory = Callable[[Callable[[httpx.Request], httpx.Response]], httpx.AsyncClient]


class RecordingSleep(Protocol):
    delays: list[float]

    async def __call__(self, seconds: float) -> None: ...


class WatchedStream(httpx.AsyncByteStream):
    """A body that records whether anything ever started reading it."""

    def __init__(self) -> None:
        self.started = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started = True
        yield b"x" * 4096

    async def aclose(self) -> None:
        return None


class CountingStream(httpx.AsyncByteStream):
    """A body that hands out chunks and remembers how many it gave."""

    def __init__(self, chunks: Sequence[bytes]) -> None:
        self.chunks = chunks
        self.yielded = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.yielded += 1
            yield chunk

    async def aclose(self) -> None:
        return None


# --- redirects --------------------------------------------------------------


async def test_a_redirect_is_refused_and_never_followed(
    mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    """A redirect can move a crawl onto a host nobody vetted, so it stops here."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://elsewhere.example/mcp"})

    async with mock_client(handler) as client:
        with pytest.raises(RedirectRefused):
            await fetch(client, "https://upstream.example/data", sleep=no_sleep)

    assert calls == ["https://upstream.example/data"]
    assert no_sleep.delays == []


# --- size caps --------------------------------------------------------------


async def test_a_declared_length_over_the_cap_is_refused_before_the_body_is_read(
    mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    stream = WatchedStream()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-length": "99999999"}, stream=stream)

    async with mock_client(handler) as client:
        with pytest.raises(ResponseTooLarge):
            await fetch(
                client, "https://upstream.example/big", max_response_bytes=1024, sleep=no_sleep
            )

    assert stream.started is False


async def test_a_body_that_crosses_the_cap_mid_flight_is_refused(
    mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    """No ``content-length`` at all, so the cap has to be enforced as it streams."""
    stream = CountingStream([b"x" * 600, b"y" * 600, b"z" * 600])

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    async with mock_client(handler) as client:
        with pytest.raises(ResponseTooLarge):
            await fetch(
                client, "https://upstream.example/big", max_response_bytes=1024, sleep=no_sleep
            )

    assert stream.yielded < 3


async def test_a_body_inside_the_cap_is_returned_whole(
    mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"ok":true}')

    async with mock_client(handler) as client:
        result = await fetch(client, "https://upstream.example/small", sleep=no_sleep)

    assert result.body == b'{"ok":true}'
    assert result.status_code == 200
    assert result.attempts == 1
    assert len(result.sha256) == 64


# --- retries ----------------------------------------------------------------


async def test_a_429_honours_a_numeric_retry_after(
    mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": "7"}, content=b"slow down")
        return httpx.Response(200, content=b"ok")

    async with mock_client(handler) as client:
        result = await fetch(client, "https://upstream.example/x", sleep=no_sleep)

    assert result.body == b"ok"
    assert result.attempts == 2
    assert no_sleep.delays == [7.0]


async def test_three_503s_back_off_by_doubling_and_then_succeed(
    mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls <= 3:
            return httpx.Response(503, content=b"unavailable")
        return httpx.Response(200, content=b"ok")

    async with mock_client(handler) as client:
        result = await fetch(client, "https://upstream.example/x", sleep=no_sleep)

    assert result.body == b"ok"
    assert no_sleep.delays == [0.5, 1.0, 2.0]


async def test_five_consecutive_500s_give_up_and_report_the_status(
    mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, content=b"boom")

    async with mock_client(handler) as client:
        with pytest.raises(FetchError) as raised:
            await fetch(client, "https://upstream.example/x", sleep=no_sleep)

    assert calls == MAX_ATTEMPTS
    assert raised.value.status_code == 500
    assert raised.value.url == "https://upstream.example/x"


async def test_a_404_is_final_on_the_first_try(
    mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    """A missing document does not become present by asking again."""
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, content=b"not found")

    async with mock_client(handler) as client:
        with pytest.raises(FetchError) as raised:
            await fetch(client, "https://upstream.example/gone", sleep=no_sleep)

    assert calls == 1
    assert raised.value.status_code == 404
    assert no_sleep.delays == []


async def test_a_transport_error_retries_under_the_same_policy(
    mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(200, content=b"ok")

    async with mock_client(handler) as client:
        result = await fetch(client, "https://upstream.example/x", sleep=no_sleep)

    assert result.body == b"ok"
    assert no_sleep.delays == [0.5]


def test_the_backoff_schedule_doubles_and_then_flattens_at_the_cap() -> None:
    assert [backoff_delay(attempt) for attempt in range(1, 9)] == [
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        30.0,
        30.0,
    ]


def test_a_retry_after_longer_than_the_computed_delay_wins() -> None:
    assert backoff_delay(1, retry_after=7.0) == 7.0


def test_a_retry_after_shorter_than_the_computed_delay_loses() -> None:
    assert backoff_delay(6, retry_after=1.0) == 16.0


def test_an_absurd_retry_after_is_clamped() -> None:
    """A server asking for an hour gets a minute; the crawl has a schedule too."""
    assert backoff_delay(1, retry_after=3600.0) == 60.0


# --- the client itself ------------------------------------------------------


async def test_the_catalog_user_agent_is_sent_and_is_not_the_stdlib_default(
    mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    """Smithery rejects ``Python-urllib`` outright, and anonymity is impolite."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["user-agent"])
        return httpx.Response(200, content=b"ok")

    async with mock_client(handler) as client:
        await fetch(client, "https://upstream.example/x", sleep=no_sleep)

    assert seen == [DEFAULT_USER_AGENT]
    assert not seen[0].startswith("Python-urllib")
    assert "jhin-catalog" in seen[0]


def test_the_client_never_follows_redirects_by_construction() -> None:
    client = build_client()
    assert client.follow_redirects is False


# --- json -------------------------------------------------------------------


async def test_fetch_json_parses_a_document_and_returns_the_raw_result(
    mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"servers":[]}')

    async with mock_client(handler) as client:
        payload, result = await fetch_json(client, "https://upstream.example/x", sleep=no_sleep)

    assert payload == {"servers": []}
    assert result.body == b'{"servers":[]}'


async def test_fetch_json_on_a_body_that_is_not_json_is_a_fetch_error(
    mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    async with mock_client(handler) as client:
        with pytest.raises(FetchError):
            await fetch_json(client, "https://upstream.example/x", sleep=no_sleep)


async def test_query_parameters_reach_the_wire(
    mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, content=b"{}")

    async with mock_client(handler) as client:
        await fetch(
            client,
            "https://upstream.example/x",
            params={"limit": 100, "seed": "42"},
            sleep=no_sleep,
        )

    assert seen[0].params["limit"] == "100"
    assert seen[0].params["seed"] == "42"
