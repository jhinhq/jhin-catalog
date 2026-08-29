"""One bounded, redirect-refusing GET, with a deterministic retry schedule.

Every byte the catalog ingests arrives through :func:`fetch`.  The posture
is the parent repo's connector client: redirects are never followed, the
response size is capped on the stream rather than only on the declared
``content-length``, and no error message ever carries a header, a body, or
the userinfo half of a URL.  Retries are jitter-free and the ``sleep``
callable is injected, so a crawl replays identically in a test.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable, Mapping
from typing import Final, NoReturn, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field

from jhin_catalog.types import CatalogError, JsonValue, payload_sha256

DEFAULT_USER_AGENT: Final[str] = "jhin-catalog/0.1.0 (+https://github.com/jhin-dev/jhin-catalog)"
DEFAULT_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(
    connect=10.0, read=30.0, write=10.0, pool=10.0
)
MAX_RESPONSE_BYTES: Final[int] = 32 * 1024 * 1024
MAX_ATTEMPTS: Final[int] = 5
RETRY_STATUSES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})
BACKOFF_BASE_SECONDS: Final[float] = 0.5
BACKOFF_CAP_SECONDS: Final[float] = 30.0
RETRY_AFTER_CAP_SECONDS: Final[float] = 60.0


class FetchError(CatalogError):
    """A display-safe fetch failure carrying the status and the URL."""

    def __init__(self, message: str, *, url: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.url = url
        self.status_code = status_code


class ResponseTooLarge(FetchError):
    """The upstream offered more bytes than the caller agreed to hold."""


class RedirectRefused(FetchError):
    """A redirect arrived; following one would leave the audited host."""


class _Retry(Exception):
    """Internal signal: this attempt failed in a way worth trying again."""

    def __init__(self, *, delay: float, error: FetchError) -> None:
        super().__init__(str(error))
        self.delay = delay
        self.error = error


class FetchResult(BaseModel):
    """One completed response: its bytes, their hash, and what it cost."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    status_code: int = Field(ge=100, le=599)
    body: bytes
    sha256: str
    attempts: int = Field(ge=1)
    content_type: str = ""


def build_client(
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """An ``AsyncClient`` with ``follow_redirects=False`` and the catalog UA.

    Smithery rejects the stdlib's default agent outright, and an anonymous
    crawler is impolite besides, so the agent names the project and links
    somewhere an operator can complain.
    """
    return httpx.AsyncClient(
        headers={"user-agent": user_agent},
        timeout=timeout,
        transport=transport,
        follow_redirects=False,
    )


def backoff_delay(attempt: int, *, retry_after: float | None = None) -> float:
    """Seconds to wait before ``attempt`` (1-based). Deterministic, no jitter.

    No jitter, because a reproducible wait schedule is worth more here than
    herd-avoidance: one crawler runs at a time, and a test asserts on the
    exact list of delays.  A server that asks for longer than the doubling
    schedule gets what it asked for, up to ``RETRY_AFTER_CAP_SECONDS``.
    """
    # The exponent is clamped before it is applied so a caller passing an
    # absurd attempt number cannot overflow the doubling into an error.
    base = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * 2.0 ** min(attempt - 1, 32))
    if retry_after:
        return max(base, min(retry_after, RETRY_AFTER_CAP_SECONDS))
    return base


def _reject_constant(name: str) -> NoReturn:
    """Refuse the JSON constants canonical encoding cannot round-trip."""
    raise ValueError(f"{name} is not a permitted JSON value")


def _safe_url(url: httpx.URL) -> str:
    """The URL minus any userinfo and any query, safe to render in a message.

    This value becomes ``FetchResult.url`` and the ``url`` on every
    ``FetchError``, all of which the CLI prints to a public workflow log. No
    credential in this repository travels in a query string today — every one
    is a header — but ``raw_path`` carries the query, so the first source that
    authenticates by parameter would leak its key here with nothing in the
    codebase objecting. Which endpoint failed is all a message needs.
    """
    path = url.raw_path.decode("ascii").split("?", 1)[0]
    return f"{url.scheme}://{url.netloc.decode('ascii')}{path}"


def _declared_length(response: httpx.Response) -> int | None:
    """The ``content-length`` header as a non-negative int, when it is one."""
    raw = response.headers.get("content-length")
    if raw is None:
        return None
    try:
        value = int(raw, 10)
    except ValueError:
        return None
    return value if value >= 0 else None


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """The ``Retry-After`` header when it is a plain number of seconds.

    The HTTP-date form is ignored on purpose: reading it would need a wall
    clock, and nothing in this package is allowed one.
    """
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(raw.strip())
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0.0:
        return None
    return value


async def _read_bounded(
    response: httpx.Response, *, safe_url: str, max_response_bytes: int
) -> bytes:
    """Accumulate the body, refusing it the moment it crosses the cap."""
    status = response.status_code
    declared = _declared_length(response)
    if declared is not None and declared > max_response_bytes:
        raise ResponseTooLarge(
            f"Response declares {declared} bytes, over the {max_response_bytes} byte cap",
            url=safe_url,
            status_code=status,
        )
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > max_response_bytes:
            raise ResponseTooLarge(
                f"Response exceeded the {max_response_bytes} byte cap mid-stream",
                url=safe_url,
                status_code=status,
            )
        body.extend(chunk)
    return bytes(body)


async def _attempt_once(
    client: httpx.AsyncClient,
    request: httpx.Request,
    *,
    safe_url: str,
    attempt: int,
    max_response_bytes: int,
) -> FetchResult:
    """One send-and-read. Raises ``_Retry`` when another go is warranted."""
    try:
        response = await client.send(request, stream=True, follow_redirects=False)
    except httpx.TransportError as exc:
        error = FetchError(f"Transport failure ({type(exc).__name__})", url=safe_url)
        raise _Retry(delay=backoff_delay(attempt), error=error) from None

    try:
        status = response.status_code
        if 300 <= status < 400:
            raise RedirectRefused(
                f"Refusing to follow the {status} redirect", url=safe_url, status_code=status
            )
        if status in RETRY_STATUSES:
            delay = backoff_delay(attempt, retry_after=_retry_after_seconds(response))
            error = FetchError(f"Upstream returned {status}", url=safe_url, status_code=status)
            raise _Retry(delay=delay, error=error)
        if not 200 <= status < 300:
            raise FetchError(f"Upstream returned {status}", url=safe_url, status_code=status)
        try:
            body = await _read_bounded(
                response, safe_url=safe_url, max_response_bytes=max_response_bytes
            )
        except httpx.TransportError as exc:
            error = FetchError(f"Transport failure ({type(exc).__name__})", url=safe_url)
            raise _Retry(delay=backoff_delay(attempt), error=error) from None
        return FetchResult(
            url=safe_url,
            status_code=status,
            body=body,
            sha256=payload_sha256(body),
            attempts=attempt,
            content_type=response.headers.get("content-type", ""),
        )
    finally:
        await response.aclose()


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Mapping[str, str | int] | None = None,
    headers: Mapping[str, str] | None = None,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
    max_attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> FetchResult:
    """Stream one bounded GET. Raises ``FetchError`` and its subclasses.

    Only transport faults and the statuses in ``RETRY_STATUSES`` are retried.
    A redirect and every other error status fail on the first response, so a
    misconfigured host costs one request rather than five.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if max_response_bytes < 0:
        raise ValueError("max_response_bytes must not be negative")

    request = client.build_request("GET", url, params=params, headers=headers)
    safe_url = _safe_url(request.url)

    for attempt in range(1, max_attempts + 1):
        try:
            return await _attempt_once(
                client,
                request,
                safe_url=safe_url,
                attempt=attempt,
                max_response_bytes=max_response_bytes,
            )
        except _Retry as signal:
            if attempt >= max_attempts:
                raise signal.error from None
            await sleep(signal.delay)
    raise FetchError(f"Gave up after {max_attempts} attempts", url=safe_url)


async def fetch_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Mapping[str, str | int] | None = None,
    headers: Mapping[str, str] | None = None,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
    max_attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[JsonValue, FetchResult]:
    """``fetch`` plus ``json.loads``. Raises ``FetchError`` on non-JSON bodies.

    ``NaN`` and the infinities are rejected here rather than at write time,
    because canonical JSON cannot encode them and a build should not travel
    the whole pipeline before finding that out.
    """
    result = await fetch(
        client,
        url,
        params=params,
        headers=headers,
        max_response_bytes=max_response_bytes,
        max_attempts=max_attempts,
        sleep=sleep,
    )
    try:
        payload = json.loads(result.body, parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError):
        raise FetchError(
            "Response body is not valid JSON", url=result.url, status_code=result.status_code
        ) from None
    return cast(JsonValue, payload), result
