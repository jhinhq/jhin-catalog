"""The contract every upstream crawler implements, and its shared parts.

A source turns one remote index into a ``SourceFetch``: the raw records it
saw, how many pages that took, and a rolling hash of the exact bytes those
pages contained.  The hash is what ``sources.lock`` commits to, so a build
can tell an upstream that genuinely changed from one that merely ran again.

``SourceLimits`` bounds a crawl, ``TokenBucket`` paces it against a clock
the caller supplies, and both the clock and the ``sleep`` callable are
injected so a full crawl replays in a test without waiting or drifting.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from typing import ClassVar, Final

import httpx
from pydantic import BaseModel, ConfigDict, Field

from jhin_catalog.types import CatalogError, SourceFetch


class SourceLimits(BaseModel):
    """How far a crawl may go, and how hard it is allowed to push."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_size: int = Field(default=100, ge=1)
    max_pages: int = Field(default=1000, ge=1)
    max_records: int = Field(default=100_000, ge=1)
    detail_top_n: int = Field(default=500, ge=0)
    github_token: str = Field(default="", repr=False)
    requests_per_minute: int = Field(default=0, ge=0)

    # Which marketplace repositories the crawl may read skills from, as
    # lowercase ``owner/name``. Topic search finds candidates; this says which
    # of them a person has actually reviewed. An empty, unrequired list means
    # crawl whatever the topic returns — the pre-review default, and not the
    # one ``curated/skills.yaml`` ships.
    marketplace_allowlist: tuple[str, ...] = ()
    require_marketplace_allowlist: bool = False


DEFAULT_LIMITS: Final[SourceLimits] = SourceLimits()


class SourceError(CatalogError):
    """A source could not produce a usable page."""

    def __init__(self, message: str, *, source_id: str) -> None:
        super().__init__(message)
        self.source_id = source_id


class Source(ABC):
    """One upstream index, crawled whole into a single ``SourceFetch``.

    Implementations own their paging, their throttle, and their own idea of
    what an ``upstream_id`` is; they do not interpret what they collect.
    Every record comes back as the upstream wrote it, and ``normalize`` is
    the only place a payload is read for meaning.
    """

    source_id: ClassVar[str]

    @abstractmethod
    async def fetch(
        self,
        client: httpx.AsyncClient,
        *,
        limits: SourceLimits = DEFAULT_LIMITS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> SourceFetch:
        """Crawl this source to exhaustion (within ``limits``)."""


class TokenBucket:
    """A deterministic client-side rate limiter driven by an injected clock.

    The bucket starts full, so a burst of a whole minute's requests goes out
    at once and only the steady state is paced.  Tokens are earned in whole
    units rather than continuously, which makes every wait an exact multiple
    of the token interval: the schedule a test asserts on cannot drift with
    the microseconds a real clock spent in between.
    """

    def __init__(
        self,
        *,
        rate_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if rate_per_minute < 0:
            raise ValueError("rate_per_minute must not be negative")
        self._interval = 60.0 / rate_per_minute if rate_per_minute > 0 else 0.0
        self._capacity = float(rate_per_minute)
        self._tokens = float(rate_per_minute)
        self._clock = clock
        self._sleep = sleep
        self._updated = clock()

    def _refill(self) -> None:
        """Credit every whole token the elapsed time has earned."""
        elapsed = self._clock() - self._updated
        if elapsed < self._interval:
            return
        earned = int(elapsed // self._interval)
        self._tokens = min(self._capacity, self._tokens + earned)
        self._updated += earned * self._interval

    async def acquire(self) -> None:
        """Wait, if waiting is what the configured rate requires."""
        if self._interval <= 0.0:
            return
        self._refill()
        if self._tokens < 1.0:
            await self._sleep(self._interval)
            self._updated += self._interval
            self._tokens += 1.0
        self._tokens -= 1.0


def rolling_sha256(bodies: Iterable[bytes]) -> str:
    """SHA-256 over the concatenation of every raw page body, in fetch order."""
    digest = hashlib.sha256()
    for body in bodies:
        digest.update(body)
    return digest.hexdigest()


class RollingDigest:
    """:func:`rolling_sha256`, folded one body at a time.

    Same hex digest over the same fetch order, and it retains none of the
    bodies. That distinction is a memory bound, not a style preference: a
    crawl's per-response cap says nothing about the total, and an upstream
    serving a thousand padded pages will exhaust a runner that is holding
    every one of them to hash at the end. Nothing here needs a body once it
    has been parsed, so nothing here keeps one.
    """

    def __init__(self) -> None:
        self._digest = hashlib.sha256()

    def update(self, body: bytes) -> None:
        """Fold one raw response body in and forget it."""
        self._digest.update(body)

    def hexdigest(self) -> str:
        """The digest over every body folded in so far."""
        return self._digest.hexdigest()
