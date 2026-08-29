"""Source limits, the token bucket's injected clock, and rolling hashes."""

from __future__ import annotations

import hashlib
from typing import Protocol

import pytest

from jhin_catalog.sources import ALL_SOURCES, source_by_id
from jhin_catalog.sources.base import (
    DEFAULT_LIMITS,
    Source,
    SourceError,
    SourceLimits,
    TokenBucket,
    rolling_sha256,
)
from jhin_catalog.sources.github_topics import GitHubTopicsSource
from jhin_catalog.sources.marketplaces import MarketplacesSource
from jhin_catalog.sources.npm import NpmSource
from jhin_catalog.sources.registry import RegistrySource
from jhin_catalog.sources.smithery import SmitherySource
from jhin_catalog.types import SOURCE_IDS


class RecordingSleep(Protocol):
    delays: list[float]

    async def __call__(self, seconds: float) -> None: ...


class ManualClock:
    """A monotonic clock the test moves by hand, so no test ever waits."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# --- limits -----------------------------------------------------------------


def test_the_default_limits_are_the_documented_ones() -> None:
    assert DEFAULT_LIMITS.page_size == 100
    assert DEFAULT_LIMITS.max_pages == 1000
    assert DEFAULT_LIMITS.max_records == 100_000
    assert DEFAULT_LIMITS.detail_top_n == 500
    assert DEFAULT_LIMITS.github_token == ""
    assert DEFAULT_LIMITS.requests_per_minute == 0


def test_no_token_is_the_default_so_an_unconfigured_crawl_stays_anonymous() -> None:
    assert SourceLimits().github_token == ""


# --- the token bucket -------------------------------------------------------


async def test_a_thirty_a_minute_bucket_lets_thirty_through_then_waits(
    no_sleep: RecordingSleep,
) -> None:
    """A burst of a full minute's budget is free; the next one costs the gap."""
    bucket = TokenBucket(rate_per_minute=30, clock=ManualClock(), sleep=no_sleep)
    for _ in range(30):
        await bucket.acquire()
    assert no_sleep.delays == []

    await bucket.acquire()
    assert no_sleep.delays == [2.0]


async def test_a_bucket_refills_as_the_clock_moves(no_sleep: RecordingSleep) -> None:
    clock = ManualClock()
    bucket = TokenBucket(rate_per_minute=60, clock=clock, sleep=no_sleep)
    for _ in range(60):
        await bucket.acquire()

    clock.now += 60.0
    await bucket.acquire()
    assert no_sleep.delays == []


async def test_a_bucket_with_no_rate_never_waits(no_sleep: RecordingSleep) -> None:
    bucket = TokenBucket(rate_per_minute=0, clock=ManualClock(), sleep=no_sleep)
    for _ in range(200):
        await bucket.acquire()
    assert no_sleep.delays == []


# --- the rolling hash -------------------------------------------------------


def test_the_rolling_hash_is_the_hash_of_the_concatenation() -> None:
    assert rolling_sha256([b"a", b"b"]) == hashlib.sha256(b"ab").hexdigest()


def test_the_rolling_hash_of_nothing_is_the_empty_digest() -> None:
    assert rolling_sha256([]) == hashlib.sha256(b"").hexdigest()


def test_the_rolling_hash_depends_on_fetch_order() -> None:
    """It is a fingerprint of one crawl, so a reordered crawl is a new one."""
    assert rolling_sha256([b"a", b"b"]) != rolling_sha256([b"b", b"a"])


def test_the_rolling_hash_changes_when_one_page_changes() -> None:
    before = rolling_sha256([b"page one", b"page two"])
    after = rolling_sha256([b"page one", b"page three"])
    assert before != after


# --- the registry of sources ------------------------------------------------


def test_every_source_id_but_curated_has_exactly_one_class() -> None:
    """``curated`` is a file in this repository, not something to crawl."""
    ids = {source.source_id for source in ALL_SOURCES}
    assert ids == set(SOURCE_IDS) - {"curated"}
    assert len(ALL_SOURCES) == len(ids)


def test_each_source_resolves_by_its_own_id() -> None:
    assert source_by_id("registry") is RegistrySource
    assert source_by_id("smithery") is SmitherySource
    assert source_by_id("npm") is NpmSource
    assert source_by_id("github_topics") is GitHubTopicsSource
    assert source_by_id("marketplaces") is MarketplacesSource


def test_an_unknown_source_id_is_a_key_error() -> None:
    with pytest.raises(KeyError):
        source_by_id("nope")


def test_curated_is_not_a_crawlable_source() -> None:
    with pytest.raises(KeyError):
        source_by_id("curated")


def test_every_source_class_really_is_a_source() -> None:
    for source in ALL_SOURCES:
        assert issubclass(source, Source)
        assert source.source_id in SOURCE_IDS


def test_a_source_error_carries_the_source_that_raised_it() -> None:
    error = SourceError("smithery served an empty page", source_id="smithery")
    assert error.source_id == "smithery"
    assert "empty page" in str(error)
