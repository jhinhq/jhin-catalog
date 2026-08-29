"""Topic search: the 1000 ceiling, the 30/min bucket, and 403 backoff."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import httpx
import pytest

from jhin_catalog.normalize import normalize_github_topics
from jhin_catalog.sources.base import DEFAULT_LIMITS, SourceError
from jhin_catalog.sources.github_topics import (
    GITHUB_API,
    MAX_PAGES_PER_TOPIC,
    SEARCH_RATE_PER_MINUTE,
    TOPICS,
    GitHubTopicsSource,
    github_headers,
    parse_search_page,
)
from jhin_catalog.types import JsonObject, JsonValue, RawRecord

type FixtureLoader = Callable[[str], JsonValue]
type ClientFactory = Callable[[Callable[[httpx.Request], httpx.Response]], httpx.AsyncClient]


class RecordingSleep(Protocol):
    delays: list[float]

    async def __call__(self, seconds: float) -> None: ...


SEARCH_URL = f"{GITHUB_API}/search/repositories"


def _object(value: JsonValue) -> JsonObject:
    assert isinstance(value, dict)
    return value


def _record(records: tuple[RawRecord, ...], name: str) -> RawRecord:
    return next(record for record in records if record.upstream_id == name)


# --- headers ---------------------------------------------------------------


def test_the_api_version_and_accept_headers_are_always_sent() -> None:
    headers = github_headers("")
    assert headers["Accept"] == "application/vnd.github+json"
    assert headers["X-GitHub-Api-Version"] == "2022-11-28"


def test_an_authorization_header_appears_only_when_a_token_is_given() -> None:
    assert "Authorization" not in github_headers("")
    assert "ghp_example" in github_headers("ghp_example")["Authorization"]


# --- parsing ---------------------------------------------------------------


def test_the_recorded_search_page_yields_one_record_per_repository(
    load_fixture: FixtureLoader,
) -> None:
    records = parse_search_page(
        load_fixture("github_topics.json"), url=SEARCH_URL, topic="mcp-server"
    )
    assert len(records) == 3
    assert all(record.source_id == "github_topics" for record in records)


def test_the_upstream_id_is_the_lowercased_full_name(load_fixture: FixtureLoader) -> None:
    records = parse_search_page(
        load_fixture("github_topics.json"), url=SEARCH_URL, topic="mcp-server"
    )
    assert {record.upstream_id for record in records} == {
        "tavily-ai/tavily-mcp",
        "modelcontextprotocol/servers",
        "acme-example/acme-skills",
    }


def test_each_record_points_at_the_repository_page(load_fixture: FixtureLoader) -> None:
    records = parse_search_page(
        load_fixture("github_topics.json"), url=SEARCH_URL, topic="mcp-server"
    )
    assert _record(records, "tavily-ai/tavily-mcp").url == "https://github.com/tavily-ai/tavily-mcp"


def test_a_page_with_no_items_list_is_a_source_error() -> None:
    with pytest.raises(SourceError):
        parse_search_page({"total_count": 0}, url=SEARCH_URL, topic="mcp-server")


# --- signals ---------------------------------------------------------------


def test_stars_forks_and_topics_all_reach_the_candidate(
    load_fixture: FixtureLoader,
) -> None:
    records = parse_search_page(
        load_fixture("github_topics.json"), url=SEARCH_URL, topic="mcp-server"
    )
    candidate = normalize_github_topics(_record(records, "tavily-ai/tavily-mcp"))
    assert candidate is not None
    assert candidate.signals.github_stars == 1400
    assert candidate.signals.github_forks == 118
    tags = candidate.fields["tags"]
    assert isinstance(tags, list)
    assert "mcp-server" in tags


def test_a_topics_row_never_claims_an_endpoint_or_a_trust_tier(
    load_fixture: FixtureLoader,
) -> None:
    """A topic label is evidence a repository exists, and nothing more."""
    records = parse_search_page(
        load_fixture("github_topics.json"), url=SEARCH_URL, topic="mcp-server"
    )
    candidate = normalize_github_topics(_record(records, "tavily-ai/tavily-mcp"))
    assert candidate is not None
    assert candidate.fields.get("mcp_url") in (None, "")
    assert candidate.fields.get("packages") in (None, [], ())
    assert candidate.trust_hint == "indexed"


def test_a_repository_with_no_licence_block_is_still_a_candidate(
    load_fixture: FixtureLoader,
) -> None:
    records = parse_search_page(
        load_fixture("github_topics.json"), url=SEARCH_URL, topic="agent-skills"
    )
    candidate = normalize_github_topics(_record(records, "acme-example/acme-skills"))
    assert candidate is not None
    assert candidate.fields.get("license", "") == ""


# --- the crawl -------------------------------------------------------------


def _search(
    load_fixture: FixtureLoader, requested: list[httpx.URL]
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url)
        assert request.url.path != "/rate_limit"
        return httpx.Response(200, json=load_fixture("github_topics.json"))

    return handler


async def test_the_crawl_never_asks_for_the_page_past_the_thousand_result_ceiling(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    """Search caps at 1000 results; page 11 answers 422 for everyone."""
    requested: list[httpx.URL] = []
    async with mock_client(_search(load_fixture, requested)) as client:
        await GitHubTopicsSource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)

    pages = [int(url.params.get("page", 1)) for url in requested]
    assert pages
    assert max(pages) <= MAX_PAGES_PER_TOPIC


async def test_the_crawl_covers_every_topic_it_was_configured_with(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    requested: list[httpx.URL] = []
    async with mock_client(_search(load_fixture, requested)) as client:
        await GitHubTopicsSource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)

    queries = " ".join(str(url) for url in requested)
    for topic in TOPICS:
        assert f"topic%3A{topic}" in queries or f"topic:{topic}" in queries


async def test_the_rate_limit_endpoint_is_never_consulted(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    """``/rate_limit`` reports a stale ``remaining``; the bucket is the truth."""
    requested: list[httpx.URL] = []
    async with mock_client(_search(load_fixture, requested)) as client:
        await GitHubTopicsSource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)

    assert requested
    assert all(url.path != "/rate_limit" for url in requested)


async def test_no_token_means_no_authorization_header_on_the_wire(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    seen: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append("authorization" in request.headers)
        return httpx.Response(200, json=load_fixture("github_topics.json"))

    async with mock_client(handler) as client:
        await GitHubTopicsSource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)

    assert seen
    assert not any(seen)


async def test_a_token_is_sent_when_the_limits_carry_one(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization", ""))
        return httpx.Response(200, json=load_fixture("github_topics.json"))

    limits = DEFAULT_LIMITS.model_copy(update={"github_token": "ghp_example"})
    async with mock_client(handler) as client:
        await GitHubTopicsSource().fetch(client, limits=limits, sleep=no_sleep)

    assert seen
    assert all("ghp_example" in header for header in seen)


async def test_a_persistent_403_cuts_the_topic_short_rather_than_the_crawl(
    mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    """A secondary rate limit ends the topic, not the build.

    Aborting here would discard every other source's work over one
    paginated page. The diff gate is what guards a truncated catalog, so
    this degrades the way npm and smithery already do: return what was
    reached and let the gate decide whether it is publishable.
    """
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(403, json={"message": "API rate limit exceeded"})

    async with mock_client(handler) as client:
        fetched = await GitHubTopicsSource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)

    assert fetched.records == ()
    # Five attempts per topic across every topic in TOPICS.
    assert calls == 5 * len(TOPICS)
    assert no_sleep.delays[:4] == [0.5, 1.0, 2.0, 4.0]


async def test_a_403_waits_the_retry_after_github_asked_for(
    mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    """``Retry-After`` beats the doubling schedule, capped at 60s."""
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            403,
            json={"message": "API rate limit exceeded"},
            headers={"Retry-After": "45"},
        )

    async with mock_client(handler) as client:
        await GitHubTopicsSource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)

    assert no_sleep.delays[:4] == [45.0, 45.0, 45.0, 45.0]


async def test_a_403_that_clears_is_retried_rather_than_fatal(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(403, json={"message": "API rate limit exceeded"})
        return httpx.Response(200, json=load_fixture("github_topics.json"))

    async with mock_client(handler) as client:
        fetched = await GitHubTopicsSource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)

    assert fetched.entry_count > 0
    assert no_sleep.delays[0] == 0.5


def test_the_search_bucket_is_the_documented_thirty_a_minute() -> None:
    assert SEARCH_RATE_PER_MINUTE == 30
