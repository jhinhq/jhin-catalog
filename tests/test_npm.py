"""npm search: the 250 clamp, the from=5000 wrap tripwire, and repo links."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import httpx
import pytest

from jhin_catalog.normalize import normalize_npm
from jhin_catalog.sources.base import DEFAULT_LIMITS, SourceError
from jhin_catalog.sources.npm import (
    KEYWORDS,
    MAX_FROM,
    MAX_PAGE_SIZE,
    NPM_SEARCH_URL,
    NpmPaginationWrap,
    NpmSource,
    parse_page,
    repo_url_from_links,
)
from jhin_catalog.types import JsonObject, JsonValue, RawRecord

type FixtureLoader = Callable[[str], JsonValue]
type ClientFactory = Callable[[Callable[[httpx.Request], httpx.Response]], httpx.AsyncClient]
type Sleeper = Callable[[float], Awaitable[None]]


def _object(value: JsonValue) -> JsonObject:
    assert isinstance(value, dict)
    return value


def _record(records: tuple[RawRecord, ...], name: str) -> RawRecord:
    return next(record for record in records if record.upstream_id == name)


# --- parsing ---------------------------------------------------------------


def test_the_recorded_search_page_reports_its_total(load_fixture: FixtureLoader) -> None:
    records, total = parse_page(load_fixture("npm_search.json"), url=NPM_SEARCH_URL, keyword="mcp")
    assert total == 4
    assert all(record.source_id == "npm" for record in records)


def test_an_insecure_package_is_dropped_rather_than_indexed(
    load_fixture: FixtureLoader,
) -> None:
    """``flags.insecure`` is npm saying it found something; that is enough."""
    records, _ = parse_page(load_fixture("npm_search.json"), url=NPM_SEARCH_URL, keyword="mcp")
    assert "@acme-example/mcp-legacy" not in {record.upstream_id for record in records}
    assert len(records) == 3


def test_each_record_points_at_the_package_page(load_fixture: FixtureLoader) -> None:
    records, _ = parse_page(load_fixture("npm_search.json"), url=NPM_SEARCH_URL, keyword="mcp")
    record = _record(records, "tavily-mcp")
    assert record.url == "https://www.npmjs.com/package/tavily-mcp"


def test_a_payload_with_no_objects_list_is_a_source_error() -> None:
    with pytest.raises(SourceError):
        parse_page({"total": 0}, url=NPM_SEARCH_URL, keyword="mcp")


# --- signals ---------------------------------------------------------------


def test_a_dependent_count_arrives_as_a_string_and_is_cast(
    load_fixture: FixtureLoader,
) -> None:
    records, _ = parse_page(load_fixture("npm_search.json"), url=NPM_SEARCH_URL, keyword="mcp")
    candidate = normalize_npm(_record(records, "tavily-mcp"))
    assert candidate is not None
    assert candidate.signals.npm_dependents == 67410


def test_a_dependent_count_that_is_not_a_number_becomes_none(
    load_fixture: FixtureLoader,
) -> None:
    records, _ = parse_page(load_fixture("npm_search.json"), url=NPM_SEARCH_URL, keyword="mcp")
    candidate = normalize_npm(_record(records, "@acme-example/mcp-notes"))
    assert candidate is not None
    assert candidate.signals.npm_dependents is None


def test_monthly_downloads_come_off_the_search_envelope(load_fixture: FixtureLoader) -> None:
    records, _ = parse_page(load_fixture("npm_search.json"), url=NPM_SEARCH_URL, keyword="mcp")
    candidate = normalize_npm(_record(records, "tavily-mcp"))
    assert candidate is not None
    assert candidate.signals.npm_downloads_monthly == 412000


def test_an_npm_record_never_claims_an_endpoint(load_fixture: FixtureLoader) -> None:
    """npm knows a package exists; it knows nothing about where it is hosted."""
    records, _ = parse_page(load_fixture("npm_search.json"), url=NPM_SEARCH_URL, keyword="mcp")
    candidate = normalize_npm(_record(records, "tavily-mcp"))
    assert candidate is not None
    assert candidate.fields.get("mcp_url") in (None, "")
    assert candidate.trust_hint == "indexed"


def test_keywords_and_licence_survive_into_the_candidate(
    load_fixture: FixtureLoader,
) -> None:
    records, _ = parse_page(load_fixture("npm_search.json"), url=NPM_SEARCH_URL, keyword="mcp")
    candidate = normalize_npm(_record(records, "tavily-mcp"))
    assert candidate is not None
    assert candidate.fields["license"] == "MIT"
    tags = candidate.fields["tags"]
    assert isinstance(tags, list)
    assert "search" in tags


# --- repository links ------------------------------------------------------


def test_repo_url_from_links_unwraps_the_git_plus_scheme() -> None:
    links: JsonObject = {"repository": "git+https://github.com/o/r.git"}
    assert repo_url_from_links(links) == "https://github.com/o/r"


def test_repo_url_from_links_passes_a_plain_https_link_through() -> None:
    assert repo_url_from_links({"repository": "https://github.com/o/r"}) == "https://github.com/o/r"


def test_repo_url_from_links_refuses_the_git_protocol() -> None:
    """``git://`` is unauthenticated and unverifiable; it is not a repo link."""
    assert repo_url_from_links({"repository": "git://github.com/o/r.git"}) == ""


def test_repo_url_from_links_refuses_a_host_outside_the_three() -> None:
    assert repo_url_from_links({"repository": "https://evil.example/o/r"}) == ""


def test_repo_url_from_links_with_no_links_at_all_is_empty() -> None:
    assert repo_url_from_links(None) == ""
    assert repo_url_from_links({}) == ""


def test_a_package_with_no_links_still_becomes_a_candidate(
    load_fixture: FixtureLoader,
) -> None:
    records, _ = parse_page(load_fixture("npm_search.json"), url=NPM_SEARCH_URL, keyword="mcp")
    candidate = normalize_npm(_record(records, "@acme-example/mcp-nolinks"))
    assert candidate is not None
    assert candidate.repo is None
    assert "mcp:npm:@acme-example/mcp-nolinks" in candidate.alias_keys


# --- the crawl -------------------------------------------------------------


def _pages(
    requested: list[httpx.URL], load_fixture: FixtureLoader, *, total: int
) -> Callable[[httpx.Request], httpx.Response]:
    """A search endpoint whose every page is genuinely a different page.

    Each offset renames its packages, so the wrap tripwire has nothing to fire
    on and the crawl stops for the reason under test rather than for that one.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url)
        payload = _object(load_fixture("npm_search.json"))
        offset = int(request.url.params.get("from", 0))
        keyword = str(request.url.params.get("text", "")).replace(":", "-")
        objects = payload["objects"]
        assert isinstance(objects, list)
        if offset >= total:
            payload["objects"] = []
        else:
            for index, item in enumerate(objects):
                package = _object(_object(item)["package"])
                package["name"] = f"{keyword}-{offset}-{index}"
        payload["total"] = total
        return httpx.Response(200, json=payload)

    return handler


async def test_the_crawl_never_asks_for_an_offset_past_the_wrap_point(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: Sleeper
) -> None:
    """Past ``from=5000`` npm silently re-serves page one, so it is never asked."""
    requested: list[httpx.URL] = []
    async with mock_client(_pages(requested, load_fixture, total=100_000)) as client:
        await NpmSource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)

    offsets = [int(url.params.get("from", 0)) for url in requested]
    assert offsets
    assert max(offsets) <= MAX_FROM


async def test_the_crawl_never_asks_for_more_than_the_page_size_npm_honours(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: Sleeper
) -> None:
    requested: list[httpx.URL] = []
    limits = DEFAULT_LIMITS.model_copy(update={"page_size": 9_000})
    async with mock_client(_pages(requested, load_fixture, total=600)) as client:
        await NpmSource().fetch(client, limits=limits, sleep=no_sleep)

    assert all(int(url.params["size"]) <= MAX_PAGE_SIZE for url in requested)


async def test_only_the_search_endpoint_is_ever_called(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: Sleeper
) -> None:
    """Download counts ride the search envelope; a second API call is a bug."""
    requested: list[httpx.URL] = []
    async with mock_client(_pages(requested, load_fixture, total=4)) as client:
        await NpmSource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)

    assert requested
    assert all(str(url).startswith(NPM_SEARCH_URL) for url in requested)


async def test_a_package_found_under_two_keywords_is_recorded_once(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: Sleeper
) -> None:
    requested: list[httpx.URL] = []
    async with mock_client(_pages(requested, load_fixture, total=4)) as client:
        fetched = await NpmSource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)

    names = [record.upstream_id for record in fetched.records]
    assert len(requested) >= len(KEYWORDS)
    assert len(names) == len(set(names))


async def test_a_repeated_first_package_trips_the_pagination_wrap_guard(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: Sleeper
) -> None:
    """npm answers an overflowed offset with page one instead of an error."""

    def handler(_request: httpx.Request) -> httpx.Response:
        payload = _object(load_fixture("npm_search.json"))
        payload["total"] = 100_000
        return httpx.Response(200, json=payload)

    async with mock_client(handler) as client:
        with pytest.raises(NpmPaginationWrap):
            await NpmSource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)
