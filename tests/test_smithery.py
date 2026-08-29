"""Smithery paging with ``seed``, nullable ``remote``, and the fixed detail shape."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol

import httpx
import pytest

from jhin_catalog.http import FetchError
from jhin_catalog.normalize import normalize_smithery
from jhin_catalog.sources.base import DEFAULT_LIMITS, SourceError
from jhin_catalog.sources.smithery import (
    SEED,
    SMITHERY_BASE,
    SmitherySource,
    parse_detail,
    parse_page,
)
from jhin_catalog.types import CATALOG_ICONS, JsonObject, JsonValue, RawRecord

type FixtureLoader = Callable[[str], JsonValue]
type ClientFactory = Callable[[Callable[[httpx.Request], httpx.Response]], httpx.AsyncClient]


class RecordingSleep(Protocol):
    delays: list[float]

    async def __call__(self, seconds: float) -> None: ...


LIST_URL = f"{SMITHERY_BASE}/servers"


def _object(value: JsonValue) -> JsonObject:
    assert isinstance(value, dict)
    return value


def _by_name(records: tuple[RawRecord, ...]) -> dict[str, RawRecord]:
    return {record.upstream_id: record for record in records}


# --- list paging -----------------------------------------------------------


def test_the_recorded_page_reports_its_pagination_block(load_fixture: FixtureLoader) -> None:
    page = parse_page(load_fixture("smithery_list.json"), url=LIST_URL)
    assert len(page.servers) == 5
    assert page.current_page == 1
    assert page.total_pages == 1
    assert page.total_count == 5


def test_a_page_with_no_servers_block_is_a_source_error() -> None:
    with pytest.raises(SourceError):
        parse_page(
            {"pagination": {"currentPage": 1, "totalPages": 1, "totalCount": 0}}, url=LIST_URL
        )


def test_a_page_with_no_pagination_block_is_a_source_error() -> None:
    with pytest.raises(SourceError):
        parse_page({"servers": []}, url=LIST_URL)


def test_a_nullable_remote_flag_reads_as_false_rather_than_crashing(
    load_fixture: FixtureLoader,
) -> None:
    page = parse_page(load_fixture("smithery_list.json"), url=LIST_URL)
    notes = next(row for row in page.servers if row["qualifiedName"] == "@acme-example/notes")
    assert notes["remote"] is None


# --- detail parsing --------------------------------------------------------


def test_the_http_detail_keeps_its_endpoint_and_tool_list(
    load_fixture: FixtureLoader,
) -> None:
    detail = parse_detail(load_fixture("smithery_detail_http.json"), qualified_name="exa")
    assert detail["deploymentUrl"] == "https://server.smithery.ai/exa/mcp"
    tools = detail["tools"]
    assert isinstance(tools, list)
    assert len(tools) == 2


def test_an_http_connection_is_the_one_that_becomes_a_remote(
    load_fixture: FixtureLoader,
) -> None:
    detail = parse_detail(load_fixture("smithery_detail_http.json"), qualified_name="exa")
    connections = detail["connections"]
    assert isinstance(connections, list)
    assert _object(connections[0])["type"] == "http"


def test_a_null_tool_list_is_not_the_same_as_an_empty_one(
    load_fixture: FixtureLoader,
) -> None:
    """``null`` means the detail pass never learned; ``[]`` means none exist."""
    detail = parse_detail(
        load_fixture("smithery_detail_stdio.json"), qualified_name="@acme-example/context"
    )
    assert detail["tools"] is None


def test_an_empty_connection_list_is_accepted(load_fixture: FixtureLoader) -> None:
    payload = _object(load_fixture("smithery_detail_stdio.json"))
    payload["connections"] = []
    detail = parse_detail(payload, qualified_name="@acme-example/context")
    assert detail["connections"] == []


def test_a_stdio_only_detail_offers_no_deployment_url(load_fixture: FixtureLoader) -> None:
    detail = parse_detail(
        load_fixture("smithery_detail_stdio.json"), qualified_name="@acme-example/context"
    )
    assert detail["deploymentUrl"] is None


def test_a_detail_that_is_not_an_object_is_a_source_error() -> None:
    with pytest.raises(SourceError):
        parse_detail(["not", "a", "detail"], qualified_name="exa")


def test_the_config_schema_is_reduced_to_names_and_header_hints(
    load_fixture: FixtureLoader,
) -> None:
    """Jhin stores pointers, not payloads: a config schema is neither.

    What survives is the property names, which say how many fields a person
    will be asked for, and any ``x-to.header``, which is the only field
    ``auth_hint`` reads. Types, descriptions, and defaults are dropped.
    """
    detail = parse_detail(load_fixture("smithery_detail_http.json"), qualified_name="exa")
    connections = detail["connections"]
    assert isinstance(connections, list)
    schema = _object(_object(connections[0])["configSchema"])
    properties = _object(schema["properties"])
    assert set(properties) == {"exaApiKey", "resultsPerQuery"}
    body = json.dumps(detail)
    assert "Your Exa API key." not in body
    assert '"integer"' not in body


def test_a_smithery_icon_url_never_becomes_the_entry_icon(
    load_fixture: FixtureLoader,
) -> None:
    """Jhin icons are token names from ``CATALOG_ICONS``, never remote images.

    Smithery serves an ``iconUrl`` and the crawl carries it, because a raw
    record is meant to be the wire object. What must not happen is that URL
    reaching ``icon``, which a Jhin client renders from a fixed token set.
    """
    page = parse_page(load_fixture("smithery_list.json"), url=LIST_URL)
    summary = dict(next(row for row in page.servers if row["qualifiedName"] == "exa"))
    summary["_detail"] = parse_detail(
        load_fixture("smithery_detail_http.json"), qualified_name="exa"
    )
    record = RawRecord(
        source_id="smithery",
        upstream_id="exa",
        url="https://smithery.ai/server/exa",
        payload=summary,
    )
    candidate = normalize_smithery(record)
    assert candidate is not None
    icon = candidate.fields.get("icon")
    assert icon is None or icon in CATALOG_ICONS


# --- the crawl -------------------------------------------------------------


def _handler(
    load_fixture: FixtureLoader, requested: list[str], *, missing: frozenset[str] = frozenset()
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        assert request.headers["user-agent"].startswith("jhin-catalog/")
        path = request.url.path
        if path.rstrip("/").endswith("/servers"):
            return httpx.Response(200, json=load_fixture("smithery_list.json"))
        name = path.split("/servers/", 1)[1]
        if name in missing:
            return httpx.Response(404, json={"error": "not found"})
        fixture = "smithery_detail_http.json" if name == "exa" else "smithery_detail_stdio.json"
        return httpx.Response(200, json=load_fixture(fixture))

    return handler


async def test_the_crawl_always_sends_the_seed_that_lifts_the_five_hundred_cap(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    """Without ``seed`` only 500 of ~11,000 rows are reachable and the overflow
    pages answer 200 with an empty list, which the gate reads as a deletion."""
    requested: list[str] = []
    limits = DEFAULT_LIMITS.model_copy(update={"detail_top_n": 0})
    async with mock_client(_handler(load_fixture, requested)) as client:
        await SmitherySource().fetch(client, limits=limits, sleep=no_sleep)

    assert requested
    assert all(f"seed={SEED}" in url for url in requested if "/servers?" in url)


async def test_only_the_busiest_servers_are_detailed_and_in_a_fixed_order(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    requested: list[str] = []
    limits = DEFAULT_LIMITS.model_copy(update={"detail_top_n": 2})
    async with mock_client(_handler(load_fixture, requested)) as client:
        await SmitherySource().fetch(client, limits=limits, sleep=no_sleep)

    details = [url for url in requested if "/servers/" in url]
    assert len(details) == 2
    assert details[0].endswith("/servers/exa")
    assert "context" in details[1]


async def test_a_detail_that_answers_404_is_skipped_rather_than_fatal(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    requested: list[str] = []
    limits = DEFAULT_LIMITS.model_copy(update={"detail_top_n": 2})
    handler = _handler(load_fixture, requested, missing=frozenset({"exa"}))
    async with mock_client(handler) as client:
        fetched = await SmitherySource().fetch(client, limits=limits, sleep=no_sleep)

    assert fetched.entry_count > 0
    assert "exa" in _by_name(fetched.records)


async def test_unlisted_and_inactive_rows_never_become_records(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    requested: list[str] = []
    limits = DEFAULT_LIMITS.model_copy(update={"detail_top_n": 0})
    async with mock_client(_handler(load_fixture, requested)) as client:
        fetched = await SmitherySource().fetch(client, limits=limits, sleep=no_sleep)

    names = set(_by_name(fetched.records))
    assert "@acme-example/hidden" not in names
    assert "@acme-example/retired" not in names
    assert "exa" in names


async def test_each_record_points_at_the_page_a_person_can_read(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    requested: list[str] = []
    limits = DEFAULT_LIMITS.model_copy(update={"detail_top_n": 0})
    async with mock_client(_handler(load_fixture, requested)) as client:
        fetched = await SmitherySource().fetch(client, limits=limits, sleep=no_sleep)

    for record in fetched.records:
        assert record.url.startswith("https://smithery.ai/server/")
        assert record.source_id == "smithery"


def _paged_handler(
    load_fixture: FixtureLoader,
    *,
    total_pages: int,
    respond: Callable[[int, int], httpx.Response | None],
) -> Callable[[httpx.Request], httpx.Response]:
    """A list walk of ``total_pages`` whose later pages the test scripts.

    ``respond`` sees ``(page, calls_so_far_for_that_page)`` and either returns
    a response or ``None`` for "serve the fixture page as usual".
    """
    calls: dict[int, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        calls[page] = calls.get(page, 0) + 1
        scripted = respond(page, calls[page])
        if scripted is not None:
            return scripted
        payload = _object(load_fixture("smithery_list.json"))
        payload["pagination"] = {
            "currentPage": page,
            "pageSize": 100,
            "totalPages": total_pages,
            "totalCount": 5 * total_pages,
        }
        return httpx.Response(200, json=payload)

    return handler


async def test_an_exhausted_throttle_cuts_the_list_walk_short_rather_than_the_build(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    """A 429 that outlives every retry ends the walk, not the crawl.

    Declaring it fatal threw away every page already read — and, at the
    build level, every other source's work — over rate limiting, which is
    the expected end of a large unauthenticated crawl rather than a broken
    upstream. ``github_topics`` degrades the same way on its 403, and the
    diff gate stays the judge of whether a short crawl is publishable.
    """

    def respond(page: int, _calls: int) -> httpx.Response | None:
        if page > 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return None

    limits = DEFAULT_LIMITS.model_copy(update={"detail_top_n": 0})
    handler = _paged_handler(load_fixture, total_pages=3, respond=respond)
    async with mock_client(handler) as client:
        fetched = await SmitherySource().fetch(client, limits=limits, sleep=no_sleep)

    # The fixture page holds five rows, two of them unlisted or inactive.
    assert fetched.entry_count == 3
    assert fetched.page_count == 1
    # ``fetch`` spent its own retries before the walk gave up.
    assert no_sleep.delays == [0.5, 1.0, 2.0, 4.0]


async def test_a_throttle_that_clears_within_the_retries_costs_nothing(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    """The shared retry inside ``fetch`` still absorbs a transient 429."""

    def respond(page: int, calls: int) -> httpx.Response | None:
        if page == 2 and calls < 3:
            return httpx.Response(429, json={"error": "rate limited"})
        return None

    limits = DEFAULT_LIMITS.model_copy(update={"detail_top_n": 0})
    handler = _paged_handler(load_fixture, total_pages=2, respond=respond)
    async with mock_client(handler) as client:
        fetched = await SmitherySource().fetch(client, limits=limits, sleep=no_sleep)

    assert fetched.page_count == 2


async def test_a_hard_failure_on_the_list_walk_is_still_a_fetch_fault(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    """Only the throttle degrades; a broken upstream still fails the crawl,
    because a 500 mid-walk is a fault someone should see, not a ceiling."""

    def respond(page: int, _calls: int) -> httpx.Response | None:
        if page > 1:
            return httpx.Response(404, json={"error": "gone"})
        return None

    handler = _paged_handler(load_fixture, total_pages=3, respond=respond)
    async with mock_client(handler) as client:
        with pytest.raises(FetchError):
            await SmitherySource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)


async def test_an_empty_page_below_the_last_one_is_a_fetch_fault(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: RecordingSleep
) -> None:
    """The symptom of the 500-result cap, and the thing the seed exists to avoid."""

    def handler(_request: httpx.Request) -> httpx.Response:
        payload = _object(load_fixture("smithery_list.json"))
        payload["servers"] = []
        payload["pagination"] = {
            "currentPage": 1,
            "pageSize": 100,
            "totalPages": 6,
            "totalCount": 550,
        }
        return httpx.Response(200, json=payload)

    async with mock_client(handler) as client:
        with pytest.raises(SourceError):
            await SmitherySource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)
