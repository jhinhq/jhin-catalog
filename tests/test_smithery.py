"""Smithery paging with ``seed``, nullable ``remote``, and the fixed detail shape."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol

import httpx
import pytest

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
