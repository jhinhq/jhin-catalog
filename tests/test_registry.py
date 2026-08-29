"""Official MCP registry paging: cursors, deleted rows, and remote election."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import httpx
import pytest

from jhin_catalog.normalize import normalize_registry
from jhin_catalog.sources.base import DEFAULT_LIMITS, SourceError
from jhin_catalog.sources.registry import (
    REGISTRY_BASE,
    SERVERS_PATH,
    RegistrySource,
    official_meta,
    parse_page,
)
from jhin_catalog.types import JsonObject, JsonValue, RawRecord

# ``tests/`` is not a package, so the shapes ``conftest`` provides are restated
# structurally here rather than imported. Each one matches a fixture exactly.
type FixtureLoader = Callable[[str], JsonValue]
type ClientFactory = Callable[[Callable[[httpx.Request], httpx.Response]], httpx.AsyncClient]


class RecordingSleep(Protocol):
    delays: list[float]

    async def __call__(self, seconds: float) -> None: ...


PAGE_URL = f"{REGISTRY_BASE}{SERVERS_PATH}"


def _record(records: tuple[RawRecord, ...], name: str) -> RawRecord:
    return next(record for record in records if record.upstream_id == name)


def _server(record: RawRecord) -> JsonObject:
    server = record.payload["server"]
    assert isinstance(server, dict)
    return server


# --- paging ----------------------------------------------------------------


def test_the_recorded_page_yields_one_record_per_live_server(
    load_fixture: FixtureLoader,
) -> None:
    records, cursor = parse_page(load_fixture("registry_page1.json"), url=PAGE_URL)
    assert len(records) == 4
    assert cursor == "eyJvIjo0fQ"
    assert [record.source_id for record in records] == ["registry"] * 4


def test_a_last_page_reports_no_cursor(load_fixture: FixtureLoader) -> None:
    _, cursor = parse_page(load_fixture("registry_page2.json"), url=PAGE_URL)
    assert cursor is None


def test_a_withdrawn_row_and_a_superseded_row_are_both_dropped(
    load_fixture: FixtureLoader,
) -> None:
    """``status: deleted`` is a tombstone and ``isLatest: false`` is history."""
    records, _ = parse_page(load_fixture("registry_page2.json"), url=PAGE_URL)
    names = {record.upstream_id for record in records}
    assert names == {"io.github.example-org/templated"}


def test_metadata_count_is_never_read_as_a_total(load_fixture: FixtureLoader) -> None:
    """``count`` is the size of the page; treating it as a total ends the walk
    after one page and the diff gate then reads the rest as a mass deletion."""
    payload = load_fixture("registry_page1.json")
    assert isinstance(payload, dict)
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    metadata["count"] = 4
    records, cursor = parse_page(payload, url=PAGE_URL)
    assert len(records) == 4
    assert cursor == "eyJvIjo0fQ"


def test_an_opaque_cursor_is_carried_through_rather_than_understood(
    load_fixture: FixtureLoader,
) -> None:
    payload = load_fixture("registry_page1.json")
    assert isinstance(payload, dict)
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    metadata["nextCursor"] = "!! not base64 !!"
    _, cursor = parse_page(payload, url=PAGE_URL)
    assert cursor == "!! not base64 !!"


def test_a_page_with_no_servers_list_is_a_source_error() -> None:
    with pytest.raises(SourceError, match="servers"):
        parse_page({"metadata": {}}, url=PAGE_URL)


def test_a_page_that_is_not_an_object_is_a_source_error() -> None:
    with pytest.raises(SourceError):
        parse_page(["not", "a", "page"], url=PAGE_URL)


# --- record content --------------------------------------------------------


def test_the_record_url_points_at_the_versions_endpoint_for_that_name(
    load_fixture: FixtureLoader,
) -> None:
    records, _ = parse_page(load_fixture("registry_page1.json"), url=PAGE_URL)
    record = _record(records, "io.github.tavily-ai/tavily-mcp")
    assert record.url.startswith(REGISTRY_BASE)
    assert record.url.endswith("/versions")


def test_official_meta_finds_the_status_block(load_fixture: FixtureLoader) -> None:
    records, _ = parse_page(load_fixture("registry_page1.json"), url=PAGE_URL)
    meta = official_meta(_record(records, "io.github.tavily-ai/tavily-mcp").payload)
    assert meta["status"] == "active"
    assert meta["isLatest"] is True


def test_official_meta_on_a_row_with_no_meta_is_empty() -> None:
    assert official_meta({"server": {"name": "x"}}) == {}


def test_a_server_with_neither_remotes_nor_packages_is_still_a_record(
    load_fixture: FixtureLoader,
) -> None:
    records, _ = parse_page(load_fixture("registry_page1.json"), url=PAGE_URL)
    bare = _server(_record(records, "io.github.example-org/bare"))
    assert "remotes" not in bare
    assert "packages" not in bare


def test_package_argument_shapes_the_registry_actually_serves_are_tolerated(
    load_fixture: FixtureLoader,
) -> None:
    """``type`` arrives empty, as a flag, as an environment name, or literal."""
    records, _ = parse_page(load_fixture("registry_page2.json"), url=PAGE_URL)
    packages = _server(_record(records, "io.github.example-org/templated"))["packages"]
    assert isinstance(packages, list)
    first = packages[0]
    assert isinstance(first, dict)
    arguments = first["packageArguments"]
    assert isinstance(arguments, list)
    assert len(arguments) == 4


# --- transport and remote election -----------------------------------------


def test_a_transport_nobody_has_defined_normalises_to_unknown(
    load_fixture: FixtureLoader,
) -> None:
    records, _ = parse_page(load_fixture("registry_page1.json"), url=PAGE_URL)
    candidate = normalize_registry(_record(records, "io.github.example-org/odd-transport"))
    assert candidate is not None
    remotes = candidate.fields["remotes"]
    assert isinstance(remotes, list)
    transports = {remote["transport"] for remote in remotes if isinstance(remote, dict)}
    assert transports == {"unknown", "sse"}


def test_the_first_concrete_https_remote_becomes_the_endpoint(
    load_fixture: FixtureLoader,
) -> None:
    records, _ = parse_page(load_fixture("registry_page1.json"), url=PAGE_URL)
    candidate = normalize_registry(_record(records, "io.github.tavily-ai/tavily-mcp"))
    assert candidate is not None
    assert candidate.fields["mcp_url"] == "https://mcp.tavily.com/mcp/"
    assert candidate.fields["transport"] == "streamable_http"


def test_an_authorization_header_elects_the_bearer_hint(
    load_fixture: FixtureLoader,
) -> None:
    records, _ = parse_page(load_fixture("registry_page1.json"), url=PAGE_URL)
    candidate = normalize_registry(_record(records, "io.github.tavily-ai/tavily-mcp"))
    assert candidate is not None
    assert candidate.fields["auth_hint"] == "bearer"


def test_a_templated_endpoint_is_refused_and_explained(load_fixture: FixtureLoader) -> None:
    """A URL with a ``{var}`` in it cannot be connected to, so it is not offered."""
    records, _ = parse_page(load_fixture("registry_page2.json"), url=PAGE_URL)
    candidate = normalize_registry(_record(records, "io.github.example-org/templated"))
    assert candidate is not None
    assert candidate.fields.get("mcp_url") in (None, "")
    note = candidate.fields.get("auth_note")
    assert isinstance(note, str)
    assert "templated" in note.lower()


def test_a_stdio_only_server_says_so_and_names_the_package(
    load_fixture: FixtureLoader,
) -> None:
    records, _ = parse_page(load_fixture("registry_page1.json"), url=PAGE_URL)
    candidate = normalize_registry(
        _record(records, "io.github.modelcontextprotocol/server-filesystem")
    )
    assert candidate is not None
    note = candidate.fields.get("setup_note")
    assert isinstance(note, str)
    assert "@modelcontextprotocol/server-filesystem" in note


def test_the_repository_url_becomes_the_repo_identity(load_fixture: FixtureLoader) -> None:
    records, _ = parse_page(load_fixture("registry_page1.json"), url=PAGE_URL)
    candidate = normalize_registry(_record(records, "io.github.tavily-ai/tavily-mcp"))
    assert candidate is not None
    assert "mcp:repo:github.com/tavily-ai/tavily-mcp" in candidate.alias_keys
    assert "mcp:registry:io.github.tavily-ai/tavily-mcp" in candidate.alias_keys


# --- the crawl -------------------------------------------------------------


async def test_the_crawl_follows_the_cursor_and_stops_when_it_runs_out(
    load_fixture: FixtureLoader,
    mock_client: ClientFactory,
    no_sleep: RecordingSleep,
) -> None:
    requested: list[str] = []
    pages: dict[str | None, JsonValue] = {
        None: load_fixture("registry_page1.json"),
        "eyJvIjo0fQ": load_fixture("registry_page2.json"),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        assert request.headers["user-agent"].startswith("jhin-catalog/")
        assert "authorization" not in request.headers
        return httpx.Response(200, json=pages[request.url.params.get("cursor")])

    async with mock_client(handler) as client:
        fetched = await RegistrySource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)

    assert len(requested) == 2
    assert "cursor=eyJvIjo0fQ" in requested[1]
    assert fetched.source_id == "registry"
    assert fetched.page_count == 2
    assert fetched.entry_count == len(fetched.records) == 5
    assert no_sleep.delays == []


async def test_the_crawl_stops_at_the_page_limit_it_was_given(
    load_fixture: FixtureLoader,
    mock_client: ClientFactory,
    no_sleep: RecordingSleep,
) -> None:
    """Each page names a cursor nobody has seen, so only the limit ends it."""
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = load_fixture("registry_page1.json")
        assert isinstance(payload, dict)
        payload["metadata"] = {"nextCursor": f"cursor-{calls}"}
        return httpx.Response(200, json=payload)

    limits = DEFAULT_LIMITS.model_copy(update={"max_pages": 3})
    async with mock_client(handler) as client:
        fetched = await RegistrySource().fetch(client, limits=limits, sleep=no_sleep)

    assert calls == 3
    assert fetched.page_count == 3


async def test_a_cursor_the_registry_repeats_ends_the_walk_rather_than_looping(
    load_fixture: FixtureLoader,
    mock_client: ClientFactory,
    no_sleep: RecordingSleep,
) -> None:
    """A registry that keeps handing back the same cursor is a loop, not a crawl."""
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=load_fixture("registry_page1.json"))

    async with mock_client(handler) as client:
        fetched = await RegistrySource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)

    assert calls == 2
    assert fetched.page_count == 2
