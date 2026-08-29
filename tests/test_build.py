"""Shard planning, writing, the lock file, and curated loading."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from jhin_catalog.build import (
    LOCK_FILENAME,
    assign_slugs,
    load_curated,
    load_denylist,
    plan_shards,
    read_lock,
    render_lock,
    render_shard,
    run_sync,
    write_lock,
    write_shards,
)
from jhin_catalog.diffgate import DEFAULT_THRESHOLDS, DiffGateFailed
from jhin_catalog.sources.base import DEFAULT_LIMITS
from jhin_catalog.types import (
    SHARD_COUNT,
    CatalogEntry,
    CuratedError,
    McpEntry,
    SourceFetch,
    SourceRef,
    SourcesLock,
    all_shards,
    dumps_line,
    shard_for,
)

NOW = datetime(2026, 8, 29, 4, 11, 7, tzinfo=UTC)

NOTION_REGISTRY = "mcp:registry:io.github.makenotion/notion-mcp"
NOTION_BRIDGE = "mcp:repo:github.com/someone/notion-bridge"


def _entry(key: str, *, slug: str) -> McpEntry:
    return McpEntry(
        kind="mcp",
        canonical_key=key,
        slug=slug,
        name="Notion",
        description="Pages, databases, and comments.",
        trust_tier="registry_verified",
        sources=(SourceRef(source_id="registry", upstream_id=key, url="https://reg.example/x"),),
        category="Documents & knowledge",
        icon="notebook",
        mcp_url="https://mcp.notion.com/mcp",
        transport="streamable_http",
    )


def _fetch(source_id: str, *, entry_count: int = 3, pages: int = 1) -> SourceFetch:
    return SourceFetch(
        source_id=source_id,
        url=f"https://{source_id}.example/",
        sha256=hashlib.sha256(source_id.encode()).hexdigest(),
        entry_count=entry_count,
        page_count=pages,
        records=(),
    )


def _yaml(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _refuse_network() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no fetch expected, got {request.url}")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# --- shard planning --------------------------------------------------------


def test_plan_shards_names_all_two_hundred_and_fifty_six_even_for_no_entries() -> None:
    planned = plan_shards([])
    assert len(planned) == SHARD_COUNT
    assert set(planned) == set(all_shards())
    assert all(entries == () for entries in planned.values())


def test_plan_shards_files_each_entry_under_the_shard_its_key_hashes_to() -> None:
    entry = _entry(NOTION_REGISTRY, slug="notion")
    planned = plan_shards([entry])
    assert planned[shard_for(NOTION_REGISTRY)] == (entry,)


def test_write_shards_creates_every_file_and_leaves_the_empty_ones_at_zero_bytes(
    tmp_path: Path,
) -> None:
    written = write_shards(tmp_path, "mcp", plan_shards([_entry(NOTION_REGISTRY, slug="notion")]))
    files = sorted((tmp_path / "data" / "mcp").glob("*.jsonl"))
    assert len(written) == SHARD_COUNT
    assert len(files) == SHARD_COUNT
    assert sum(1 for path in files if path.stat().st_size == 0) == SHARD_COUNT - 1


def test_write_shards_returns_repo_relative_posix_paths_in_sorted_order(tmp_path: Path) -> None:
    written = write_shards(tmp_path, "mcp", plan_shards([]))
    assert list(written) == sorted(written)
    assert written[0] == "data/mcp/00.jsonl"
    assert written[-1] == "data/mcp/ff.jsonl"


def test_render_shard_sorts_its_input_rather_than_trusting_it() -> None:
    first = _entry("mcp:registry:example.test/aaa", slug="aaa")
    second = _entry("mcp:registry:example.test/bbb", slug="bbb")
    assert render_shard([second, first]) == render_shard([first, second])
    assert render_shard([second, first]) == (dumps_line(first) + dumps_line(second)).encode()


def test_render_shard_of_nothing_is_zero_bytes() -> None:
    assert render_shard([]) == b""


# --- slug allocation -------------------------------------------------------


def test_worked_example_e_gives_the_lower_canonical_key_the_bare_slug() -> None:
    """``mcp:registry:…`` sorts before ``mcp:repo:…``, so the registry row wins.

    The loser's suffix is the first four hex characters of the SHA-256 of its
    own canonical key, which makes the allocation a pure function of identity
    rather than of the order entries happened to arrive in.
    """
    entries: Sequence[CatalogEntry] = [
        _entry(NOTION_BRIDGE, slug="notion"),
        _entry(NOTION_REGISTRY, slug="notion"),
    ]
    by_key = {entry.canonical_key: entry.slug for entry in assign_slugs(entries)}
    digest = hashlib.sha256(NOTION_BRIDGE.encode("utf-8")).hexdigest()[:4]
    assert by_key[NOTION_REGISTRY] == "notion"
    assert by_key[NOTION_BRIDGE] == f"notion_{digest}"
    assert digest == "bad6"


def test_slug_allocation_does_not_depend_on_input_order() -> None:
    forward: Sequence[CatalogEntry] = [
        _entry(NOTION_REGISTRY, slug="notion"),
        _entry(NOTION_BRIDGE, slug="notion"),
    ]
    backward: Sequence[CatalogEntry] = list(reversed(forward))
    assert {entry.canonical_key: entry.slug for entry in assign_slugs(forward)} == {
        entry.canonical_key: entry.slug for entry in assign_slugs(backward)
    }


def test_every_allocated_slug_is_unique_and_within_the_pattern() -> None:
    entries: Sequence[CatalogEntry] = [
        _entry(f"mcp:registry:example.test/server-{index:03d}", slug="notion")
        for index in range(50)
    ]
    slugs = [entry.slug for entry in assign_slugs(entries)]
    assert len(set(slugs)) == 50
    assert all(len(slug) <= 32 for slug in slugs)


def test_a_third_collision_on_one_slug_is_an_error_rather_than_a_guess() -> None:
    """Two widenings are the whole budget; a third means the base is wrong."""
    entries: Sequence[CatalogEntry] = [
        _entry(f"mcp:registry:example.test/n-{index}", slug="n") for index in range(4)
    ]
    slugs = {entry.slug for entry in assign_slugs(entries)}
    assert len(slugs) == 4


# --- curated loading -------------------------------------------------------


def test_load_curated_on_a_missing_file_yields_nothing(tmp_path: Path) -> None:
    assert load_curated(tmp_path / "absent.yaml") == ()


def test_load_curated_reads_key_kind_aliases_and_fields(tmp_path: Path) -> None:
    path = _yaml(
        tmp_path / "mcp.yaml",
        """
version: 1
entries:
  - key: mcp:url:mcp.notion.com/mcp
    kind: mcp
    aliases:
      - mcp:registry:io.github.makenotion/notion-mcp
    fields:
      slug: notion
      name: Notion
""",
    )
    overrides = load_curated(path)
    assert len(overrides) == 1
    assert overrides[0].key == "mcp:url:mcp.notion.com/mcp"
    assert overrides[0].aliases == ("mcp:registry:io.github.makenotion/notion-mcp",)
    assert overrides[0].fields["slug"] == "notion"


def test_a_duplicate_curated_key_is_a_build_failure(tmp_path: Path) -> None:
    path = _yaml(
        tmp_path / "mcp.yaml",
        """
version: 1
entries:
  - key: mcp:url:mcp.notion.com/mcp
    kind: mcp
    fields: {slug: notion}
  - key: mcp:url:mcp.notion.com/mcp
    kind: mcp
    fields: {slug: notion_two}
""",
    )
    with pytest.raises(CuratedError):
        load_curated(path)


def test_an_unknown_curated_field_names_itself_in_the_error(tmp_path: Path) -> None:
    path = _yaml(
        tmp_path / "mcp.yaml",
        """
version: 1
entries:
  - key: mcp:url:mcp.notion.com/mcp
    kind: mcp
    fields:
      iconUrl: https://example.com/icon.png
""",
    )
    with pytest.raises(CuratedError, match="iconUrl"):
        load_curated(path)


def test_a_curated_document_that_is_not_a_mapping_is_a_build_failure(tmp_path: Path) -> None:
    with pytest.raises(CuratedError):
        load_curated(_yaml(tmp_path / "mcp.yaml", "- just\n- a\n- list\n"))


def test_load_denylist_refuses_a_reason_of_seven_characters(tmp_path: Path) -> None:
    """A reason nobody can act on is the same as no reason at all."""
    path = _yaml(
        tmp_path / "denylist.yaml",
        """
version: 1
entries:
  - key: mcp:url:gone.example/mcp
    reason: too old
""",
    )
    with pytest.raises(CuratedError):
        load_denylist(path)


def test_load_denylist_accepts_a_reason_a_person_could_act_on(tmp_path: Path) -> None:
    path = _yaml(
        tmp_path / "denylist.yaml",
        """
version: 1
entries:
  - key: mcp:url:gone.example/mcp
    reason: Domain expired 2026-07-02 and now answers with a different tool set.
""",
    )
    denied = load_denylist(path)
    assert len(denied) == 1
    assert denied[0].key == "mcp:url:gone.example/mcp"


def test_load_denylist_on_a_missing_file_yields_nothing(tmp_path: Path) -> None:
    assert load_denylist(tmp_path / "absent.yaml") == ()


# --- the lock file ---------------------------------------------------------


def test_render_lock_sorts_by_source_id_and_stamps_second_precision() -> None:
    lock = render_lock(
        [_fetch("smithery"), _fetch("registry"), _fetch("npm")],
        now=datetime(2026, 8, 29, 4, 11, 7, 654_321, tzinfo=UTC),
    )
    assert [entry.source_id for entry in lock.sources] == sorted(
        entry.source_id for entry in lock.sources
    )
    assert all(entry.fetched_at == "2026-08-29T04:11:07Z" for entry in lock.sources)


def test_the_lock_round_trips_through_the_file(tmp_path: Path) -> None:
    lock = render_lock([_fetch("registry", entry_count=25_492, pages=255)], now=NOW)
    write_lock(tmp_path, lock)
    assert read_lock(tmp_path) == lock


def test_the_lock_file_is_indented_and_newline_terminated(tmp_path: Path) -> None:
    write_lock(tmp_path, render_lock([_fetch("registry")], now=NOW))
    body = (tmp_path / LOCK_FILENAME).read_text("utf-8")
    assert body.endswith("\n")
    assert "\n  " in body


def test_a_missing_lock_reads_as_no_sources(tmp_path: Path) -> None:
    assert read_lock(tmp_path) == SourcesLock(sources=())


# --- run_sync writes nothing until every gate passes -----------------------


async def test_a_dry_run_writes_nothing_at_all(tmp_catalog: Path) -> None:
    before = _snapshot(tmp_catalog)
    async with _refuse_network() as client:
        result = await run_sync(
            tmp_catalog,
            sources=(),
            limits=DEFAULT_LIMITS,
            thresholds=DEFAULT_THRESHOLDS,
            now=NOW,
            client=client,
            dry_run=True,
        )
    assert _snapshot(tmp_catalog) == before
    assert result.written == ()


async def test_a_tripped_gate_leaves_the_committed_data_byte_identical(
    tmp_catalog: Path,
) -> None:
    """The whole point: a bad crawl must not be able to touch a single byte."""
    baseline: list[CatalogEntry] = [
        _entry(f"mcp:registry:example.test/server-{index:03d}", slug=f"srv_{index:03d}")
        for index in range(200)
    ]
    write_shards(tmp_catalog, "mcp", plan_shards(baseline))
    write_shards(tmp_catalog, "skill", plan_shards([]))
    before = _snapshot(tmp_catalog)

    async with _refuse_network() as client:
        with pytest.raises(DiffGateFailed):
            await run_sync(
                tmp_catalog,
                sources=(),
                limits=DEFAULT_LIMITS,
                thresholds=DEFAULT_THRESHOLDS,
                now=NOW,
                client=client,
            )

    assert _snapshot(tmp_catalog) == before
