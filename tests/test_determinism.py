"""Two builds of the same input produce the same bytes."""

from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from jhin_catalog.build import (
    entries_from_fetches,
    plan_shards,
    render_lock,
    render_shard,
    write_lock,
    write_shards,
)
from jhin_catalog.types import (
    SHARD_COUNT,
    CatalogEntry,
    JsonObject,
    McpEntry,
    RawRecord,
    SkillEntry,
    SourceFetch,
    SourceRef,
    all_shards,
    loads_line,
    shard_for,
)

NOW = datetime(2026, 8, 29, 4, 11, 7, tzinfo=UTC)
LATER = datetime(2026, 8, 30, 5, 22, 18, tzinfo=UTC)


def _mcp(index: int) -> McpEntry:
    key = f"mcp:registry:example.test/server-{index:03d}"
    return McpEntry(
        kind="mcp",
        canonical_key=key,
        slug=f"srv_{index:03d}",
        name=f"Server {index:03d} — ünicode",
        description="A synthetic server, written twice and compared byte for byte.",
        trust_tier="registry_verified",
        sources=(SourceRef(source_id="registry", upstream_id=key, url="https://reg.example/x"),),
        category="Developer tools",
        icon="terminal",
        mcp_url=f"https://server-{index:03d}.example/mcp",
        transport="streamable_http",
        tags=("mcp-server", "synthetic"),
        popularity=round(index / 1000, 4),
    )


def _server_payload(index: int) -> JsonObject:
    """One registry list item, in the shape ``parse_page`` keeps."""
    return {
        "server": {
            "name": f"io.github.example-org/server-{index:03d}",
            "title": f"Server {index:03d}",
            "description": "A synthetic registry row for the determinism build.",
            "version": "1.0.0",
            "repository": {
                "url": f"https://github.com/example-org/server-{index:03d}",
                "source": "github",
            },
            "remotes": [
                {
                    "type": "streamable-http",
                    "url": f"https://server-{index:03d}.example/mcp",
                }
            ],
        },
        "_meta": {
            "io.modelcontextprotocol.registry/official": {
                "status": "active",
                "isLatest": True,
            }
        },
    }


def _registry_fetch(count: int = 12) -> SourceFetch:
    records = tuple(
        RawRecord(
            source_id="registry",
            upstream_id=f"io.github.example-org/server-{index:03d}",
            url=f"https://registry.example/v0.1/servers/server-{index:03d}/versions",
            payload=_server_payload(index),
        )
        for index in range(count)
    )
    return SourceFetch(
        source_id="registry",
        url="https://registry.modelcontextprotocol.io/v0.1/servers",
        sha256="a" * 64,
        entry_count=count,
        page_count=1,
        records=records,
    )


def _write_both(root: Path, entries: Sequence[CatalogEntry]) -> None:
    servers = [entry for entry in entries if isinstance(entry, McpEntry)]
    skills = [entry for entry in entries if isinstance(entry, SkillEntry)]
    write_shards(root, "mcp", plan_shards(servers))
    write_shards(root, "skill", plan_shards(skills))


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*.jsonl"))
    }


def test_two_builds_of_the_same_input_write_the_same_five_hundred_and_twelve_files(
    tmp_path: Path,
) -> None:
    entries = [_mcp(index) for index in range(40)]
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_both(first, entries)
    _write_both(second, entries)

    left = _snapshot(first)
    right = _snapshot(second)
    assert len(left) == 2 * SHARD_COUNT
    assert left == right


def test_shuffling_the_input_changes_nothing_about_the_output(tmp_path: Path) -> None:
    """Sort order is a property of the data, not of the order it arrived in."""
    entries = [_mcp(index) for index in range(40)]
    shuffled = list(entries)
    random.Random(20260829).shuffle(shuffled)
    assert [entry.canonical_key for entry in shuffled] != [entry.canonical_key for entry in entries]

    ordered = tmp_path / "ordered"
    jumbled = tmp_path / "jumbled"
    _write_both(ordered, entries)
    _write_both(jumbled, shuffled)
    assert _snapshot(ordered) == _snapshot(jumbled)


def test_the_whole_pipeline_is_a_function_of_its_recorded_input(tmp_path: Path) -> None:
    fetches = [_registry_fetch()]
    first = entries_from_fetches(fetches, overrides=[], denylist=[])
    second = entries_from_fetches(fetches, overrides=[], denylist=[])
    assert first == second
    assert first

    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_both(left, first)
    _write_both(right, second)
    assert _snapshot(left) == _snapshot(right)


def test_no_written_byte_is_ever_a_carriage_return(tmp_path: Path) -> None:
    _write_both(tmp_path, [_mcp(index) for index in range(40)])
    for body in _snapshot(tmp_path).values():
        assert b"\r" not in body


def test_an_empty_shard_is_an_empty_file_rather_than_a_blank_line(tmp_path: Path) -> None:
    """One entry, so exactly one of the 512 files has anything in it."""
    _write_both(tmp_path, [_mcp(0)])
    bodies = _snapshot(tmp_path)
    assert len(bodies) == 2 * SHARD_COUNT
    assert sum(1 for body in bodies.values() if not body) == 2 * SHARD_COUNT - 1


def test_every_populated_shard_ends_in_one_newline_and_holds_no_blank_line(
    tmp_path: Path,
) -> None:
    _write_both(tmp_path, [_mcp(index) for index in range(40)])
    for body in _snapshot(tmp_path).values():
        if not body:
            continue
        assert body.endswith(b"\n")
        assert b"\n\n" not in body


def test_every_file_is_named_by_the_shard_its_records_hash_to(tmp_path: Path) -> None:
    _write_both(tmp_path, [_mcp(index) for index in range(40)])
    for name, body in _snapshot(tmp_path).items():
        shard = Path(name).stem
        for line in body.decode("utf-8").splitlines():
            assert shard_for(loads_line(line + "\n").canonical_key) == shard


def test_all_two_hundred_and_fifty_six_names_are_written_even_with_no_data(
    tmp_path: Path,
) -> None:
    written = write_shards(tmp_path, "mcp", plan_shards([]))
    assert len(written) == SHARD_COUNT
    assert [Path(path).stem for path in written] == list(all_shards())


def test_rendering_a_shard_from_its_own_parsed_contents_is_idempotent(
    tmp_path: Path,
) -> None:
    """The strongest statement of §5: the file is a pure function of what is in it."""
    _write_both(tmp_path, [_mcp(index) for index in range(40)])
    for body in _snapshot(tmp_path).values():
        parsed = [loads_line(line + "\n") for line in body.decode("utf-8").splitlines()]
        assert render_shard(parsed) == body


def test_records_inside_a_shard_are_sorted_by_canonical_key(tmp_path: Path) -> None:
    _write_both(tmp_path, [_mcp(index) for index in range(200)])
    for body in _snapshot(tmp_path).values():
        if not body:
            continue
        keys = [loads_line(line + "\n").canonical_key for line in body.decode("utf-8").splitlines()]
        assert keys == sorted(keys)


def test_the_lock_file_differs_only_in_its_timestamp_when_the_clock_moves(
    tmp_path: Path,
) -> None:
    """``sources.lock`` is the one output a clock may reach."""
    fetches = [_registry_fetch()]
    early = tmp_path / "early"
    late = tmp_path / "late"
    early.mkdir()
    late.mkdir()
    write_lock(early, render_lock(fetches, now=NOW))
    write_lock(late, render_lock(fetches, now=LATER))

    first = (early / "sources.lock").read_text("utf-8")
    second = (late / "sources.lock").read_text("utf-8")
    assert first != second
    assert first.replace("2026-08-29T04:11:07Z", "") == second.replace("2026-08-30T05:22:18Z", "")


def test_the_lock_file_is_byte_identical_for_the_same_clock(tmp_path: Path) -> None:
    fetches = [_registry_fetch()]
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    write_lock(left, render_lock(fetches, now=NOW))
    write_lock(right, render_lock(fetches, now=NOW))
    assert (left / "sources.lock").read_bytes() == (right / "sources.lock").read_bytes()


def test_the_timestamp_is_second_precision_utc(tmp_path: Path) -> None:
    lock = render_lock([_registry_fetch()], now=datetime(2026, 8, 29, 4, 11, 7, 987654, tzinfo=UTC))
    assert lock.sources[0].fetched_at == "2026-08-29T04:11:07Z"
