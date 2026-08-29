"""The gate that stops an empty upstream page from deleting the catalog."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from jhin_catalog.build import plan_shards, write_shards
from jhin_catalog.diffgate import (
    DEFAULT_THRESHOLDS,
    DiffGateFailed,
    check,
    check_source_counts,
    compare,
    load_shards,
)
from jhin_catalog.types import (
    CatalogEntry,
    CatalogError,
    DiffThresholds,
    LockEntry,
    McpEntry,
    SourceRef,
    SourcesLock,
    dumps_line,
    shard_for,
)

FETCHED_AT = "2026-08-29T04:11:07Z"
SHA = "0" * 64


def _entry(index: int, *, name: str | None = None) -> McpEntry:
    key = f"mcp:registry:example.test/server-{index:03d}"
    return McpEntry(
        kind="mcp",
        canonical_key=key,
        slug=f"srv_{index:03d}",
        name=name or f"Server {index:03d}",
        description="A synthetic server, present so the gate has something to measure.",
        trust_tier="registry_verified",
        sources=(SourceRef(source_id="registry", upstream_id=key, url="https://reg.example/x"),),
        category="Developer tools",
        icon="terminal",
        mcp_url=f"https://server-{index:03d}.example/mcp",
        transport="streamable_http",
    )


def _keyed(entries: Sequence[CatalogEntry]) -> dict[str, CatalogEntry]:
    return {entry.canonical_key: entry for entry in entries}


def _write(root: Path, entries: Sequence[CatalogEntry]) -> None:
    write_shards(root, "mcp", plan_shards(entries))


def _lock(**counts: int) -> SourcesLock:
    return SourcesLock(
        sources=tuple(
            LockEntry(
                source_id=source_id,
                url=f"https://{source_id}.example/",
                fetched_at=FETCHED_AT,
                sha256=SHA,
                entry_count=count,
                page_count=1,
            )
            for source_id, count in sorted(counts.items())
        )
    )


# --- the drop threshold ----------------------------------------------------


def test_six_dropped_of_a_hundred_trips_the_gate() -> None:
    baseline = _keyed([_entry(index) for index in range(100)])
    candidate = _keyed([_entry(index) for index in range(94)])
    report = compare(baseline, candidate, kind="mcp")

    assert report.baseline_count == 100
    assert report.candidate_count == 94
    assert len(report.dropped) == 6
    assert report.drop_fraction == 0.06

    with pytest.raises(DiffGateFailed) as raised:
        check(report)
    assert raised.value.report.drop_fraction == 0.06


def test_four_dropped_of_a_hundred_passes() -> None:
    baseline = _keyed([_entry(index) for index in range(100)])
    candidate = _keyed([_entry(index) for index in range(96)])
    report = compare(baseline, candidate, kind="mcp")
    assert report.drop_fraction == 0.04
    check(report)


# --- the change threshold --------------------------------------------------


def _changed(count: int) -> tuple[dict[str, CatalogEntry], dict[str, CatalogEntry]]:
    baseline = [_entry(index) for index in range(100)]
    candidate = [
        _entry(index, name=f"Renamed {index:03d}") if index < count else _entry(index)
        for index in range(100)
    ]
    return _keyed(baseline), _keyed(candidate)


def test_twenty_one_changed_of_a_hundred_trips_the_gate() -> None:
    report = compare(*_changed(21), kind="mcp")
    assert len(report.changed) == 21
    assert report.change_fraction == 0.21
    with pytest.raises(DiffGateFailed):
        check(report)


def test_twenty_changed_of_a_hundred_is_exactly_tolerable() -> None:
    report = compare(*_changed(20), kind="mcp")
    assert report.change_fraction == 0.20
    check(report)


def test_a_changed_entry_is_found_by_comparing_bytes_not_identity() -> None:
    baseline, candidate = _changed(1)
    report = compare(baseline, candidate, kind="mcp")
    assert report.changed == ("mcp:registry:example.test/server-000",)
    assert report.added == ()
    assert report.dropped == ()


# --- additions and bootstraps ----------------------------------------------


def test_additions_alone_never_trip_the_gate() -> None:
    """Growth is the point. Only losses and rewrites are suspicious."""
    baseline = _keyed([_entry(index) for index in range(100)])
    candidate = _keyed([_entry(index) for index in range(1000)])
    report = compare(baseline, candidate, kind="mcp")
    assert len(report.added) == 900
    assert report.drop_fraction == 0.0
    assert report.change_fraction == 0.0
    check(report)


def test_an_empty_baseline_skips_the_gate_entirely() -> None:
    """A bootstrap must not be permanently un-buildable."""
    report = compare({}, _keyed([_entry(0)]), kind="mcp")
    assert report.baseline_count == 0
    assert report.drop_fraction == 0.0
    check(report)


def test_a_baseline_below_the_minimum_skips_the_gate() -> None:
    baseline = _keyed([_entry(index) for index in range(99)])
    report = compare(baseline, {}, kind="mcp")
    assert report.drop_fraction == 1.0
    check(report, DiffThresholds(min_baseline_entries=100))


def test_the_same_report_fails_once_the_baseline_is_big_enough() -> None:
    baseline = _keyed([_entry(index) for index in range(99)])
    report = compare(baseline, {}, kind="mcp")
    with pytest.raises(DiffGateFailed):
        check(report, DiffThresholds(min_baseline_entries=50))


def test_the_default_thresholds_are_the_documented_ones() -> None:
    assert DEFAULT_THRESHOLDS.max_drop_fraction == 0.05
    assert DEFAULT_THRESHOLDS.max_change_fraction == 0.20
    assert DEFAULT_THRESHOLDS.min_baseline_entries == 100


# --- reading what is committed ---------------------------------------------


def test_a_missing_data_directory_reads_as_an_empty_baseline(tmp_path: Path) -> None:
    assert load_shards(tmp_path, "mcp") == {}


def test_load_shards_reads_back_exactly_what_write_shards_wrote(tmp_path: Path) -> None:
    entries = [_entry(index) for index in range(20)]
    _write(tmp_path, entries)
    assert load_shards(tmp_path, "mcp") == _keyed(entries)


def test_a_malformed_line_is_a_read_error(tmp_path: Path) -> None:
    _write(tmp_path, [_entry(0)])
    (tmp_path / "data" / "mcp" / "00.jsonl").write_bytes(b"{not json}\n")
    with pytest.raises(CatalogError):
        load_shards(tmp_path, "mcp")


def test_an_entry_filed_in_the_wrong_shard_is_a_read_error(tmp_path: Path) -> None:
    """The shard name is derived from the key, so a mismatch means a hand edit."""
    entry = _entry(0)
    _write(tmp_path, [entry])
    correct = shard_for(entry.canonical_key)
    wrong = "00" if correct != "00" else "01"
    directory = tmp_path / "data" / "mcp"
    (directory / f"{correct}.jsonl").write_bytes(b"")
    (directory / f"{wrong}.jsonl").write_bytes(dumps_line(entry).encode("utf-8"))
    with pytest.raises(CatalogError):
        load_shards(tmp_path, "mcp")


def test_one_key_in_two_shards_is_a_read_error(tmp_path: Path) -> None:
    entry = _entry(0)
    _write(tmp_path, [entry])
    correct = shard_for(entry.canonical_key)
    other = "00" if correct != "00" else "01"
    directory = tmp_path / "data" / "mcp"
    (directory / f"{other}.jsonl").write_bytes(dumps_line(entry).encode("utf-8"))
    with pytest.raises(CatalogError):
        load_shards(tmp_path, "mcp")


# --- the source-collapse tripwire ------------------------------------------


def test_a_source_that_collapsed_to_zero_is_flagged() -> None:
    """Twenty-five thousand servers do not vanish overnight; a page does."""
    previous = _lock(registry=25_492, smithery=11_000)
    current = _lock(registry=0, smithery=11_000)
    assert check_source_counts(previous, current) == ("registry",)


def test_a_source_that_was_already_empty_is_not_flagged() -> None:
    assert check_source_counts(_lock(marketplaces=0), _lock(marketplaces=0)) == ()


def test_a_source_that_merely_shrank_is_not_flagged() -> None:
    assert check_source_counts(_lock(registry=25_492), _lock(registry=12_000)) == ()


def test_a_source_that_appeared_for_the_first_time_is_not_flagged() -> None:
    assert check_source_counts(_lock(registry=10), _lock(registry=10, npm=0)) == ()


def test_every_collapsed_source_is_reported_not_just_the_first() -> None:
    previous = _lock(npm=900, registry=25_492, smithery=11_000)
    current = _lock(npm=0, registry=0, smithery=11_000)
    flagged = check_source_counts(previous, current)
    assert set(flagged) == {"npm", "registry"}
    assert len(flagged) == 2
