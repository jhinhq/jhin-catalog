"""The gate that stops an empty upstream page from deleting the catalog.

Every upstream in this index has been observed to answer a perfectly valid
request with an empty list. A build that trusted such a page would rewrite
``data/**`` with most of the corpus removed, and the removal would look
exactly like a legitimate one. So a build compares itself against what is
already committed before it writes anything: too large a drop or too much
churn refuses the write and names what changed, and a source that returned
records yesterday and none today is read as a fetch fault rather than as a
mass deletion. Additions are never blocked — growth is the normal case.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

from jhin_catalog.types import (
    CatalogEntry,
    CatalogError,
    DiffReport,
    DiffThresholds,
    SourcesLock,
    all_shards,
    dumps_line,
    loads_line,
    shard_for,
)

DEFAULT_THRESHOLDS: Final[DiffThresholds] = DiffThresholds()

_DATA_DIRNAME: Final[str] = "data"

# ``kind`` is singular because a record is one MCP server or one skill; the
# directory is plural because it holds many. The two spellings are fixed by
# the repository manifest, so the mapping is explicit rather than inferred.
_KIND_DIRNAMES: Final[Mapping[str, str]] = {"mcp": "mcp", "skill": "skills"}

_FRACTION_DECIMALS: Final[int] = 6
_SAMPLE_KEYS: Final[int] = 3


class DiffGateFailed(CatalogError):
    """The build would replace too much of the committed data to be trusted."""

    def __init__(self, message: str, *, report: DiffReport) -> None:
        super().__init__(message)
        self.report = report


def load_shards(root: Path, kind: str) -> dict[str, CatalogEntry]:
    """Every committed entry for ``kind``, keyed by ``canonical_key``.

    A missing ``root/data/{kind}`` directory yields ``{}``, which is how a
    bootstrap build starts. Everything else about the layout is checked here
    rather than trusted: a line that will not parse, a key that hashes to a
    different shard than the file it sits in, an entry of the wrong kind, and
    a key that appears twice are all faults, because each one means the
    committed bytes are no longer what this code would have written.
    """
    directory = _kind_directory(root, kind)
    if not directory.is_dir():
        return {}

    entries: dict[str, CatalogEntry] = {}
    for shard in all_shards():
        path = directory / f"{shard}.jsonl"
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if not raw:
            continue
        for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line:
                raise CatalogError(f"{_relative(root, path)} line {number} is empty")
            entry = loads_line(line)
            if entry.kind != kind:
                raise CatalogError(
                    f"{_relative(root, path)} line {number} holds a {entry.kind} entry"
                )
            if shard_for(entry.canonical_key) != shard:
                raise CatalogError(
                    f"{entry.canonical_key} belongs in shard "
                    f"{shard_for(entry.canonical_key)}, not {shard}"
                )
            if entry.canonical_key in entries:
                raise CatalogError(f"{entry.canonical_key} appears in more than one shard")
            entries[entry.canonical_key] = entry
    return entries


def compare(
    baseline: Mapping[str, CatalogEntry],
    candidate: Mapping[str, CatalogEntry],
    *,
    kind: str,
) -> DiffReport:
    """Set difference plus a byte comparison of ``dumps_line`` for shared keys.

    Comparing the serialised line rather than the model is the whole point: a
    field the schema gained but the data has not is a change the gate should
    see, and it is the bytes on disk that a deployment syncs.
    """
    baseline_keys = set(baseline)
    candidate_keys = set(candidate)
    added = tuple(sorted(candidate_keys - baseline_keys))
    dropped = tuple(sorted(baseline_keys - candidate_keys))
    changed = tuple(
        sorted(
            key
            for key in baseline_keys & candidate_keys
            if dumps_line(baseline[key]) != dumps_line(candidate[key])
        )
    )
    total = len(baseline)
    return DiffReport(
        kind=kind,
        baseline_count=total,
        candidate_count=len(candidate),
        added=added,
        dropped=dropped,
        changed=changed,
        drop_fraction=_fraction(len(dropped), total),
        change_fraction=_fraction(len(changed), total),
    )


def check(report: DiffReport, thresholds: DiffThresholds = DEFAULT_THRESHOLDS) -> None:
    """Raise ``DiffGateFailed`` when a threshold is exceeded.

    The gate is skipped when ``report.baseline_count <
    thresholds.min_baseline_entries``. A bootstrap has nothing to compare
    against, and a deliberately tiny catalog would otherwise be permanently
    un-buildable: with four entries committed, one legitimate removal is a
    twenty-five per cent drop.
    """
    if report.baseline_count < thresholds.min_baseline_entries:
        return
    if report.drop_fraction > thresholds.max_drop_fraction:
        raise DiffGateFailed(
            f"{report.kind}: {len(report.dropped)} of {report.baseline_count} entries would be "
            f"dropped ({report.drop_fraction:.1%}), over the "
            f"{thresholds.max_drop_fraction:.1%} limit"
            f"{_sample(report.dropped)}",
            report=report,
        )
    if report.change_fraction > thresholds.max_change_fraction:
        raise DiffGateFailed(
            f"{report.kind}: {len(report.changed)} of {report.baseline_count} entries would "
            f"change ({report.change_fraction:.1%}), over the "
            f"{thresholds.max_change_fraction:.1%} limit"
            f"{_sample(report.changed)}",
            report=report,
        )


def check_source_counts(previous: SourcesLock, current: SourcesLock) -> tuple[str, ...]:
    """Source ids that returned zero records after previously returning some.

    Upstream APIs are observed to serve empty pages intermittently, so a
    source collapsing to zero is treated as a fetch fault. A source that
    reported nothing last time and nothing this time is not news, and a source
    appearing for the first time cannot have collapsed.
    """
    before = {entry.source_id: entry.entry_count for entry in previous.sources}
    return tuple(
        sorted(
            entry.source_id
            for entry in current.sources
            if entry.entry_count == 0 and before.get(entry.source_id, 0) > 0
        )
    )


def _kind_directory(root: Path, kind: str) -> Path:
    """Where the shards for one kind live under a repository root.

    ``build`` and ``cli`` share this rather than each spelling the layout
    out again, so the directory a build writes is by construction the one a
    verify reads.
    """
    return root / _DATA_DIRNAME / _KIND_DIRNAMES.get(kind, kind)


def _fraction(part: int, total: int) -> float:
    """A share of the baseline, rounded so two builds agree on the value."""
    if total <= 0:
        return 0.0
    return round(part / total, _FRACTION_DECIMALS)


def _sample(keys: tuple[str, ...]) -> str:
    """A few of the affected keys, so the failure names something concrete."""
    if not keys:
        return ""
    shown = ", ".join(keys[:_SAMPLE_KEYS])
    if len(keys) > _SAMPLE_KEYS:
        return f"; for example {shown}, and {len(keys) - _SAMPLE_KEYS} more"
    return f"; namely {shown}"


def _relative(root: Path, path: Path) -> str:
    """A repository-relative POSIX path for a message, absolute as a fallback."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
