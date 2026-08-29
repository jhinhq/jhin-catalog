"""The command surface: sync, build, verify, export, diff, and stats.

Six commands over one pipeline. ``sync`` crawls and writes, ``build`` does
the same from recorded pages with no network, ``verify`` re-derives the
committed bytes and refuses to agree that a hand edit is data, ``export``
projects the strongest records into Jhin's ``catalog.json``, ``diff`` reports
what a build would change without changing it, and ``stats`` counts what is
there. Every known failure maps to its own exit code so a workflow can tell a
refused write from a broken upstream, and ``--json`` makes any command's
answer machine-readable in the one canonical encoding this repo emits.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import traceback
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from jhin_catalog.build import (
    _DENYLIST_FILENAME,
    CURATED_DIRNAME,
    DEFAULT_EXPORT_LIMIT,
    SCHEMA_PATH,
    _finish_build,
    _load_all_curated,
    export_catalog_json,
    load_denylist,
    load_marketplace_policy,
    render_schema,
    render_shard,
    run_sync,
)
from jhin_catalog.diffgate import (
    DEFAULT_THRESHOLDS,
    DiffGateFailed,
    _kind_directory,
    check,
    compare,
    load_shards,
)
from jhin_catalog.http import FetchError, build_client
from jhin_catalog.sources import ALL_SOURCES, source_by_id
from jhin_catalog.sources.base import DEFAULT_LIMITS, Source, SourceError, SourceLimits
from jhin_catalog.types import (
    KINDS,
    SOURCE_RANK,
    BuildResult,
    CatalogEntry,
    CatalogError,
    CuratedError,
    DedupeError,
    DiffReport,
    DiffThresholds,
    JsonObject,
    JsonValue,
    NormalizeError,
    SourceFetch,
    SourcesLock,
    all_shards,
    canonical_json,
    loads_line,
    shard_for,
)

_EXIT_OK: Final[int] = 0
_EXIT_INTERNAL: Final[int] = 1
_EXIT_USAGE: Final[int] = 2
_EXIT_GATE: Final[int] = 3
_EXIT_FETCH: Final[int] = 4
_EXIT_VIOLATION: Final[int] = 5
_EXIT_CURATED: Final[int] = 6

_DEFAULT_TOKEN_ENV: Final[str] = "GITHUB_TOKEN"
_DECILES: Final[int] = 10
_PROGRAM: Final[str] = "jhin-catalog"


def build_parser() -> argparse.ArgumentParser:
    """The whole command surface, including every flag section 6 lists."""
    parser = argparse.ArgumentParser(
        prog=_PROGRAM,
        description="Build and verify the jhin-catalog index of MCP servers and Agent Skills.",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    sync = commands.add_parser("sync", help="crawl every source, gate the result, and write it")
    _add_root(sync)
    _add_thresholds(sync)
    sync.add_argument(
        "--source",
        action="append",
        choices=sorted(source.source_id for source in ALL_SOURCES),
        help="crawl only this source; repeatable, defaults to all of them",
    )
    sync.add_argument("--limit-pages", type=int, help="stop each source after this many pages")
    sync.add_argument("--limit-records", type=int, help="stop each source after this many records")
    sync.add_argument("--detail-top-n", type=int, help="how many servers earn a detail request")
    sync.add_argument("--page-size", type=int, help="rows per upstream page")
    sync.add_argument(
        "--github-token-env",
        default=_DEFAULT_TOKEN_ENV,
        help="environment variable holding the GitHub token (default: GITHUB_TOKEN)",
    )
    sync.add_argument("--allow-breaking", action="store_true", help="write even if a gate fails")
    sync.add_argument("--dry-run", action="store_true", help="run every gate but write nothing")
    sync.add_argument("--now", help="RFC3339 instant to stamp into sources.lock")
    _add_json(sync)

    build = commands.add_parser("build", help="run the same pipeline from recorded pages")
    _add_root(build)
    _add_thresholds(build)
    build.add_argument(
        "--from-cache",
        required=True,
        help="directory of recorded SourceFetch JSON documents",
    )
    build.add_argument("--allow-breaking", action="store_true", help="write even if a gate fails")
    build.add_argument("--dry-run", action="store_true", help="run every gate but write nothing")
    build.add_argument("--now", help="RFC3339 instant to stamp into sources.lock")
    _add_json(build)

    verify = commands.add_parser("verify", help="re-derive the committed data and compare")
    _add_root(verify)
    _add_json(verify)

    export = commands.add_parser("export", help="write Jhin's catalog.json projection")
    _add_root(export)
    export.add_argument("--out", help="file to write (default: standard output)")
    export.add_argument(
        "--limit", type=int, default=DEFAULT_EXPORT_LIMIT, help="how many records to publish"
    )

    diff = commands.add_parser("diff", help="report what another build would change")
    _add_root(diff)
    _add_thresholds(diff)
    diff.add_argument("--against", required=True, help="a second repository root or data directory")
    _add_json(diff)

    stats = commands.add_parser("stats", help="count what is committed")
    _add_root(stats)
    _add_json(stats)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code; never raises for a known failure.

    Every deliberate failure in this package is a ``CatalogError`` subclass,
    and each subclass has its own code so a workflow can distinguish a refused
    write from a broken upstream without parsing text. Anything else is a bug
    and prints its traceback under code 1.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(None if argv is None else list(argv))
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else _EXIT_USAGE
    if args.command is None:
        parser.print_help(sys.stderr)
        return _EXIT_USAGE

    try:
        return _dispatch(args)
    except DiffGateFailed as exc:
        return _fail(exc, _EXIT_GATE)
    except (SourceError, FetchError) as exc:
        return _fail(exc, _EXIT_FETCH)
    except CuratedError as exc:
        return _fail(exc, _EXIT_CURATED)
    except (NormalizeError, DedupeError, CatalogError) as exc:
        return _fail(exc, _EXIT_VIOLATION)
    except Exception:
        traceback.print_exc()
        return _EXIT_INTERNAL


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _dispatch(args: argparse.Namespace) -> int:
    """Route a parsed command to its handler."""
    match args.command:
        case "sync":
            return _run_sync(args)
        case "build":
            return _run_build(args)
        case "verify":
            return _run_verify(args)
        case "export":
            return _run_export(args)
        case "diff":
            return _run_diff(args)
        case "stats":
            return _run_stats(args)
        case _:
            return _EXIT_USAGE


def _run_sync(args: argparse.Namespace) -> int:
    """Crawl every selected source, gate the result, and write it."""
    root = _root(args)
    sources = _selected_sources(args.source)
    limits = _limits(args, root)
    thresholds = _thresholds(args)
    now = _instant(args.now)

    async def _crawl() -> BuildResult:
        async with build_client() as client:
            return await run_sync(
                root,
                sources=sources,
                limits=limits,
                thresholds=thresholds,
                now=now,
                client=client,
                allow_breaking=args.allow_breaking,
                dry_run=args.dry_run,
            )

    return _report_build(asyncio.run(_crawl()), json_mode=args.json, dry_run=args.dry_run)


def _run_build(args: argparse.Namespace) -> int:
    """Run the same pipeline over recorded pages, with no network at all."""
    root = _root(args)
    result = _finish_build(
        root,
        _cached_fetches(Path(args.from_cache)),
        thresholds=_thresholds(args),
        now=_instant(args.now),
        allow_breaking=args.allow_breaking,
        dry_run=args.dry_run,
    )
    return _report_build(result, json_mode=args.json, dry_run=args.dry_run)


def _run_verify(args: argparse.Namespace) -> int:
    """Re-derive every committed byte and report anything that disagrees.

    A hand edit to ``data/**`` is the failure this exists to catch: the files
    are a build artefact, and an edit that survives is an edit the next sync
    silently reverts. Everything checked here is something a build guarantees,
    so a violation means the tree and the code have drifted apart. A tree that
    has never been built is not such a drift: a repository whose data
    directories are still empty passes, because there is nothing yet to
    disagree with.
    """
    root = _root(args)
    violations: list[str] = []
    counts: dict[str, int] = {}
    keys: dict[str, str] = {}

    for kind in KINDS:
        counts[kind] = _verify_kind(root, kind, violations=violations, keys=keys)

    violations.extend(_verify_schema(root))
    violations.extend(_verify_curated(root, keys))

    if args.json:
        reported: list[JsonValue] = list(violations)
        payload: JsonObject = {
            "ok": not violations,
            "counts": {kind: counts.get(kind, 0) for kind in KINDS},
            "violations": reported,
        }
        _emit_json(payload)
    else:
        for violation in violations:
            sys.stderr.write(f"{violation}\n")
        totals = ", ".join(f"{kind}={counts.get(kind, 0)}" for kind in KINDS)
        sys.stdout.write(f"{'ok' if not violations else 'FAILED'}: {totals}\n")
    return _EXIT_VIOLATION if violations else _EXIT_OK


def _run_export(args: argparse.Namespace) -> int:
    """Write the ``catalog.json`` projection to a file or to standard output."""
    root = _root(args)
    entries = _all_entries(root)
    body = export_catalog_json(entries, limit=args.limit)
    if args.out:
        Path(args.out).write_bytes(body.encode("utf-8"))
    else:
        sys.stdout.write(body)
    return _EXIT_OK


def _run_diff(args: argparse.Namespace) -> int:
    """Report what a second tree would change here, and never write."""
    baseline_root = _root(args)
    candidate_root = _resolve_root(Path(args.against))
    thresholds = _thresholds(args)
    reports: list[DiffReport] = []
    breached = False
    for kind in KINDS:
        report = compare(
            load_shards(baseline_root, kind), load_shards(candidate_root, kind), kind=kind
        )
        reports.append(report)
        try:
            check(report, thresholds)
        except DiffGateFailed as exc:
            breached = True
            if not args.json:
                sys.stderr.write(f"{exc}\n")

    if args.json:
        _emit_json({"breached": breached, "reports": [_report_json(item) for item in reports]})
    else:
        for report in reports:
            sys.stdout.write(
                f"{report.kind}: {len(report.added)} added, {len(report.dropped)} dropped, "
                f"{len(report.changed)} changed of {report.baseline_count}\n"
            )
    return _EXIT_GATE if breached else _EXIT_OK


def _run_stats(args: argparse.Namespace) -> int:
    """Count the committed corpus by kind, tier, category, source, and decile."""
    root = _root(args)
    entries = _all_entries(root)
    kinds: dict[str, int] = dict.fromkeys(KINDS, 0)
    tiers: dict[str, int] = {}
    categories: dict[str, int] = {}
    sources: dict[str, int] = {}
    deciles = [0] * _DECILES
    for entry in entries:
        kinds[entry.kind] = kinds.get(entry.kind, 0) + 1
        tiers[entry.trust_tier] = tiers.get(entry.trust_tier, 0) + 1
        categories[entry.category] = categories.get(entry.category, 0) + 1
        for ref in entry.sources:
            sources[ref.source_id] = sources.get(ref.source_id, 0) + 1
        deciles[min(int(entry.popularity * _DECILES), _DECILES - 1)] += 1

    histogram: list[JsonValue] = list(deciles)
    payload: JsonObject = {
        "entries": {key: kinds[key] for key in sorted(kinds)},
        "trust_tiers": {key: tiers[key] for key in sorted(tiers)},
        "categories": {key: categories[key] for key in sorted(categories)},
        "sources": {key: sources[key] for key in sorted(sources)},
        "popularity_deciles": histogram,
    }
    if args.json:
        _emit_json(payload)
    else:
        sys.stdout.write(f"entries: {', '.join(f'{k}={kinds[k]}' for k in sorted(kinds))}\n")
        sys.stdout.write(f"tiers: {', '.join(f'{k}={tiers[k]}' for k in sorted(tiers))}\n")
        sys.stdout.write(
            f"categories: {', '.join(f'{k}={categories[k]}' for k in sorted(categories))}\n"
        )
        sys.stdout.write(f"sources: {', '.join(f'{k}={sources[k]}' for k in sorted(sources))}\n")
        sys.stdout.write(f"popularity deciles: {deciles}\n")
    return _EXIT_OK


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def _verify_kind(root: Path, kind: str, *, violations: list[str], keys: dict[str, str]) -> int:
    """Check the shard files of one kind and return how many records they hold.

    A build writes all 256 files, so once any of them exists the rest must
    too, and one that has gone missing is reported. A kind with no files at
    all has simply never been built, which a fresh clone is entitled to be.
    """
    directory = _kind_directory(root, kind)
    present = {shard for shard in all_shards() if (directory / f"{shard}.jsonl").is_file()}
    total = 0
    for shard in all_shards():
        path = directory / f"{shard}.jsonl"
        name = _relative(root, path)
        if shard not in present:
            if present:
                violations.append(f"{name}: the shard file is missing")
            continue
        raw = path.read_bytes()
        if b"\r" in raw:
            violations.append(f"{name}: holds a carriage return")
        try:
            entries = [loads_line(line) for line in raw.decode("utf-8").splitlines()]
        except (CatalogError, UnicodeDecodeError) as exc:
            violations.append(f"{name}: {exc}")
            continue
        if render_shard(entries) != raw:
            violations.append(f"{name}: is not the canonical rendering of its own records")
        for entry in entries:
            total += 1
            if entry.kind != kind:
                violations.append(f"{name}: holds a {entry.kind} record")
            expected = shard_for(entry.canonical_key)
            if expected != shard:
                violations.append(f"{entry.canonical_key}: belongs in shard {expected}")
            if entry.canonical_key in keys:
                violations.append(
                    f"{entry.canonical_key}: also appears in {keys[entry.canonical_key]}"
                )
            for key in (entry.canonical_key, *entry.alias_keys):
                keys.setdefault(key, name)
    return total


def _verify_schema(root: Path) -> list[str]:
    """Whether the committed schema is still what ``render_schema`` produces."""
    path = root / SCHEMA_PATH
    if not path.is_file():
        return [f"{SCHEMA_PATH}: the generated schema is missing"]
    if path.read_text(encoding="utf-8") != render_schema():
        return [f"{SCHEMA_PATH}: is not what render_schema() produces; regenerate it"]
    return []


def _verify_curated(root: Path, keys: dict[str, str]) -> list[str]:
    """Whether every curated key resolves and every denial states a reason.

    Both files are always parsed, so a missing reason or a misspelled field
    name fails even on a tree that has never been built. Key resolution is
    only asked for once something has been built: on a fresh clone every
    curated key is unresolved, and that is a build that has not happened
    rather than a curated file that is wrong.
    """
    try:
        overrides = _load_all_curated(root)
        load_denylist(root / CURATED_DIRNAME / _DENYLIST_FILENAME)
    except CuratedError as exc:
        return [str(exc)]
    if not keys:
        return []
    return [
        f"{override.key}: curated but present in no shard"
        for override in overrides
        if override.key not in keys
    ]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _add_root(parser: argparse.ArgumentParser) -> None:
    """Add ``--root``, which every command takes."""
    parser.add_argument("--root", help="repository root (default: the working directory)")


def _add_json(parser: argparse.ArgumentParser) -> None:
    """Add ``--json``, which replaces all human output with one JSON object."""
    parser.add_argument(
        "--json", action="store_true", help="print one canonical JSON object and nothing else"
    )


def _add_thresholds(parser: argparse.ArgumentParser) -> None:
    """Add the three diff-gate knobs."""
    parser.add_argument("--max-drop-fraction", type=float, help="largest tolerable share dropped")
    parser.add_argument("--max-change-fraction", type=float, help="largest tolerable share changed")
    parser.add_argument(
        "--min-baseline-entries", type=int, help="baseline size below which the gate is skipped"
    )


def _root(args: argparse.Namespace) -> Path:
    """The repository root this command works on."""
    return Path(args.root) if args.root else Path.cwd()


def _thresholds(args: argparse.Namespace) -> DiffThresholds:
    """The diff-gate thresholds, taking each unset flag from the defaults."""
    return DiffThresholds(
        max_drop_fraction=_or_default(args.max_drop_fraction, DEFAULT_THRESHOLDS.max_drop_fraction),
        max_change_fraction=_or_default(
            args.max_change_fraction, DEFAULT_THRESHOLDS.max_change_fraction
        ),
        min_baseline_entries=_or_default(
            args.min_baseline_entries, DEFAULT_THRESHOLDS.min_baseline_entries
        ),
    )


def _limits(args: argparse.Namespace, root: Path) -> SourceLimits:
    """The crawl limits, taking each unset flag from the defaults.

    The GitHub token is read from the environment rather than from a flag, so
    it never reaches a process listing or a shell history, and it changes only
    what the crawl is allowed to ask for — never what the build writes.

    The marketplace allowlist is read from ``curated/skills.yaml`` for the
    same reason the denylist is: which repositories a person has reviewed is a
    reviewed fact in the repository, not something a caller talks the crawl
    out of with a flag.
    """
    policy = load_marketplace_policy(root)
    return SourceLimits(
        page_size=_or_default(args.page_size, DEFAULT_LIMITS.page_size),
        max_pages=_or_default(args.limit_pages, DEFAULT_LIMITS.max_pages),
        max_records=_or_default(args.limit_records, DEFAULT_LIMITS.max_records),
        detail_top_n=_or_default(args.detail_top_n, DEFAULT_LIMITS.detail_top_n),
        github_token=os.environ.get(str(args.github_token_env), ""),
        requests_per_minute=DEFAULT_LIMITS.requests_per_minute,
        marketplace_allowlist=policy.allow,
        require_marketplace_allowlist=policy.require_allowlist,
        marketplace_reviewed=policy.reviewed,
    )


def _instant(value: str | None) -> datetime:
    """The injected clock: an RFC3339 flag, or the real one when absent."""
    if not value:
        return datetime.now(UTC)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CatalogError(f"--now {value!r} is not an RFC3339 instant") from exc
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _selected_sources(requested: Sequence[str] | None) -> tuple[type[Source], ...]:
    """The sources to crawl, always in ``SOURCE_IDS`` order.

    Ordering the selection rather than honouring the order the flags were
    typed in keeps two runs of the same set identical, which matters because
    a crawl's page order is what ``sources.lock`` hashes.
    """
    names = sorted(
        set(requested) if requested else {source.source_id for source in ALL_SOURCES},
        key=lambda name: (SOURCE_RANK.get(name, len(SOURCE_RANK)), name),
    )
    return tuple(source_by_id(name) for name in names)


def _cached_fetches(directory: Path) -> tuple[SourceFetch, ...]:
    """Recorded ``SourceFetch`` documents, read in filename order."""
    if not directory.is_dir():
        raise CatalogError(f"{directory} is not a directory of recorded fetches")
    fetches: list[SourceFetch] = []
    for path in sorted(directory.glob("*.json")):
        try:
            fetches.append(SourceFetch.model_validate_json(path.read_bytes()))
        except ValueError as exc:
            raise CatalogError(f"{path.name} is not a recorded SourceFetch: {exc}") from exc
    if not fetches:
        raise CatalogError(f"{directory} holds no recorded fetches")
    fetches.sort(key=lambda fetch: fetch.source_id)
    return tuple(fetches)


def _all_entries(root: Path) -> list[CatalogEntry]:
    """Every committed entry of every kind, in canonical-key order."""
    entries: list[CatalogEntry] = []
    for kind in KINDS:
        entries.extend(load_shards(root, kind).values())
    entries.sort(key=lambda entry: entry.canonical_key)
    return entries


def _resolve_root(path: Path) -> Path:
    """A repository root, given either the root itself or its ``data`` directory."""
    if (path / "data").is_dir():
        return path
    if path.name == "data":
        return path.parent
    raise CatalogError(f"{path} is neither a repository root nor a data directory")


def _report_build(result: BuildResult, *, json_mode: bool, dry_run: bool) -> int:
    """Print what a build did, in one JSON object or in a few lines."""
    if json_mode:
        written: list[JsonValue] = list(result.written)
        _emit_json(
            {
                "dry_run": dry_run,
                "entry_counts": {kind: result.entry_counts[kind] for kind in KINDS},
                "written": written,
                "reports": [_report_json(report) for report in result.reports],
                "lock": _lock_json(result.lock),
            }
        )
    else:
        counts = ", ".join(f"{kind}={result.entry_counts[kind]}" for kind in KINDS)
        action = "would write" if dry_run else "wrote"
        sys.stdout.write(f"{counts}; {action} {len(result.written)} files\n")
        for report in result.reports:
            sys.stdout.write(
                f"{report.kind}: {len(report.added)} added, {len(report.dropped)} dropped, "
                f"{len(report.changed)} changed of {report.baseline_count}\n"
            )
    return _EXIT_OK


def _report_json(report: DiffReport) -> JsonValue:
    """One diff report as plain JSON, keys and all."""
    return {
        "kind": report.kind,
        "baseline_count": report.baseline_count,
        "candidate_count": report.candidate_count,
        "added": len(report.added),
        "dropped": len(report.dropped),
        "changed": len(report.changed),
        "drop_fraction": report.drop_fraction,
        "change_fraction": report.change_fraction,
    }


def _lock_json(lock: SourcesLock) -> JsonValue:
    """One lock file as plain JSON, source by source."""
    return {
        entry.source_id: {
            "entry_count": entry.entry_count,
            "fetched_at": entry.fetched_at,
            "page_count": entry.page_count,
            "sha256": entry.sha256,
        }
        for entry in lock.sources
    }


def _emit_json(payload: JsonObject) -> None:
    """The one canonical JSON encoding this repo emits, and nothing else."""
    sys.stdout.write(canonical_json(payload) + "\n")


def _fail(exc: BaseException, code: int) -> int:
    """Print a display-safe failure to standard error and return its code."""
    sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
    return code


def _or_default[T](value: T | None, fallback: T) -> T:
    """The flag's value when it was given, the default when it was not."""
    return fallback if value is None else value


def _relative(root: Path, path: Path) -> str:
    """A repository-relative POSIX path, absolute when the path is outside."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
