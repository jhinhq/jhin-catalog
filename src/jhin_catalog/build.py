"""The whole pipeline: fetch, normalise, merge, score, curate, gate, write.

One build reads five upstreams, projects every record into a candidate,
merges the candidates that describe the same thing, scores what is left,
overlays the curated file and the denylist, and only then compares itself
against the committed data. Nothing reaches the disk until every gate has
passed, and what does reach it is byte-deterministic: the same records always
produce the same 512 shard files, and ``sources.lock`` is the single output
that carries a timestamp, injected rather than read from a clock.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx
import yaml
from pydantic import ValidationError

from jhin_catalog.dedupe import (
    apply_curated,
    elect_primary_key,
    merge_candidates,
    unresolved_denylist_keys,
)
from jhin_catalog.diffgate import (
    DEFAULT_THRESHOLDS,
    DiffGateFailed,
    _kind_directory,
    check,
    check_source_counts,
    compare,
    load_shards,
)
from jhin_catalog.normalize import build_entry, normalize, slugify, summarize
from jhin_catalog.score import popularity, rank_score
from jhin_catalog.sources.base import Source, SourceLimits
from jhin_catalog.types import (
    CATALOG_CATEGORIES,
    CATALOG_ICONS,
    ENTRY_ADAPTER,
    ENTRY_MODELS,
    KINDS,
    SERVER_SLUG_RE,
    TRUST_RANK,
    TRUST_TIERS,
    BuildResult,
    Candidate,
    CatalogEntry,
    CatalogError,
    CuratedError,
    CuratedOverride,
    DenylistItem,
    DiffReport,
    DiffThresholds,
    JsonObject,
    JsonValue,
    LockEntry,
    MarketplacePolicy,
    McpEntry,
    MergedCandidate,
    NormalizeError,
    PopularitySignals,
    SourceFetch,
    SourceRef,
    SourcesLock,
    all_shards,
    dumps_line,
    entry_sort_key,
    shard_for,
)

DATA_DIRNAME: Final[str] = "data"
CURATED_DIRNAME: Final[str] = "curated"
LOCK_FILENAME: Final[str] = "sources.lock"
SCHEMA_PATH: Final[str] = "schema/catalog.schema.json"
DEFAULT_EXPORT_LIMIT: Final[int] = 200

_MCP_CURATED_FILENAME: Final[str] = "mcp.yaml"
_SKILLS_CURATED_FILENAME: Final[str] = "skills.yaml"
_DENYLIST_FILENAME: Final[str] = "denylist.yaml"

_SCHEMA_ID: Final[str] = (
    "https://raw.githubusercontent.com/jhin-dev/jhin-catalog/main/schema/catalog.schema.json"
)
_SCHEMA_DIALECT: Final[str] = "https://json-schema.org/draft/2020-12/schema"
_SCHEMA_TITLE: Final[str] = "CatalogEntry"
_SCHEMA_DESCRIPTION: Final[str] = (
    "One line of a jhin-catalog JSONL shard: either an MCP server or an Agent Skill, "
    "discriminated by ``kind``. Records are written with sorted keys, no separator "
    "spaces, and no omitted keys — every field is materialised at its default, so "
    "``required`` here lists them all. Omit-when-default applies only to the exported "
    "``catalog.json`` projection, never to ``data/**``."
)

_CURATED_SOURCE_URL: Final[str] = "https://github.com/jhin-dev/jhin-catalog/tree/main/curated"
_MAX_APP_NAME_CHARS: Final[int] = 60
_MIN_DENY_REASON_CHARS: Final[int] = 8
_MAX_DENY_REASON_CHARS: Final[int] = 300
_SLUG_HASH_SHORT: Final[int] = 4
_SLUG_HASH_LONG: Final[int] = 8
_SLUG_MAX_CHARS: Final[int] = 32
_SERVER_PREFIXES: Final[tuple[str, ...]] = ("server-", "mcp-")
_SERVER_SUFFIXES: Final[tuple[str, ...]] = ("-mcp", "-server")
_UNKNOWN_TRANSPORT: Final[str] = "unknown"

# How many components may fail validation before the build calls it a code
# fault rather than a handful of malformed upstream rows. Small on purpose:
# the point is to survive a hostile publisher, not to normalise breakage.
_MAX_UNBUILDABLE: Final[int] = 25
_UNBUILDABLE_SAMPLE: Final[int] = 3


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe loader that refuses a mapping holding the same key twice.

    ``yaml.safe_load`` lets the later of two identical keys win in silence,
    which in a curated file means an override nobody can see is overruling one
    somebody wrote on purpose.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        """The mapping, after checking that no key was written twice."""
        seen: set[str] = set()
        for key_node, _ in node.value:
            key = str(key_node.value)
            if key in seen:
                raise CuratedError(f"duplicate key {key!r} at {node.start_mark}")
            seen.add(key)
        return super().construct_mapping(node, deep)


# ---------------------------------------------------------------------------
# Curated input
# ---------------------------------------------------------------------------


def load_curated(path: Path) -> tuple[CuratedOverride, ...]:
    """Parse ``curated/mcp.yaml`` or ``curated/skills.yaml``.

    The document is a mapping with an ``entries`` list, each item naming a
    ``key``, an optional ``kind`` and ``aliases``, and a ``fields`` mapping of
    partial entry values. A missing file yields ``()``, because a deployment
    is allowed to have curated nothing yet. Raises ``CuratedError`` on a
    non-mapping document, a duplicate ``key``, or a field name that is not on
    the model — a typo in a curated file must fail the build rather than be
    quietly dropped on the floor.
    """
    document = _load_yaml(path)
    if document is None:
        return ()
    if not isinstance(document, dict):
        raise CuratedError(f"{path.name} is not a mapping")
    raw = document.get("entries")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise CuratedError(f"{path.name} has an ``entries`` field that is not a list")

    default_kind = _kind_for_filename(path.stem)
    overrides: list[CuratedOverride] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise CuratedError(f"{path.name} entry {index} is not a mapping")
        key = _required_text(item.get("key"), f"{path.name} entry {index}", "key")
        if key in seen:
            raise CuratedError(f"curated key {key!r} appears twice in {path.name}")
        seen.add(key)
        kind = _override_kind(item.get("kind"), key=key, default=default_kind, source=path.name)
        fields = item.get("fields") or {}
        if not isinstance(fields, dict):
            raise CuratedError(f"curated entry {key!r} has a ``fields`` value that is not a map")
        unknown = sorted(set(map(str, fields)) - _settable_fields(kind))
        if unknown:
            raise CuratedError(
                f"curated entry {key!r} names unknown field(s): {', '.join(unknown)}"
            )
        aliases = _alias_list(item.get("aliases"), key=key)
        try:
            overrides.append(CuratedOverride(key=key, kind=kind, aliases=aliases, fields=fields))
        except ValidationError as exc:
            raise CuratedError(f"curated entry {key!r} is not loadable: {exc}") from exc
    return tuple(overrides)


def load_marketplace_policy(root: Path) -> MarketplacePolicy:
    """The ``marketplaces:`` block of ``curated/skills.yaml``.

    That block declares which repositories a person has reviewed and whether
    the crawl is confined to them. It was written, documented, and tested as
    text, and never read by any code: ``MarketplacesSource`` crawled every
    repository carrying the topic, which is the exact thing the file says it
    prevents. This is the function that makes the declaration load-bearing.

    A missing file, or one with no ``marketplaces`` block, leaves discovery
    open — the same behaviour a repository with no curated policy had before.
    """
    document = _load_yaml(root / CURATED_DIRNAME / _SKILLS_CURATED_FILENAME)
    if document is None:
        return MarketplacePolicy()
    if not isinstance(document, dict):
        raise CuratedError(f"{_SKILLS_CURATED_FILENAME} is not a mapping")
    block = document.get("marketplaces")
    if block is None:
        return MarketplacePolicy()
    if not isinstance(block, dict):
        raise CuratedError(f"{_SKILLS_CURATED_FILENAME} has a ``marketplaces`` that is not a map")

    allow: set[str] = set()
    for group in ("allow", "community"):
        listed = block.get(group)
        if listed is None:
            continue
        if not isinstance(listed, list):
            raise CuratedError(
                f"{_SKILLS_CURATED_FILENAME}: ``marketplaces.{group}`` is not a list"
            )
        for index, item in enumerate(listed):
            if not isinstance(item, dict):
                raise CuratedError(
                    f"{_SKILLS_CURATED_FILENAME}: ``marketplaces.{group}[{index}]`` is not a map"
                )
            where = f"{_SKILLS_CURATED_FILENAME} marketplaces.{group}[{index}]"
            allow.add(_required_text(item.get("repo"), where, "repo").lower())

    discovery = block.get("discovery")
    if discovery is not None and not isinstance(discovery, dict):
        raise CuratedError(f"{_SKILLS_CURATED_FILENAME}: ``marketplaces.discovery`` is not a map")
    required = bool((discovery or {}).get("require_allowlist", False))
    return MarketplacePolicy(allow=tuple(sorted(allow)), require_allowlist=required)


def load_denylist(path: Path) -> tuple[DenylistItem, ...]:
    """Parse ``curated/denylist.yaml``.

    Raises ``CuratedError`` when an entry omits ``reason`` or gives one
    shorter than eight characters. A denial with no stated reason is
    indistinguishable from a mistake six months later, and nobody will dare
    remove it.
    """
    document = _load_yaml(path)
    if document is None:
        return ()
    if not isinstance(document, dict):
        raise CuratedError(f"{path.name} is not a mapping")
    raw = document.get("entries")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise CuratedError(f"{path.name} has an ``entries`` field that is not a list")

    items: list[DenylistItem] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise CuratedError(f"{path.name} entry {index} is not a mapping")
        key = _required_text(item.get("key"), f"{path.name} entry {index}", "key")
        if key in seen:
            raise CuratedError(f"denylist key {key!r} appears twice in {path.name}")
        seen.add(key)
        reason = item.get("reason")
        if not isinstance(reason, str) or len(reason.strip()) < _MIN_DENY_REASON_CHARS:
            raise CuratedError(
                f"denylist entry {key!r} needs a reason of at least "
                f"{_MIN_DENY_REASON_CHARS} characters"
            )
        items.append(DenylistItem(key=key, reason=reason.strip()[:_MAX_DENY_REASON_CHARS]))
    return tuple(items)


# ---------------------------------------------------------------------------
# Slugs and shards
# ---------------------------------------------------------------------------


def assign_slugs(entries: Sequence[CatalogEntry]) -> tuple[CatalogEntry, ...]:
    """Deterministic global slug allocation. Raises ``NormalizeError``.

    The slug is the name Jhin's UI and its stored connections resolve by, so
    the strongest claimant keeps the plain one: entries are processed by
    descending trust tier and then by ascending ``canonical_key``. Without the
    tier, a third-party proxy that happens to sort first takes ``linear`` and
    the hand-checked Linear record — the one carrying the real endpoint and
    the ``oauth`` auth hint — is renamed out from under every consumer.

    Both keys are properties of the entries themselves, never of the order a
    crawl returned them in, so the allocation stays byte-deterministic. A
    loser is suffixed with four hex characters of the SHA-256 of its own key,
    widened to eight if those four also collide.
    """
    taken: dict[str, str] = {}
    assigned: list[CatalogEntry] = []
    for entry in sorted(entries, key=_slug_priority):
        slug = _allocate_slug(entry, taken)
        taken[slug] = entry.canonical_key
        assigned.append(entry if slug == entry.slug else entry.model_copy(update={"slug": slug}))
    assigned.sort(key=entry_sort_key)
    return tuple(assigned)


def _slug_priority(entry: CatalogEntry) -> tuple[int, str]:
    """Who gets first refusal on a contested slug: trust, then identity."""
    return (TRUST_RANK.get(entry.trust_tier, len(TRUST_TIERS)), entry.canonical_key)


def plan_shards(entries: Sequence[CatalogEntry]) -> dict[str, tuple[CatalogEntry, ...]]:
    """All 256 shard names → their sorted entries (possibly empty)."""
    buckets: dict[str, list[CatalogEntry]] = {shard: [] for shard in all_shards()}
    for entry in entries:
        buckets[shard_for(entry.canonical_key)].append(entry)
    return {shard: tuple(sorted(rows, key=entry_sort_key)) for shard, rows in buckets.items()}


def render_shard(entries: Sequence[CatalogEntry]) -> bytes:
    """The exact bytes of one shard file.

    Sorted by canonical key, one JSON object per line, ``\\n`` terminated, and
    nothing else — an empty shard renders as zero bytes rather than as a blank
    line, so a shard that emptied is visible in a diff as a deletion.
    """
    return b"".join(
        dumps_line(entry).encode("utf-8") for entry in sorted(entries, key=entry_sort_key)
    )


def write_shards(
    root: Path, kind: str, shards: Mapping[str, Sequence[CatalogEntry]]
) -> tuple[str, ...]:
    """Write all 256 files. Returns the repo-relative paths, sorted.

    Every shard is written on every build, including the empty ones, so the
    file set is a property of the schema rather than of the data.
    """
    directory = _kind_directory(root, kind)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for shard in all_shards():
        path = directory / f"{shard}.jsonl"
        path.write_bytes(render_shard(shards.get(shard, ())))
        written.append(_relative(root, path))
    return tuple(sorted(written))


# ---------------------------------------------------------------------------
# The lock file
# ---------------------------------------------------------------------------


def render_lock(fetches: Sequence[SourceFetch], *, now: datetime) -> SourcesLock:
    """One lock entry per source, sorted by ``source_id``.

    ``fetched_at`` is the injected clock at second precision, and it is the
    only timestamp any build output carries. A naive datetime is read as UTC
    rather than as local time, so a lock file does not depend on the machine
    that wrote it.
    """
    stamped = (now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    entries = [
        LockEntry(
            source_id=fetch.source_id,
            url=fetch.url,
            fetched_at=stamped,
            sha256=fetch.sha256,
            entry_count=fetch.entry_count,
            page_count=fetch.page_count,
        )
        for fetch in fetches
    ]
    entries.sort(key=lambda entry: entry.source_id)
    return SourcesLock(sources=tuple(entries))


def write_lock(root: Path, lock: SourcesLock) -> str:
    """Write ``sources.lock`` and return its repo-relative path."""
    body = json.dumps(lock.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False)
    path = root / LOCK_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((body + "\n").encode("utf-8"))
    return _relative(root, path)


def read_lock(root: Path) -> SourcesLock:
    """The committed lock, or an empty one when there is no file yet."""
    path = root / LOCK_FILENAME
    if not path.is_file():
        return SourcesLock(sources=())
    try:
        return SourcesLock.model_validate_json(path.read_bytes())
    except (ValidationError, ValueError) as exc:
        raise CatalogError(f"{LOCK_FILENAME} is not a readable lock file: {exc}") from exc


# ---------------------------------------------------------------------------
# Projection into Jhin's catalog
# ---------------------------------------------------------------------------


def is_publishable(entry: CatalogEntry) -> bool:
    """Whether an entry is strong enough to reach Jhin's Apps library.

    Publication is a recommendation, not an index, so an entry only a topic
    search has heard of does not qualify however popular it is, and neither
    does one with nothing a person could actually connect to. Skills are never
    projected: the library lists servers.

    A curated record is exempt from the last test. Half of ``curated/mcp.yaml``
    describes a server a person self-hosts, so it carries a ``setup_note`` and
    ``url_unverified: true`` instead of an endpoint — deliberately, and with a
    human's name on the judgement. Applying the "nothing to connect to" rule to
    those would silently drop fourteen of the fifty shipped connectors from the
    library the file exists to populate.
    """
    if not isinstance(entry, McpEntry):
        return False
    if entry.trust_tier == "indexed" or entry.deprecated or not entry.description:
        return False
    if entry.trust_tier != "curated" and (
        entry.mcp_url is None and entry.connector_type is None and not entry.stdio_only
    ):
        return False
    return entry.icon in CATALOG_ICONS and entry.category in CATALOG_CATEGORIES


def project_catalog_app(entry: McpEntry) -> JsonObject:
    """One entry as a Jhin ``CatalogApp``: fifteen keys, omit-when-default.

    A value equal to the ``CatalogApp`` default is left out, which is the
    convention the shipped ``catalog.json`` already follows. ``auth_hint`` is
    the exception and is always written, because "this server wants a bearer
    token" and "nobody has said" are different claims and the reader of the
    file cannot tell them apart from an absence.
    """
    app: JsonObject = {
        "slug": entry.slug,
        "name": entry.name[:_MAX_APP_NAME_CHARS],
        "category": entry.category,
        "icon": entry.icon,
        "description": summarize(entry.description),
    }
    if entry.connector_type is not None:
        app["connector_type"] = entry.connector_type
    if entry.mcp_url is not None:
        app["mcp_url"] = entry.mcp_url
    if entry.url_unverified:
        app["url_unverified"] = True
    if entry.transport != _UNKNOWN_TRANSPORT:
        app["transport"] = entry.transport
    app["auth_hint"] = entry.auth_hint
    if entry.auth_note:
        app["auth_note"] = entry.auth_note
    if entry.docs_url:
        app["docs_url"] = entry.docs_url
    if entry.setup_note:
        app["setup_note"] = entry.setup_note
    if entry.stdio_only:
        app["stdio_only"] = True
    if entry.connector_config:
        config: JsonObject = {
            key: entry.connector_config[key] for key in sorted(entry.connector_config)
        }
        app["connector_config"] = config
    return app


def export_catalog_json(
    entries: Sequence[CatalogEntry], *, limit: int = DEFAULT_EXPORT_LIMIT
) -> str:
    """The Jhin ``catalog.json`` body: a JSON array, ``indent=2``, trailing ``\\n``.

    Entries are filtered by ``is_publishable``, ordered by ``rank_score``, and
    truncated to ``limit``, so the file is the strongest few hundred records
    rather than the whole index. Slugs are unique because ``assign_slugs``
    already made them so.
    """
    publishable = [
        entry for entry in entries if isinstance(entry, McpEntry) and is_publishable(entry)
    ]
    publishable.sort(key=rank_score)
    projected = [project_catalog_app(entry) for entry in publishable[: max(limit, 0)]]
    return json.dumps(projected, indent=2, ensure_ascii=False) + "\n"


def render_schema() -> str:
    """The ``catalog.schema.json`` body, ``indent=2``, ``sort_keys=True``, trailing ``\\n``.

    Generated from the models in serialisation mode, which is why every field
    is ``required``: a shard line materialises every default rather than
    omitting it, so a consumer never has to guess what an absence meant.
    """
    schema: dict[str, Any] = dict(ENTRY_ADAPTER.json_schema(mode="serialization"))
    schema["$schema"] = _SCHEMA_DIALECT
    schema["$id"] = _SCHEMA_ID
    schema["title"] = _SCHEMA_TITLE
    schema["description"] = _SCHEMA_DESCRIPTION
    _materialise_defaults(schema)
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _materialise_defaults(schema: dict[str, Any]) -> None:
    """Mark every field required and drop the generated per-field titles.

    A shard line writes every field, defaults included, so ``required`` here
    genuinely lists them all — which the generator does not assume, because a
    field with a default is optional to a validator even when it is mandatory
    to a serialiser. The per-property titles pydantic derives from the field
    names ("Canonical Key") say nothing the key does not, and they triple the
    size of a file people read.
    """
    for definition in schema.get("$defs", {}).values():
        properties = definition.get("properties")
        if not isinstance(properties, dict):
            continue
        for field_schema in properties.values():
            if isinstance(field_schema, dict):
                field_schema.pop("title", None)
        definition["required"] = sorted(properties)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


async def fetch_all(
    *,
    sources: Sequence[type[Source]],
    limits: SourceLimits,
    client: httpx.AsyncClient,
) -> tuple[SourceFetch, ...]:
    """Run each source in the given order. Raises ``SourceError``.

    Sequentially, not concurrently: every upstream here rate-limits by client
    address, and three crawls in flight is how one build earns the throttling
    that truncates the next.
    """
    fetches: list[SourceFetch] = []
    for source in sources:
        fetches.append(await source().fetch(client, limits=limits))
    return tuple(fetches)


def entries_from_fetches(
    fetches: Sequence[SourceFetch],
    *,
    overrides: Sequence[CuratedOverride],
    denylist: Sequence[DenylistItem],
) -> tuple[CatalogEntry, ...]:
    """The whole pure pipeline: normalize → merge → score → curate → slug."""
    built = merged_entries(fetches, overrides=overrides)
    return assign_slugs(apply_curated(built, overrides=overrides, denylist=denylist))


def merged_entries(
    fetches: Sequence[SourceFetch], *, overrides: Sequence[CuratedOverride]
) -> tuple[CatalogEntry, ...]:
    """Everything before curation: normalize → merge → score → validate.

    Split from ``entries_from_fetches`` so a caller can ask what the denylist
    resolved against, which is only answerable before the denial removed it.

    A curated ``aliases`` list is injected as a synthetic candidate before the
    merge, which is how a human can force together two components no crawl
    connected. The injection is skipped when none of the alias keys matches
    anything real, because an override that matches nothing becomes an entry
    of its own in ``apply_curated`` rather than an empty component here.

    A component that will not validate is dropped rather than fatal, up to
    ``_MAX_UNBUILDABLE``. Upstreams serve rows their own schemas forbid, and
    anyone may publish one; letting a single such row abort the nightly hands
    a stranger an indefinite denial of service. Past the bound the fault is no
    longer one bad row and the build stops, which is what the exit code means.
    """
    candidates: list[Candidate] = []
    for fetch in fetches:
        for record in fetch.records:
            candidate = normalize(record)
            if candidate is not None:
                candidates.append(candidate)

    known = {key for candidate in candidates for key in candidate.alias_keys}
    for override in sorted(overrides, key=lambda item: item.key):
        keys = {override.key, *override.aliases}
        if len(keys) > 1 and not keys.isdisjoint(known):
            candidates.append(_alias_candidate(override))

    curated_keys = frozenset(
        key for override in overrides for key in (override.key, *override.aliases)
    )
    merged = merge_candidates(candidates, curated_keys=curated_keys)
    built: list[CatalogEntry] = []
    unbuildable: list[str] = []
    for item in merged:
        try:
            built.append(
                build_entry(item, popularity=popularity(item.signals), slug=_provisional_slug(item))
            )
        except NormalizeError as exc:
            unbuildable.append(f"{item.canonical_key}: {exc}")
            if len(unbuildable) > _MAX_UNBUILDABLE:
                joined = "; ".join(unbuildable[:_UNBUILDABLE_SAMPLE])
                raise NormalizeError(
                    f"{len(unbuildable)} components would not validate, over the "
                    f"{_MAX_UNBUILDABLE} this build tolerates; for example {joined}"
                ) from exc
    return tuple(built)


async def run_sync(
    root: Path,
    *,
    sources: Sequence[type[Source]],
    limits: SourceLimits,
    thresholds: DiffThresholds,
    now: datetime,
    client: httpx.AsyncClient,
    allow_breaking: bool = False,
    dry_run: bool = False,
) -> BuildResult:
    """Fetch, build, gate, and (unless ``dry_run``) write.

    Raises ``DiffGateFailed`` before any file is written, ``SourceError`` on a
    fetch fault, and ``CuratedError`` on bad curated input.
    """
    fetches = await fetch_all(sources=sources, limits=limits, client=client)
    return _finish_build(
        root,
        fetches,
        thresholds=thresholds,
        now=now,
        allow_breaking=allow_breaking,
        dry_run=dry_run,
    )


def _finish_build(
    root: Path,
    fetches: Sequence[SourceFetch],
    *,
    thresholds: DiffThresholds = DEFAULT_THRESHOLDS,
    now: datetime,
    allow_breaking: bool = False,
    dry_run: bool = False,
) -> BuildResult:
    """Everything after the fetch: build, gate, and write, in that order.

    Split out from ``run_sync`` so a build from recorded fetches takes exactly
    the same path as a build from the network — a re-run that skipped the gate
    would defeat the point of having one. Nothing is written until every gate
    for every kind has passed.
    """
    overrides = _load_all_curated(root)
    denylist = load_denylist(root / CURATED_DIRNAME / _DENYLIST_FILENAME)
    built = merged_entries(fetches, overrides=overrides)
    stale_denylist_keys = unresolved_denylist_keys(built, denylist)
    entries = assign_slugs(apply_curated(built, overrides=overrides, denylist=denylist))

    by_kind: dict[str, dict[str, CatalogEntry]] = {
        kind: {entry.canonical_key: entry for entry in entries if entry.kind == kind}
        for kind in KINDS
    }
    reports: list[DiffReport] = []
    for kind in KINDS:
        report = compare(load_shards(root, kind), by_kind[kind], kind=kind)
        reports.append(report)
        if not allow_breaking:
            check(report, thresholds)

    lock = render_lock(fetches, now=now)
    collapsed = check_source_counts(read_lock(root), lock)
    if collapsed and not allow_breaking:
        raise DiffGateFailed(
            f"{', '.join(collapsed)} returned no records after previously returning some; "
            "that is a fetch fault, not a deletion",
            report=_collapse_report(collapsed, lock),
        )

    written: tuple[str, ...] = ()
    if not dry_run:
        paths: list[str] = []
        for kind in KINDS:
            paths.extend(write_shards(root, kind, plan_shards(list(by_kind[kind].values()))))
        paths.append(write_lock(root, lock))
        written = tuple(sorted(paths))

    return BuildResult(
        entry_counts={kind: len(by_kind[kind]) for kind in KINDS},
        written=written,
        reports=tuple(reports),
        lock=lock,
        stale_denylist_keys=stale_denylist_keys,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_all_curated(root: Path) -> tuple[CuratedOverride, ...]:
    """Both curated files, with a key claimed by both treated as a conflict."""
    directory = root / CURATED_DIRNAME
    overrides = (
        *load_curated(directory / _MCP_CURATED_FILENAME),
        *load_curated(directory / _SKILLS_CURATED_FILENAME),
    )
    seen: set[str] = set()
    for override in overrides:
        if override.key in seen:
            raise CuratedError(f"curated key {override.key!r} appears in both curated files")
        seen.add(override.key)
    return overrides


def _alias_candidate(override: CuratedOverride) -> Candidate:
    """A key-only candidate that forces an override's aliases into one component."""
    keys = tuple(sorted({override.key, *override.aliases}))
    return Candidate(
        kind=override.kind,
        source_id="curated",
        upstream_id=override.key[:200],
        primary_key=elect_primary_key(keys),
        alias_keys=keys,
        repo=None,
        signals=PopularitySignals(),
        trust_hint="curated",
        source_ref=SourceRef(
            source_id="curated", upstream_id=override.key[:200], url=_CURATED_SOURCE_URL
        ),
        fields={},
    )


def _provisional_slug(merged: MergedCandidate) -> str:
    """The slug a merged record asks for, before collisions are resolved.

    The order runs from the most deliberate name to the least: a registry
    namespace was chosen by the publisher, a package name by the author, a
    repository name by whoever created it, and a display name by nobody in
    particular. ``assign_slugs`` settles ties afterwards.
    """
    fields = merged.fields
    options: list[str] = []
    registry_name = _text(fields.get("registry_name"))
    if registry_name:
        options.append(_strip_server_affixes(registry_name.rsplit("/", 1)[-1]))
    qualified = _text(fields.get("smithery_qualified_name"))
    if qualified:
        options.append(qualified.rsplit("/", 1)[-1])
    npm_package = _text(fields.get("npm_package"))
    if npm_package:
        options.append(
            npm_package.split("/", 1)[-1] if npm_package.startswith("@") else npm_package
        )
    options.append(_text(fields.get("skill_name")))
    repo = fields.get("repo")
    if isinstance(repo, dict):
        options.append(_text(repo.get("repo")))
    options.append(_text(fields.get("name")))
    options.append(merged.canonical_key.rsplit(":", 1)[-1].rsplit("/", 1)[-1])

    for option in options:
        if not option:
            continue
        try:
            return slugify(option)
        except NormalizeError:
            continue
    raise NormalizeError(f"no slug survives {merged.canonical_key!r}")


def _strip_server_affixes(value: str) -> str:
    """``server-filesystem`` → ``filesystem``, ``tavily-mcp`` → ``tavily``."""
    text = value
    for prefix in _SERVER_PREFIXES:
        if text.startswith(prefix) and len(text) > len(prefix):
            text = text[len(prefix) :]
            break
    for suffix in _SERVER_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)]
            break
    return text or value


def _allocate_slug(entry: CatalogEntry, taken: Mapping[str, str]) -> str:
    """This entry's slug, suffixed with a hash of its key if one is needed."""
    if entry.slug not in taken:
        return entry.slug
    digest = hashlib.sha256(entry.canonical_key.encode("utf-8")).hexdigest()
    short = f"{entry.slug[: _SLUG_MAX_CHARS - _SLUG_HASH_SHORT - 1]}_{digest[:_SLUG_HASH_SHORT]}"
    if short not in taken and SERVER_SLUG_RE.fullmatch(short):
        return short
    long = f"{entry.slug[: _SLUG_MAX_CHARS - _SLUG_HASH_LONG - 1]}_{digest[:_SLUG_HASH_LONG]}"
    if long not in taken and SERVER_SLUG_RE.fullmatch(long):
        return long
    raise NormalizeError(
        f"slug {entry.slug!r} collides for {entry.canonical_key!r} even after hashing"
    )


def _collapse_report(collapsed: Sequence[str], lock: SourcesLock) -> DiffReport:
    """The report attached to a source that stopped returning records."""
    return DiffReport(
        kind="sources",
        baseline_count=len(lock.sources),
        candidate_count=len(lock.sources) - len(collapsed),
        added=(),
        dropped=tuple(collapsed),
        changed=(),
        drop_fraction=1.0,
        change_fraction=0.0,
    )


def _load_yaml(path: Path) -> object:
    """One YAML document, or ``None`` when the file is not there."""
    if not path.is_file():
        return None
    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise CuratedError(f"{path.name} is not readable YAML: {exc}") from exc


def _kind_for_filename(stem: str) -> str:
    """The kind a curated filename implies, or ``""`` when it implies none."""
    if stem == "mcp":
        return "mcp"
    if stem in {"skill", "skills"}:
        return "skill"
    return ""


def _override_kind(value: object, *, key: str, default: str, source: str) -> str:
    """The kind an override declares, or the one its key or its file implies."""
    if isinstance(value, str) and value.strip() in KINDS:
        return value.strip()
    prefix = key.split(":", 1)[0]
    if prefix in KINDS:
        return prefix
    if default in KINDS:
        return default
    raise CuratedError(f"curated entry {key!r} in {source} does not say what kind it is")


def _alias_list(value: object, *, key: str) -> tuple[str, ...]:
    """An override's aliases, trimmed, deduped, and never repeating its key."""
    if value is None:
        return ()
    if not isinstance(value, list):
        raise CuratedError(f"curated entry {key!r} has an ``aliases`` value that is not a list")
    aliases = {item.strip() for item in value if isinstance(item, str) and item.strip()}
    return tuple(sorted(aliases - {key}))


def _settable_fields(kind: str) -> frozenset[str]:
    """Every field name a curated override of this kind may name."""
    return frozenset(ENTRY_MODELS[kind].model_fields)


def _required_text(value: object, where: str, name: str) -> str:
    """One non-blank string, or a ``CuratedError`` naming what was missing."""
    if not isinstance(value, str) or not value.strip():
        raise CuratedError(f"{where} has no ``{name}``")
    return value.strip()


def _text(value: JsonValue) -> str:
    """A trimmed string, or ``""`` when the value is not a string."""
    return value.strip() if isinstance(value, str) else ""


def _relative(root: Path, path: Path) -> str:
    """A repository-relative POSIX path, absolute when the path is outside."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
