"""jhin-catalog: an open index of MCP servers and agent skills.

Crawls the official MCP registry, Smithery, npm, GitHub topics, and Claude
Code plugin marketplaces; normalises every record into one canonical entry,
merges duplicates by identity key, and publishes sorted JSONL shards that a
Jhin deployment syncs into Postgres. The index stores pointers and metadata
only — never skill text, never server code. See ``README.md``.

Public surface: :class:`McpEntry` and :class:`SkillEntry` are the record
types, :func:`dumps_line` and :func:`shard_for` fix the on-disk form,
:func:`run_sync` drives a whole build, and :mod:`jhin_catalog.sources` holds
one :class:`Source` per upstream.
"""

from __future__ import annotations

from jhin_catalog.build import (
    DEFAULT_EXPORT_LIMIT,
    entries_from_fetches,
    export_catalog_json,
    is_publishable,
    load_curated,
    load_denylist,
    project_catalog_app,
    run_sync,
)
from jhin_catalog.dedupe import apply_curated, merge_candidates
from jhin_catalog.diffgate import DEFAULT_THRESHOLDS, DiffGateFailed, check, compare, load_shards
from jhin_catalog.http import (
    DEFAULT_USER_AGENT,
    FetchError,
    FetchResult,
    ResponseTooLarge,
    build_client,
    fetch,
    fetch_json,
)
from jhin_catalog.normalize import normalize, slugify, summarize
from jhin_catalog.score import popularity, rank_score
from jhin_catalog.types import (
    CATALOG_CATEGORIES,
    CATALOG_ICONS,
    SCHEMA_VERSION,
    SHARD_COUNT,
    TRUST_TIERS,
    CatalogEntry,
    CatalogError,
    DiffReport,
    DiffThresholds,
    McpEntry,
    PopularitySignals,
    SkillEntry,
    SourceRef,
    SourcesLock,
    all_shards,
    dumps_line,
    loads_line,
    shard_for,
)

__all__ = [
    "CATALOG_CATEGORIES",
    "CATALOG_ICONS",
    "DEFAULT_EXPORT_LIMIT",
    "DEFAULT_THRESHOLDS",
    "DEFAULT_USER_AGENT",
    "SCHEMA_VERSION",
    "SHARD_COUNT",
    "TRUST_TIERS",
    "CatalogEntry",
    "CatalogError",
    "DiffGateFailed",
    "DiffReport",
    "DiffThresholds",
    "FetchError",
    "FetchResult",
    "McpEntry",
    "PopularitySignals",
    "ResponseTooLarge",
    "SkillEntry",
    "SourceRef",
    "SourcesLock",
    "all_shards",
    "apply_curated",
    "build_client",
    "check",
    "compare",
    "dumps_line",
    "entries_from_fetches",
    "export_catalog_json",
    "fetch",
    "fetch_json",
    "is_publishable",
    "load_curated",
    "load_denylist",
    "load_shards",
    "loads_line",
    "merge_candidates",
    "normalize",
    "popularity",
    "project_catalog_app",
    "rank_score",
    "run_sync",
    "shard_for",
    "slugify",
    "summarize",
]
