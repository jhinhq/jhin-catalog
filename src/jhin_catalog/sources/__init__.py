"""Every upstream the catalog crawls, reachable by its ``source_id``.

Each submodule owns one index -- the official MCP registry, Smithery, npm
search, GitHub topic search, and Claude Code plugin marketplaces -- and
exposes exactly one :class:`Source` subclass.  ``ALL_SOURCES`` fixes the
order a full build runs them in; ``source_by_id`` is how the CLI resolves
a ``--source`` flag.  A source collects, it never interprets: turning a
payload into an entry is ``normalize``'s job, not a crawler's.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from jhin_catalog.sources.base import (
    DEFAULT_LIMITS,
    RollingDigest,
    Source,
    SourceError,
    SourceLimits,
    TokenBucket,
    rolling_sha256,
)
from jhin_catalog.sources.github_topics import GitHubTopicsSource
from jhin_catalog.sources.marketplaces import MarketplacesSource
from jhin_catalog.sources.npm import NpmSource
from jhin_catalog.sources.registry import RegistrySource
from jhin_catalog.sources.smithery import SmitherySource

ALL_SOURCES: Final[tuple[type[Source], ...]] = (
    GitHubTopicsSource,
    MarketplacesSource,
    NpmSource,
    RegistrySource,
    SmitherySource,
)

_BY_ID: Final[Mapping[str, type[Source]]] = {source.source_id: source for source in ALL_SOURCES}


def source_by_id(source_id: str) -> type[Source]:
    """Look up a source class by its ``source_id``. Raises ``KeyError``."""
    return _BY_ID[source_id]


__all__ = [
    "ALL_SOURCES",
    "DEFAULT_LIMITS",
    "GitHubTopicsSource",
    "MarketplacesSource",
    "NpmSource",
    "RegistrySource",
    "RollingDigest",
    "SmitherySource",
    "Source",
    "SourceError",
    "SourceLimits",
    "TokenBucket",
    "rolling_sha256",
    "source_by_id",
]
