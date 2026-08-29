"""The Smithery crawl: seeded list paging plus a bounded detail pass.

Smithery indexes an order of magnitude more servers than the official
registry and almost none of them are name-verified, so this module treats
it as breadth: it walks ``/servers`` with a ``seed`` because the endpoint
otherwise reaches only the first 500 rows, drops unlisted and inactive
servers, and then details the busiest ``detail_top_n`` qualified names to
learn their transports and tool counts. Details run concurrently under a
small semaphore and degrade one at a time — a detail that 404s or fails to
parse leaves its summary record intact, without ``_detail``, rather than
losing the server. Only connection shape survives: ``configSchema`` is
pruned to its property names and each tool to its ``name``, so a schema
body never enters a record.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import ClassVar, Final
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict

from jhin_catalog.http import FetchError, fetch_json
from jhin_catalog.sources.base import (
    DEFAULT_LIMITS,
    RollingDigest,
    Source,
    SourceError,
    SourceLimits,
    TokenBucket,
)
from jhin_catalog.types import JsonObject, JsonValue, RawRecord, SourceFetch

SMITHERY_BASE: Final[str] = "https://registry.smithery.ai"
LIST_PATH: Final[str] = "/servers"
MAX_PAGE_SIZE: Final[int] = 100
SEED: Final[int] = 42
UNCAPPED_NOTE: Final[str] = "seed lifts the 500-result cap"

_SOURCE_ID: Final[str] = "smithery"
_LIST_URL: Final[str] = f"{SMITHERY_BASE}{LIST_PATH}"
_SERVER_PAGE: Final[str] = "https://smithery.ai/server/"
_REQUEST_HEADERS: Final[Mapping[str, str]] = {"accept": "application/json"}

# How many detail requests may be in flight at once. Smithery answers the
# detail route far more slowly than the list route, so the pass is worth
# overlapping, but a wide fan-out is how an unauthenticated client earns a
# 429 and turns a cheap enrichment into a fetch fault.
_DETAIL_CONCURRENCY: Final[int] = 8

# The detail route answers with a fixed object. ``parse_detail`` projects
# onto exactly these names, so an upstream addition cannot enlarge what the
# crawl carries without this tuple being changed first.
_DETAIL_KEYS: Final[tuple[str, ...]] = (
    "connections",
    "deploymentUrl",
    "description",
    "displayName",
    "iconUrl",
    "qualifiedName",
    "remote",
    "security",
    "tools",
    "useCount",
    "verified",
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)


class SmitheryPage(BaseModel):
    """One page of ``/servers``: its rows and its place in the walk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    servers: tuple[JsonObject, ...]
    current_page: int
    total_pages: int
    total_count: int


def parse_page(payload: JsonValue, *, url: str) -> SmitheryPage:
    """Raises ``SourceError`` on a missing ``servers`` or ``pagination`` block.

    A page without pagination cannot say whether the walk has finished or
    has been capped, and a row that is not an object cannot carry a
    qualified name, so both are wire faults rather than empty results.
    """
    if not isinstance(payload, dict):
        raise SourceError(f"smithery page {url} is not a JSON object", source_id=_SOURCE_ID)
    servers = payload.get("servers")
    if not isinstance(servers, list):
        raise SourceError(f"smithery page {url} has no ``servers`` list", source_id=_SOURCE_ID)
    pagination = payload.get("pagination")
    if not isinstance(pagination, dict):
        raise SourceError(f"smithery page {url} has no ``pagination`` block", source_id=_SOURCE_ID)

    rows: list[JsonObject] = []
    for index, row in enumerate(servers):
        if not isinstance(row, dict):
            raise SourceError(
                f"smithery page {url} row {index} is not an object", source_id=_SOURCE_ID
            )
        rows.append(row)

    return SmitheryPage(
        servers=tuple(rows),
        current_page=_as_int(pagination.get("currentPage"), default=1),
        total_pages=_as_int(pagination.get("totalPages"), default=1),
        total_count=_as_int(pagination.get("totalCount"), default=len(rows)),
    )


def parse_detail(payload: JsonValue, *, qualified_name: str) -> JsonObject:
    """Validate the fixed 11-key detail object. Raises ``SourceError``.

    The result always carries exactly the eleven names Smithery serves, with
    an absent one present as ``null``, so a consumer never has to ask whether
    a key exists. Two of them are reduced on the way through: every tool
    keeps its ``name`` and loses its input schema, and every connection's
    ``configSchema`` keeps its property names, each property's ``x-to``
    header, and the ``required`` list, and loses the rest. That is the whole
    of what ``normalize`` reads for ``auth_hint``, ``auth_note`` and
    ``tool_count``, and it means no schema body is ever stored.
    """
    if not isinstance(payload, dict):
        raise SourceError(
            f"smithery detail for {qualified_name!r} is not a JSON object", source_id=_SOURCE_ID
        )
    name = payload.get("qualifiedName")
    if not isinstance(name, str) or not name.strip():
        raise SourceError(
            f"smithery detail for {qualified_name!r} lacks ``qualifiedName``",
            source_id=_SOURCE_ID,
        )

    detail: JsonObject = {
        "connections": _prune_connections(payload.get("connections"), name=qualified_name),
        "deploymentUrl": _optional_str(payload.get("deploymentUrl")),
        "description": _optional_str(payload.get("description")),
        "displayName": _optional_str(payload.get("displayName")),
        "iconUrl": _optional_str(payload.get("iconUrl")),
        "qualifiedName": name.strip(),
        "remote": _optional_bool(payload.get("remote")),
        "security": _optional_object(payload.get("security")),
        "tools": _prune_tools(payload.get("tools"), name=qualified_name),
        "useCount": _optional_int(payload.get("useCount")),
        "verified": _optional_bool(payload.get("verified")),
    }
    return {key: detail[key] for key in _DETAIL_KEYS}


class SmitherySource(Source):
    """Smithery, walked with a seed and enriched by a bounded detail pass.

    ``detail_concurrency`` bounds the fan-out of the second pass; the pass
    itself is sized by ``SourceLimits.detail_top_n``, and the two are
    separate because how many servers deserve a detail is a catalog
    question while how many may be in flight is a politeness one.
    """

    source_id: ClassVar[str] = "smithery"

    def __init__(self, *, detail_concurrency: int = _DETAIL_CONCURRENCY) -> None:
        self.detail_concurrency = max(detail_concurrency, 1)

    async def fetch(
        self,
        client: httpx.AsyncClient,
        *,
        limits: SourceLimits = DEFAULT_LIMITS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> SourceFetch:
        """Crawl every seeded page, then detail the busiest servers.

        The list walk ends on the last page Smithery reports, on
        ``limits.max_pages``, or on ``limits.max_records``, and each early
        end is logged with the number of servers it left behind. The detail
        pass then runs over the top ``limits.detail_top_n`` qualified names
        by ``(-useCount, qualifiedName)`` and requests them in that fixed
        order, so the rolling hash never depends on which response landed
        first. ``page_count`` counts every response the hash covers, list
        pages and details alike.
        """
        bucket = (
            TokenBucket(rate_per_minute=limits.requests_per_minute, sleep=sleep)
            if limits.requests_per_minute > 0
            else None
        )
        page_size = min(max(limits.page_size, 1), MAX_PAGE_SIZE)
        digest = RollingDigest()
        summaries: dict[str, JsonObject] = {}
        dropped = 0
        page_count = 0
        page = 1
        total_pages = 1
        total_count = 0

        while page <= total_pages:
            if page > limits.max_pages:
                _LOG.warning(
                    "smithery: stopping after page %d of %d; about %d servers were not reached",
                    limits.max_pages,
                    total_pages,
                    max(total_count - len(summaries) - dropped, 0),
                )
                break
            if bucket is not None:
                await bucket.acquire()
            payload, result = await fetch_json(
                client,
                _LIST_URL,
                params={"page": page, "pageSize": page_size, "seed": SEED},
                headers=_REQUEST_HEADERS,
                sleep=sleep,
            )
            digest.update(result.body)
            page_count += 1
            parsed = parse_page(payload, url=result.url)

            # An empty page below the reported last page is the 500-result
            # cap answering with a 200. Read as data it is a mass deletion,
            # so it fails the fetch instead of reaching the diff gate.
            if not parsed.servers and page < parsed.total_pages:
                raise SourceError(
                    f"smithery page {page} of {parsed.total_pages} is empty; the result cap "
                    f"is back in force ({UNCAPPED_NOTE})",
                    source_id=_SOURCE_ID,
                )

            if page == 1:
                total_count = parsed.total_count
                _LOG.info(
                    "smithery: %d servers across %d pages (%s)",
                    parsed.total_count,
                    parsed.total_pages,
                    UNCAPPED_NOTE,
                )
            total_pages = parsed.total_pages
            for row in parsed.servers:
                if not _collect(row, summaries):
                    dropped += 1

            if len(summaries) >= limits.max_records:
                _LOG.warning(
                    "smithery: record limit %d reached at page %d of %d; about %d servers "
                    "were not reached",
                    limits.max_records,
                    page,
                    total_pages,
                    max(total_count - len(summaries) - dropped, 0),
                )
                break
            page += 1

        if dropped:
            _LOG.info("smithery: %d unlisted or inactive servers were dropped", dropped)

        names = list(summaries)[: limits.max_records]
        detail_names = sorted(names, key=lambda name: (-_use_count(summaries[name]), name))[
            : max(limits.detail_top_n, 0)
        ]
        details, detail_pages = await self._fetch_details(
            client, detail_names, bucket=bucket, sleep=sleep, digest=digest
        )
        page_count += detail_pages
        if len(details) < len(detail_names):
            _LOG.warning(
                "smithery: %d of %d details were unavailable; those servers land without "
                "transports or a tool count",
                len(detail_names) - len(details),
                len(detail_names),
            )

        records = tuple(
            RawRecord(
                source_id=_SOURCE_ID,
                upstream_id=name,
                url=f"{_SERVER_PAGE}{name}",
                payload=_with_detail(summaries[name], details.get(name)),
            )
            for name in names
        )
        return SourceFetch(
            source_id=_SOURCE_ID,
            url=_LIST_URL,
            sha256=digest.hexdigest(),
            entry_count=len(records),
            page_count=page_count,
            records=records,
        )

    async def _fetch_details(
        self,
        client: httpx.AsyncClient,
        names: Sequence[str],
        *,
        bucket: TokenBucket | None,
        sleep: Callable[[float], Awaitable[None]],
        digest: RollingDigest,
    ) -> tuple[dict[str, JsonObject], int]:
        """Detail every name, at most ``detail_concurrency`` in flight.

        One failure never costs more than one detail: a refused, oversized,
        missing or malformed response is logged and skipped, and its server
        still reaches the catalog from its summary alone.

        Bodies are folded into the caller's digest in ``names`` order rather
        than in completion order, which is what keeps ``sources.lock``
        reproducible across runs, and none is kept afterwards. The count of
        responses consumed comes back so the caller can bill them as pages.
        """
        if not names:
            return {}, 0
        semaphore = asyncio.Semaphore(self.detail_concurrency)

        async def one(name: str) -> tuple[JsonObject | None, bytes | None]:
            async with semaphore:
                if bucket is not None:
                    await bucket.acquire()
                url = f"{_LIST_URL}/{quote(name, safe='')}"
                try:
                    payload, result = await fetch_json(
                        client, url, headers=_REQUEST_HEADERS, sleep=sleep
                    )
                    return parse_detail(payload, qualified_name=name), result.body
                except (FetchError, SourceError) as exc:
                    _LOG.warning("smithery: detail for %r is unavailable (%s)", name, exc)
                    return None, None

        outcomes: list[tuple[JsonObject | None, bytes | None]] = list(
            await asyncio.gather(*(one(name) for name in names))
        )
        details: dict[str, JsonObject] = {}
        consumed = 0
        for name, (detail, body) in zip(names, outcomes, strict=True):
            if detail is not None:
                details[name] = detail
            if body is not None:
                digest.update(body)
                consumed += 1
        return details, consumed


def _collect(row: JsonObject, summaries: dict[str, JsonObject]) -> bool:
    """Keep one summary row, reporting whether it was kept.

    ``unlisted`` and ``inactive`` rows are refused here rather than carried
    and filtered later, because an inactive server has no endpoint to offer
    and a repeat of a qualified name is the seeded sort re-serving a row.
    """
    name = row.get("qualifiedName")
    if not isinstance(name, str) or not name.strip():
        _LOG.warning("smithery: a row without a ``qualifiedName`` was skipped")
        return False
    if row.get("unlisted") is True or row.get("inactive") is True:
        return False
    summaries.setdefault(name.strip(), row)
    return True


def _with_detail(summary: JsonObject, detail: JsonObject | None) -> JsonObject:
    """The summary, carrying its detail under ``_detail`` when one landed."""
    if detail is None:
        return summary
    merged = dict(summary)
    merged["_detail"] = detail
    return merged


def _use_count(row: JsonObject) -> int:
    """``useCount``, which orders the detail pass. Absent reads as zero."""
    return _as_int(row.get("useCount"), default=0)


def _prune_connections(value: JsonValue, *, name: str) -> JsonValue:
    """Every connection, reduced to type, endpoint, and config shape.

    Raises ``SourceError`` when ``connections`` is present but is not a
    list, because the transport a server offers is the one thing the detail
    pass exists to learn.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise SourceError(
            f"smithery detail for {name!r} has a non-list ``connections``", source_id=_SOURCE_ID
        )
    return [_prune_connection(item) for item in value if isinstance(item, dict)]


def _prune_connection(connection: JsonObject) -> JsonObject:
    """One connection: its type, its two URL spellings, its config shape."""
    return {
        "configSchema": _prune_config_schema(connection.get("configSchema")),
        "deploymentUrl": _optional_str(connection.get("deploymentUrl")),
        "type": _optional_str(connection.get("type")),
        "url": _optional_str(connection.get("url")),
    }


def _prune_config_schema(value: JsonValue) -> JsonValue:
    """A config schema reduced to property names and header hints.

    Property keys are emitted in sorted order so two crawls of the same
    server produce the same bytes, and every property keeps only an
    ``x-to.header``, which is the one field ``auth_hint`` reads.
    """
    if not isinstance(value, dict):
        return None
    raw = value.get("properties")
    properties: JsonObject = {}
    if isinstance(raw, dict):
        for key in sorted(raw):
            properties[key] = _prune_property(raw[key])
    required = value.get("required")
    names: list[JsonValue] = []
    if isinstance(required, list):
        names.extend(sorted(item for item in required if isinstance(item, str)))
    return {"properties": properties, "required": names}


def _prune_property(value: JsonValue) -> JsonObject:
    """One config property, keeping its ``x-to`` header and nothing else."""
    if not isinstance(value, dict):
        return {}
    x_to = value.get("x-to")
    if not isinstance(x_to, dict):
        return {}
    header = _optional_str(x_to.get("header"))
    if header is None:
        return {}
    return {"x-to": {"header": header}}


def _prune_tools(value: JsonValue, *, name: str) -> JsonValue:
    """Tool names only. ``null`` survives as ``null``, which is not ``[]``.

    The distinction matters: an absent list means the detail pass never
    learned the tools, while an empty one means the server exposes none, and
    only the second should become ``tool_count == 0``.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise SourceError(
            f"smithery detail for {name!r} has a non-list ``tools``", source_id=_SOURCE_ID
        )
    return [{"name": _optional_str(item.get("name"))} for item in value if isinstance(item, dict)]


def _optional_str(value: JsonValue) -> str | None:
    """``value`` when it is a string, else ``None``."""
    return value if isinstance(value, str) else None


def _optional_bool(value: JsonValue) -> bool | None:
    """``value`` when it is a boolean, else ``None``.

    Smithery serves ``remote: null`` for a server it has not deployed, which
    is not the same claim as ``false``; both reach ``normalize`` unchanged.
    """
    return value if isinstance(value, bool) else None


def _optional_int(value: JsonValue) -> int | None:
    """``value`` when it is a whole number that is not a boolean."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_object(value: JsonValue) -> JsonObject | None:
    """``value`` when it is an object, else ``None``."""
    return value if isinstance(value, dict) else None


def _as_int(value: JsonValue, *, default: int) -> int:
    """``value`` as a whole number, falling back to ``default``."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("-").isdigit():
            return int(text)
    return default
