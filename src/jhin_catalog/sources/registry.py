"""The official MCP registry crawl: cursor paging over latest versions.

The registry is the strongest of the five upstreams, because a namespace is
proved by DNS or by GitHub before anyone may publish under it, so its rows
carry the identity every other source is merged onto. This module walks
``/v0.1/servers`` by opaque cursor, keeps one ``RawRecord`` per latest
server version, and hands the wire object downstream untouched: mapping a
record into an ``McpEntry`` belongs to ``normalize``, and electing
``registry_verified`` belongs to ``dedupe``. The crawl reads nothing but the
page in front of it and the cursor that page names, so the same pages always
produce the same records in the same order.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import ClassVar, Final
from urllib.parse import quote

import httpx

from jhin_catalog.http import FetchError, FetchResult, backoff_delay, fetch_json
from jhin_catalog.sources.base import (
    DEFAULT_LIMITS,
    RollingDigest,
    Source,
    SourceError,
    SourceLimits,
    TokenBucket,
)
from jhin_catalog.types import JsonObject, JsonValue, RawRecord, SourceFetch

REGISTRY_BASE: Final[str] = "https://registry.modelcontextprotocol.io"
SERVERS_PATH: Final[str] = "/v0.1/servers"
MAX_PAGE_SIZE: Final[int] = 100
_SOURCE_ID: Final[str] = "registry"
_SERVERS_URL: Final[str] = f"{REGISTRY_BASE}{SERVERS_PATH}"
_OFFICIAL_META_KEY: Final[str] = "io.modelcontextprotocol.registry/official"
_DELETED_STATUS: Final[str] = "deleted"
_REQUEST_HEADERS: Final[Mapping[str, str]] = {"accept": "application/json"}

# A cursor the registry no longer honours is answered with a client error
# rather than an empty page. It ends the walk; it does not fail the build.
_STALE_CURSOR_STATUSES: Final[frozenset[int]] = frozenset({400, 410, 422})

# ``http.fetch`` already retries a 429 and honours ``Retry-After``. These are
# the extra rounds this source spends before it calls a throttled registry a
# fetch fault, so a long window of rate limiting cannot truncate the crawl
# into what the diff gate would read as a mass deletion.
_RATE_LIMIT_STATUS: Final[int] = 429
_RATE_LIMIT_ATTEMPTS: Final[int] = 3


def official_meta(item: JsonObject) -> JsonObject:
    """``_meta["io.modelcontextprotocol.registry/official"]`` or ``{}``.

    The block is read from the list item first and from the nested ``server``
    object second, which covers both shapes the registry has served.
    """
    for holder in (item, item.get("server")):
        if not isinstance(holder, dict):
            continue
        meta = holder.get("_meta")
        if not isinstance(meta, dict):
            continue
        official = meta.get(_OFFICIAL_META_KEY)
        if isinstance(official, dict):
            return official
    return {}


def parse_page(payload: JsonValue, *, url: str) -> tuple[tuple[RawRecord, ...], str | None]:
    """Records plus ``metadata.nextCursor`` (``None`` on the last page).

    Raises ``SourceError`` when ``payload`` is not an object, ``servers`` is
    missing or not a list, or an item lacks ``server.name``.

    A row whose official status is ``deleted``, and a row whose ``isLatest``
    is ``false``, are dropped here rather than carried and filtered later.
    ``metadata.count`` is never read: it is the size of the page, not a total,
    and treating it as a total would end the walk early. An explicit
    ``"servers": null`` is an empty page, which the wire allows and this
    reads as no records rather than as a fault.
    """
    return _parse_page(payload, url=url, include_deleted=False)


def _parse_page(
    payload: JsonValue, *, url: str, include_deleted: bool
) -> tuple[tuple[RawRecord, ...], str | None]:
    """``parse_page``, with the tombstone filter under caller control."""
    if not isinstance(payload, dict):
        raise SourceError(f"registry page {url} is not a JSON object", source_id=_SOURCE_ID)
    if "servers" not in payload:
        raise SourceError(
            f"registry page {url} has no ``servers`` list",
            source_id=_SOURCE_ID,
        )
    # The wire types ``servers`` as ``array | null``, and a null is how an
    # exhausted filter answers. It is an empty page, not a fetch fault.
    servers = payload["servers"] if payload["servers"] is not None else []
    if not isinstance(servers, list):
        raise SourceError(
            f"registry page {url} has a ``servers`` field that is not a list",
            source_id=_SOURCE_ID,
        )

    records: list[RawRecord] = []
    for index, item in enumerate(servers):
        if not isinstance(item, dict):
            raise SourceError(
                f"registry page {url} item {index} is not an object",
                source_id=_SOURCE_ID,
            )
        server = _server_object(item, url=url, index=index)
        name = _server_name(server, url=url, index=index)
        if not _is_latest(item, server):
            continue
        if _status(item, server) == _DELETED_STATUS and not include_deleted:
            continue
        records.append(
            RawRecord(
                source_id=_SOURCE_ID,
                upstream_id=name,
                url=_versions_url(name),
                payload=item,
            )
        )
    return tuple(records), _next_cursor(payload)


def _server_object(item: JsonObject, *, url: str, index: int) -> JsonObject:
    """The ``server`` block, tolerating the older un-wrapped list item."""
    server = item.get("server")
    if isinstance(server, dict):
        return server
    if isinstance(item.get("name"), str):
        return item
    raise SourceError(
        f"registry page {url} item {index} lacks ``server.name``",
        source_id=_SOURCE_ID,
    )


def _server_name(server: JsonObject, *, url: str, index: int) -> str:
    """The reverse-DNS server name, which is the registry's primary id."""
    name = server.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SourceError(
            f"registry page {url} item {index} lacks ``server.name``",
            source_id=_SOURCE_ID,
        )
    return name.strip()


def _is_latest(item: JsonObject, server: JsonObject) -> bool:
    """Whether this row is the latest published version of its server.

    The crawl asks for ``version=latest``, so a superseded row is a wire
    anomaly; it is dropped whatever ``include_deleted`` says, because it
    would otherwise merge over the current row's fields.

    Either holder saying ``false`` is enough, which is what ``normalize``
    already does with the same two flags. Reading only the first holder made
    the two disagree whenever ``_meta.isLatest`` was true and the server's own
    was false: the crawl kept the row and the normaliser then dropped it.
    """
    return not any(holder.get("isLatest") is False for holder in (official_meta(item), server))


def _status(item: JsonObject, server: JsonObject) -> str:
    """The lifecycle status, lowercased, or ``""`` when the row omits one."""
    for holder in (official_meta(item), server):
        status = holder.get("status")
        if isinstance(status, str):
            return status.strip().lower()
    return ""


def _next_cursor(payload: JsonObject) -> str | None:
    """``metadata.nextCursor`` when it is a non-empty string, else ``None``.

    The cursor is opaque. Anything that is not a usable string, including a
    ``null`` and an empty string, is the end of the walk.
    """
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    cursor = metadata.get("nextCursor")
    if not isinstance(cursor, str) or not cursor:
        return None
    return cursor


def _versions_url(name: str) -> str:
    """The canonical human page for one server: its version history."""
    return f"{_SERVERS_URL}/{quote(name, safe='')}/versions"


def _rfc3339(value: datetime) -> str:
    """``value`` as a second-precision RFC 3339 instant in UTC.

    A naive datetime is read as UTC rather than as local time, so the crawl
    window never depends on the machine that runs it.
    """
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class RegistrySource(Source):
    """The official MCP registry, walked latest-version-first by cursor.

    A default instance crawls the whole registry. ``updated_since`` narrows
    the walk to servers touched after an instant, which is how a refresh run
    stays cheap, and ``include_deleted`` asks the registry for its tombstones
    and keeps them in the fetch so a caller can reconcile removals; neither
    changes what ``normalize`` is willing to index.

    The two interact on the wire: the registry forces ``include_deleted`` on
    whenever ``updated_since`` is given, because a window that hid deletions
    could never report one. An incremental crawl therefore always receives
    tombstones, and it is the filter here, not the query, that keeps them out
    of a default fetch.
    """

    source_id: ClassVar[str] = _SOURCE_ID

    def __init__(
        self, *, updated_since: datetime | None = None, include_deleted: bool = False
    ) -> None:
        self.updated_since = updated_since
        self.include_deleted = include_deleted

    async def fetch(
        self,
        client: httpx.AsyncClient,
        *,
        limits: SourceLimits = DEFAULT_LIMITS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> SourceFetch:
        """Crawl ``/v0.1/servers`` to exhaustion (within ``limits``).

        The walk ends on an absent ``nextCursor``, on ``limits.max_pages``, on
        ``limits.max_records``, on a cursor the registry has forgotten, or on
        a cursor it repeats. An empty page mid-walk is legitimate under a
        filter and is followed, unlike Smithery's empty page, which is a bug.
        """
        bucket = (
            TokenBucket(rate_per_minute=limits.requests_per_minute, sleep=sleep)
            if limits.requests_per_minute > 0
            else None
        )
        base_params = self._base_params(limits)
        digest = RollingDigest()
        records: list[RawRecord] = []
        seen_ids: set[str] = set()
        seen_cursors: set[str] = set()
        cursor: str | None = None
        page_count = 0

        while page_count < limits.max_pages:
            params = dict(base_params)
            if cursor is not None:
                params["cursor"] = cursor
            if bucket is not None:
                await bucket.acquire()
            try:
                payload, result = await self._fetch_page(client, params, sleep=sleep)
            except FetchError as exc:
                if cursor is not None and exc.status_code in _STALE_CURSOR_STATUSES:
                    break
                raise SourceError(
                    f"registry page {page_count + 1} could not be fetched: {exc}",
                    source_id=_SOURCE_ID,
                ) from exc

            digest.update(result.body)
            page_count += 1
            page_records, next_cursor = _parse_page(
                payload, url=result.url, include_deleted=self.include_deleted
            )
            for record in page_records:
                if record.upstream_id in seen_ids:
                    continue
                seen_ids.add(record.upstream_id)
                records.append(record)

            if len(records) >= limits.max_records:
                records = records[: limits.max_records]
                break
            if next_cursor is None or next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        return SourceFetch(
            source_id=_SOURCE_ID,
            url=_SERVERS_URL,
            sha256=digest.hexdigest(),
            entry_count=len(records),
            page_count=page_count,
            records=tuple(records),
        )

    def _base_params(self, limits: SourceLimits) -> dict[str, str | int]:
        """The query every page of this crawl carries, in emitted order."""
        params: dict[str, str | int] = {
            "limit": min(max(limits.page_size, 1), MAX_PAGE_SIZE),
            "version": "latest",
        }
        if self.updated_since is not None:
            params["updated_since"] = _rfc3339(self.updated_since)
        if self.include_deleted:
            params["include_deleted"] = "true"
        return params

    async def _fetch_page(
        self,
        client: httpx.AsyncClient,
        params: Mapping[str, str | int],
        *,
        sleep: Callable[[float], Awaitable[None]],
    ) -> tuple[JsonValue, FetchResult]:
        """One page, with a bounded second round of rate-limit patience."""
        attempt = 1
        while True:
            try:
                return await fetch_json(
                    client,
                    _SERVERS_URL,
                    params=params,
                    headers=_REQUEST_HEADERS,
                    sleep=sleep,
                )
            except FetchError as exc:
                if exc.status_code != _RATE_LIMIT_STATUS or attempt >= _RATE_LIMIT_ATTEMPTS:
                    raise
                await sleep(backoff_delay(attempt))
                attempt += 1
