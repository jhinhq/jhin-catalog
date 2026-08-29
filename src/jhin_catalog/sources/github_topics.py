"""GitHub topic search: the 1000-result ceiling, walked in star windows.

Code search will serve ten pages of any one query and no more, which for a
topic like ``mcp-server`` is a small fraction of the repositories carrying
it. Rather than truncate at the ceiling and let the missing tail read as a
deletion, this module partitions: it takes the ten pages a query is worth,
notes the lowest star count it saw, and re-asks the same topic with a
``stars:`` upper bound at that mark, so each window covers the next slice
down. Windows are bounded and every genuinely unreachable remainder is
logged. Requests pass through a ``TokenBucket`` at the documented search
rate and a ``403`` — GitHub's secondary rate limit — is waited out on its
own ``Retry-After``; a topic that stays refused is cut short and reported
rather than failing the crawl. ``GET /rate_limit`` is never consulted,
because it reports a stale ``remaining`` and scheduling against it is how
a crawl earns a ban.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import ClassVar, Final, NamedTuple

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

GITHUB_API: Final[str] = "https://api.github.com"
TOPICS: Final[tuple[str, ...]] = (
    "mcp-server",
    "modelcontextprotocol",
    "claude-code-plugin",
    "agent-skills",
    "claude-skills",
)
MAX_PAGE_SIZE: Final[int] = 100
MAX_PAGES_PER_TOPIC: Final[int] = 10
SEARCH_RATE_PER_MINUTE: Final[int] = 30
CORE_RATE_PER_MINUTE: Final[int] = 80

_SOURCE_ID: Final[str] = "github_topics"
_SEARCH_URL: Final[str] = f"{GITHUB_API}/search/repositories"
_ACCEPT: Final[str] = "application/vnd.github+json"
_API_VERSION: Final[str] = "2022-11-28"

# A refused search is not in ``RETRY_STATUSES`` — a 403 is how GitHub says
# "secondary rate limit", not "forbidden" — so this source spends its own
# rounds on it, honouring the ``Retry-After`` GitHub sends with it. Five
# attempts on the doubling schedule is ~7s of its own patience, and far more
# when the header asks for longer, which on a secondary limit it usually
# does.
_FORBIDDEN_STATUS: Final[int] = 403
_FORBIDDEN_ATTEMPTS: Final[int] = 5
_UNPROCESSABLE_STATUS: Final[int] = 422


class _RateLimited(Exception):
    """Internal signal: GitHub kept refusing this page, so stop this topic.

    Deliberately not a ``SourceError``. A secondary rate limit is the
    expected end of a large crawl rather than a broken upstream, and the
    other sources already treat their own ceilings that way — npm reports
    what it skipped past ``from=5000``, smithery reports the details it
    could not reach. Aborting the whole build here would throw away every
    other source's work over one paginated page, and the diff gate is what
    actually guards against publishing a truncated catalog.
    """


# How many ``stars:`` windows one topic may be split into. Five topics at
# ten pages of a hundred is 5,000 repositories per window pass; twelve
# windows is far more headroom than any observed topic needs, and it is
# what stops a pathological star distribution from looping.
_MAX_STAR_WINDOWS: Final[int] = 12

_LOG: Final[logging.Logger] = logging.getLogger(__name__)


class _QueryWindow(NamedTuple):
    """What one bounded query returned, and how far down it reached."""

    records: tuple[RawRecord, ...]
    total_count: int
    min_stars: int | None
    pages: int


def github_headers(token: str) -> dict[str, str]:
    """``Accept``, ``X-GitHub-Api-Version``, and ``Authorization`` when a token is given.

    An empty token is the unauthenticated crawl, which is legitimate and
    much slower; no header is invented for it, because an ``Authorization``
    carrying nothing is answered with a 401 rather than with the anonymous
    rate limit.
    """
    headers = {"Accept": _ACCEPT, "X-GitHub-Api-Version": _API_VERSION}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def parse_search_page(payload: JsonValue, *, url: str, topic: str) -> tuple[RawRecord, ...]:
    """Raises ``SourceError`` when ``items`` is missing or not a list.

    A repository is keyed by its ``full_name`` lowercased, which is the form
    the repo identity key is built from, and is pointed at its own
    ``html_url``. Everything else on the item — stars, forks, topics,
    licence — is left exactly as GitHub served it for ``normalize`` to read.
    """
    if not isinstance(payload, dict):
        raise SourceError(
            f"github search for {topic!r} at {url} is not a JSON object", source_id=_SOURCE_ID
        )
    items = payload.get("items")
    if not isinstance(items, list):
        raise SourceError(
            f"github search for {topic!r} at {url} has no ``items`` list", source_id=_SOURCE_ID
        )

    records: list[RawRecord] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise SourceError(
                f"github search for {topic!r} at {url} item {index} is not an object",
                source_id=_SOURCE_ID,
            )
        full_name = item.get("full_name")
        if not isinstance(full_name, str) or "/" not in full_name:
            raise SourceError(
                f"github search for {topic!r} at {url} item {index} lacks ``full_name``",
                source_id=_SOURCE_ID,
            )
        slug = full_name.strip()
        html_url = item.get("html_url")
        page = html_url if isinstance(html_url, str) and html_url else f"https://github.com/{slug}"
        records.append(
            RawRecord(
                source_id=_SOURCE_ID,
                upstream_id=slug.lower(),
                url=page,
                payload=item,
            )
        )
    return tuple(records)


class GitHubTopicsSource(Source):
    """GitHub repository search over the topics that name this ecosystem."""

    source_id: ClassVar[str] = "github_topics"

    async def fetch(
        self,
        client: httpx.AsyncClient,
        *,
        limits: SourceLimits = DEFAULT_LIMITS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> SourceFetch:
        """Search every topic, partitioning by stars when one overflows.

        A topic whose ``total_count`` fits inside ten pages costs exactly
        one query, which is the common case and the one every fixture
        exercises. A topic that overflows is re-asked with a descending
        ``stars:`` bound until it fits, until the bound reaches zero, or
        until ``_MAX_STAR_WINDOWS`` is spent; whatever remains unreachable
        is logged with its count rather than dropped in silence. The whole
        crawl shares one bucket, one page budget and one record budget, so
        a greedy first topic cannot starve the last one without saying so.
        """
        headers = github_headers(limits.github_token)
        rate = (
            min(limits.requests_per_minute, SEARCH_RATE_PER_MINUTE)
            if limits.requests_per_minute > 0
            else SEARCH_RATE_PER_MINUTE
        )
        bucket = TokenBucket(rate_per_minute=rate, sleep=sleep)
        per_page = min(max(limits.page_size, 1), MAX_PAGE_SIZE)
        reachable = MAX_PAGES_PER_TOPIC * per_page
        digest = RollingDigest()
        seen: dict[str, RawRecord] = {}
        page_count = 0
        spent = False

        for topic in TOPICS:
            if spent:
                break
            query = f"topic:{topic}"
            previous_min: int | None = None
            for window in range(_MAX_STAR_WINDOWS):
                if page_count >= limits.max_pages or len(seen) >= limits.max_records:
                    _LOG.warning(
                        "github_topics: budget spent during topic %r window %d; the remaining "
                        "topics were not searched",
                        topic,
                        window + 1,
                    )
                    spent = True
                    break

                try:
                    result = await self._crawl_query(
                        client,
                        query=query,
                        topic=topic,
                        per_page=per_page,
                        page_budget=limits.max_pages - page_count,
                        headers=headers,
                        bucket=bucket,
                        sleep=sleep,
                        digest=digest,
                    )
                except _RateLimited as exc:
                    _LOG.warning(
                        "github_topics: topic %r hit the search rate limit below stars<=%s "
                        "and was cut short (%s); the tail was skipped",
                        topic,
                        previous_min,
                        exc,
                    )
                    break
                page_count += result.pages
                added = 0
                for record in result.records:
                    if record.upstream_id in seen:
                        continue
                    seen[record.upstream_id] = record
                    added += 1

                if result.total_count <= reachable:
                    break
                if result.min_stars is None or result.min_stars <= 0:
                    _LOG.warning(
                        "github_topics: topic %r holds %d repositories and cannot be narrowed "
                        "further; about %d were skipped",
                        topic,
                        result.total_count,
                        result.total_count - reachable,
                    )
                    break
                if added == 0 and previous_min == result.min_stars:
                    _LOG.warning(
                        "github_topics: topic %r stalled at stars<=%d; about %d repositories "
                        "were skipped",
                        topic,
                        result.min_stars,
                        result.total_count - reachable,
                    )
                    break
                previous_min = result.min_stars
                query = f"topic:{topic} stars:<={result.min_stars}"
            else:
                _LOG.warning(
                    "github_topics: topic %r used all %d star windows; the tail below "
                    "stars<=%s was skipped",
                    topic,
                    _MAX_STAR_WINDOWS,
                    previous_min,
                )

        records = tuple(seen.values())[: limits.max_records]
        return SourceFetch(
            source_id=_SOURCE_ID,
            url=_SEARCH_URL,
            sha256=digest.hexdigest(),
            entry_count=len(records),
            page_count=page_count,
            records=records,
        )

    async def _crawl_query(
        self,
        client: httpx.AsyncClient,
        *,
        query: str,
        topic: str,
        per_page: int,
        page_budget: int,
        headers: Mapping[str, str],
        bucket: TokenBucket,
        sleep: Callable[[float], Awaitable[None]],
        digest: RollingDigest,
    ) -> _QueryWindow:
        """One query, walked to its own ceiling.

        Page eleven is never requested: the search API answers it with a 422
        whatever the query, so the walk stops at ``MAX_PAGES_PER_TOPIC``, at
        a short page, at the reported total, or at the caller's remaining
        page budget. The lowest star count seen comes back with the records,
        because it is the bound the next window is cut at.
        """
        records: list[RawRecord] = []
        total_count = 0
        min_stars: int | None = None
        pages = 0
        ceiling = min(MAX_PAGES_PER_TOPIC, max(page_budget, 0))

        page = 1
        while page <= ceiling:
            await bucket.acquire()
            payload, result = await self._fetch_search_page(
                client,
                params={
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": per_page,
                    "page": page,
                },
                headers=headers,
                sleep=sleep,
                topic=topic,
                page=page,
            )
            digest.update(result.body)
            pages += 1
            page_records = parse_search_page(payload, url=result.url, topic=topic)
            if page == 1:
                total_count = _total_count(payload)
                if _incomplete(payload):
                    _LOG.info(
                        "github_topics: %r returned incomplete results; the window may be short",
                        query,
                    )
            records.extend(page_records)
            for record in page_records:
                stars = _stars(record.payload)
                if stars is not None and (min_stars is None or stars < min_stars):
                    min_stars = stars

            if len(page_records) < per_page or pages * per_page >= total_count:
                break
            page += 1

        return _QueryWindow(
            records=tuple(records),
            total_count=total_count,
            min_stars=min_stars,
            pages=pages,
        )

    async def _fetch_search_page(
        self,
        client: httpx.AsyncClient,
        *,
        params: Mapping[str, str | int],
        headers: Mapping[str, str],
        sleep: Callable[[float], Awaitable[None]],
        topic: str,
        page: int,
    ) -> tuple[JsonValue, FetchResult]:
        """One search page, patient with a 403 and fatal on a 422.

        A 403 is the secondary rate limit rather than a refusal, so it is
        slept off up to ``_FORBIDDEN_ATTEMPTS`` tries, waiting whatever
        ``Retry-After`` asked for when GitHub sent one and the doubling
        schedule otherwise. Outliving that raises ``_RateLimited``, which
        ends this topic rather than the crawl. A 422 means the query itself
        is unacceptable — past the result ceiling, or malformed — and no
        amount of waiting changes that.
        """
        attempt = 1
        while True:
            try:
                return await fetch_json(
                    client, _SEARCH_URL, params=params, headers=headers, sleep=sleep
                )
            except FetchError as exc:
                if exc.status_code == _FORBIDDEN_STATUS and attempt < _FORBIDDEN_ATTEMPTS:
                    await sleep(backoff_delay(attempt, retry_after=exc.retry_after))
                    attempt += 1
                    continue
                if exc.status_code == _FORBIDDEN_STATUS:
                    raise _RateLimited(
                        f"github search for {topic!r} page {page} was refused "
                        f"{_FORBIDDEN_ATTEMPTS} times: {exc}"
                    ) from exc
                if exc.status_code == _UNPROCESSABLE_STATUS:
                    raise SourceError(
                        f"github search for {topic!r} page {page} was rejected: {exc}",
                        source_id=_SOURCE_ID,
                    ) from exc
                raise


def _total_count(payload: JsonValue) -> int:
    """``total_count``, which says whether the query overflowed its ceiling."""
    if not isinstance(payload, dict):
        return 0
    value = payload.get("total_count")
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _incomplete(payload: JsonValue) -> bool:
    """Whether GitHub timed out mid-query and served a partial window."""
    return isinstance(payload, dict) and payload.get("incomplete_results") is True


def _stars(item: JsonObject) -> int | None:
    """``stargazers_count`` when the item carries a usable one."""
    value = item.get("stargazers_count")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
