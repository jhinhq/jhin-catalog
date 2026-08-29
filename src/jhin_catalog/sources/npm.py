"""The npm search crawl: keyword partitions inside the ``from`` ceiling.

npm's search endpoint will not page past ``from=5000`` for any one query —
it stops erroring and starts re-serving the first page instead — so this
module treats the keyword list as a partition of the query space rather
than as a set of synonyms, runs each keyword to that ceiling, and counts
and logs whatever the ceiling puts out of reach. A page that repeats the
first package name trips a mandatory tripwire and fails the fetch, because
silently re-ingesting page one under a fresh offset would look downstream
like the long tail had been deleted. Records leave here as npm served
them: casting ``dependents`` and reading ``downloads`` belong to
``normalize``, and the only judgement made in this module is that an
insecure package is not worth indexing.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import ClassVar, Final
from urllib.parse import urlsplit

import httpx

from jhin_catalog.http import fetch_json
from jhin_catalog.sources.base import (
    DEFAULT_LIMITS,
    RollingDigest,
    Source,
    SourceError,
    SourceLimits,
    TokenBucket,
)
from jhin_catalog.types import JsonObject, JsonValue, RawRecord, SourceFetch

NPM_SEARCH_URL: Final[str] = "https://registry.npmjs.org/-/v1/search"
MAX_PAGE_SIZE: Final[int] = 250
MAX_FROM: Final[int] = 5000
KEYWORDS: Final[tuple[str, ...]] = (
    "mcp",
    "mcp-server",
    "modelcontextprotocol",
    "model-context-protocol",
    "claude-mcp",
    "mcp-client",
)

_SOURCE_ID: Final[str] = "npm"
_PACKAGE_PAGE: Final[str] = "https://www.npmjs.com/package/"
_REQUEST_HEADERS: Final[Mapping[str, str]] = {"accept": "application/json"}

# The hosts a repository link may name. Anything else — a self-hosted
# forge, a shortener, a tarball — is dropped rather than guessed at,
# because a wrong repo key would merge two unrelated servers into one.
_REPO_HOSTS: Final[frozenset[str]] = frozenset({"github.com", "gitlab.com", "bitbucket.org"})
_GIT_PREFIX: Final[str] = "git+"
_GIT_SUFFIX: Final[str] = ".git"

_LOG: Final[logging.Logger] = logging.getLogger(__name__)


class NpmPaginationWrap(SourceError):
    """``from`` overflowed and the API silently re-served page 1."""


def parse_page(payload: JsonValue, *, url: str, keyword: str) -> tuple[tuple[RawRecord, ...], int]:
    """Records plus the reported ``total``. Raises ``SourceError``.

    The envelope must be an object carrying an ``objects`` list, and every
    object in it must name a package, because a search hit without a name
    cannot be keyed, merged, or linked. A package flagged ``insecure`` is
    dropped here: npm has already judged it, and the catalog has no reason
    to publish a pointer to it.
    """
    if not isinstance(payload, dict):
        raise SourceError(
            f"npm search for {keyword!r} at {url} is not a JSON object", source_id=_SOURCE_ID
        )
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise SourceError(
            f"npm search for {keyword!r} at {url} has no ``objects`` list", source_id=_SOURCE_ID
        )

    records: list[RawRecord] = []
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise SourceError(
                f"npm search for {keyword!r} at {url} object {index} is not an object",
                source_id=_SOURCE_ID,
            )
        package = item.get("package")
        if not isinstance(package, dict):
            raise SourceError(
                f"npm search for {keyword!r} at {url} object {index} lacks ``package``",
                source_id=_SOURCE_ID,
            )
        name = package.get("name")
        if not isinstance(name, str) or not name.strip():
            raise SourceError(
                f"npm search for {keyword!r} at {url} object {index} lacks ``package.name``",
                source_id=_SOURCE_ID,
            )
        if _is_insecure(item):
            continue
        records.append(
            RawRecord(
                source_id=_SOURCE_ID,
                upstream_id=name.strip(),
                url=f"{_PACKAGE_PAGE}{name.strip()}",
                payload=item,
            )
        )

    return tuple(records), _as_int(payload.get("total"), default=len(records))


def repo_url_from_links(links: JsonValue) -> str:
    """``git+https://github.com/o/r.git`` → ``https://github.com/o/r``, else ``""``.

    Only the three hosts the catalog knows how to key are accepted, and
    only over ``https``: a ``git://`` or ``ssh://`` link names the same repo
    but cannot be shown to a person, and a link with no owner and name is
    not a repo at all. A deep link into a monorepo is narrowed to the
    repository it lives in, which the ambiguous-repo rule then demotes if
    two packages land on the same one.
    """
    if not isinstance(links, dict):
        return ""
    raw = links.get("repository")
    if not isinstance(raw, str):
        return ""
    candidate = raw.strip()
    if candidate.startswith(_GIT_PREFIX):
        candidate = candidate[len(_GIT_PREFIX) :]
    if not candidate.startswith("https://"):
        return ""
    parts = urlsplit(candidate)
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host not in _REPO_HOSTS:
        return ""
    segments = [segment for segment in parts.path.split("/") if segment]
    if len(segments) < 2:
        return ""
    owner, repo = segments[0], segments[1]
    if repo.endswith(_GIT_SUFFIX):
        repo = repo[: -len(_GIT_SUFFIX)]
    if not owner or not repo:
        return ""
    return f"https://{host}/{owner}/{repo}"


class NpmSource(Source):
    """npm search, run once per keyword and stopped at the ``from`` ceiling."""

    source_id: ClassVar[str] = "npm"

    async def fetch(
        self,
        client: httpx.AsyncClient,
        *,
        limits: SourceLimits = DEFAULT_LIMITS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> SourceFetch:
        """Page every keyword to exhaustion or to ``MAX_FROM``, whichever comes first.

        ``from`` is never asked to exceed ``MAX_FROM``, so the wrap tripwire
        guards a bug rather than an expected condition. A keyword whose
        ``total`` exceeds what the ceiling can reach is logged with the
        number of packages left behind, which is the honest form of a
        partition that is not quite fine enough. A package already seen
        under an earlier keyword is skipped, so the crawl order fixes which
        keyword a package is attributed to.
        """
        bucket = (
            TokenBucket(rate_per_minute=limits.requests_per_minute, sleep=sleep)
            if limits.requests_per_minute > 0
            else None
        )
        size = min(max(limits.page_size, 1), MAX_PAGE_SIZE)
        digest = RollingDigest()
        seen: dict[str, RawRecord] = {}
        page_count = 0

        for keyword in KEYWORDS:
            if page_count >= limits.max_pages or len(seen) >= limits.max_records:
                _LOG.warning(
                    "npm: budget spent before keyword %r; %d keywords were not searched",
                    keyword,
                    len(KEYWORDS) - KEYWORDS.index(keyword),
                )
                break

            first_name = ""
            offset = 0
            total = 0
            while offset <= MAX_FROM:
                if page_count >= limits.max_pages or len(seen) >= limits.max_records:
                    break
                if bucket is not None:
                    await bucket.acquire()
                payload, result = await fetch_json(
                    client,
                    NPM_SEARCH_URL,
                    params={"text": f"keywords:{keyword}", "size": size, "from": offset},
                    headers=_REQUEST_HEADERS,
                    sleep=sleep,
                )
                digest.update(result.body)
                page_count += 1
                records, total = parse_page(payload, url=result.url, keyword=keyword)
                if not records:
                    break

                head = records[0].upstream_id
                if offset == 0:
                    first_name = head
                elif head == first_name:
                    raise NpmPaginationWrap(
                        f"npm re-served {head!r} as the first result of keyword {keyword!r} "
                        f"at from={offset}; the search window has wrapped",
                        source_id=_SOURCE_ID,
                    )

                for record in records:
                    seen.setdefault(record.upstream_id, record)
                offset += size
                if offset >= total:
                    break

            reachable = MAX_FROM + size
            if total > reachable:
                _LOG.warning(
                    "npm: keyword %r reports %d packages but only %d are reachable; %d were "
                    "skipped past from=%d",
                    keyword,
                    total,
                    reachable,
                    total - reachable,
                    MAX_FROM,
                )

        records_out = tuple(seen.values())[: limits.max_records]
        return SourceFetch(
            source_id=_SOURCE_ID,
            url=NPM_SEARCH_URL,
            sha256=digest.hexdigest(),
            entry_count=len(records_out),
            page_count=page_count,
            records=records_out,
        )


def _is_insecure(item: JsonObject) -> bool:
    """Whether npm has flagged this search hit as insecure."""
    flags = item.get("flags")
    if not isinstance(flags, dict):
        return False
    return bool(flags.get("insecure"))


def _as_int(value: JsonValue, *, default: int) -> int:
    """``value`` as a whole number, falling back to ``default``.

    npm serves ``total`` as a number and ``dependents`` as a string in the
    same envelope, so a reader that accepts both is the only one that works
    on both.
    """
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
