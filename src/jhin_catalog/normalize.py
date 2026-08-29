"""Per-source projection into the canonical entry, and the identity keys.

Every upstream describes a server in its own vocabulary. This module reads
one ``RawRecord`` at a time and emits a ``Candidate``: the identity keys that
record proves, the popularity counts it reports, and a partial entry payload.
Nothing here merges, scores, or writes — a candidate is a claim about a
single record, and ``dedupe`` decides which claims describe the same thing.
``build_entry`` is the other end of the module: the one place a validated
``McpEntry`` or ``SkillEntry`` is constructed, where every derived field is
recomputed from merged data rather than trusted from it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Final, NamedTuple
from urllib.parse import urlsplit

from pydantic import ValidationError

from jhin_catalog.score import POPULARITY_DECIMALS
from jhin_catalog.types import (
    CATALOG_CATEGORIES,
    CATALOG_ICONS,
    CONNECTOR_TYPES,
    DEFAULT_SKILL_CATEGORY,
    ENTRY_ADAPTER,
    ICON_URL_GITHUB_RE,
    ICON_URL_SMITHERY_RE,
    MAX_ALIAS_KEYS,
    MAX_DESCRIPTION_CHARS,
    MAX_NAME_CHARS,
    MAX_NOTE_CHARS,
    MAX_PACKAGES,
    MAX_REMOTES,
    MAX_SOURCES,
    MAX_SUMMARY_CHARS,
    MAX_TAG_CHARS,
    MAX_TAGS,
    MAX_URL_CHARS,
    SCHEMA_VERSION,
    SERVER_SLUG_RE,
    SHA1_RE,
    SKILL_NAME_RE,
    SOURCE_RANK,
    Candidate,
    CatalogEntry,
    JsonObject,
    JsonValue,
    MergedCandidate,
    NormalizeError,
    PopularitySignals,
    RawRecord,
    RepoRef,
    SourceRef,
)

# ---------------------------------------------------------------------------
# Identity key spaces
# ---------------------------------------------------------------------------

# Section 2.1 of the build specification. Lower wins when a component holds
# several keys: a repository is the most durable identity anyone publishes, a
# concrete endpoint the next, and a package name the weakest, because a name
# can be transferred to a different maintainer. The table lives beside the
# functions that mint the keys, and ``dedupe`` re-exports it for its callers.
KEY_SPACE_RANK: Final[Mapping[str, int]] = {
    "repo": 0,
    "url": 1,
    "registry": 2,
    "npm": 3,
    "pypi": 4,
    "smithery": 5,
    "skill": 0,
    "plugin": 1,
}

_UNKNOWN_SPACE_RANK: Final[int] = 99

# ---------------------------------------------------------------------------
# Category and icon derivation
# ---------------------------------------------------------------------------

# Scanned in order; the first rule with a whole-word hit wins. The order is
# the point: "Stripe payments for developers" is a payments server, so the
# narrow verticals are asked before the catch-all developer bucket.
CATEGORY_RULES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (
        "Payments & commerce",
        (
            "stripe",
            "paypal",
            "payment",
            "invoice",
            "billing",
            "checkout",
            "shopify",
            "square",
            "commerce",
        ),
    ),
    (
        "CRM & support",
        (
            "crm",
            "salesforce",
            "hubspot",
            "zendesk",
            "intercom",
            "helpdesk",
            "ticketing",
            "support",
        ),
    ),
    ("Design", ("figma", "canva", "design", "sketch", "prototype", "wireframe")),
    (
        "Project management",
        (
            "jira",
            "linear",
            "asana",
            "trello",
            "clickup",
            "monday",
            "kanban",
            "sprint",
            "backlog",
            "issue tracker",
            "project management",
        ),
    ),
    (
        "Communication",
        (
            "slack",
            "discord",
            "telegram",
            "email",
            "gmail",
            "smtp",
            "sms",
            "twilio",
            "chat",
            "messaging",
            "resend",
        ),
    ),
    (
        "Documents & knowledge",
        (
            "notion",
            "confluence",
            "wiki",
            "docs",
            "document",
            "markdown",
            "pdf",
            "knowledge base",
            "obsidian",
        ),
    ),
    (
        "Storage",
        ("s3", "dropbox", "box", "google drive", "bucket", "blob", "filesystem", "file storage"),
    ),
    (
        "Search & web",
        (
            "search",
            "browse",
            "browser",
            "scrape",
            "crawl",
            "serp",
            "brave",
            "tavily",
            "exa",
            "firecrawl",
            "playwright",
            "puppeteer",
            "web",
        ),
    ),
    (
        "Data & infrastructure",
        (
            "postgres",
            "postgresql",
            "mysql",
            "sqlite",
            "database",
            "warehouse",
            "bigquery",
            "snowflake",
            "redis",
            "kafka",
            "kubernetes",
            "terraform",
            "aws",
            "gcp",
            "azure",
            "cloudflare",
            "supabase",
            "neon",
        ),
    ),
    ("Automation", ("zapier", "workflow", "automation", "n8n", "cron", "scheduler", "pipeline")),
    ("Productivity", ("calendar", "todo", "todoist", "notes", "reminder", "task", "productivity")),
    (
        "Developer tools",
        (
            "github",
            "gitlab",
            "git",
            "ci",
            "cd",
            "deploy",
            "vercel",
            "netlify",
            "lint",
            "test",
            "debug",
            "sentry",
            "observability",
            "sdk",
            "compiler",
            "ide",
        ),
    ),
)

SLUG_ICONS: Final[Mapping[str, str]] = {
    "github": "github",
    "linear": "linear",
    "vercel": "vercel",
    "notion": "notebook",
    "slack": "message-square",
    "discord": "message-circle",
    "telegram": "send",
    "stripe": "credit-card",
    "paypal": "credit-card",
    "square": "credit-card",
    "hubspot": "users",
    "intercom": "life-buoy",
    "sentry": "bug",
    "cloudflare": "cloud",
    "zapier": "zap",
    "asana": "check-square",
    "monday": "kanban",
    "clickup": "kanban",
    "trello": "kanban",
    "todoist": "check-square",
    "canva": "palette",
    "figma": "pen-tool",
    "google_drive": "folder",
    "dropbox": "folder",
    "box": "folder",
    "google_calendar": "calendar",
    "gmail": "mail",
    "airtable": "table",
    "postgres": "database",
    "supabase": "database",
    "neon": "database",
    "filesystem": "hard-drive",
    "playwright": "globe",
    "brave_search": "search",
    "tavily": "search",
    "exa": "search",
    "firecrawl": "flame",
    "context7": "book-open",
    "huggingface": "cpu",
    "twilio": "phone",
    "resend": "mail",
    "deepwiki": "book-open",
    "microsoft_learn": "book-open",
    "http": "terminal",
}

CATEGORY_ICONS: Final[Mapping[str, str]] = {
    "Developer tools": "terminal",
    "Project management": "kanban",
    "Communication": "message-square",
    "Documents & knowledge": "notebook",
    "Payments & commerce": "credit-card",
    "CRM & support": "users",
    "Design": "palette",
    "Search & web": "search",
    "Data & infrastructure": "database",
    "Automation": "zap",
    "Productivity": "check-square",
    "Storage": "folder",
}

_CATEGORY_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = tuple(
    (category, re.compile(r"\b(?:" + "|".join(re.escape(word) for word in words) + r")\b"))
    for category, words in CATEGORY_RULES
)

# ---------------------------------------------------------------------------
# Patterns and bounds
# ---------------------------------------------------------------------------

_CONTROL_RE: Final[re.Pattern[str]] = re.compile(
    "[\\x00-\\x1f\\x7f-\\x9f\\u200b-\\u200f\\u2028\\u2029\\ufeff]"
)
_SLUG_GAP_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9_]+")
_SLUG_RUN_RE: Final[re.Pattern[str]] = re.compile(r"_{2,}")
_TAG_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_KEY_GAP_RE: Final[re.Pattern[str]] = re.compile(r"[^\x21-\x7e]+")

_MD_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"`{3,}[A-Za-z0-9_+.-]*")
_MD_IMAGE_RE: Final[re.Pattern[str]] = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK_RE: Final[re.Pattern[str]] = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_CODE_RE: Final[re.Pattern[str]] = re.compile(r"`+([^`]*)`+")
_MD_STRIKE_RE: Final[re.Pattern[str]] = re.compile(r"~~(.+?)~~", re.DOTALL)
_MD_STAR_RE: Final[re.Pattern[str]] = re.compile(r"\*{1,3}(\S(?:.*?\S)?)\*{1,3}", re.DOTALL)
_MD_UNDER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9_])_{1,3}([^_]+?)_{1,3}(?![A-Za-z0-9_])"
)
_MD_PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"(?m)^[ \t]{0,3}(?:#{1,6}[ \t]+|>[ \t]?|[-*+][ \t]+|\d{1,3}[.)][ \t]+)"
)

_REPO_HOSTS: Final[frozenset[str]] = frozenset({"github.com", "gitlab.com", "bitbucket.org"})
_OWNER_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_REPO_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
_DEEP_LINK_SEGMENTS: Final[frozenset[str]] = frozenset({"tree", "blob", "src"})

_TRANSPORTS: Final[frozenset[str]] = frozenset({"streamable_http", "sse", "unknown"})
_AUTH_HINTS: Final[frozenset[str]] = frozenset({"none", "bearer", "header", "oauth"})
_REMOTE_TRANSPORT_MAP: Final[Mapping[str, str]] = {
    "streamable-http": "streamable_http",
    "streamable_http": "streamable_http",
    "http": "streamable_http",
    "sse": "sse",
}
_PACKAGE_TRANSPORT_MAP: Final[Mapping[str, str]] = {
    "streamable-http": "streamable_http",
    "streamable_http": "streamable_http",
    "sse": "sse",
    "stdio": "stdio",
}
_REMOTE_TRANSPORT_RANK: Final[Mapping[str, int]] = {
    "streamable_http": 0,
    "sse": 1,
    "unknown": 2,
}
_SECRET_WORDS: Final[tuple[str, ...]] = ("key", "token", "auth")

_OFFICIAL_META_KEY: Final[str] = "io.modelcontextprotocol.registry/official"
_PUBLISHER_META_KEY: Final[str] = "io.modelcontextprotocol.registry/publisher-provided"
_SKILL_SUFFIX: Final[str] = "/SKILL.md"

_MAX_KEY_VALUE_CHARS: Final[int] = 220
_MAX_UPSTREAM_ID_CHARS: Final[int] = 200
_MAX_HEADER_NAMES: Final[int] = 8
_MAX_TOOL_COUNT: Final[int] = 200
_MAX_LICENSE_CHARS: Final[int] = 64
_MAX_SUBPATH_CHARS: Final[int] = 255
_MAX_SKILL_PATH_CHARS: Final[int] = 255
_MAX_SOURCE_REF_CHARS: Final[int] = 300
_MAX_SKILL_NAME_CHARS: Final[int] = 64
_MAX_SKILL_CATEGORY_CHARS: Final[int] = 64
_MAX_SKILL_VERSION_CHARS: Final[int] = 32
_MAX_ALLOWED_TOOLS: Final[int] = 32
_MAX_ALLOWED_TOOL_CHARS: Final[int] = 64
_MAX_FRONTMATTER_BYTES: Final[int] = 8192
_MAX_IDENTIFIER_CHARS: Final[int] = 200
_MAX_REGISTRY_TYPE_CHARS: Final[int] = 32
_UNKNOWN_REGISTRY_TYPE: Final[str] = "unknown"
_MAX_VERSION_CHARS: Final[int] = 64
_MAX_RUNTIME_HINT_CHARS: Final[int] = 32
_MAX_REGISTRY_NAME_CHARS: Final[int] = 132
_MAX_QUALIFIED_NAME_CHARS: Final[int] = 200
_MAX_NPM_PACKAGE_CHARS: Final[int] = 214
_MAX_MARKETPLACE_CHARS: Final[int] = 100
# ``types._OWNER_REPO_RE`` accepts 64 + 1 + 100. A ``marketplace_repo`` past
# that is dropped rather than cut, because a cut one still parses.
_MAX_OWNER_REPO_CHARS: Final[int] = 165
# ``types.PluginRef.source_kind``: anything else is recorded as unknown rather
# than failing the entry, since the manifest field is free text upstream.
_PLUGIN_SOURCE_KINDS: Final[frozenset[str]] = frozenset(
    {"relative", "url", "git-subdir", "github", "npm", "unknown"}
)
_MAX_CONNECTOR_CONFIG_PAIRS: Final[int] = 10
_MAX_CONNECTOR_CONFIG_KEY_CHARS: Final[int] = 64
_MAX_CONNECTOR_CONFIG_VALUE_CHARS: Final[int] = 500

# Fields ``build_entry`` always recomputes. A merged payload carrying any of
# them is carrying a stale answer, so they never survive into validation.
_RECOMPUTED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "kind",
        "canonical_key",
        "alias_keys",
        "slug",
        "trust_tier",
        "popularity",
        "popularity_signals",
        "sources",
        "stdio_only",
        "url_unverified",
    }
)


class _Remote(NamedTuple):
    """One published endpoint, before it becomes a ``RemoteRef``."""

    transport: str
    url: str
    templated: bool
    header_names: tuple[str, ...]

    def as_json(self) -> JsonObject:
        """The ``RemoteRef`` payload this remote validates into."""
        names: list[JsonValue] = list(self.header_names)
        return {
            "transport": self.transport,
            "url": self.url,
            "templated": self.templated,
            "header_names": names,
        }


class _Package(NamedTuple):
    """One installable artefact, before it becomes a ``PackageRef``."""

    registry_type: str
    identifier: str
    version: str
    runtime_hint: str
    transport: str

    def as_json(self) -> JsonObject:
        """The ``PackageRef`` payload this package validates into."""
        return {
            "registry_type": self.registry_type,
            "identifier": self.identifier,
            "version": self.version,
            "runtime_hint": self.runtime_hint,
            "transport": self.transport,
        }


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def slugify(value: str) -> str:
    """A ``SERVER_SLUG_RE`` slug. Raises ``NormalizeError`` when nothing survives.

    Every run of anything outside ``[a-z0-9_]`` becomes one underscore, the
    result is trimmed at both ends, and the truncation to 32 characters is
    trimmed again so a slug never ends in the separator a cut left behind.
    """
    lowered = collapse(value, limit=len(value) + 1).lower()
    gapped = _SLUG_GAP_RE.sub("_", lowered)
    collapsed = _SLUG_RUN_RE.sub("_", gapped).strip("_")
    truncated = collapsed[:32].rstrip("_")
    if not truncated or not SERVER_SLUG_RE.fullmatch(truncated):
        raise NormalizeError(f"no slug survives {value!r}")
    return truncated


def collapse(text: str, *, limit: int) -> str:
    """Strip control characters, collapse whitespace runs to one space, truncate.

    A control character becomes a space rather than disappearing, so a value
    that arrived with a newline between two words does not come back with the
    two words fused into one.
    """
    spaced = _CONTROL_RE.sub(" ", text)
    return " ".join(spaced.split())[: max(limit, 0)].rstrip()


def summarize(text: str) -> str:
    """A one-line ≤``MAX_SUMMARY_CHARS`` summary for the projected catalog.

    Markdown emphasis, links, images, and code fences are flattened to the
    words a reader would have seen, because a description in the Apps library
    is rendered as plain prose. An over-long result is cut at the last word
    boundary and closed with ``…`` rather than mid-word.
    """
    flattened = collapse(_flatten_markdown(text), limit=MAX_SUMMARY_CHARS * 8)
    if len(flattened) <= MAX_SUMMARY_CHARS:
        return flattened
    head = flattened[: MAX_SUMMARY_CHARS - 1]
    boundary = head.rfind(" ")
    if boundary > 0:
        head = head[:boundary]
    return head.rstrip().rstrip(",;:") + "…"


def _flatten_markdown(text: str) -> str:
    """Markdown reduced to the words inside it, with the syntax removed."""
    flat = _MD_FENCE_RE.sub(" ", text)
    flat = _MD_PREFIX_RE.sub("", flat)
    flat = _MD_IMAGE_RE.sub(r"\1", flat)
    flat = _MD_LINK_RE.sub(r"\1", flat)
    flat = _MD_CODE_RE.sub(r"\1", flat)
    flat = _MD_STRIKE_RE.sub(r"\1", flat)
    flat = _MD_STAR_RE.sub(r"\1", flat)
    return _MD_UNDER_RE.sub(r"\1", flat)


# ---------------------------------------------------------------------------
# Repositories, URLs, and keys
# ---------------------------------------------------------------------------


def parse_repo_url(url: str) -> RepoRef | None:
    """``https://github.com/o/r.git`` → ``RepoRef``. ``None`` for anything else.

    Only the three hosts the catalog can key are accepted, and only over
    ``http`` or ``https``: a link this function cannot resolve to an owner and
    a name is not an identity, and guessing one would merge two servers that
    have nothing to do with each other. A ``/tree/{ref}/{path}`` deep link
    keeps its path as the subpath.
    """
    candidate = collapse(url, limit=MAX_URL_CHARS + 1)
    candidate = candidate.removeprefix("git+")
    if not candidate.startswith(("https://", "http://")):
        return None
    try:
        parts = urlsplit(candidate)
        host = (parts.hostname or "").lower()
    except ValueError:
        return None
    host = host.removeprefix("www.")
    if host not in _REPO_HOSTS:
        return None
    segments = [segment for segment in parts.path.split("/") if segment]
    if len(segments) < 2:
        return None
    owner = segments[0]
    repo = segments[1].removesuffix(".git")
    if not _OWNER_RE.fullmatch(owner) or not _REPO_RE.fullmatch(repo):
        return None
    rest = segments[2:]
    if len(rest) >= 2 and rest[0] in _DEEP_LINK_SEGMENTS:
        rest = rest[2:]
    subpath = _posix_relpath("/".join(rest), limit=_MAX_SUBPATH_CHARS)
    try:
        return RepoRef(host=host, owner=owner.lower(), repo=repo.lower(), subpath=subpath)
    except ValidationError:
        return None


def repo_key(repo: RepoRef, *, kind: str) -> str:
    """The ``repo`` key for one repository, with ``#subpath`` when it has one."""
    value = f"{repo.host}/{repo.owner}/{repo.repo}"
    if repo.subpath:
        value = f"{value}#{repo.subpath}"
    return _key(kind, "repo", value)


def url_key(url: str, *, kind: str) -> str:
    """The ``url`` key for one endpoint: host and path, and nothing else.

    The query and the fragment are dropped and a trailing ``/`` is removed, so
    ``https://brave.run.tools`` and ``https://brave.run.tools/`` are one
    identity rather than two entries that never merge.
    """
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return _key(kind, "url", url.strip())
    if not host:
        return _key(kind, "url", url.strip())
    if port is not None and port not in {80, 443}:
        host = f"{host}:{port}"
    return _key(kind, "url", f"{host}{parts.path.rstrip('/')}")


def name_key(space: str, value: str, *, kind: str) -> str:
    """A key in a name-shaped space: ``registry``, ``npm``, ``smithery``, and friends."""
    return _key(kind, space, value)


def _key(kind: str, space: str, value: str) -> str:
    """One ``{kind}:{space}:{value}`` key, lowercased and bounded.

    Anything outside printable ASCII becomes an underscore so the key always
    matches ``CANONICAL_KEY_RE``. Upstream ids are ASCII in practice, and a
    key that cannot be written down is a key nobody can reconcile against.
    """
    cleaned = _KEY_GAP_RE.sub("_", value.strip().lower())[:_MAX_KEY_VALUE_CHARS]
    if not cleaned:
        raise NormalizeError(f"no {space} key survives {value!r}")
    return f"{kind}:{space}:{cleaned}"


def _key_space(key: str) -> str:
    """The middle segment of a key, or ``""`` when the key is malformed."""
    parts = key.split(":", 2)
    return parts[1] if len(parts) == 3 else ""


def _elect(keys: Sequence[str]) -> str:
    """The strongest key one candidate holds, under ``KEY_SPACE_RANK``."""
    if not keys:
        raise NormalizeError("a candidate must hold at least one identity key")
    return min(
        keys, key=lambda key: (KEY_SPACE_RANK.get(_key_space(key), _UNKNOWN_SPACE_RANK), key)
    )


# ---------------------------------------------------------------------------
# Category and icon
# ---------------------------------------------------------------------------


def choose_category(*, name: str, description: str, tags: Sequence[str], extra: str = "") -> str:
    """The first ``CATEGORY_RULES`` bucket whose keyword appears as a whole word.

    The haystack is the name, the description, the tags, and whatever else the
    caller can offer — registry and npm names carry the vendor when the prose
    does not. Matching is on word boundaries so ``cd`` does not fire on
    ``cdn``, and a record matching nothing lands in ``Developer tools``.
    """
    haystack = " ".join([name, description, " ".join(tags), extra]).lower()
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(haystack):
            return category
    return "Developer tools"


def choose_icon(*, slug: str, category: str) -> str:
    """A known vendor mark when there is one, the category's mark otherwise."""
    named = SLUG_ICONS.get(slug)
    if named is not None:
        return named
    mapped = CATEGORY_ICONS.get(category)
    if mapped is None:
        raise NormalizeError(f"no icon is defined for category {category!r}")
    return mapped


def github_avatar_url(owner: str) -> str:
    """The GitHub avatar URL for one repository owner, or ``""``.

    Only the shape a consumer's icon proxy will actually dial: one to
    thirty-nine characters of letters, digits, and hyphens — GitHub's own
    username grammar. An owner outside it (a dotted GitLab group, say) yields
    no URL rather than a URL nothing will ever fetch. Lowercased, like the
    skill side's ``_owner_avatar_url``: GitHub resolves either spelling, and
    two spellings of one URL would read as a change on every rebuild.
    """
    candidate = f"https://github.com/{owner.lower()}.png?size=128"
    return candidate if ICON_URL_GITHUB_RE.fullmatch(candidate) is not None else ""


def elect_icon_url(payload: JsonObject) -> str:
    """The one upstream logo this record may be proxied from, or ``""``.

    In order: a Smithery-sourced entry gets Smithery's own icon route, since
    that serves the mark the server's publisher uploaded; anything with a
    GitHub repository gets its owner's avatar; a registry- or npm-only record
    gets nothing, because neither upstream serves an image this index would
    let a proxy dial. A constructed URL that fails the shape check falls
    through rather than failing the entry — a qualified name with a ``?`` in
    it is upstream noise, not a reason to drop a server.
    """
    qualified = _text(payload.get("smithery_qualified_name"))
    if qualified:
        candidate = f"https://api.smithery.ai/servers/{qualified}/icon"
        if len(candidate) <= MAX_URL_CHARS and ICON_URL_SMITHERY_RE.fullmatch(candidate):
            return candidate
    repo = _obj(payload.get("repo"))
    if _text(repo.get("host")) == "github.com":
        owner = _text(repo.get("owner"))
        if owner:
            return github_avatar_url(owner)
    return ""


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize(record: RawRecord) -> Candidate | None:
    """Dispatch on ``record.source_id``. ``None`` means "not catalogable".

    Raises ``NormalizeError`` only for a record that claims a known source and
    is then structurally impossible — a registry row with no server name, an
    npm object with no package name. A record from a source this module does
    not know is not an error; it is simply not something the catalog indexes.
    """
    match record.source_id:
        case "registry":
            return normalize_registry(record)
        case "smithery":
            return normalize_smithery(record)
        case "npm":
            return normalize_npm(record)
        case "github_topics":
            return normalize_github_topics(record)
        case "marketplaces":
            return normalize_marketplace_skill(record)
        case _:
            return None


def normalize_registry(record: RawRecord) -> Candidate | None:
    """One official-registry row: the strongest identity any source publishes.

    A deleted tombstone and a superseded version are dropped. Everything else
    is read whole — the repository, the packages, the remotes, and the
    endpoint elected from those remotes, which is the only endpoint in the
    corpus a person can trust without completing a handshake first.
    """
    item = record.payload
    server = _obj(item.get("server")) or item
    name = _text(server.get("name"))
    if not name:
        raise NormalizeError(f"registry record {record.upstream_id!r} has no ``server.name``")

    meta = _obj(_obj(item.get("_meta")).get(_OFFICIAL_META_KEY)) or _obj(
        _obj(server.get("_meta")).get(_OFFICIAL_META_KEY)
    )
    status = _text(meta.get("status")).lower() or "active"
    if status == "deleted":
        return None
    if meta.get("isLatest") is False or server.get("isLatest") is False:
        return None

    repository = _obj(server.get("repository"))
    repo = _with_subpath(
        parse_repo_url(_text(repository.get("url"))), _text(repository.get("subfolder"))
    )
    remotes = _registry_remotes(server)
    packages = _registry_packages(server)
    ordered = sorted(remotes, key=lambda one: (_REMOTE_TRANSPORT_RANK[one.transport], one.url))
    elected = next((one for one in ordered if _endpoint_url(one.url) is not None), None)

    mcp_url = _endpoint_url(elected.url) if elected is not None else None
    auth_hint, auth_note = _registry_auth(elected, ordered)
    stdio_only = mcp_url is None and bool(packages)
    homepage = _https_url(_text(server.get("websiteUrl")))
    publisher = _obj(_obj(server.get("_meta")).get(_PUBLISHER_META_KEY))
    title = _text(server.get("title")) or name.rsplit("/", 1)[-1]

    fields: JsonObject = {
        "name": collapse(title, limit=MAX_NAME_CHARS),
        "description": collapse(_text(server.get("description")), limit=MAX_DESCRIPTION_CHARS),
        "homepage": homepage,
        "docs_url": _first_url(homepage, _repo_web_url(repo), _text(repository.get("url"))),
        "tags": _tags(_strings(publisher.get("tags"))),
        "deprecated": status == "deprecated",
        "mcp_url": mcp_url,
        "transport": elected.transport if elected is not None else "unknown",
        "auth_hint": auth_hint,
        "auth_note": auth_note,
        "setup_note": _stdio_setup_note(packages) if stdio_only else "",
        "packages": [package.as_json() for package in packages],
        "remotes": [remote.as_json() for remote in ordered[:MAX_REMOTES]],
        "registry_name": name[:_MAX_REGISTRY_NAME_CHARS],
        "verified_upstream": status == "active" and bool(remotes),
    }
    if repo is not None:
        fields["repo"] = repo.model_dump(mode="json")

    keys = {name_key("registry", name, kind="mcp")}
    if repo is not None:
        keys.add(repo_key(repo, kind="mcp"))
    if mcp_url is not None:
        keys.add(url_key(mcp_url, kind="mcp"))
    for package in packages:
        if package.registry_type in {"npm", "pypi"}:
            keys.add(name_key(package.registry_type, package.identifier, kind="mcp"))

    verified = status == "active" and bool(remotes) and mcp_url is not None
    return _candidate(
        record,
        kind="mcp",
        keys=keys,
        repo=repo,
        signals=PopularitySignals(),
        trust_hint="registry_verified" if verified else "indexed",
        fields=fields,
    )


def normalize_smithery(record: RawRecord) -> Candidate | None:
    """One Smithery row, plus the detail pass when the server earned one.

    Smithery knows how often a server is actually used, which no other source
    reports, and it knows the deployment URL. It does not know the repository,
    so a Smithery candidate merges through its endpoint or not at all. The
    config schema is read for an auth hint and a field count and is then
    discarded: it is a payload, and this index stores pointers.
    """
    row = record.payload
    qualified = _text(row.get("qualifiedName"))
    if not qualified:
        raise NormalizeError(f"smithery record {record.upstream_id!r} has no ``qualifiedName``")
    if row.get("unlisted") is True or row.get("inactive") is True:
        return None

    detail = _obj(row.get("_detail"))
    connections = [_obj(item) for item in _list(detail.get("connections"))]
    http_connections = [item for item in connections if _text(item.get("type")).lower() == "http"]
    # The endpoint is on the detail document for most servers and only on the
    # HTTP connection for the rest. ``sources.smithery`` preserves both
    # spellings; reading only the first published those servers with no
    # ``mcp_url`` and nothing a person could dial.
    deployment = _text(detail.get("deploymentUrl")) or next(
        (url for item in http_connections if (url := _first_text(item, "deploymentUrl", "url"))),
        "",
    )
    mcp_url = _endpoint_url(deployment)

    remotes: list[_Remote] = []
    if http_connections and deployment:
        remotes.append(
            _Remote(
                transport="streamable_http",
                url=deployment[:MAX_URL_CHARS],
                templated="{" in deployment,
                header_names=(),
            )
        )

    # A stdio-only Smithery server has no artefact name of its own, so its
    # qualified name stands in as the identifier. That is what keeps
    # ``stdio_only`` recomputable from packages alone in ``build_entry``, and
    # it is also the name the setup note tells a person to go and host.
    packages: list[_Package] = []
    if connections and not http_connections and mcp_url is None:
        packages.append(
            _Package(
                registry_type="smithery",
                identifier=qualified[:_MAX_IDENTIFIER_CHARS],
                version="",
                runtime_hint="",
                transport="stdio",
            )
        )

    auth_hint, auth_note = _smithery_auth(connections) if detail else ("bearer", "")
    homepage = _https_url(_text(row.get("homepage")))
    name = _text(row.get("displayName")) or _text(detail.get("displayName")) or qualified
    description = _text(row.get("description")) or _text(detail.get("description"))
    verified = row.get("verified") is True or detail.get("verified") is True

    fields: JsonObject = {
        "name": collapse(name, limit=MAX_NAME_CHARS),
        "description": collapse(description, limit=MAX_DESCRIPTION_CHARS),
        "homepage": homepage,
        "docs_url": _first_url(homepage, record.url),
        "deprecated": row.get("inactive") is True,
        "mcp_url": mcp_url,
        "transport": "streamable_http" if mcp_url is not None else "unknown",
        "auth_hint": auth_hint,
        "auth_note": auth_note,
        "setup_note": _stdio_setup_note(packages),
        "packages": [package.as_json() for package in packages],
        "remotes": [remote.as_json() for remote in remotes],
        "smithery_qualified_name": qualified[:_MAX_QUALIFIED_NAME_CHARS],
        "verified_upstream": verified,
    }
    tool_count = _tool_count(detail.get("tools"))
    if tool_count is not None:
        fields["tool_count"] = tool_count

    keys = {name_key("smithery", qualified, kind="mcp")}
    if mcp_url is not None:
        keys.add(url_key(mcp_url, kind="mcp"))

    use_count = _count(row.get("useCount"))
    if use_count is None:
        use_count = _count(detail.get("useCount"))

    return _candidate(
        record,
        kind="mcp",
        keys=keys,
        repo=None,
        signals=PopularitySignals(smithery_use_count=use_count),
        trust_hint="smithery_verified" if verified else "indexed",
        fields=fields,
    )


def normalize_npm(record: RawRecord) -> Candidate | None:
    """One npm search hit: download counts, keywords, and a repository link.

    npm never publishes an endpoint, so an npm candidate contributes signals
    and metadata and leaves ``mcp_url`` to a source that actually knows one.
    ``dependents`` arrives as a decimal string and is cast; a value that will
    not parse is recorded as unknown rather than as zero.
    """
    item = record.payload
    package = _obj(item.get("package"))
    name = _text(package.get("name"))
    if not name:
        raise NormalizeError(f"npm record {record.upstream_id!r} has no ``package.name``")
    if _truthy(_obj(item.get("flags")).get("insecure")):
        return None

    links = _obj(package.get("links"))
    repo = parse_repo_url(_text(links.get("repository")))
    homepage = _https_url(_text(links.get("homepage")))
    artefact = _Package(
        registry_type="npm",
        identifier=name[:_MAX_IDENTIFIER_CHARS],
        version=_text(package.get("version"))[:_MAX_VERSION_CHARS],
        runtime_hint="",
        transport="stdio",
    )

    fields: JsonObject = {
        "name": collapse(name, limit=MAX_NAME_CHARS),
        "description": collapse(_text(package.get("description")), limit=MAX_DESCRIPTION_CHARS),
        "homepage": homepage,
        "docs_url": _first_url(homepage, _repo_web_url(repo), record.url),
        "tags": _tags(_strings(package.get("keywords"))),
        "license": _text(package.get("license"))[:_MAX_LICENSE_CHARS],
        "npm_package": name[:_MAX_NPM_PACKAGE_CHARS],
        "packages": [artefact.as_json()],
        "setup_note": _stdio_setup_note([artefact]),
    }
    if repo is not None:
        fields["repo"] = repo.model_dump(mode="json")

    keys = {name_key("npm", name, kind="mcp")}
    if repo is not None:
        keys.add(repo_key(repo, kind="mcp"))

    return _candidate(
        record,
        kind="mcp",
        keys=keys,
        repo=repo,
        signals=PopularitySignals(
            npm_downloads_monthly=_count(_obj(item.get("downloads")).get("monthly")),
            npm_dependents=_count(item.get("dependents")),
        ),
        trust_hint="indexed",
        fields=fields,
    )


def normalize_github_topics(record: RawRecord) -> Candidate | None:
    """One topic-search repository: stars, forks, topics, and nothing else.

    A topic is a label anyone can apply, so this source never proves an
    endpoint and never lifts an entry above ``indexed``. What it does prove is
    attention, which is the signal the whole ranking leans on.
    """
    item = record.payload
    full_name = _text(item.get("full_name"))
    html_url = _text(item.get("html_url"))
    if not full_name and not html_url:
        raise NormalizeError(f"github record {record.upstream_id!r} has no ``full_name``")

    repo = parse_repo_url(html_url) or _repo_from_full_name(full_name)
    if repo is None:
        return None

    homepage = _https_url(_text(item.get("homepage")))
    fields: JsonObject = {
        "name": collapse(_text(item.get("name")) or repo.repo, limit=MAX_NAME_CHARS),
        "description": collapse(_text(item.get("description")), limit=MAX_DESCRIPTION_CHARS),
        "homepage": homepage,
        "docs_url": _first_url(homepage, html_url, _repo_web_url(repo)),
        "tags": _tags(_strings(item.get("topics"))),
        "license": _text(_obj(item.get("license")).get("spdx_id"))[:_MAX_LICENSE_CHARS],
        "deprecated": item.get("archived") is True,
        "repo": repo.model_dump(mode="json"),
    }

    return _candidate(
        record,
        kind="mcp",
        keys={repo_key(repo, kind="mcp")},
        repo=repo,
        signals=PopularitySignals(
            github_stars=_count(item.get("stargazers_count")),
            github_forks=_count(item.get("forks_count")),
        ),
        trust_hint="indexed",
        fields=fields,
    )


def normalize_marketplace_skill(record: RawRecord) -> Candidate | None:
    """One ``SKILL.md`` found under one plugin of one marketplace manifest.

    A skill's identity is where its directory lives, not what the manifest
    that advertised it is currently called, so the repository path is the
    primary key and the marketplace-and-plugin path is an alias. A rename adds
    a second alias, which is how a renamed plugin reconciles to the row a
    previous build wrote instead of deleting it and inserting another. A skill
    whose repository cannot be resolved is dropped, because there would be
    nothing to point a reader at.
    """
    payload = record.payload
    frontmatter = _obj(_first_value(payload, "frontmatter", "skill_frontmatter", "skill"))
    plugin_obj = _obj(_first_value(payload, "plugin", "plugin_ref"))
    skill_path = _posix_relpath(
        _first_text(payload, "skill_path", "skillPath", "path"), limit=_MAX_SKILL_PATH_CHARS
    )
    if not skill_path:
        raise NormalizeError(f"marketplace record {record.upstream_id!r} has no ``skill_path``")
    if not skill_path.endswith(_SKILL_SUFFIX):
        return None
    skill_dir = skill_path[: -len(_SKILL_SUFFIX)]

    repo = _skill_repo(payload)
    if repo is None:
        return None

    declared_name = _meta_text(payload, frontmatter, "name")
    skill_name = _skill_name(declared_name, fallback=skill_dir.rsplit("/", 1)[-1])
    if not skill_name:
        return None

    marketplace = _first_text(payload, "marketplace", "marketplace_name") or _text(
        plugin_obj.get("marketplace")
    )
    marketplace = marketplace[:_MAX_MARKETPLACE_CHARS]
    marketplace_repo = _first_text(payload, "marketplace_repo", "marketplaceRepo") or _text(
        plugin_obj.get("marketplace_repo")
    )
    plugin_name = (_text(plugin_obj.get("plugin")) or _first_text(payload, "plugin_name"))[
        :_MAX_MARKETPLACE_CHARS
    ]
    commit_sha = _first_text(payload, "commit_sha", "sha", "commitSha").lower()

    fields: JsonObject = {
        "name": collapse(declared_name or skill_name, limit=MAX_NAME_CHARS),
        "description": collapse(
            _meta_text(payload, frontmatter, "description"), limit=MAX_DESCRIPTION_CHARS
        ),
        "docs_url": _first_url(record.url),
        "license": _meta_text(payload, frontmatter, "license")[:_MAX_LICENSE_CHARS],
        "repo": repo.model_dump(mode="json"),
        "skill_name": skill_name,
        "category": _skill_category(_first_text(payload, "category")),
        "source_ref": f"{repo.owner}/{repo.repo}/{skill_dir}"[:_MAX_SOURCE_REF_CHARS],
        "skill_path": skill_path,
        "commit_sha": commit_sha if SHA1_RE.fullmatch(commit_sha) else "",
        "model_invocable": _model_invocable(payload, frontmatter),
        "allowed_tools": _allowed_tools(
            _meta_value(payload, frontmatter, "allowed_tools", "allowed-tools")
        ),
        "skill_version": _meta_text(payload, frontmatter, "skill_version", "version")[
            :_MAX_SKILL_VERSION_CHARS
        ],
        "frontmatter_bytes": min(
            max(_count(_meta_value(payload, frontmatter, "frontmatter_bytes")) or 0, 0),
            _MAX_FRONTMATTER_BYTES,
        ),
        "tags": _tags(_strings(_first_value(payload, "tags", "keywords"))),
        "marketplace_reviewed": _truthy(payload.get("marketplace_reviewed")),
        "icon_url": _skill_icon_url(payload, repo),
    }
    plugin = _plugin_ref(payload, plugin_obj, marketplace, marketplace_repo, plugin_name)
    if plugin is not None:
        fields["plugin"] = plugin

    keys = {name_key("skill", f"{repo.host}/{repo.owner}/{repo.repo}/{skill_dir}", kind="skill")}
    if marketplace and plugin_name:
        aliases = _strings(_first_value(payload, "aliases", "renamed_from", "previous_plugins"))
        for alias in (plugin_name, *aliases):
            keys.add(name_key("plugin", f"{marketplace}/{alias}/{skill_name}", kind="skill"))

    return _candidate(
        record,
        kind="skill",
        keys=keys,
        repo=repo,
        signals=PopularitySignals(),
        trust_hint="indexed",
        fields=fields,
    )


# ---------------------------------------------------------------------------
# Entry construction
# ---------------------------------------------------------------------------


def build_entry(merged: MergedCandidate, *, popularity: float, slug: str) -> CatalogEntry:
    """Materialise the final, validated entry. Raises ``NormalizeError``.

    Applied in order: the merged fields, then the identity and the score the
    caller computed, then ``category`` and ``icon`` derivation, then the
    ``stdio_only`` and ``url_unverified`` recomputation, then note synthesis,
    then pydantic validation. The two recomputations are deliberate — a merge
    can carry a stale ``stdio_only`` forward from a source that never saw the
    endpoint another source published, and an entry claiming a verified URL it
    did not earn is worse than one that admits the URL is a guess.
    """
    payload: JsonObject = {
        key: value for key, value in merged.fields.items() if key not in _RECOMPUTED_FIELDS
    }
    payload["schema_version"] = SCHEMA_VERSION
    payload["kind"] = merged.kind
    payload["canonical_key"] = merged.canonical_key
    payload["alias_keys"] = _json_strings(merged.alias_keys[:MAX_ALIAS_KEYS])
    payload["slug"] = slug
    payload["trust_tier"] = merged.trust_tier
    payload["popularity"] = round(popularity, POPULARITY_DECIMALS)
    payload["popularity_signals"] = merged.signals.model_dump(mode="json")
    payload["sources"] = [ref.model_dump(mode="json") for ref in _sources(merged.candidates)]

    name = collapse(_text(payload.get("name")), limit=MAX_NAME_CHARS) or slug
    description = collapse(_text(payload.get("description")), limit=MAX_DESCRIPTION_CHARS)
    payload["name"] = name
    payload["description"] = description
    payload["homepage"] = _https_url(_text(payload.get("homepage")))
    payload["docs_url"] = _first_url(_text(payload.get("docs_url")))
    payload["license"] = _text(payload.get("license"))[:_MAX_LICENSE_CHARS]
    payload["tags"] = _tags(_strings(payload.get("tags")))
    payload["curated_fields"] = _json_strings(sorted(set(_strings(payload.get("curated_fields")))))

    if merged.kind == "mcp":
        _finish_mcp(payload, name=name, description=description, slug=slug)
    else:
        category = _text(payload.get("category"))[:_MAX_SKILL_CATEGORY_CHARS]
        payload["category"] = category or DEFAULT_SKILL_CATEGORY
        # A skill's icon is always its repository owner's avatar (rule 2 of
        # the election); a merged value of that shape stands, and anything
        # else is re-elected from the repository rather than trusted.
        icon_url = _text(payload.get("icon_url"))
        if ICON_URL_GITHUB_RE.fullmatch(icon_url) is None:
            payload["icon_url"] = elect_icon_url(payload)

    try:
        return ENTRY_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise NormalizeError(f"entry {merged.canonical_key!r} does not validate: {exc}") from exc


def _finish_mcp(payload: JsonObject, *, name: str, description: str, slug: str) -> None:
    """The MCP-only tail of ``build_entry``, applied in place on the payload.

    The two notes live here rather than in the shared prologue because a skill
    has neither: ``SkillEntry`` forbids extra keys, so writing them for every
    kind would make every skill in the corpus fail validation.
    """
    payload["auth_note"] = collapse(_text(payload.get("auth_note")), limit=MAX_NOTE_CHARS)
    payload["setup_note"] = collapse(_text(payload.get("setup_note")), limit=MAX_NOTE_CHARS)
    packages = _package_list(payload.get("packages"))
    remotes = _remote_list(payload.get("remotes"))
    payload["packages"] = packages
    payload["remotes"] = remotes

    category = _text(payload.get("category"))
    if category not in CATALOG_CATEGORIES:
        category = choose_category(
            name=name,
            description=description,
            tags=_strings(payload.get("tags")),
            extra=" ".join(
                [_text(payload.get("registry_name")), _text(payload.get("npm_package"))]
            ),
        )
    payload["category"] = category

    icon = _text(payload.get("icon"))
    payload["icon"] = icon if icon in CATALOG_ICONS else choose_icon(slug=slug, category=category)
    # Recomputed rather than merged, like the other derived fields: the
    # election is a pure function of the identity fields already on the
    # payload, and a merge could otherwise carry a stale answer forward from
    # a source that never saw the Smithery name another source published.
    payload["icon_url"] = elect_icon_url(payload)

    connector_type = _text(payload.get("connector_type"))
    payload["connector_type"] = connector_type if connector_type in CONNECTOR_TYPES else None
    payload["connector_config"] = _connector_config(payload.get("connector_config"))

    mcp_url = _endpoint_url(_text(payload.get("mcp_url")))
    payload["mcp_url"] = mcp_url
    payload["transport"] = _elected_transport(mcp_url, remotes, _text(payload.get("transport")))

    auth_hint = _text(payload.get("auth_hint"))
    payload["auth_hint"] = auth_hint if auth_hint in _AUTH_HINTS else "bearer"

    stdio_only = mcp_url is None and bool(packages)
    payload["stdio_only"] = stdio_only

    remote_urls = {_text(_obj(remote).get("url")) for remote in remotes}
    tier = _text(payload.get("trust_tier"))
    trusted = tier == "curated" or (tier == "registry_verified" and mcp_url in remote_urls)
    payload["url_unverified"] = _url_unverified(
        mcp_url, connector_type=payload["connector_type"], trusted=trusted
    )

    if stdio_only and not payload.get("setup_note"):
        identifier = _text(_obj(packages[0]).get("identifier"))
        payload["setup_note"] = collapse(
            f"Jhin does not spawn stdio servers. Host {identifier} yourself "
            "and connect over HTTPS.",
            limit=MAX_NOTE_CHARS,
        )

    tool_count = _count(payload.get("tool_count"))
    payload["tool_count"] = None if tool_count is None else min(tool_count, _MAX_TOOL_COUNT)


def _url_unverified(mcp_url: str | None, *, connector_type: JsonValue, trusted: bool) -> bool:
    """Whether the endpoint on offer is a guess a person should check.

    An endpoint that came from a registry remote or from the curated file is
    verified; anything else that names a URL is not. A record with no URL at
    all but a native connector has nothing to verify — the connection does not
    go through a published endpoint — while a record with neither is offering
    a guess by omission, and says so.
    """
    if mcp_url is not None:
        return not trusted
    return connector_type is None


def _elected_transport(mcp_url: str | None, remotes: Sequence[JsonValue], declared: str) -> str:
    """The transport of the remote the endpoint actually came from.

    A merge can take ``mcp_url`` from one source and ``transport`` from a
    higher-ranked source that never published an endpoint, which would leave
    an entry advertising an HTTPS URL over ``unknown``. The remote list is the
    tie-breaker, and an endpoint no remote claims keeps whatever the sources
    declared.
    """
    if mcp_url is None:
        return "unknown"
    for remote in remotes:
        entry = _obj(remote)
        if _text(entry.get("url")) == mcp_url:
            transport = _text(entry.get("transport"))
            if transport in _TRANSPORTS:
                return transport
    return declared if declared in _TRANSPORTS else "unknown"


# ---------------------------------------------------------------------------
# Private helpers: candidates
# ---------------------------------------------------------------------------


def _candidate(
    record: RawRecord,
    *,
    kind: str,
    keys: set[str],
    repo: RepoRef | None,
    signals: PopularitySignals,
    trust_hint: str,
    fields: JsonObject,
) -> Candidate:
    """One ``Candidate``, with its keys sorted and its primary key elected."""
    alias_keys = tuple(sorted(keys))
    upstream_id = record.upstream_id[:_MAX_UPSTREAM_ID_CHARS]
    try:
        source_ref = SourceRef(
            source_id=record.source_id,
            upstream_id=upstream_id,
            url=record.url[:MAX_URL_CHARS],
        )
    except ValidationError as exc:
        raise NormalizeError(f"record {record.upstream_id!r} has an unusable source URL") from exc
    return Candidate(
        kind=kind,
        source_id=record.source_id,
        upstream_id=upstream_id,
        primary_key=_elect(alias_keys),
        alias_keys=alias_keys,
        repo=repo,
        signals=signals,
        trust_hint=trust_hint,
        source_ref=source_ref,
        fields={key: value for key, value in fields.items() if key not in _RECOMPUTED_FIELDS},
    )


def _sources(candidates: Sequence[Candidate]) -> tuple[SourceRef, ...]:
    """One reference per contributing record, deduped and source-ranked."""
    seen: dict[tuple[str, str], SourceRef] = {}
    for candidate in candidates:
        ref = candidate.source_ref
        seen.setdefault((ref.source_id, ref.upstream_id), ref)
    ordered = sorted(
        seen.values(),
        key=lambda ref: (SOURCE_RANK.get(ref.source_id, len(SOURCE_RANK)), ref.upstream_id),
    )
    return tuple(ordered[:MAX_SOURCES])


# ---------------------------------------------------------------------------
# Private helpers: registry and smithery detail
# ---------------------------------------------------------------------------


def _registry_remotes(server: JsonObject) -> list[_Remote]:
    """``server.remotes[]`` as ``_Remote`` values, with header names bounded."""
    remotes: list[_Remote] = []
    for raw in _list(server.get("remotes")):
        entry = _obj(raw)
        url = collapse(_text(entry.get("url")), limit=MAX_URL_CHARS)
        if not url:
            continue
        names: list[str] = []
        lowered: set[str] = set()
        for header in _list(entry.get("headers")):
            header_name = _text(_obj(header).get("name"))
            if header_name and header_name.lower() not in lowered:
                lowered.add(header_name.lower())
                names.append(header_name)
        remotes.append(
            _Remote(
                transport=_REMOTE_TRANSPORT_MAP.get(_text(entry.get("type")).lower(), "unknown"),
                url=url,
                templated="{" in url,
                header_names=tuple(sorted(names))[:_MAX_HEADER_NAMES],
            )
        )
    return remotes


def _registry_packages(server: JsonObject) -> list[_Package]:
    """``server.packages[]`` as ``_Package`` values, sorted and bounded."""
    packages: list[_Package] = []
    for raw in _list(server.get("packages")):
        entry = _obj(raw)
        identifier = _text(entry.get("identifier")) or _text(entry.get("name"))
        if not identifier:
            continue
        # ``registryType`` is required by the registry's own schema and is
        # nonetheless observed missing. ``PackageRef`` will not accept an
        # empty one, and one malformed row must not abort the whole build,
        # so an unstated registry is recorded as unknown.
        registry_type = (
            _text(entry.get("registryType"))
            or _text(entry.get("registry_type"))
            or _text(entry.get("registryName"))
            or _text(entry.get("registry_name"))
            or _UNKNOWN_REGISTRY_TYPE
        ).lower()[:_MAX_REGISTRY_TYPE_CHARS]
        runtime_hint = _text(entry.get("runtimeHint")) or _text(entry.get("runtime_hint"))
        declared = _text(_obj(entry.get("transport")).get("type")).lower()
        packages.append(
            _Package(
                registry_type=registry_type,
                identifier=identifier[:_MAX_IDENTIFIER_CHARS],
                version=_text(entry.get("version"))[:_MAX_VERSION_CHARS],
                runtime_hint=runtime_hint[:_MAX_RUNTIME_HINT_CHARS],
                transport=_PACKAGE_TRANSPORT_MAP.get(declared, "unknown") if declared else "stdio",
            )
        )
    packages.sort(key=lambda package: (package.registry_type, package.identifier))
    return packages[:MAX_PACKAGES]


def _registry_auth(elected: _Remote | None, ordered: Sequence[_Remote]) -> tuple[str, str]:
    """The auth hint and note implied by the elected remote's headers.

    A remote naming ``Authorization`` wants a bearer token; a remote naming
    some other header wants that header, and the note says which one. When no
    remote could be elected because the published endpoint is a template, the
    note says that instead — a templated URL is a real answer to "what is the
    endpoint", just not one any client can dial.
    """
    if elected is not None:
        lowered = {header.lower() for header in elected.header_names}
        if "authorization" in lowered:
            return "bearer", ""
        if elected.header_names:
            note = f"Set the {elected.header_names[0]} header from the provider's dashboard."
            return "header", collapse(note, limit=MAX_NOTE_CHARS)
        return "none", ""
    if ordered and ordered[0].templated:
        note = (
            f"The published endpoint is templated ({ordered[0].url}); "
            "enter the concrete URL from the provider's docs."
        )
        return "bearer", collapse(note, limit=MAX_NOTE_CHARS)
    return "bearer", ""


def _smithery_auth(connections: Sequence[JsonObject]) -> tuple[str, str]:
    """The auth hint and note implied by a Smithery connection's config schema.

    Only the property names and each property's ``x-to.header`` are read; the
    schema itself is never stored. A property mentioning a key, a token, or
    auth means a secret, and a secret on an MCP connection is a bearer token
    far more often than it is anything else.

    A server that declares no connections at all has told us nothing, which is
    not the same as telling us it needs no credential. Publishing ``none``
    there invites a person to dial an endpoint unauthenticated and be refused,
    so silence falls back to the same conservative guess a missing detail
    document gets. ``none`` is reserved for a server that declared a
    connection and put no configuration on it.
    """
    if not connections:
        return "bearer", ""
    properties: dict[str, JsonObject] = {}
    for connection in connections:
        schema = _obj(connection.get("configSchema"))
        for key, value in _obj(schema.get("properties")).items():
            properties.setdefault(key, _obj(value))
    if not properties:
        return "none", ""

    parts: list[str] = []
    for key in sorted(properties):
        parts.append(key)
        header = _text(_obj(properties[key].get("x-to")).get("header"))
        if header:
            parts.append(header)
    haystack = " ".join(parts).lower()
    hint = "bearer" if any(word in haystack for word in _SECRET_WORDS) else "header"
    note = (
        f"This server takes configuration ({len(properties)} field(s)); "
        "supply values from the provider's docs."
    )
    return hint, collapse(note, limit=MAX_NOTE_CHARS)


def _stdio_setup_note(packages: Sequence[_Package]) -> str:
    """What a person has to do before a stdio-only server is reachable."""
    if not packages:
        return ""
    note = (
        f"Jhin does not spawn stdio servers. Host {packages[0].identifier} "
        "yourself and connect over HTTPS."
    )
    return collapse(note, limit=MAX_NOTE_CHARS)


def _tool_count(value: JsonValue) -> int | None:
    """``len(tools)`` when the detail pass saw a list, ``None`` when it did not."""
    if not isinstance(value, list):
        return None
    return min(len(value), _MAX_TOOL_COUNT)


# ---------------------------------------------------------------------------
# Private helpers: skills
# ---------------------------------------------------------------------------


def _skill_icon_url(payload: JsonObject, repo: RepoRef) -> str:
    """A skill's icon URL: the crawl's value when it is well-shaped, else rule 2.

    The crawl writes the repository owner's avatar onto the payload, and this
    keeps it. The re-election covers a payload recorded before the crawl
    learnt to — the field is derivable from the repository the skill already
    proves, so an older recording loses nothing.
    """
    declared = _text(payload.get("icon_url"))
    if ICON_URL_GITHUB_RE.fullmatch(declared) is not None:
        return declared
    return github_avatar_url(repo.owner) if repo.host == "github.com" else ""


def _skill_repo(payload: JsonObject) -> RepoRef | None:
    """The repository a skill's text actually lives in, from whatever says so."""
    raw = _first_value(payload, "repo", "repository")
    if isinstance(raw, dict):
        try:
            return RepoRef.model_validate(raw)
        except ValidationError:
            pass
    if isinstance(raw, str) and raw.strip():
        resolved = parse_repo_url(raw) or _repo_from_full_name(raw)
        if resolved is not None:
            return resolved
    source = _obj(_first_value(payload, "plugin_source", "source")) or _obj(
        _first_value(payload, "plugin", "plugin_ref")
    )
    for name in ("source_value", "url", "repo"):
        value = _text(source.get(name))
        if value:
            resolved = parse_repo_url(value) or _repo_from_full_name(value)
            if resolved is not None:
                return resolved
    return _repo_from_full_name(
        _first_text(payload, "marketplace_repo", "marketplaceRepo")
        or _text(source.get("marketplace_repo"))
    )


def _plugin_ref(
    payload: JsonObject,
    plugin_obj: JsonObject,
    marketplace: str,
    marketplace_repo: str,
    plugin: str,
) -> JsonObject | None:
    """The ``PluginRef`` payload, or ``None`` when the manifest did not say.

    ``marketplace_repo`` is used whole or not at all: it is validated against
    the owner/repo grammar, and a truncated repository name would either fail
    that check or — worse — pass it while naming a different repository.
    """
    if not (marketplace and marketplace_repo and plugin):
        return None
    if len(marketplace_repo) > _MAX_OWNER_REPO_CHARS:
        return None
    source = _obj(_first_value(payload, "plugin_source", "source")) or plugin_obj
    kind = _text(source.get("source_kind")) or _text(source.get("kind"))
    sha = (_text(source.get("sha")) or _first_text(payload, "commit_sha", "sha")).lower()
    return {
        "marketplace": marketplace,
        "marketplace_repo": marketplace_repo,
        "plugin": plugin,
        "source_kind": kind if kind in _PLUGIN_SOURCE_KINDS else "unknown",
        "source_value": _text(source.get("source_value"))[:MAX_URL_CHARS],
        "sha": sha if SHA1_RE.fullmatch(sha) else "",
    }


def _model_invocable(payload: JsonObject, frontmatter: JsonObject) -> bool:
    """Whether the model may invoke this skill unprompted.

    The crawl states the resolved answer as ``model_invocable``; raw
    frontmatter states the opposite as ``disable-model-invocation``. Reading
    only the second would publish every opted-out skill as invocable.
    """
    stated = _meta_value(payload, frontmatter, "model_invocable")
    if stated is not None:
        return _truthy(stated)
    return not _truthy(
        _meta_value(payload, frontmatter, "disable_model_invocation", "disable-model-invocation")
    )


def _skill_name(raw: str, *, fallback: str) -> str:
    """The frontmatter name, or the directory basename when it will not do."""
    for candidate in (raw.strip().lower(), fallback.strip().lower()):
        if not candidate or len(candidate) > _MAX_SKILL_NAME_CHARS:
            continue
        if SKILL_NAME_RE.fullmatch(candidate):
            return candidate
    return ""


def _skill_category(value: str) -> str:
    """A marketplace category, title-cased, or the default when there is none.

    A word that already carries capitals keeps them, so ``CRM`` does not come
    back as ``Crm``.
    """
    words = collapse(value, limit=_MAX_SKILL_CATEGORY_CHARS).split()
    if not words:
        return DEFAULT_SKILL_CATEGORY
    titled = " ".join(word.capitalize() if word.islower() else word for word in words)
    return titled[:_MAX_SKILL_CATEGORY_CHARS]


def _allowed_tools(value: JsonValue) -> list[JsonValue]:
    """``allowed-tools`` as a sorted, unique list, from a string or a list."""
    raw: list[str] = []
    if isinstance(value, str):
        raw = value.split(",")
    elif isinstance(value, list):
        raw = [item for item in value if isinstance(item, str)]
    names = {tool.strip()[:_MAX_ALLOWED_TOOL_CHARS] for tool in raw if tool.strip()}
    return _json_strings(sorted(names)[:_MAX_ALLOWED_TOOLS])


# ---------------------------------------------------------------------------
# Private helpers: values
# ---------------------------------------------------------------------------


def _obj(value: JsonValue) -> JsonObject:
    """A JSON object, or an empty one when the value is anything else."""
    return value if isinstance(value, dict) else {}


def _list(value: JsonValue) -> list[JsonValue]:
    """A JSON array, or an empty one when the value is anything else."""
    return value if isinstance(value, list) else []


def _text(value: JsonValue) -> str:
    """A trimmed string, or ``""`` when the value is not a string."""
    return value.strip() if isinstance(value, str) else ""


def _strings(value: JsonValue) -> list[str]:
    """Every non-blank string in a JSON array, in order, others discarded."""
    return [item.strip() for item in _list(value) if isinstance(item, str) and item.strip()]


def _json_strings(values: Iterable[str]) -> list[JsonValue]:
    """Strings widened to JSON values so they can sit in an entry payload."""
    return list(values)


def _count(value: JsonValue) -> int | None:
    """A non-negative integer from an int or a decimal string, else ``None``.

    npm reports ``dependents`` as a string, so the cast is not optional; a
    value that will not parse is unknown rather than zero, because zero is a
    claim about the world and silence is not.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _truthy(value: JsonValue) -> bool:
    """Whether a JSON flag means yes, tolerating ``1``, ``"true"``, and ``"yes"``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _first_value(payload: JsonObject, *names: str) -> JsonValue:
    """The first of several spellings a source might have used for one field."""
    for name in names:
        value = payload.get(name)
        if value is not None:
            return value
    return None


def _first_text(payload: JsonObject, *names: str) -> str:
    """``_first_value`` narrowed to the first non-blank string."""
    for name in names:
        text = _text(payload.get(name))
        if text:
            return text
    return ""


def _meta_value(payload: JsonObject, frontmatter: JsonObject, *names: str) -> JsonValue:
    """One ``SKILL.md`` metadata field, wherever the record chose to put it.

    ``sources.marketplaces`` clips the seven frontmatter scalars as it reads
    them and writes them flat on the payload; a record that carries the raw
    block instead nests them under ``frontmatter``. Both are read, flat first,
    so neither shape silently loses a field.
    """
    value = _first_value(payload, *names)
    return value if value is not None else _first_value(frontmatter, *names)


def _meta_text(payload: JsonObject, frontmatter: JsonObject, *names: str) -> str:
    """``_meta_value`` narrowed to the first non-blank string in either place."""
    return _first_text(payload, *names) or _first_text(frontmatter, *names)


def _tags(values: Sequence[str]) -> list[JsonValue]:
    """Tags lowercased, filtered to the tag grammar, sorted, and bounded."""
    kept = {
        tag.strip().lower()[:MAX_TAG_CHARS]
        for tag in values
        if _TAG_RE.fullmatch(tag.strip().lower()[:MAX_TAG_CHARS])
    }
    return _json_strings(sorted(kept)[:MAX_TAGS])


def _connector_config(value: JsonValue) -> JsonObject:
    """Non-secret pre-fill values, bounded to the ``CatalogApp`` limits."""
    pairs = {
        key[:_MAX_CONNECTOR_CONFIG_KEY_CHARS]: item[:_MAX_CONNECTOR_CONFIG_VALUE_CHARS]
        for key, item in _obj(value).items()
        if isinstance(item, str) and key
    }
    return {key: pairs[key] for key in sorted(pairs)[:_MAX_CONNECTOR_CONFIG_PAIRS]}


def _package_list(value: JsonValue) -> list[JsonValue]:
    """Merged packages, re-sorted and re-bounded before validation."""
    items = [item for item in _list(value) if isinstance(item, dict)]
    items.sort(key=lambda item: (_text(item.get("registry_type")), _text(item.get("identifier"))))
    return list(items[:MAX_PACKAGES])


def _remote_list(value: JsonValue) -> list[JsonValue]:
    """Merged remotes, re-sorted and re-bounded before validation."""
    items = [item for item in _list(value) if isinstance(item, dict)]
    items.sort(key=lambda item: (_text(item.get("transport")), _text(item.get("url"))))
    return list(items[:MAX_REMOTES])


def _https_url(value: str) -> str:
    """An ``https://`` URL within the length bound, or ``""``."""
    text = collapse(value, limit=MAX_URL_CHARS + 1)
    if len(text) > MAX_URL_CHARS or not text.startswith("https://"):
        return ""
    return text


def _first_url(*candidates: str) -> str:
    """The first ``http(s)://`` URL among the candidates, or ``""``."""
    for candidate in candidates:
        text = collapse(candidate, limit=MAX_URL_CHARS + 1)
        if len(text) <= MAX_URL_CHARS and text.startswith(("https://", "http://")):
            return text
    return ""


def _endpoint_url(value: str) -> str | None:
    """A dialable ``mcp_url``, or ``None`` when the URL is not one.

    Rejected: anything not ``https``, anything holding a ``{template}``
    segment, anything carrying userinfo before the host, and anything with a
    fragment. Each of those is a URL a client cannot connect to as written,
    and offering it as an endpoint would only waste a person's time.
    """
    text = collapse(value, limit=MAX_URL_CHARS + 1)
    if len(text) > MAX_URL_CHARS or not text.startswith("https://"):
        return None
    if "{" in text or "#" in text:
        return None
    authority = text[len("https://") :].split("/", 1)[0]
    if not authority or "@" in authority:
        return None
    return text


def _repo_web_url(repo: RepoRef | None) -> str:
    """The human page for a repository, or ``""`` when there is no repository."""
    if repo is None:
        return ""
    return f"https://{repo.host}/{repo.owner}/{repo.repo}"


def _repo_from_full_name(value: str) -> RepoRef | None:
    """``owner/repo`` on GitHub, which is how several sources name a repository."""
    text = value.strip().removesuffix(".git")
    segments = [segment for segment in text.split("/") if segment]
    if len(segments) < 2:
        return None
    owner, repo = segments[0], segments[1]
    if not _OWNER_RE.fullmatch(owner) or not _REPO_RE.fullmatch(repo):
        return None
    try:
        return RepoRef(host="github.com", owner=owner.lower(), repo=repo.lower(), subpath="")
    except ValidationError:
        return None


def _with_subpath(repo: RepoRef | None, subpath: str) -> RepoRef | None:
    """The same repository, narrowed to a subdirectory when one was declared."""
    cleaned = _posix_relpath(subpath, limit=_MAX_SUBPATH_CHARS)
    if repo is None or not cleaned or repo.subpath:
        return repo
    try:
        return RepoRef(host=repo.host, owner=repo.owner, repo=repo.repo, subpath=cleaned)
    except ValidationError:
        return repo


def _posix_relpath(value: str, *, limit: int) -> str:
    """A relative POSIX path with no ``..``, no anchors, and no over-long tail."""
    text = collapse(value, limit=limit + 1).replace("\\", "/").strip("/")
    if not text or len(text) > limit:
        return ""
    segments = [segment for segment in text.split("/") if segment and segment != "."]
    if any(segment == ".." for segment in segments):
        return ""
    return "/".join(segments)
