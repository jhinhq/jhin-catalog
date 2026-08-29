"""Canonical record types, shared constants, and the byte-exact JSONL form.

Everything else in the package depends on this module and it depends on
nothing in the package.  It fixes ``McpEntry`` and ``SkillEntry``, the
identity and provenance sub-models they are built from, the intermediate
shapes the pipeline passes between stages, and the single JSON encoding
``data/**`` is written in.  Nothing here reads a clock, an environment
variable, or a file, so the same inputs always yield the same bytes.

Two validation postures run side by side.  Free text that upstreams write
carelessly -- names, descriptions, notes, tags -- is collapsed, filtered,
and truncated into range.  Identifiers, enums, URLs, and keys are checked
and rejected, because a malformed one is a bug in the pipeline rather than
noise in a crawl.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Annotated, Any, Final, Literal, NoReturn, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

SCHEMA_VERSION: Final[int] = 1
SHARD_HEX_WIDTH: Final[int] = 2
SHARD_COUNT: Final[int] = 256

KINDS: Final[tuple[str, ...]] = ("mcp", "skill")

TRUST_TIERS: Final[tuple[str, ...]] = (
    "curated",
    "registry_verified",
    "smithery_verified",
    "indexed",
)
TRUST_RANK: Final[Mapping[str, int]] = {t: i for i, t in enumerate(TRUST_TIERS)}

SOURCE_IDS: Final[tuple[str, ...]] = (
    "curated",
    "registry",
    "smithery",
    "npm",
    "github_topics",
    "marketplaces",
)
SOURCE_RANK: Final[Mapping[str, int]] = {s: i for i, s in enumerate(SOURCE_IDS)}

CATALOG_CATEGORIES: Final[tuple[str, ...]] = (
    "Developer tools",
    "Project management",
    "Communication",
    "Documents & knowledge",
    "Payments & commerce",
    "CRM & support",
    "Design",
    "Search & web",
    "Data & infrastructure",
    "Automation",
    "Productivity",
    "Storage",
)
CATALOG_ICONS: Final[frozenset[str]] = frozenset(
    {
        "github",
        "linear",
        "vercel",
        "terminal",
        "mcp",
        "notebook",
        "message-square",
        "message-circle",
        "kanban",
        "credit-card",
        "users",
        "bug",
        "cloud",
        "life-buoy",
        "zap",
        "check-square",
        "palette",
        "pen-tool",
        "folder",
        "calendar",
        "mail",
        "table",
        "database",
        "globe",
        "hard-drive",
        "search",
        "web",
        "flame",
        "book-open",
        "send",
        "phone",
        "cpu",
        "flask",
    }
)
CONNECTOR_TYPES: Final[frozenset[str]] = frozenset(
    {"github", "linear", "supabase", "vercel", "http", "web", "mcp", "cli", "example"}
)
DEFAULT_SKILL_CATEGORY: Final[str] = "General"

# The only two URL shapes ``icon_url`` may hold. The bound is the SSRF
# posture: a consumer's icon proxy dials whatever this field names, so the
# field names nothing but Smithery's own icon route and GitHub's owner
# avatar — two hosts a person has reviewed, not whatever a publisher typed
# into a manifest. Anchored and full-matched, so ``api.smithery.ai.evil.com``
# and a path that wanders off ``/icon`` both fail.
ICON_URL_SMITHERY_RE: Final[re.Pattern[str]] = re.compile(
    r"^https://api\.smithery\.ai/servers/[^/?#\s]+(/[^/?#\s]+)*/icon$"
)
ICON_URL_GITHUB_RE: Final[re.Pattern[str]] = re.compile(
    r"^https://github\.com/[A-Za-z0-9-]{1,39}\.png\?size=128$"
)

SERVER_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9_]{1,32}$")
SKILL_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
SHA1_RE: Final[re.Pattern[str]] = re.compile(r"^[a-f0-9]{40}$")
SOURCE_REF_RE: Final[re.Pattern[str]] = re.compile(
    r"^([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)(?:/(.+))?$"
)
CANONICAL_KEY_RE: Final[re.Pattern[str]] = re.compile(r"^(mcp|skill):[a-z0-9]+:[\x21-\x7e]{1,220}$")

MAX_NAME_CHARS: Final[int] = 120
MAX_DESCRIPTION_CHARS: Final[int] = 500
MAX_SUMMARY_CHARS: Final[int] = 160
MAX_NOTE_CHARS: Final[int] = 500
MAX_URL_CHARS: Final[int] = 512
MAX_TAGS: Final[int] = 20
MAX_TAG_CHARS: Final[int] = 40
MAX_SOURCES: Final[int] = 8
MAX_ALIAS_KEYS: Final[int] = 16
MAX_PACKAGES: Final[int] = 8
MAX_REMOTES: Final[int] = 4
MAX_CONNECTOR_CONFIG: Final[int] = 10
MAX_CONNECTOR_CONFIG_KEY_CHARS: Final[int] = 64
MAX_CONNECTOR_CONFIG_VALUE_CHARS: Final[int] = 500

# ``None`` trails the union so ruff's RUF036 is satisfied; the members are
# otherwise exactly the JSON value space.
type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None
type JsonObject = dict[str, JsonValue]

_MODEL_CONFIG: Final[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

_REPO_HOSTS: Final[frozenset[str]] = frozenset({"github.com", "gitlab.com", "bitbucket.org"})

_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_OWNER_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_REPO_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
_OWNER_REPO_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.-]{1,64}/[A-Za-z0-9_.-]{1,100}$")
# A note whose last word is a URL or a ``host:port``. ``_terminate`` leaves
# those alone: the full stop would become part of what the reader pastes.
_TRAILING_ADDRESS_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:\w+://\S+|\b[\w.-]+:\d{1,5}(?:/\S*)?)$"
)
_TAG_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_HEX64_RE: Final[re.Pattern[str]] = re.compile(r"^[a-f0-9]{64}$")
_RFC3339_SECONDS_RE: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

# ---------------------------------------------------------------------------
# Constraints the published JSON Schema has to carry
# ---------------------------------------------------------------------------
#
# ``build.render_schema`` derives ``schema/catalog.schema.json`` from these
# models, and pydantic can only express what a field's *type* says. Every rule
# below lives in a ``field_validator``, so without these fragments the schema
# published for downstream consumers placed no constraint at all on the
# category, the icon, the connector type, the slug, or the endpoint — and that
# schema is what a Jhin deployment validates against before it writes a row
# into Postgres. The validators remain the authority; these restate them for
# readers who only ever see the schema.
#
# Adding a rule to a validator without adding it here silently weakens the
# published schema, which is why they sit next to each other in this file.

_CONST_SCHEMA_VERSION: Final[dict[str, Any]] = {"const": SCHEMA_VERSION}
_KEY_SCHEMA: Final[dict[str, Any]] = {
    "pattern": CANONICAL_KEY_RE.pattern,
    "description": "``<kind>:<space>:<identity>``, the stable identity of one record.",
}
_ALIAS_KEYS_SCHEMA: Final[dict[str, Any]] = {
    "items": {"type": "string", "pattern": CANONICAL_KEY_RE.pattern},
    "maxItems": MAX_ALIAS_KEYS,
    "uniqueItems": True,
    "description": "Other keys that resolve to this record, sorted and deduplicated.",
}
_SLUG_SCHEMA: Final[dict[str, Any]] = {
    "pattern": SERVER_SLUG_RE.pattern,
    "description": "The name Jhin's UI and its stored connections resolve by.",
}
_SOURCES_SCHEMA: Final[dict[str, Any]] = {
    "minItems": 1,
    "maxItems": MAX_SOURCES,
    "description": "Every upstream that contributed to this record, strongest first.",
}
_CATEGORY_SCHEMA: Final[dict[str, Any]] = {
    "enum": list(CATALOG_CATEGORIES),
    "description": "One of Jhin's twelve Apps-library categories.",
}
_ICON_SCHEMA: Final[dict[str, Any]] = {
    "enum": sorted(CATALOG_ICONS),
    "description": "One of the icon tokens Jhin's Apps library can render.",
}
_CONNECTOR_TYPE_SCHEMA: Final[dict[str, Any]] = {
    "enum": [*sorted(CONNECTOR_TYPES), None],
    "description": "The native Jhin connector this server maps to, when one exists.",
}
_MCP_URL_SCHEMA: Final[dict[str, Any]] = {
    "pattern": r"^https://[^\s@{}#]+$",
    "maxLength": MAX_URL_CHARS,
    "description": (
        "A concrete https endpoint: no template placeholder, no fragment, no userinfo. "
        "``null`` when the record names no endpoint a person could dial."
    ),
}
_SKILL_NAME_SCHEMA: Final[dict[str, Any]] = {
    "pattern": SKILL_NAME_RE.pattern,
    "description": "The skill's own name, as its ``SKILL.md`` frontmatter declares it.",
}
_ICON_URL_SCHEMA: Final[dict[str, Any]] = {
    "maxLength": MAX_URL_CHARS,
    "description": (
        "The upstream image a deployment's icon proxy may fetch for this record: "
        "a Smithery server icon or a GitHub owner avatar, and nothing else. "
        '``""`` when no logo can be named. Consumers re-validate and proxy this '
        "URL server-side; it is never handed to a browser."
    ),
}


class CatalogError(Exception):
    """Base for every failure this package raises deliberately."""


class NormalizeError(CatalogError):
    """A record could not be turned into a valid entry."""


class DedupeError(CatalogError):
    """Two records that cannot be the same thing were asked to merge."""


class CuratedError(CatalogError):
    """A hand-written overlay in ``curated/`` is wrong or stale."""


def _clean(value: str) -> str:
    """Drop control characters and collapse every whitespace run to one space."""
    kept = "".join(ch for ch in value if ch.isprintable() or ch.isspace())
    return _WHITESPACE_RE.sub(" ", kept).strip()


def _collapse(value: str, limit: int) -> str:
    """Clean ``value`` and cut it to ``limit`` characters without a ragged edge."""
    return _clean(value)[:limit].rstrip()


def _terminate(value: str) -> str:
    """End a non-empty note with a full stop so it reads as a sentence.

    A note ending in a URL or a host:port is left alone. Half the curated
    setup notes end by naming the address a person is meant to paste, and a
    full stop welded onto ``http://fake-mcp:8080/mcp`` is not punctuation any
    more — it is part of what the reader copies, and it is wrong.
    """
    if not value or value.endswith(".") or _TRAILING_ADDRESS_RE.search(value):
        return value
    return f"{value}."


def _require(condition: bool, message: str) -> None:
    """Raise ``ValueError`` with ``message`` unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


def _match(pattern: re.Pattern[str], value: str, label: str) -> str:
    """Full-match ``value`` against ``pattern``.

    ``fullmatch`` rather than ``match``, because Python's ``$`` also matches
    in front of a trailing newline and a key carrying one would land in the
    wrong shard file.
    """
    if pattern.fullmatch(value) is None:
        raise ValueError(f"{label} must match {pattern.pattern!r}")
    return value


def _https_url(value: str, label: str, *, allow_http: bool = False) -> str:
    """Check an absolute, length-bounded web URL, or the empty string."""
    if value == "":
        return value
    schemes = ("https://", "http://") if allow_http else ("https://",)
    _require(value.startswith(schemes), f"{label} must start with {' or '.join(schemes)}")
    _require(len(value) <= MAX_URL_CHARS, f"{label} exceeds {MAX_URL_CHARS} characters")
    return value


def _reject_constant(name: str) -> NoReturn:
    """Refuse ``NaN`` and the infinities, which canonical JSON cannot encode."""
    raise ValueError(f"{name} is not a permitted JSON value")


def _relative_posix(value: str, label: str, *, limit: int) -> str:
    """Check a relative POSIX path with no escape upwards and no empty segment."""
    _require(len(value) <= limit, f"{label} exceeds {limit} characters")
    _require("\\" not in value, f"{label} must use POSIX separators")
    _require(not value.startswith("/"), f"{label} must be relative")
    _require(not value.endswith("/"), f"{label} must not end with a separator")
    segments = value.split("/")
    _require(all(segment not in ("", ".", "..") for segment in segments), f"{label} is not normal")
    return value


class RepoRef(BaseModel):
    """The source repository an entry points at, reduced to a stable triple."""

    model_config = _MODEL_CONFIG

    host: str
    owner: str
    repo: str
    subpath: str = ""

    @field_validator("host")
    @classmethod
    def _check_host(cls, value: str) -> str:
        lowered = value.lower()
        _require(lowered in _REPO_HOSTS, f"host must be one of {sorted(_REPO_HOSTS)}")
        return lowered

    @field_validator("owner")
    @classmethod
    def _check_owner(cls, value: str) -> str:
        return _match(_OWNER_RE, value, "owner").lower()

    @field_validator("repo")
    @classmethod
    def _check_repo(cls, value: str) -> str:
        trimmed = value.removesuffix(".git")
        return _match(_REPO_NAME_RE, trimmed, "repo").lower()

    @field_validator("subpath")
    @classmethod
    def _check_subpath(cls, value: str) -> str:
        if value == "":
            return value
        return _relative_posix(value, "subpath", limit=255)


class SourceRef(BaseModel):
    """Which upstream contributed a record, and where a human can read it."""

    model_config = _MODEL_CONFIG

    source_id: str
    upstream_id: str = Field(min_length=1, max_length=200)
    url: str

    @field_validator("source_id")
    @classmethod
    def _check_source_id(cls, value: str) -> str:
        _require(value in SOURCE_RANK, f"source_id must be one of {list(SOURCE_IDS)}")
        return value

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        _require(value != "", "url is required")
        return _https_url(value, "url")


class PopularitySignals(BaseModel):
    """Raw counts from the upstreams, stored unscaled so scoring can change."""

    model_config = _MODEL_CONFIG

    github_stars: int | None = Field(default=None, ge=0)
    github_forks: int | None = Field(default=None, ge=0)
    npm_downloads_monthly: int | None = Field(default=None, ge=0)
    npm_dependents: int | None = Field(default=None, ge=0)
    smithery_use_count: int | None = Field(default=None, ge=0)
    registry_version_count: int | None = Field(default=None, ge=0)


class PackageRef(BaseModel):
    """One installable artefact a server publishes, as a pointer only."""

    model_config = _MODEL_CONFIG

    registry_type: str = Field(min_length=1, max_length=32)
    identifier: str = Field(min_length=1, max_length=200)
    version: str = Field(default="", max_length=64)
    runtime_hint: str = Field(default="", max_length=32)
    transport: Literal["stdio", "streamable_http", "sse", "unknown"] = "stdio"

    @field_validator("registry_type")
    @classmethod
    def _lower_registry_type(cls, value: str) -> str:
        return value.lower()


class RemoteRef(BaseModel):
    """One hosted endpoint a server advertises, template segments included."""

    model_config = _MODEL_CONFIG

    transport: Literal["streamable_http", "sse", "unknown"]
    url: str = Field(min_length=1, max_length=MAX_URL_CHARS)
    templated: bool = False
    header_names: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _derive_templated(cls, data: object) -> object:
        """Force ``templated`` to agree with ``url`` rather than trusting input."""
        if isinstance(data, dict):
            url = data.get("url")
            if isinstance(url, str):
                return {**data, "templated": "{" in url}
        return data

    @field_validator("header_names")
    @classmethod
    def _check_header_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        seen: dict[str, str] = {}
        for name in value:
            cleaned = name.strip()
            if cleaned and cleaned.lower() not in seen:
                seen[cleaned.lower()] = cleaned
        return tuple(sorted(seen.values()))[:8]


class PluginRef(BaseModel):
    """The Claude Code plugin a skill was published inside."""

    model_config = _MODEL_CONFIG

    marketplace: str = Field(min_length=1, max_length=100)
    marketplace_repo: str
    plugin: str = Field(min_length=1, max_length=100)
    source_kind: Literal["relative", "url", "git-subdir", "github", "npm", "unknown"]
    source_value: str = Field(max_length=MAX_URL_CHARS)
    sha: str = ""

    @field_validator("marketplace_repo")
    @classmethod
    def _check_marketplace_repo(cls, value: str) -> str:
        return _match(_OWNER_REPO_RE, value, "marketplace_repo")

    @field_validator("sha")
    @classmethod
    def _check_sha(cls, value: str) -> str:
        if value == "":
            return value
        return _match(SHA1_RE, value.lower(), "sha")


class _EntryBase(BaseModel):
    """The identity, provenance, and reputation fields both kinds carry."""

    model_config = _MODEL_CONFIG

    schema_version: int = Field(default=SCHEMA_VERSION, json_schema_extra=_CONST_SCHEMA_VERSION)
    canonical_key: str = Field(json_schema_extra=_KEY_SCHEMA)
    alias_keys: tuple[str, ...] = Field(default=(), json_schema_extra=_ALIAS_KEYS_SCHEMA)
    slug: str = Field(json_schema_extra=_SLUG_SCHEMA)
    name: str = Field(min_length=1, max_length=MAX_NAME_CHARS)
    description: str = Field(max_length=MAX_DESCRIPTION_CHARS)
    homepage: str = ""
    docs_url: str = ""
    icon_url: str = Field(default="", json_schema_extra=_ICON_URL_SCHEMA)
    # Whether the marketplace this record was crawled from is one a person
    # marked ``trust: reviewed`` in ``curated/skills.yaml``. An additive flag
    # rather than a trust tier, so an old consumer validating ``trust_tier``
    # against its own Literal keeps working; the consumer elects its own tier
    # from it.
    marketplace_reviewed: bool = False
    trust_tier: Literal["curated", "registry_verified", "smithery_verified", "indexed"]
    popularity: float = Field(default=0.0, ge=0.0, le=1.0)
    popularity_signals: PopularitySignals = PopularitySignals()
    sources: tuple[SourceRef, ...] = Field(json_schema_extra=_SOURCES_SCHEMA)
    tags: tuple[str, ...] = ()
    license: str = Field(default="", max_length=64)
    deprecated: bool = False
    curated_fields: tuple[str, ...] = ()

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, value: int) -> int:
        _require(value == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION}")
        return value

    @field_validator("name", "description", mode="before")
    @classmethod
    def _normalize_free_text(cls, value: object, info: ValidationInfo) -> object:
        if not isinstance(value, str):
            return value
        limit = MAX_NAME_CHARS if info.field_name == "name" else MAX_DESCRIPTION_CHARS
        return _collapse(value, limit)

    @field_validator("canonical_key")
    @classmethod
    def _check_canonical_key(cls, value: str) -> str:
        return _match(CANONICAL_KEY_RE, value, "canonical_key")

    @field_validator("alias_keys")
    @classmethod
    def _check_alias_keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for key in value:
            _match(CANONICAL_KEY_RE, key, "alias key")
        return tuple(sorted(set(value)))[:MAX_ALIAS_KEYS]

    @field_validator("slug")
    @classmethod
    def _check_slug(cls, value: str) -> str:
        return _match(SERVER_SLUG_RE, value, "slug")

    @field_validator("homepage")
    @classmethod
    def _check_homepage(cls, value: str) -> str:
        return _https_url(value, "homepage")

    @field_validator("docs_url")
    @classmethod
    def _check_docs_url(cls, value: str) -> str:
        return _https_url(value, "docs_url", allow_http=True)

    @field_validator("icon_url")
    @classmethod
    def _check_icon_url(cls, value: str) -> str:
        """Checked and rejected, never coerced: a bad icon URL is a pipeline bug.

        The election in ``normalize`` only ever constructs the two permitted
        shapes, so anything else arriving here means code upstream drifted —
        or a curated file tried to point the icon proxy somewhere new, which
        is a decision for this validator, not for a YAML edit.
        """
        if value == "":
            return value
        _require(len(value) <= MAX_URL_CHARS, f"icon_url exceeds {MAX_URL_CHARS} characters")
        _require(
            ICON_URL_SMITHERY_RE.fullmatch(value) is not None
            or ICON_URL_GITHUB_RE.fullmatch(value) is not None,
            "icon_url must be a Smithery server icon or a GitHub owner avatar URL",
        )
        return value

    @field_validator("popularity")
    @classmethod
    def _round_popularity(cls, value: float) -> float:
        """Pin the one float in the record to four places before it is serialised."""
        return round(value, 4)

    @field_validator("sources")
    @classmethod
    def _check_sources(cls, value: tuple[SourceRef, ...]) -> tuple[SourceRef, ...]:
        _require(len(value) > 0, "an entry needs at least one source")
        _require(len(value) <= MAX_SOURCES, f"at most {MAX_SOURCES} sources are allowed")
        return tuple(sorted(value, key=lambda ref: (SOURCE_RANK[ref.source_id], ref.upstream_id)))

    @field_validator("tags")
    @classmethod
    def _check_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Filter rather than reject: tags are whatever a publisher typed."""
        kept = {
            lowered
            for lowered in (tag.strip().lower() for tag in value)
            if len(lowered) <= MAX_TAG_CHARS and _TAG_RE.fullmatch(lowered) is not None
        }
        return tuple(sorted(kept))[:MAX_TAGS]

    @field_validator("curated_fields")
    @classmethod
    def _sort_curated_fields(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(set(value)))

    @model_validator(mode="after")
    def _check_entry(self) -> Self:
        _require(
            self.canonical_key not in self.alias_keys,
            "alias_keys must not repeat canonical_key",
        )
        unknown = sorted(set(self.curated_fields) - set(type(self).model_fields))
        _require(not unknown, f"curated_fields names no such field: {unknown}")
        return self


class McpEntry(_EntryBase):
    """One MCP server: how to reach it, who vouches for it, how popular it is."""

    kind: Literal["mcp"]
    repo: RepoRef | None = None
    category: str = Field(json_schema_extra=_CATEGORY_SCHEMA)
    icon: str = Field(json_schema_extra=_ICON_SCHEMA)
    connector_type: str | None = Field(default=None, json_schema_extra=_CONNECTOR_TYPE_SCHEMA)
    mcp_url: str | None = Field(default=None, json_schema_extra=_MCP_URL_SCHEMA)
    url_unverified: bool = False
    transport: Literal["streamable_http", "sse", "unknown"] = "unknown"
    auth_hint: Literal["none", "bearer", "header", "oauth"] = "bearer"
    auth_note: str = Field(default="", max_length=MAX_NOTE_CHARS)
    setup_note: str = Field(default="", max_length=MAX_NOTE_CHARS)
    stdio_only: bool = False
    connector_config: dict[str, str] = Field(default_factory=dict)
    packages: tuple[PackageRef, ...] = ()
    remotes: tuple[RemoteRef, ...] = ()
    tool_count: int | None = Field(default=None, ge=0, le=200)
    registry_name: str = Field(default="", max_length=132)
    smithery_qualified_name: str = Field(default="", max_length=200)
    npm_package: str = Field(default="", max_length=214)
    verified_upstream: bool = False

    @field_validator("category")
    @classmethod
    def _check_category(cls, value: str) -> str:
        _require(value in CATALOG_CATEGORIES, f"category must be one of {list(CATALOG_CATEGORIES)}")
        return value

    @field_validator("icon")
    @classmethod
    def _check_icon(cls, value: str) -> str:
        _require(value in CATALOG_ICONS, f"icon must be one of {sorted(CATALOG_ICONS)}")
        return value

    @field_validator("connector_type")
    @classmethod
    def _check_connector_type(cls, value: str | None) -> str | None:
        if value is None:
            return None
        _require(
            value in CONNECTOR_TYPES, f"connector_type must be one of {sorted(CONNECTOR_TYPES)}"
        )
        return value

    @field_validator("mcp_url")
    @classmethod
    def _check_mcp_url(cls, value: str | None) -> str | None:
        """A URL Jhin will actually dial: concrete, secure, and free of userinfo."""
        if value is None:
            return None
        _require(value != "", "mcp_url must be a URL or None")
        _https_url(value, "mcp_url")
        _require("{" not in value, "mcp_url must not be templated")
        _require("#" not in value, "mcp_url must not carry a fragment")
        authority = value.removeprefix("https://").split("/", 1)[0].split("?", 1)[0]
        _require("@" not in authority, "mcp_url must not carry userinfo")
        return value

    @field_validator("auth_note", "setup_note", mode="before")
    @classmethod
    def _normalize_note(cls, value: object) -> object:
        """Notes are shown verbatim in Jhin's UI, so they end as sentences."""
        if not isinstance(value, str):
            return value
        return _terminate(_collapse(value, MAX_NOTE_CHARS - 1))

    @field_validator("connector_config")
    @classmethod
    def _check_connector_config(cls, value: dict[str, str]) -> dict[str, str]:
        _require(
            len(value) <= MAX_CONNECTOR_CONFIG,
            f"at most {MAX_CONNECTOR_CONFIG} connector_config pairs are allowed",
        )
        for key, item in value.items():
            _require(
                1 <= len(key) <= MAX_CONNECTOR_CONFIG_KEY_CHARS,
                f"connector_config key {key!r} exceeds {MAX_CONNECTOR_CONFIG_KEY_CHARS} characters",
            )
            _require(
                len(item) <= MAX_CONNECTOR_CONFIG_VALUE_CHARS,
                f"connector_config value for {key!r} exceeds "
                f"{MAX_CONNECTOR_CONFIG_VALUE_CHARS} characters",
            )
        return dict(sorted(value.items()))

    @field_validator("packages")
    @classmethod
    def _check_packages(cls, value: tuple[PackageRef, ...]) -> tuple[PackageRef, ...]:
        ordered = sorted(value, key=lambda pkg: (pkg.registry_type, pkg.identifier))
        return tuple(ordered)[:MAX_PACKAGES]

    @field_validator("remotes")
    @classmethod
    def _check_remotes(cls, value: tuple[RemoteRef, ...]) -> tuple[RemoteRef, ...]:
        ordered = sorted(value, key=lambda remote: (remote.transport, remote.url))
        return tuple(ordered)[:MAX_REMOTES]

    @model_validator(mode="after")
    def _check_kind_prefix(self) -> Self:
        _require(self.canonical_key.startswith("mcp:"), "canonical_key must start with 'mcp:'")
        return self


class SkillEntry(_EntryBase):
    """One Agent Skill: where its ``SKILL.md`` lives, never what it says."""

    kind: Literal["skill"]
    repo: RepoRef
    skill_name: str = Field(json_schema_extra=_SKILL_NAME_SCHEMA)
    category: str = Field(default=DEFAULT_SKILL_CATEGORY, min_length=1, max_length=64)
    source_ref: str
    skill_path: str
    plugin: PluginRef | None = None
    commit_sha: str = ""
    model_invocable: bool = True
    allowed_tools: tuple[str, ...] = ()
    skill_version: str = Field(default="", max_length=32)
    frontmatter_bytes: int = Field(default=0, ge=0, le=8192)

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return _collapse(value, 64) or DEFAULT_SKILL_CATEGORY

    @field_validator("skill_name")
    @classmethod
    def _check_skill_name(cls, value: str) -> str:
        return _match(SKILL_NAME_RE, value, "skill_name")

    @field_validator("source_ref")
    @classmethod
    def _check_source_ref(cls, value: str) -> str:
        _require(len(value) <= 300, "source_ref exceeds 300 characters")
        return _match(SOURCE_REF_RE, value, "source_ref")

    @field_validator("skill_path")
    @classmethod
    def _check_skill_path(cls, value: str) -> str:
        _require(value.endswith("/SKILL.md"), "skill_path must end with '/SKILL.md'")
        return _relative_posix(value, "skill_path", limit=255)

    @field_validator("commit_sha")
    @classmethod
    def _check_commit_sha(cls, value: str) -> str:
        if value == "":
            return value
        return _match(SHA1_RE, value.lower(), "commit_sha")

    @field_validator("allowed_tools")
    @classmethod
    def _check_allowed_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        kept = {name.strip() for name in value if name.strip() and len(name.strip()) <= 64}
        return tuple(sorted(kept))[:32]

    @model_validator(mode="after")
    def _check_kind_prefix(self) -> Self:
        _require(self.canonical_key.startswith("skill:"), "canonical_key must start with 'skill:'")
        return self


type CatalogEntry = Annotated[McpEntry | SkillEntry, Field(discriminator="kind")]
ENTRY_ADAPTER: Final[TypeAdapter[CatalogEntry]] = TypeAdapter(CatalogEntry)

# The two concrete models behind the union, keyed by discriminator. Defined
# here, once, so ``build`` and ``dedupe`` cannot disagree about what an
# unrecognised kind means — one used to raise on it and the other to treat it
# silently as a skill.
ENTRY_MODELS: Final[Mapping[str, type[McpEntry] | type[SkillEntry]]] = {
    "mcp": McpEntry,
    "skill": SkillEntry,
}


class RawRecord(BaseModel):
    """One upstream object, untouched, tagged with where it came from."""

    model_config = _MODEL_CONFIG

    source_id: str = Field(min_length=1)
    upstream_id: str = Field(min_length=1, max_length=300)
    url: str = Field(min_length=1, max_length=MAX_URL_CHARS)
    payload: JsonObject


class SourceFetch(BaseModel):
    """Everything one crawl of one upstream produced, plus its content hash."""

    model_config = _MODEL_CONFIG

    source_id: str = Field(min_length=1)
    url: str = Field(min_length=1, max_length=MAX_URL_CHARS)
    sha256: str
    entry_count: int = Field(ge=0)
    page_count: int = Field(ge=0)
    records: tuple[RawRecord, ...]

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, value: str) -> str:
        return _match(_HEX64_RE, value.lower(), "sha256")


class Candidate(BaseModel):
    """One source's opinion about one thing, before identities are joined.

    ``fields`` is a partial entry payload; its keys are expected to name real
    model fields, and ``normalize.build_entry`` is where a stray one fails.
    """

    model_config = _MODEL_CONFIG

    kind: Literal["mcp", "skill"]
    source_id: str = Field(min_length=1)
    upstream_id: str = Field(min_length=1, max_length=300)
    primary_key: str
    alias_keys: tuple[str, ...]
    repo: RepoRef | None = None
    signals: PopularitySignals = PopularitySignals()
    trust_hint: str
    source_ref: SourceRef
    fields: JsonObject = Field(default_factory=dict)

    @field_validator("primary_key")
    @classmethod
    def _check_primary_key(cls, value: str) -> str:
        return _match(CANONICAL_KEY_RE, value, "primary_key")

    @field_validator("alias_keys")
    @classmethod
    def _check_alias_keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _require(len(value) > 0, "a candidate needs at least one key")
        for key in value:
            _match(CANONICAL_KEY_RE, key, "alias key")
        return tuple(sorted(set(value)))

    @field_validator("trust_hint")
    @classmethod
    def _check_trust_hint(cls, value: str) -> str:
        _require(value in TRUST_RANK, f"trust_hint must be one of {list(TRUST_TIERS)}")
        return value

    @model_validator(mode="after")
    def _check_candidate(self) -> Self:
        _require(self.primary_key in self.alias_keys, "alias_keys must contain primary_key")
        _require(
            all(key.startswith(f"{self.kind}:") for key in self.alias_keys),
            f"every key must be prefixed {self.kind!r}",
        )
        return self


class MergedCandidate(BaseModel):
    """One identity, assembled from every candidate that turned out to be it."""

    model_config = _MODEL_CONFIG

    kind: Literal["mcp", "skill"]
    canonical_key: str
    alias_keys: tuple[str, ...] = ()
    ambiguous_repo_keys: tuple[str, ...] = ()
    candidates: tuple[Candidate, ...]
    signals: PopularitySignals = PopularitySignals()
    trust_tier: str
    fields: JsonObject = Field(default_factory=dict)

    @field_validator("canonical_key")
    @classmethod
    def _check_canonical_key(cls, value: str) -> str:
        return _match(CANONICAL_KEY_RE, value, "canonical_key")

    @field_validator("alias_keys", "ambiguous_repo_keys")
    @classmethod
    def _sort_keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for key in value:
            _match(CANONICAL_KEY_RE, key, "key")
        return tuple(sorted(set(value)))

    @field_validator("candidates")
    @classmethod
    def _check_candidates(cls, value: tuple[Candidate, ...]) -> tuple[Candidate, ...]:
        _require(len(value) > 0, "a merged candidate needs at least one candidate")
        return tuple(
            sorted(
                value, key=lambda c: (SOURCE_RANK.get(c.source_id, len(SOURCE_IDS)), c.upstream_id)
            )
        )

    @field_validator("trust_tier")
    @classmethod
    def _check_trust_tier(cls, value: str) -> str:
        _require(value in TRUST_RANK, f"trust_tier must be one of {list(TRUST_TIERS)}")
        return value

    @model_validator(mode="after")
    def _check_merged(self) -> Self:
        _require(
            self.canonical_key not in self.alias_keys,
            "alias_keys must not repeat canonical_key",
        )
        return self


class CuratedOverride(BaseModel):
    """A hand-written correction applied on top of whatever the crawl found."""

    model_config = _MODEL_CONFIG

    key: str
    kind: Literal["mcp", "skill"]
    aliases: tuple[str, ...] = ()
    fields: JsonObject = Field(default_factory=dict)

    @field_validator("key")
    @classmethod
    def _check_key(cls, value: str) -> str:
        return _match(CANONICAL_KEY_RE, value, "key")

    @field_validator("aliases")
    @classmethod
    def _check_aliases(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for alias in value:
            _match(CANONICAL_KEY_RE, alias, "alias")
        return tuple(sorted(set(value)))


class DenylistItem(BaseModel):
    """One key the catalog refuses to publish, and why."""

    model_config = _MODEL_CONFIG

    key: str
    reason: str = Field(min_length=8, max_length=300)

    @field_validator("key")
    @classmethod
    def _check_key(cls, value: str) -> str:
        return _match(CANONICAL_KEY_RE, value, "key")


class MarketplacePolicy(BaseModel):
    """Which plugin marketplaces a person has reviewed, and whether that binds.

    Topic search finds candidate repositories; this says which of them may
    actually put text into the index. It is the only control standing between
    a stranger's commit and a ``description`` that Jhin's agents will read,
    because the diff gate deliberately never blocks additions.
    """

    model_config = _MODEL_CONFIG

    allow: tuple[str, ...] = ()
    # The subset of ``allow`` a person marked ``trust: reviewed``: skills
    # crawled from these repositories carry ``marketplace_reviewed`` on their
    # records, which is what a consumer elects its own reviewed tier from.
    reviewed: tuple[str, ...] = ()
    require_allowlist: bool = False

    @field_validator("allow", "reviewed")
    @classmethod
    def _normalise(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted({item.strip().lower() for item in value if item.strip()}))


class LockEntry(BaseModel):
    """What one source returned on the crawl that produced the committed data."""

    model_config = _MODEL_CONFIG

    source_id: str = Field(min_length=1)
    url: str = Field(min_length=1, max_length=MAX_URL_CHARS)
    fetched_at: str
    sha256: str
    entry_count: int = Field(ge=0)
    page_count: int = Field(ge=0)

    @field_validator("fetched_at")
    @classmethod
    def _check_fetched_at(cls, value: str) -> str:
        return _match(_RFC3339_SECONDS_RE, value, "fetched_at")

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, value: str) -> str:
        return _match(_HEX64_RE, value.lower(), "sha256")


class SourcesLock(BaseModel):
    """``sources.lock``: the only build output that carries a timestamp."""

    model_config = _MODEL_CONFIG

    schema_version: int = SCHEMA_VERSION
    sources: tuple[LockEntry, ...] = ()

    @field_validator("sources")
    @classmethod
    def _sort_sources(cls, value: tuple[LockEntry, ...]) -> tuple[LockEntry, ...]:
        return tuple(sorted(value, key=lambda entry: entry.source_id))


class DiffThresholds(BaseModel):
    """How much of the committed catalog one build is allowed to replace."""

    model_config = _MODEL_CONFIG

    max_drop_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    max_change_fraction: float = Field(default=0.20, ge=0.0, le=1.0)
    min_baseline_entries: int = Field(default=100, ge=0)


class DiffReport(BaseModel):
    """What one build would do to one kind's committed shards."""

    model_config = _MODEL_CONFIG

    kind: str = Field(min_length=1)
    baseline_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    added: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    drop_fraction: float = Field(default=0.0, ge=0.0)
    change_fraction: float = Field(default=0.0, ge=0.0)

    @field_validator("added", "dropped", "changed")
    @classmethod
    def _sort_keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(value))

    @field_validator("drop_fraction", "change_fraction")
    @classmethod
    def _round_fraction(cls, value: float) -> float:
        return round(value, 6)


class BuildResult(BaseModel):
    """What one ``run_sync`` produced, for the CLI to print and tests to read."""

    model_config = _MODEL_CONFIG

    entry_counts: Mapping[str, int]
    written: tuple[str, ...] = ()
    reports: tuple[DiffReport, ...] = ()
    lock: SourcesLock = SourcesLock()
    # Denylist keys that resolved to nothing: a warning for a human to read,
    # not a failure. ``dedupe.unresolved_denylist_keys`` says why.
    stale_denylist_keys: tuple[str, ...] = ()

    @field_validator("entry_counts")
    @classmethod
    def _sort_counts(cls, value: Mapping[str, int]) -> Mapping[str, int]:
        return dict(sorted(value.items()))

    @field_validator("written")
    @classmethod
    def _sort_written(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(value))


def canonical_json(obj: JsonObject) -> str:
    """The one JSON encoding this repo emits. No trailing newline."""
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def dumps_line(entry: CatalogEntry) -> str:
    """One JSONL record including its terminating ``\\n``."""
    return canonical_json(cast(JsonObject, entry.model_dump(mode="json"))) + "\n"


def loads_line(line: str) -> CatalogEntry:
    """Parse one JSONL record. Raises ``CatalogError`` on bad JSON or schema."""
    try:
        payload = json.loads(line, parse_constant=_reject_constant)
    except ValueError as exc:
        raise CatalogError(f"Line is not valid JSON: {exc}") from None
    try:
        return ENTRY_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise CatalogError(f"Line is not a valid catalog entry: {exc}") from None


def shard_for(canonical_key: str) -> str:
    """The two lowercase hex characters naming this key's shard."""
    return hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:SHARD_HEX_WIDTH]


def all_shards() -> tuple[str, ...]:
    """``("00", "01", …, "ff")`` — every shard name, in file order."""
    return tuple(format(index, f"0{SHARD_HEX_WIDTH}x") for index in range(SHARD_COUNT))


def entry_sort_key(entry: CatalogEntry) -> str:
    """``entry.canonical_key`` — the total order inside a shard."""
    return entry.canonical_key


def payload_sha256(body: bytes) -> str:
    """Lowercase hex SHA-256 of a raw response body."""
    return hashlib.sha256(body).hexdigest()
