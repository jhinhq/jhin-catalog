"""Claude Code plugin marketplaces, crawled for the skills they point at.

Reads ``.claude-plugin/marketplace.json`` — and the three cross-agent
aliases some repositories ship instead — from the seed marketplaces and from
every repository carrying the ``claude-code-plugin`` topic, resolves each
``plugins[].source`` form to a GitHub repository and a plugin root, lists
that repository's tree once, and emits one :class:`RawRecord` per
``**/SKILL.md`` blob beneath the root.

Pointer, never payload. A ``SKILL.md`` body is agent instructions, and
vendoring it here would turn an index into a distributor of prompt-injection
payloads. The refusal is structural rather than conventional: the fetched
text is read inside :meth:`_Crawl._frontmatter` and the only value that
leaves that frame is a :class:`SkillFrontmatter`, a frozen model of seven
short scalars with ``extra="forbid"``; every assembled payload then passes
:func:`_assert_pointer_only`, which rejects an unlisted key or any string
longer than one URL; and response bodies are folded into the page digest as
they arrive rather than collected, so no body outlives the call that read
it.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections.abc import Awaitable, Callable, Mapping
from typing import ClassVar, Final, cast

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from jhin_catalog import http
from jhin_catalog.sources.base import (
    DEFAULT_LIMITS,
    RollingDigest,
    Source,
    SourceError,
    SourceLimits,
    TokenBucket,
)
from jhin_catalog.sources.github_topics import (
    CORE_RATE_PER_MINUTE,
    SEARCH_RATE_PER_MINUTE,
    github_headers,
)
from jhin_catalog.types import (
    DEFAULT_SKILL_CATEGORY,
    MAX_DESCRIPTION_CHARS,
    MAX_TAG_CHARS,
    MAX_TAGS,
    MAX_URL_CHARS,
    SHA1_RE,
    SKILL_NAME_RE,
    JsonObject,
    JsonValue,
    RawRecord,
    SourceFetch,
)

RAW_BASE: Final[str] = "https://raw.githubusercontent.com"
GITHUB_API: Final[str] = "https://api.github.com"
MARKETPLACE_PATH: Final[str] = ".claude-plugin/marketplace.json"
ALT_MARKETPLACE_PATHS: Final[tuple[str, ...]] = (
    ".agents/plugins/marketplace.json",
    ".cursor-plugin/marketplace.json",
    ".codex-plugin/marketplace.json",
)
SEED_REPOS: Final[tuple[str, ...]] = (
    "anthropics/claude-plugins-official",
    "anthropics/claude-code",
)
SKILL_FILENAME: Final[str] = "SKILL.md"
MAX_FRONTMATTER_BYTES: Final[int] = 8192
MAX_SKILL_MD_BYTES: Final[int] = 262_144
MAX_SKILLS_PER_REPO: Final[int] = 500
# raw.githubusercontent.com publishes no documented rate limit, and no API
# bucket covers it. This is the ceiling the crawl imposes on itself so a
# manifest declaring five hundred skills cannot spend the whole job's budget.
RAW_RATE_PER_MINUTE: Final[int] = 120

_SOURCE_ID: Final[str] = "marketplaces"
_PLUGIN_TOPIC: Final[str] = "claude-code-plugin"
_SEARCH_PATH: Final[str] = "/search/repositories"
_SEARCH_PAGE_SIZE: Final[int] = 100
_DISCOVERY_MAX_PAGES: Final[int] = 10
_MAX_FORBIDDEN_ATTEMPTS: Final[int] = 3
_MAX_MANIFEST_BYTES: Final[int] = 1_048_576
_MAX_PATH_CHARS: Final[int] = 255
_MAX_ALLOWED_TOOLS: Final[int] = 32
_MAX_SOURCE_REF_CHARS: Final[int] = 300
_FENCE: Final[str] = "---"
_DEFAULT_REF: Final[str] = "HEAD"

_CONTROL_RE: Final[re.Pattern[str]] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TAG_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_FULL_NAME_RE: Final[re.Pattern[str]] = re.compile(
    r"^([A-Za-z0-9_.-]{1,64})/([A-Za-z0-9_.-]{1,100})$"
)
_GITHUB_URL_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:git\+)?(?:https?|ssh|git)://(?:[^@/]+@)?github\.com/"
    r"([A-Za-z0-9_.-]{1,64})/([A-Za-z0-9_.-]{1,100}?)(?:\.git)?/?$"
)
_GITHUB_SCP_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:git@)?github\.com:([A-Za-z0-9_.-]{1,64})/([A-Za-z0-9_.-]{1,100}?)(?:\.git)?/?$"
)

_SKILL_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "allowed_tools",
        "category",
        "commit_sha",
        "description",
        "docs_url",
        "frontmatter_bytes",
        "license",
        "marketplace",
        "marketplace_repo",
        "model_invocable",
        "name",
        "plugin",
        "renamed_from",
        "repo",
        "skill_name",
        "skill_path",
        "skill_version",
        "source_ref",
        "tags",
    }
)


def _error(message: str) -> SourceError:
    """A :class:`SourceError` already tagged with this module's source id."""
    return SourceError(message, source_id=_SOURCE_ID)


class PluginSource(BaseModel):
    """Where one ``plugins[]`` row says its plugin actually lives."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_kind: str
    source_value: str
    path: str = ""
    ref: str = ""
    sha: str = ""


class SkillFrontmatter(BaseModel):
    """The whole of a ``SKILL.md`` this index is willing to remember.

    Seven short scalars with ``extra="forbid"``: the model is the structural
    barrier that keeps a skill's instruction body out of every record, since
    nothing else crosses out of the function that read the file.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    version: str = ""
    license: str = ""
    allowed_tools: tuple[str, ...] = ()
    disable_model_invocation: bool = False
    frontmatter_bytes: int = 0


def _clip(value: object, *, limit: int) -> str:
    """One line of ``value``: control characters gone, whitespace collapsed.

    Metadata fields are clipped on the way in rather than on the way out.
    A ``description`` past a few hundred characters is not a description, it
    is body text arriving through a metadata field, and this index does not
    carry body text.
    """
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, int | float):
        text = str(value)
    else:
        return ""
    return " ".join(_CONTROL_RE.sub(" ", text).split())[:limit]


def _safe_path(value: str) -> str | None:
    """``value`` as a repo-relative POSIX path, or ``None`` when it escapes.

    A ``..`` segment in a manifest is a traversal attempt, so the path is
    refused outright rather than silently flattened into something else.
    """
    parts = [part for part in value.replace("\\", "/").split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        return None
    joined = "/".join(parts)
    return joined if len(joined) <= _MAX_PATH_CHARS else None


def _tags(value: JsonValue) -> tuple[str, ...]:
    """A plugin's ``keywords`` as catalog tags: lowercase, sorted, bounded."""
    raw: list[str] = []
    if isinstance(value, str):
        raw = value.split(",")
    elif isinstance(value, list):
        raw = [item for item in value if isinstance(item, str)]
    kept = {
        tag
        for item in raw
        if (tag := _clip(item, limit=MAX_TAG_CHARS).lower()) and _TAG_RE.match(tag)
    }
    return tuple(sorted(kept))[:MAX_TAGS]


def _tool_names(value: object) -> tuple[str, ...]:
    """``allowed-tools`` as a sorted tuple, from a comma string or a list."""
    raw: list[str] = []
    if isinstance(value, str):
        raw = value.split(",")
    elif isinstance(value, list):
        raw = [item for item in value if isinstance(item, str)]
    kept = {name for item in raw if (name := _clip(item, limit=64))}
    return tuple(sorted(kept))[:_MAX_ALLOWED_TOOLS]


def _truthy(value: object) -> bool:
    """A YAML flag read the way a human wrote it, defaulting to false."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "on", "1"}
    return False


def _sha1_or_empty(value: object) -> str:
    """``value`` as a lowercase 40-hex commit sha, or ``""`` when it is not one."""
    text = _clip(value, limit=64).lower()
    return text if SHA1_RE.match(text) else ""


def _category(value: JsonValue) -> str:
    """A marketplace ``category``, title-cased, falling back to the default.

    The first letter of each word is raised and the rest is left alone, so an
    acronym a marketplace deliberately capitalised survives the pass.
    """
    text = _clip(value, limit=64)
    if not text:
        return DEFAULT_SKILL_CATEGORY
    titled = " ".join(word[:1].upper() + word[1:] for word in text.split(" "))
    return titled[:64]


def _split_full_name(value: str) -> tuple[str, str] | None:
    """``"owner/repo"`` split in two, or ``None`` when it is not that shape."""
    match = _FULL_NAME_RE.match(value.strip())
    if match is None:
        return None
    return match.group(1), match.group(2)


def _github_repo_from_url(url: str) -> tuple[str, str] | None:
    """The owner and repository a GitHub clone URL names, or ``None``.

    Only GitHub resolves here. Plugin marketplaces are a GitHub-hosted
    convention and the tree and raw endpoints this crawl uses are GitHub's,
    so a GitLab or Bitbucket source is recorded as unusable rather than
    guessed at.
    """
    text = url.strip()
    for pattern in (_GITHUB_URL_RE, _GITHUB_SCP_RE):
        match = pattern.match(text)
        if match is not None:
            return match.group(1), match.group(2)
    return None


def _unknown_source(value: JsonValue) -> PluginSource:
    """The disposal bin: a shape this crawl cannot resolve to a repository."""
    dumped = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return PluginSource(source_kind="unknown", source_value=dumped[:MAX_URL_CHARS])


def _relative_value(value: str) -> str | None:
    """The repo-relative directory a ``"./…"`` plugin source names.

    ``"./"`` normalises to ``""`` — the repository root — and anything with a
    ``..`` segment normalises to ``None`` so the caller can refuse it.
    """
    return _safe_path(value.strip())


def parse_plugin_source(value: JsonValue) -> PluginSource:
    """Every observed ``plugins[].source`` form, including the schema-invalid
    ``url`` + ``path`` hybrid. Unrecognised shapes become ``source_kind="unknown"``.

    A path that climbs out of its repository is unresolvable rather than
    merely unrecognised, and lands in the same bin.
    """
    if isinstance(value, str):
        relative = _relative_value(value)
        if relative is None:
            return _unknown_source(value)
        return PluginSource(source_kind="relative", source_value=relative)
    if not isinstance(value, dict):
        return _unknown_source(value)

    declared = value.get("source")
    if not isinstance(declared, str):
        return _unknown_source(value)
    kind = declared.strip().lower()
    url = _clip(value.get("url"), limit=MAX_URL_CHARS)
    raw_path = value.get("path")
    subpath = _relative_value(raw_path) if isinstance(raw_path, str) else ""
    if subpath is None:
        return _unknown_source(value)
    ref = _clip(value.get("ref"), limit=200)
    sha = _sha1_or_empty(value.get("sha"))

    if kind == "url" and url:
        if subpath:
            return PluginSource(
                source_kind="git-subdir", source_value=url, path=subpath, ref=ref, sha=sha
            )
        return PluginSource(source_kind="url", source_value=url, ref=ref, sha=sha)
    if kind == "git-subdir" and url:
        return PluginSource(
            source_kind="git-subdir", source_value=url, path=subpath, ref=ref, sha=sha
        )
    if kind == "github":
        repo = value.get("repo")
        full_name = _split_full_name(repo) if isinstance(repo, str) else None
        if full_name is not None:
            owner, name = full_name
            return PluginSource(
                source_kind="github",
                source_value=f"https://github.com/{owner}/{name}",
                path=subpath,
                ref=ref,
                sha=sha,
            )
    if kind == "npm":
        package = value.get("package")
        identifier = _clip(package, limit=214)
        if identifier:
            return PluginSource(source_kind="npm", source_value=identifier, ref=ref, sha=sha)
    return _unknown_source(value)


def parse_skill_frontmatter(text: str) -> SkillFrontmatter:
    """Parse the leading ``---`` block with ``yaml.safe_load``.

    Raises ``SourceError`` when the block is absent, over
    ``MAX_FRONTMATTER_BYTES``, not a mapping, or missing ``name``.

    Real YAML, not a line regex: a ``description: >`` block scalar and a
    quoted value carried over two lines are both common in the corpus and
    both defeat a regex reader. Only the named scalars survive the parse, so
    a frontmatter key nobody asked for cannot reach a record.
    """
    normalized = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith(f"{_FENCE}\n"):
        raise _error("SKILL.md has no leading --- frontmatter block")

    lines = normalized.split("\n")
    scanned = 0
    end = -1
    for index in range(1, len(lines)):
        if lines[index].rstrip() == _FENCE:
            end = index
            break
        scanned += len(lines[index].encode("utf-8")) + 1
        if scanned > MAX_FRONTMATTER_BYTES:
            raise _error(f"SKILL.md frontmatter exceeds {MAX_FRONTMATTER_BYTES} bytes")
    if end < 0:
        raise _error("SKILL.md frontmatter block is never closed")

    block = "\n".join(lines[1:end])
    measured = len(block.encode("utf-8"))
    if measured > MAX_FRONTMATTER_BYTES:
        raise _error(f"SKILL.md frontmatter exceeds {MAX_FRONTMATTER_BYTES} bytes")
    try:
        parsed = yaml.safe_load(block)
    except yaml.YAMLError as exc:
        raise _error("SKILL.md frontmatter is not valid YAML") from exc
    if not isinstance(parsed, dict):
        raise _error("SKILL.md frontmatter is not a mapping")

    name = _clip(parsed.get("name"), limit=64).lower()
    if SKILL_NAME_RE.match(name) is None:
        raise _error(f"SKILL.md frontmatter name {name!r} is not a usable skill name")
    tools = parsed.get("allowed-tools", parsed.get("allowed_tools"))
    disabled = parsed.get("disable-model-invocation", parsed.get("disable_model_invocation"))
    return SkillFrontmatter(
        name=name,
        description=_clip(parsed.get("description"), limit=MAX_DESCRIPTION_CHARS),
        version=_clip(parsed.get("version"), limit=32),
        license=_clip(parsed.get("license"), limit=64),
        allowed_tools=_tool_names(tools),
        disable_model_invocation=_truthy(disabled),
        frontmatter_bytes=min(measured, MAX_FRONTMATTER_BYTES),
    )


def skill_paths(tree: JsonValue, *, plugin_root: str) -> tuple[str, ...]:
    """Every ``**/SKILL.md`` blob path under ``plugin_root``, sorted.

    A flat ``skills/*/SKILL.md`` match would miss the ~38 % of skills nested
    deeper, so the match is on the basename at any depth. A bare ``SKILL.md``
    at the repository root is not a match: a skill is identified by the
    directory holding it, and a root-level file has none.
    """
    entries = tree.get("tree") if isinstance(tree, dict) else tree
    if not isinstance(entries, list):
        raise _error('git tree response carries no "tree" list')
    prefix = f"{plugin_root}/" if plugin_root else ""
    suffix = f"/{SKILL_FILENAME}"
    found: set[str] = set()
    for item in entries:
        if not isinstance(item, dict) or item.get("type") != "blob":
            continue
        path = item.get("path")
        if not isinstance(path, str):
            continue
        cleaned = _safe_path(path)
        if cleaned is None or not cleaned.endswith(suffix):
            continue
        if prefix and not cleaned.startswith(prefix):
            continue
        found.add(cleaned)
    return tuple(sorted(found))


class _Plugin(BaseModel):
    """One ``plugins[]`` row, already resolved against ``metadata.pluginRoot``.

    :func:`parse_marketplace` dumps this into the plugin record's payload and
    the crawl validates it straight back, so the manifest is read once and
    the round trip is itself a check on what the record carries.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    marketplace: str
    marketplace_repo: str
    marketplace_owner: str
    plugin: str
    description: str
    category: str
    license: str
    keywords: tuple[str, ...]
    declared_skills: tuple[str, ...]
    source_kind: str
    source_value: str
    source_git_ref: str
    source_sha: str
    plugin_root: str


def _declared_skill_dirs(value: JsonValue) -> tuple[str, ...]:
    """The directories a plugin's ``skills`` field names, relative to its root.

    A value that already ends in ``SKILL.md`` names the file rather than the
    directory holding it; the filename is dropped so the caller can append it
    exactly once.
    """
    raw: list[str] = []
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = [item for item in value if isinstance(item, str)]
    dirs: set[str] = set()
    suffix = f"/{SKILL_FILENAME}"
    for item in raw:
        cleaned = _safe_path(item)
        if cleaned is None:
            continue
        if cleaned == SKILL_FILENAME:
            cleaned = ""
        elif cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
        dirs.add(cleaned)
    return tuple(sorted(dirs))


def _plugin_root(source: PluginSource, *, prefix: str) -> str:
    """Where inside its own repository a plugin's files start.

    ``metadata.pluginRoot`` prefixes a relative source and only a relative
    source: an external repository knows its own layout and the manifest's
    root has no authority over it.
    """
    if source.source_kind == "relative":
        return "/".join(part for part in (prefix, source.source_value) if part)
    return source.path


def parse_marketplace(
    payload: JsonValue, *, marketplace_repo: str, url: str
) -> tuple[tuple[RawRecord, ...], Mapping[str, str]]:
    """Plugin records plus the ``renames`` map (empty when absent).

    Raises ``SourceError`` when ``name``, ``owner``, or ``plugins`` is missing.

    ``renames`` stays out of the records on purpose: an old plugin name is an
    alias for a skill that already exists, never a second skill. A row with
    no usable ``name``, or a name a previous row already claimed, is skipped
    rather than fatal — one malformed plugin does not cost a whole
    marketplace.

    ``plugin.json`` is never fetched. Every field this index keeps comes from
    ``marketplace.json`` or from the skill's own frontmatter, so the
    ``strict: false`` case — a plugin that ships no manifest of its own — is
    satisfied by construction instead of by a tolerated 404.
    """
    if not isinstance(payload, dict):
        raise _error(f"{url}: marketplace manifest is not an object")
    name = _clip(payload.get("name"), limit=100)
    if not name:
        raise _error(f'{url}: marketplace manifest has no "name"')
    owner = payload.get("owner")
    if owner is None:
        raise _error(f'{url}: marketplace manifest has no "owner"')
    if isinstance(owner, dict):
        owner_name = _clip(owner.get("name"), limit=100)
    elif isinstance(owner, str):
        owner_name = _clip(owner, limit=100)
    else:
        raise _error(f'{url}: marketplace manifest "owner" is neither text nor an object')
    plugins = payload.get("plugins")
    if not isinstance(plugins, list):
        raise _error(f'{url}: marketplace manifest has no "plugins" list')

    metadata = payload.get("metadata")
    prefix = ""
    if isinstance(metadata, dict):
        declared_root = metadata.get("pluginRoot")
        if isinstance(declared_root, str):
            prefix = _safe_path(declared_root) or ""

    renames: dict[str, str] = {}
    declared_renames = payload.get("renames")
    if isinstance(declared_renames, dict):
        for old, new in sorted(declared_renames.items()):
            if isinstance(new, str) and old.strip() and new.strip():
                renames[old.strip()] = new.strip()

    records: list[RawRecord] = []
    claimed: set[str] = set()
    for item in plugins:
        if not isinstance(item, dict):
            continue
        plugin_name = _clip(item.get("name"), limit=100)
        if not plugin_name or plugin_name in claimed:
            continue
        claimed.add(plugin_name)
        source = parse_plugin_source(item.get("source"))
        plugin = _Plugin(
            marketplace=name,
            marketplace_repo=marketplace_repo,
            marketplace_owner=owner_name,
            plugin=plugin_name,
            description=_clip(item.get("description"), limit=MAX_DESCRIPTION_CHARS),
            category=_category(item.get("category")),
            license=_clip(item.get("license"), limit=64),
            keywords=_tags(item.get("keywords")),
            declared_skills=_declared_skill_dirs(item.get("skills")),
            source_kind=source.source_kind,
            source_value=source.source_value,
            source_git_ref=source.ref,
            source_sha=source.sha,
            plugin_root=_plugin_root(source, prefix=prefix),
        )
        records.append(
            RawRecord(
                source_id=_SOURCE_ID,
                upstream_id=f"{marketplace_repo}#{plugin_name}",
                url=url,
                payload=cast("JsonObject", plugin.model_dump(mode="json")),
            )
        )
    return tuple(records), renames


def _reverse_renames(renames: Mapping[str, str]) -> Mapping[str, tuple[str, ...]]:
    """Current plugin name to the old names that still point at it."""
    grouped: dict[str, list[str]] = {}
    for old, new in sorted(renames.items()):
        grouped.setdefault(new, []).append(old)
    return {new: tuple(sorted(olds)) for new, olds in sorted(grouped.items())}


def _assert_pointer_value(label: str, value: JsonValue) -> None:
    """Refuse one payload value that has grown past pointer size."""
    if isinstance(value, str):
        if len(value) > MAX_URL_CHARS:
            raise _error(
                f"record field {label} holds {len(value)} characters; "
                f"nothing in this index may exceed {MAX_URL_CHARS}"
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_pointer_value(f"{label}[{index}]", item)
    elif isinstance(value, dict):
        for key in sorted(value):
            _assert_pointer_value(f"{label}.{key}", value[key])


def _assert_pointer_only(payload: JsonObject) -> None:
    """The tripwire that keeps skill instructions out of this index.

    Every key is checked against a closed allowlist and every string against
    the length of one URL. Nothing the crawl builds today can trip it — the
    values are clipped as they are read — which is the point: an edit that
    starts shovelling ``SKILL.md`` text into a record fails here instead of
    quietly publishing a prompt-injection payload.
    """
    unexpected = sorted(set(payload) - _SKILL_PAYLOAD_KEYS)
    if unexpected:
        raise _error(f"record payload carries unindexable keys: {', '.join(unexpected)}")
    for key in sorted(payload):
        _assert_pointer_value(key, payload[key])


def _docs_url(*, owner: str, repo: str, skill_path: str) -> str:
    """Where a reader goes to read this ``SKILL.md`` themselves.

    One function so the length the crawl budgets for a path and the length the
    record actually spends on one cannot drift apart.
    """
    return f"https://github.com/{owner}/{repo}/blob/{_DEFAULT_REF}/{skill_path}"


def _skill_record(
    plugin: _Plugin,
    *,
    owner: str,
    repo: str,
    skill_path: str,
    commit_sha: str,
    frontmatter: SkillFrontmatter,
    renamed_from: tuple[str, ...],
) -> RawRecord:
    """One skill, as identity and pointers and nothing else.

    ``skill_path`` always names a file inside a directory, because that
    directory is what gives the skill an identity distinct from its
    neighbours in the same repository.
    """
    suffix = f"/{SKILL_FILENAME}"
    if not skill_path.endswith(suffix):
        raise _error(f"{skill_path} does not name a {SKILL_FILENAME} inside a directory")
    skill_dir = skill_path[: -len(suffix)]
    slug = f"{owner.lower()}/{repo.lower()}"
    source_ref = f"{slug}/{skill_dir}"
    html_url = _docs_url(owner=owner, repo=repo, skill_path=skill_path)
    payload: JsonObject = {
        "skill_name": frontmatter.name,
        "name": frontmatter.name,
        "description": frontmatter.description or plugin.description,
        "category": plugin.category,
        "source_ref": source_ref[:_MAX_SOURCE_REF_CHARS],
        "skill_path": skill_path,
        "commit_sha": commit_sha,
        "model_invocable": not frontmatter.disable_model_invocation,
        "allowed_tools": list(frontmatter.allowed_tools),
        "skill_version": frontmatter.version,
        "frontmatter_bytes": frontmatter.frontmatter_bytes,
        "license": frontmatter.license or plugin.license,
        "tags": list(plugin.keywords),
        "docs_url": html_url,
        "marketplace": plugin.marketplace,
        "marketplace_repo": plugin.marketplace_repo,
        "repo": {
            "host": "github.com",
            "owner": owner.lower(),
            "repo": repo.lower(),
            "subpath": skill_dir,
        },
        "plugin": {
            "marketplace": plugin.marketplace,
            "marketplace_repo": plugin.marketplace_repo,
            "plugin": plugin.plugin,
            "source_kind": plugin.source_kind,
            "source_value": plugin.source_value,
            "sha": plugin.source_sha or commit_sha,
        },
        "renamed_from": list(renamed_from),
    }
    _assert_pointer_only(payload)
    return RawRecord(
        source_id=_SOURCE_ID,
        upstream_id=f"{plugin.marketplace_repo}#{plugin.plugin}#{skill_dir}",
        url=html_url,
        payload=payload,
    )


class _Crawl:
    """One run of :meth:`MarketplacesSource.fetch` and the state it carries.

    Repositories are visited in a fixed order — the seeds as listed, then the
    topic search's results sorted by full name — and each repository's tree
    is listed once and reused by every plugin that resolves into it, so the
    same corpus produces the same records in the same order every time.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        limits: SourceLimits,
        sleep: Callable[[float], Awaitable[None]],
    ) -> None:
        self._client = client
        self._limits = limits
        self._sleep = sleep
        self._headers = github_headers(limits.github_token)
        self._search_bucket = TokenBucket(rate_per_minute=SEARCH_RATE_PER_MINUTE, sleep=sleep)
        self._core_bucket = TokenBucket(rate_per_minute=CORE_RATE_PER_MINUTE, sleep=sleep)
        # Unlike every other source, this one fetches a file per *skill* from
        # a CDN that no API bucket covers, and the number of those a manifest
        # can ask for is attacker-chosen. So the raw bucket is never absent:
        # an unset rate falls back to a ceiling rather than to no limit at all.
        self._raw_bucket = TokenBucket(
            rate_per_minute=limits.requests_per_minute or RAW_RATE_PER_MINUTE, sleep=sleep
        )
        # The whole of the injection surface this source has. A skill's
        # ``description`` is attacker-authored free text that ends up in front
        # of Jhin's agents, the diff gate never blocks additions, and one
        # commit adding a topic is all it takes to be crawled. So when
        # ``curated/skills.yaml`` says ``require_allowlist``, a repository
        # nobody reviewed produces no entry — which is what that file has
        # always claimed happens.
        self._allowlist: frozenset[str] | None = (
            frozenset(name.lower() for name in (*limits.marketplace_allowlist, *SEED_REPOS))
            if limits.require_marketplace_allowlist
            else None
        )
        self._rejected: set[str] = set()
        self._digest = RollingDigest()
        self._pages = 0
        self._records: list[RawRecord] = []
        self._trees: dict[str, JsonObject] = {}
        self._tree_shas: dict[str, str] = {}
        self._tree_failures: set[str] = set()
        self._budget: dict[str, int] = {}

    def _count(self, result: http.FetchResult) -> None:
        """Fold one consumed response into the page count and the digest."""
        self._pages += 1
        self._digest.update(result.body)

    async def _api_json(
        self, url: str, *, params: Mapping[str, str | int] | None = None, search: bool = False
    ) -> JsonValue:
        """One ``api.github.com`` GET, retried past the secondary-rate 403.

        A 403 is not in ``http.RETRY_STATUSES`` — it is usually a permission
        answer, not a transient one — so the abuse-detection variant GitHub
        serves under load is backed off here instead.
        """
        bucket = self._search_bucket if search else self._core_bucket
        for attempt in range(1, _MAX_FORBIDDEN_ATTEMPTS + 1):
            await bucket.acquire()
            try:
                payload, result = await http.fetch_json(
                    self._client, url, params=params, headers=self._headers, sleep=self._sleep
                )
            except http.FetchError as exc:
                if exc.status_code == 403 and attempt < _MAX_FORBIDDEN_ATTEMPTS:
                    await self._sleep(http.backoff_delay(attempt))
                    continue
                raise
            self._count(result)
            return payload
        raise _error(f"{url} answered 403 on every one of {_MAX_FORBIDDEN_ATTEMPTS} attempts")

    async def _raw_bytes(self, url: str, *, max_bytes: int) -> bytes:
        """One ``raw.githubusercontent.com`` GET, carrying no credentials.

        The token this crawl may hold is for ``api.github.com``. Everything
        read from the CDN is public, so the header never travels there.
        """
        if self._raw_bucket is not None:
            await self._raw_bucket.acquire()
        result = await http.fetch(
            self._client, url, max_response_bytes=max_bytes, sleep=self._sleep
        )
        self._count(result)
        return result.body

    async def _discover(self) -> tuple[str, ...]:
        """Repositories carrying the plugin topic, sorted by full name.

        A discovery failure is fatal. Silently crawling the seeds alone would
        drop most of the corpus and reach the diff gate as a mass deletion,
        which is a far worse way to learn that the search API is down.
        """
        pages = max(1, min(_DISCOVERY_MAX_PAGES, self._limits.max_pages))
        per_page = max(1, min(self._limits.page_size, _SEARCH_PAGE_SIZE))
        found: dict[str, str] = {}
        for page in range(1, pages + 1):
            params: Mapping[str, str | int] = {
                "q": f"topic:{_PLUGIN_TOPIC}",
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            }
            try:
                payload = await self._api_json(
                    f"{GITHUB_API}{_SEARCH_PATH}", params=params, search=True
                )
            except http.FetchError as exc:
                raise _error(f"plugin-topic search failed at page {page}: {exc}") from exc
            items = payload.get("items") if isinstance(payload, dict) else None
            if not isinstance(items, list):
                raise _error(f'plugin-topic search page {page} carries no "items" list')
            if not items:
                break
            for item in items:
                if not isinstance(item, dict):
                    continue
                full_name = _clip(item.get("full_name"), limit=165)
                if _split_full_name(full_name) is None:
                    continue
                if self._allowlist is not None and full_name.lower() not in self._allowlist:
                    self._rejected.add(full_name.lower())
                    continue
                found.setdefault(full_name.lower(), full_name)
            if len(items) < per_page:
                break
        return tuple(found[key] for key in sorted(found))

    async def _manifest(self, repo_full_name: str) -> tuple[JsonValue, str] | None:
        """The first marketplace manifest this repository answers with.

        ``.claude-plugin/marketplace.json`` first, then the cross-agent
        aliases in order, so a repository publishing several of them is
        ingested once. All-404 means the repository is simply not a
        marketplace, which is an absence rather than a fault.
        """
        for path in (MARKETPLACE_PATH, *ALT_MARKETPLACE_PATHS):
            url = f"{RAW_BASE}/{repo_full_name}/{_DEFAULT_REF}/{path}"
            try:
                body = await self._raw_bytes(url, max_bytes=_MAX_MANIFEST_BYTES)
            except http.FetchError as exc:
                if exc.status_code == 404:
                    continue
                raise
            try:
                payload = cast("JsonValue", json.loads(body))
            except json.JSONDecodeError as exc:
                raise _error(f"{url} is not valid JSON") from exc
            return payload, path
        return None

    async def _tree(self, owner: str, repo: str) -> JsonObject | None:
        """This repository's recursive tree, listed once per crawl."""
        key = f"{owner}/{repo}".lower()
        cached = self._trees.get(key)
        if cached is not None:
            return cached
        if key in self._tree_failures:
            return None
        url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{_DEFAULT_REF}"
        try:
            payload = await self._api_json(url, params={"recursive": 1})
        except http.FetchError:
            self._tree_failures.add(key)
            return None
        if not isinstance(payload, dict):
            self._tree_failures.add(key)
            return None
        self._trees[key] = payload
        self._tree_shas[key] = _sha1_or_empty(payload.get("sha"))
        return payload

    async def _frontmatter(self, *, owner: str, repo: str, path: str) -> SkillFrontmatter | None:
        """Read one ``SKILL.md`` and return nothing but its frontmatter.

        The body is bound to ``text`` for the length of this call and to
        nothing else: no caller, no record, and no cache holds it, and the
        digest folds it in and forgets it. That is what keeps agent
        instructions out of this index — the barrier is the return type, not
        a convention a later edit can drift away from.
        """
        url = f"{RAW_BASE}/{owner}/{repo}/{_DEFAULT_REF}/{path}"
        try:
            body = await self._raw_bytes(url, max_bytes=MAX_SKILL_MD_BYTES)
        except http.FetchError:
            return None
        text = body.decode("utf-8", errors="replace")
        try:
            return parse_skill_frontmatter(text)
        except SourceError:
            return None

    def _target_repo(self, plugin: _Plugin) -> tuple[str, str] | None:
        """The GitHub repository holding this plugin's files, when there is one.

        An ``npm`` plugin has no git tree to enumerate and an ``unknown``
        source has no resolvable location at all; both yield no skills rather
        than a guess at where their files might be.
        """
        if plugin.source_kind == "relative":
            return _split_full_name(plugin.marketplace_repo)
        if plugin.source_kind in {"url", "git-subdir", "github"}:
            return _github_repo_from_url(plugin.source_value)
        return None

    def _candidate_paths(
        self, plugin: _Plugin, tree: JsonObject, *, owner: str, repo: str
    ) -> tuple[str, ...]:
        """Every ``SKILL.md`` this plugin claims, from the tree and the manifest.

        The tree scan finds what is committed; the manifest's ``skills``
        field names paths that may sit outside the plugin root, and both sets
        are tried. A path the manifest names but the tree does not hold is
        fetched anyway and skipped on its 404.

        A path too long for the ``docs_url`` built from it is dropped here.
        ``plugin_root`` is itself two bounded values concatenated, so a
        manifest can name a path twice as long as either bound suggests, and
        the record built from one used to raise out of the entire crawl.
        """
        candidates = set(skill_paths(tree, plugin_root=plugin.plugin_root))
        for declared in plugin.declared_skills:
            directory = "/".join(part for part in (plugin.plugin_root, declared) if part)
            if directory:
                candidates.add(f"{directory}/{SKILL_FILENAME}")
        room = MAX_URL_CHARS - len(_docs_url(owner=owner, repo=repo, skill_path=""))
        return tuple(sorted(path for path in candidates if len(path) <= room))

    async def _walk_plugin(self, plugin: _Plugin, *, renamed_from: tuple[str, ...]) -> None:
        """Append one record per readable ``SKILL.md`` this plugin points at."""
        target = self._target_repo(plugin)
        if target is None:
            return
        owner, repo = target
        tree = await self._tree(owner, repo)
        if tree is None:
            return
        key = f"{owner}/{repo}".lower()
        room = MAX_SKILLS_PER_REPO - self._budget.get(key, 0)
        if room <= 0:
            return
        commit_sha = plugin.source_sha or self._tree_shas.get(key, "")
        for path in self._candidate_paths(plugin, tree, owner=owner, repo=repo)[:room]:
            if len(self._records) >= self._limits.max_records:
                return
            self._budget[key] = self._budget.get(key, 0) + 1
            frontmatter = await self._frontmatter(owner=owner, repo=repo, path=path)
            if frontmatter is None:
                continue
            self._records.append(
                _skill_record(
                    plugin,
                    owner=owner,
                    repo=repo,
                    skill_path=path,
                    commit_sha=commit_sha,
                    frontmatter=frontmatter,
                    renamed_from=renamed_from,
                )
            )

    async def _walk_repo(self, repo_full_name: str, *, seed: bool) -> None:
        """Read one repository's marketplace and every skill it points at.

        A seed marketplace failing to answer is a fetch fault and stops the
        crawl; a discovered repository failing is skipped. The seeds are the
        allowlisted core, so their absence is never a real deletion, whereas
        one flaky repository out of hundreds should not cost the run.
        """
        try:
            manifest = await self._manifest(repo_full_name)
        except (http.FetchError, SourceError) as exc:
            if seed:
                raise _error(f"seed marketplace {repo_full_name} is unreadable: {exc}") from exc
            return
        if manifest is None:
            return
        payload, path = manifest
        url = f"https://github.com/{repo_full_name}/blob/{_DEFAULT_REF}/{path}"
        try:
            plugin_records, renames = parse_marketplace(
                payload, marketplace_repo=repo_full_name, url=url
            )
        except SourceError:
            if seed:
                raise
            return
        reverse = _reverse_renames(renames)
        for record in plugin_records:
            if len(self._records) >= self._limits.max_records:
                return
            plugin = _Plugin.model_validate(record.payload)
            # One plugin that will not produce a valid record costs that
            # plugin, not the crawl. A discovered repository is a stranger's;
            # letting its manifest raise past here handed anyone with a GitHub
            # account a way to fail every nightly build.
            try:
                await self._walk_plugin(plugin, renamed_from=reverse.get(plugin.plugin, ()))
            except (SourceError, ValidationError):
                if seed:
                    raise
                continue

    def _report_rejected(self) -> None:
        """Name the candidates the allowlist turned away, for a human to read.

        ``curated/skills.yaml`` promises that a repository the topic search
        finds but nobody reviewed "is reported for review and produces no
        entry". This is the reporting half. It goes to stderr so it never
        contaminates the ``--json`` document on stdout, and the names are
        sorted so two runs of the same corpus print the same lines.
        """
        if not self._rejected:
            return
        listed = ", ".join(sorted(self._rejected))
        print(
            f"{_SOURCE_ID}: {len(self._rejected)} repository/-ies carrying "
            f"topic:{_PLUGIN_TOPIC} are not on the reviewed allowlist in "
            f"curated/skills.yaml and produced no entries: {listed}",
            file=sys.stderr,
        )

    async def _repos(self) -> tuple[tuple[str, bool], ...]:
        """Every repository to visit, in crawl order, flagged seed or not."""
        ordered: list[tuple[str, bool]] = []
        seen: set[str] = set()
        for full_name in SEED_REPOS:
            if full_name.lower() not in seen:
                seen.add(full_name.lower())
                ordered.append((full_name, True))
        for full_name in await self._discover():
            if full_name.lower() not in seen:
                seen.add(full_name.lower())
                ordered.append((full_name, False))
        return tuple(ordered)

    async def run(self) -> SourceFetch:
        """Crawl every marketplace and return the skills they point at."""
        for full_name, seed in await self._repos():
            if len(self._records) >= self._limits.max_records:
                break
            await self._walk_repo(full_name, seed=seed)
        self._report_rejected()
        return SourceFetch(
            source_id=_SOURCE_ID,
            url=f"{RAW_BASE}/{SEED_REPOS[0]}/{_DEFAULT_REF}/{MARKETPLACE_PATH}",
            sha256=self._digest.hexdigest(),
            entry_count=len(self._records),
            page_count=self._pages,
            records=tuple(self._records),
        )


class MarketplacesSource(Source):
    """The Claude Code plugin marketplaces, read as a source of skill pointers.

    ``limits.max_pages`` bounds the topic search's pages, the only paginated
    listing here; ``limits.max_records`` bounds the skills emitted;
    ``limits.detail_top_n`` has no meaning for this source and is ignored.
    """

    source_id: ClassVar[str] = "marketplaces"

    async def fetch(
        self,
        client: httpx.AsyncClient,
        *,
        limits: SourceLimits = DEFAULT_LIMITS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> SourceFetch:
        """Crawl this source to exhaustion (within ``limits``)."""
        return await _Crawl(client, limits=limits, sleep=sleep).run()
