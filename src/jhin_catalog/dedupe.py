"""Identity: which records are the same thing, and which field wins.

Every candidate arrives holding the identity keys its source could prove.
This module unions candidates that share a key, demotes a repository key that
several distinct servers claim, elects one canonical key per component, and
merges the fields under a fixed source precedence. It then applies the two
places a human overrules a crawl: the curated overlay and the denylist.
Nothing here reads the network or the clock, and no step depends on the order
the candidates arrived in — the same corpus always yields the same components
with the same keys.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Final

from pydantic import ValidationError

from jhin_catalog.normalize import (
    KEY_SPACE_RANK,
    build_entry,
    repo_key,
    slugify,
)
from jhin_catalog.types import (
    ENTRY_ADAPTER,
    ENTRY_MODELS,
    MAX_ALIAS_KEYS,
    MAX_PACKAGES,
    MAX_REMOTES,
    MAX_SOURCES,
    MAX_TAG_CHARS,
    MAX_TAGS,
    SOURCE_RANK,
    Candidate,
    CatalogEntry,
    CuratedError,
    CuratedOverride,
    DedupeError,
    DenylistItem,
    JsonObject,
    JsonValue,
    McpEntry,
    MergedCandidate,
    NormalizeError,
    PopularitySignals,
    SourceRef,
)

__all__ = [
    "KEY_SPACE_RANK",
    "ambiguous_repo_keys",
    "apply_curated",
    "elect_primary_key",
    "elect_trust_tier",
    "merge_candidates",
    "merge_fields",
    "merge_signals",
    "repo_fanout",
    "unresolved_denylist_keys",
]

_UNKNOWN_SPACE_RANK: Final[int] = 99

# The page a curated record points at when it has no upstream of its own.
_CURATED_SOURCE_URL: Final[str] = "https://github.com/jhin-dev/jhin-catalog/tree/main/curated"
_CURATED_SOURCE_ID: Final[str] = "curated"

_SIGNAL_FIELDS: Final[tuple[str, ...]] = (
    "github_stars",
    "github_forks",
    "npm_downloads_monthly",
    "npm_dependents",
    "smithery_use_count",
    "registry_version_count",
)

# Fields the generic precedence rule never decides: identity and score are
# assigned by ``build_entry``, and the rest have their own rules below.
_MERGE_EXCLUDED: Final[frozenset[str]] = frozenset(
    {
        "alias_keys",
        "canonical_key",
        "curated_fields",
        "deprecated",
        "icon_url",
        "kind",
        "marketplace_reviewed",
        "packages",
        "popularity",
        "popularity_signals",
        "remotes",
        "schema_version",
        "slug",
        "sources",
        "stdio_only",
        "tags",
        "trust_tier",
        "url_unverified",
    }
)

# An override may correct any statement the catalog makes about a record. It
# may not change what the record *is*: the discriminator and the identity are
# what a consumer reconciles against, and a curated file that could rewrite
# them could silently retarget a row a deployment already synced.
_UNSETTABLE_FIELDS: Final[frozenset[str]] = frozenset({"canonical_key", "kind", "schema_version"})


def repo_fanout(candidates: Sequence[Candidate]) -> Mapping[str, int]:
    """How many candidates claim each repository key."""
    counts: Counter[str] = Counter()
    for candidate in candidates:
        for key in candidate.alias_keys:
            if _key_space(key) == "repo":
                counts[key] += 1
    return dict(sorted(counts.items()))


def ambiguous_repo_keys(candidates: Sequence[Candidate]) -> frozenset[str]:
    """Repo keys claimed by more than one candidate with conflicting identities.

    A monorepo publishes many servers from one repository, so a repository key
    alone cannot decide identity there. What separates the two cases is
    whether the claimants contradict each other: ``server-filesystem`` and
    ``server-git`` both name a registry server and name different ones, which
    is a conflict, while a registry row, an npm row, and a topic-search row
    for the same Tavily server name different *kinds* of thing and contradict
    nothing. So the test is per key space — two claimants that both name a
    registry server, both name a package, or both name an endpoint, and share
    none of it, are two different things sharing an address. A claimant whose
    only evidence is the repository, such as a topic-search hit, is not
    evidence of a conflict and cannot split a component two richer records
    agree on.
    """
    claims: dict[str, list[dict[str, frozenset[str]]]] = {}
    for candidate in candidates:
        spaces: dict[str, set[str]] = {}
        for key in candidate.alias_keys:
            space = _key_space(key)
            if space != "repo":
                spaces.setdefault(space, set()).add(key)
        held = {space: frozenset(keys) for space, keys in spaces.items()}
        for key in candidate.alias_keys:
            if _key_space(key) == "repo":
                claims.setdefault(key, []).append(held)

    ambiguous = {key for key, holders in claims.items() if _claimants_conflict(holders)}
    return frozenset(ambiguous)


def _claimants_conflict(holders: Sequence[Mapping[str, frozenset[str]]]) -> bool:
    """Whether two claimants of one repository name different things in one space."""
    if len(holders) < 2:
        return False
    spaces = sorted({space for held in holders for space in held})
    for space in spaces:
        declaring = [held[space] for held in holders if space in held]
        if len(declaring) < 2 or all(held == declaring[0] for held in declaring[1:]):
            continue
        for index, first in enumerate(declaring):
            if any(first.isdisjoint(other) for other in declaring[index + 1 :]):
                return True
    return False


def elect_primary_key(keys: Iterable[str]) -> str:
    """``min`` under ``(KEY_SPACE_RANK[space], key)``. Raises ``DedupeError`` on empty."""
    ordered = sorted(set(keys))
    if not ordered:
        raise DedupeError("a component must hold at least one identity key")
    return min(
        ordered,
        key=lambda key: (KEY_SPACE_RANK.get(_key_space(key), _UNKNOWN_SPACE_RANK), key),
    )


def merge_signals(candidates: Sequence[Candidate]) -> PopularitySignals:
    """Per-field maximum over non-``None`` values.

    The maximum rather than the latest, because two sources counting the same
    repository disagree by staleness far more often than by substance, and the
    fresher crawl is the larger number.
    """
    merged = PopularitySignals()
    for candidate in candidates:
        merged = _max_signals(merged, candidate.signals)
    return merged


def elect_trust_tier(candidates: Sequence[Candidate], *, curated_keys: AbstractSet[str]) -> str:
    """The strongest tier any contributing candidate earns, top rule first."""
    for candidate in candidates:
        if candidate.trust_hint == "curated":
            return "curated"
        if curated_keys and not curated_keys.isdisjoint(candidate.alias_keys):
            return "curated"
    for candidate in candidates:
        if candidate.source_id == "registry" and candidate.trust_hint == "registry_verified":
            return "registry_verified"
    for candidate in candidates:
        if candidate.source_id == "smithery" and candidate.trust_hint == "smithery_verified":
            return "smithery_verified"
    return "indexed"


def merge_fields(candidates: Sequence[Candidate]) -> JsonObject:
    """Source-ranked, informative-value-wins field merge.

    Candidates are ordered by ``(SOURCE_RANK, upstream_id)`` and the first
    informative value for each field wins, where informative means not
    ``None``, not empty, and — for a boolean — not ``False``. The asymmetry on
    booleans is deliberate: every flag in the schema is an assertion, so a
    source that says nothing must not overwrite a source that says something.
    Seven fields have their own rules: ``deprecated`` and
    ``marketplace_reviewed`` are logical ORs, ``icon_url`` prefers the
    Smithery icon route over an owner avatar, ``tags`` is a union,
    ``packages`` and ``remotes`` are unions on their identity, and the
    derived flags are dropped for ``build_entry`` to recompute.
    """
    ordered = sorted(candidates, key=_candidate_order)
    merged: JsonObject = {}
    for candidate in ordered:
        for name in sorted(candidate.fields):
            if name in _MERGE_EXCLUDED or name in merged:
                continue
            value = candidate.fields[name]
            if _informative(value):
                merged[name] = value

    merged["deprecated"] = any(candidate.fields.get("deprecated") is True for candidate in ordered)
    merged["marketplace_reviewed"] = any(
        candidate.fields.get("marketplace_reviewed") is True for candidate in ordered
    )
    icon_url = _merge_icon_url(ordered)
    if icon_url:
        merged["icon_url"] = icon_url
    tags = _merge_tags(ordered)
    if tags:
        merged["tags"] = tags
    packages = _merge_keyed(ordered, "packages", ("registry_type", "identifier"), MAX_PACKAGES)
    if packages:
        merged["packages"] = packages
    remotes = _merge_keyed(ordered, "remotes", ("transport", "url"), MAX_REMOTES)
    if remotes:
        merged["remotes"] = remotes
    return merged


def merge_candidates(
    candidates: Iterable[Candidate], *, curated_keys: AbstractSet[str] = frozenset()
) -> tuple[MergedCandidate, ...]:
    """Union-find over alias keys, sorted by ``canonical_key``.

    Signals join before identity does, and independently of it: a candidate
    carrying GitHub counts lends them to every candidate on the same
    repository, whether or not that repository is strong enough to merge them.
    That is what lets the sixty thousand stars of a monorepo reach each of the
    servers inside it while still keeping those servers apart. A candidate
    left with no key at all after demotion is dropped, because it has nothing
    left that distinguishes it from its siblings.

    Raises ``DedupeError`` if a component ever spans two ``kind`` values.
    """
    items = list(candidates)
    for candidate in items:
        prefix = f"{candidate.kind}:"
        for key in candidate.alias_keys:
            if not key.startswith(prefix):
                raise DedupeError(
                    f"candidate {candidate.primary_key!r} holds {key!r}, a key of another kind"
                )

    items = _join_repo_signals(items, _repo_signal_index(items))
    ambiguous = ambiguous_repo_keys(items)

    surviving: list[tuple[Candidate, tuple[str, ...]]] = []
    for candidate in items:
        keys = tuple(key for key in candidate.alias_keys if key not in ambiguous)
        if keys:
            surviving.append((candidate, keys))

    union = _DisjointSet()
    for _, keys in surviving:
        for key in keys:
            union.add(key)
    for _, keys in surviving:
        for key in keys[1:]:
            union.merge(keys[0], key)

    groups: dict[str, list[tuple[Candidate, tuple[str, ...]]]] = {}
    for candidate, keys in surviving:
        groups.setdefault(union.root(keys[0]), []).append((candidate, keys))

    merged: list[MergedCandidate] = []
    for members in groups.values():
        kinds = {candidate.kind for candidate, _ in members}
        if len(kinds) != 1:
            raise DedupeError(f"one component spans several kinds: {sorted(kinds)}")
        component_keys = sorted({key for _, keys in members for key in keys})
        canonical_key = elect_primary_key(component_keys)
        aliases = tuple(key for key in component_keys if key != canonical_key)
        group = tuple(sorted((candidate for candidate, _ in members), key=_candidate_order))
        demoted = sorted(
            {key for candidate, _ in members for key in candidate.alias_keys if key in ambiguous}
        )
        merged.append(
            MergedCandidate(
                kind=group[0].kind,
                canonical_key=canonical_key,
                alias_keys=aliases[:MAX_ALIAS_KEYS],
                ambiguous_repo_keys=tuple(demoted),
                candidates=group,
                signals=merge_signals(group),
                trust_tier=elect_trust_tier(group, curated_keys=curated_keys),
                fields=merge_fields(group),
            )
        )
    merged.sort(key=lambda item: item.canonical_key)
    return tuple(merged)


def apply_curated(
    entries: Sequence[CatalogEntry],
    *,
    overrides: Sequence[CuratedOverride],
    denylist: Sequence[DenylistItem],
) -> tuple[CatalogEntry, ...]:
    """Drop denylisted entries, then overlay curated fields per field.

    An override whose ``key`` matches no entry's ``canonical_key`` or
    ``alias_keys`` creates a new entry from its ``fields`` alone, which is how
    a hand-written record for a native connector reaches the catalog at all.
    Every field an override sets is recorded in ``curated_fields`` so a reader
    can tell a human judgement from a crawled one.

    A denylist key that matches nothing is not fatal. It was, and that made a
    denial into a liability: the typosquats this list exists to remove are
    exactly the packages an upstream later takes down, and a key that stopped
    resolving would then fail every nightly build until a human edited the
    file. Worse, anyone could force it — publishing a second package that
    claims the same repository demotes a denied repo key out of the corpus.
    Unresolved keys are returned to the caller to report instead.

    A denial removes an identity, not one row. Every key of the entry it
    resolved to is remembered, so a curated override naming the entry's
    canonical key cannot republish it at the strongest tier there is, and no
    surviving entry can pick a denied key up as an alias.
    """
    locate = _locate_index(entries)
    surviving: dict[str, CatalogEntry] = {entry.canonical_key: entry for entry in entries}
    denied: set[str] = set()
    for item in denylist:
        denied.add(item.key)
        canonical_key = locate.get(item.key)
        if canonical_key is None:
            continue
        removed = surviving.pop(canonical_key, None)
        denied.add(canonical_key)
        if removed is not None:
            denied.update(removed.alias_keys)

    for override in sorted(overrides, key=lambda item: item.key):
        clashing = sorted(denied.intersection({override.key, *override.aliases}))
        if clashing:
            raise CuratedError(
                f"curated key {override.key!r} is also denylisted (via {', '.join(clashing)})"
            )
        canonical_key = locate.get(override.key)
        updated = (
            _create_curated(override)
            if canonical_key is None or canonical_key not in surviving
            else _overlay(surviving[canonical_key], override, denied=denied)
        )
        surviving[updated.canonical_key] = updated
        for key in (updated.canonical_key, *updated.alias_keys):
            locate.setdefault(key, updated.canonical_key)

    return tuple(sorted(surviving.values(), key=lambda entry: entry.canonical_key))


def unresolved_denylist_keys(
    entries: Sequence[CatalogEntry], denylist: Sequence[DenylistItem]
) -> tuple[str, ...]:
    """Denylist keys that name nothing in this corpus, sorted.

    Reported rather than raised. A stale denial is worth a human's attention —
    it may mean the denial silently stopped working — but it is not worth
    refusing to publish a catalog over, because the same symptom is produced
    by an upstream doing the right thing and taking a bad package down.
    """
    locate = _locate_index(entries)
    return tuple(sorted({item.key for item in denylist if item.key not in locate}))


# ---------------------------------------------------------------------------
# Curated overlay
# ---------------------------------------------------------------------------


def _create_curated(override: CuratedOverride) -> CatalogEntry:
    """A whole entry from an override alone, then overlaid with its own fields.

    The base is built through ``build_entry`` so a curated record gets the
    same derivation, the same bounds, and the same validation as a crawled
    one. The overlay then runs a second time over that base, because the
    derived flags ``build_entry`` recomputes are exactly the ones a curator
    most often needs to state by hand.
    """
    candidate = _curated_candidate(override)
    merged = MergedCandidate(
        kind=override.kind,
        canonical_key=override.key,
        alias_keys=tuple(sorted(set(override.aliases) - {override.key}))[:MAX_ALIAS_KEYS],
        ambiguous_repo_keys=(),
        candidates=(candidate,),
        signals=PopularitySignals(),
        trust_tier="curated",
        fields={
            name: value for name, value in override.fields.items() if name not in _UNSETTABLE_FIELDS
        },
    )
    try:
        base = build_entry(merged, popularity=0.0, slug=_curated_slug(override))
    except NormalizeError as exc:
        raise CuratedError(f"curated entry {override.key!r} is not buildable: {exc}") from exc
    return _overlay(base, override)


def _curated_candidate(override: CuratedOverride) -> Candidate:
    """The synthetic candidate a curated-only entry is built from.

    It carries the override's key and aliases and nothing else, so the entry
    that comes out of ``build_entry`` is entirely the curator's statement.
    """
    keys = tuple(sorted({override.key, *override.aliases}))
    return Candidate(
        kind=override.kind,
        source_id=_CURATED_SOURCE_ID,
        upstream_id=override.key[:200],
        primary_key=elect_primary_key(keys),
        alias_keys=keys,
        repo=None,
        signals=PopularitySignals(),
        trust_hint="curated",
        source_ref=_curated_source_ref(override),
        fields={},
    )


def _overlay(
    entry: CatalogEntry, override: CuratedOverride, *, denied: AbstractSet[str] = frozenset()
) -> CatalogEntry:
    """One entry with every field the override states, then re-validated.

    ``denied`` keys are kept out of the alias union: an override may widen an
    entry's identity but may not re-attach one a denial removed, or any
    consumer resolving by that alias would resurrect the denied record.
    """
    allowed = _settable_fields(override.kind)
    payload = entry.model_dump(mode="json")
    touched: list[str] = []
    for name in sorted(override.fields):
        if name in _UNSETTABLE_FIELDS:
            raise CuratedError(f"curated override {override.key!r} may not set {name!r}")
        if name not in allowed:
            raise CuratedError(f"curated override {override.key!r} names unknown field {name!r}")
        payload[name] = override.fields[name]
        touched.append(name)

    if "trust_tier" not in touched:
        payload["trust_tier"] = "curated"
    payload["curated_fields"] = _as_json_strings(sorted(set(entry.curated_fields) | set(touched)))
    payload["alias_keys"] = _as_json_strings(
        sorted(
            (set(entry.alias_keys) | set(override.aliases)) - {entry.canonical_key} - set(denied)
        )[:MAX_ALIAS_KEYS]
    )
    payload["sources"] = _with_curated_source(entry.sources, override)
    if isinstance(entry, McpEntry):
        _reconcile_mcp(payload, touched)

    try:
        return ENTRY_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise CuratedError(f"curated override {override.key!r} is invalid: {exc}") from exc


def _reconcile_mcp(payload: JsonObject, touched: Sequence[str]) -> None:
    """Recompute the derived flags an override did not state for itself."""
    mcp_url = payload.get("mcp_url")
    packages = payload.get("packages")
    if "stdio_only" not in touched:
        payload["stdio_only"] = mcp_url is None and bool(packages)
    if "url_unverified" not in touched:
        payload["url_unverified"] = _url_unverified(mcp_url, payload.get("connector_type"))


def _url_unverified(mcp_url: JsonValue, connector_type: JsonValue) -> bool:
    """A curated endpoint is verified by the curation; a native one has none.

    A record that names no endpoint and no connector is offering nothing a
    person could dial, so the flag stands as a warning; one whose connection
    is a native Jhin connector has no published URL to be wrong about.
    """
    if mcp_url is not None:
        return False
    return connector_type is None


def _curated_slug(override: CuratedOverride) -> str:
    """The slug a curated record asks for, or one derived from what it says."""
    for source in (override.fields.get("slug"), override.fields.get("name"), override.key):
        if isinstance(source, str) and source.strip():
            try:
                return slugify(source.rsplit(":", 1)[-1].rsplit("/", 1)[-1])
            except NormalizeError:
                continue
    raise CuratedError(f"curated entry {override.key!r} yields no slug")


def _curated_source_ref(override: CuratedOverride) -> SourceRef:
    """The reference a curated record carries, pointing at the curated file."""
    for name in ("docs_url", "homepage", "mcp_url"):
        value = override.fields.get(name)
        if isinstance(value, str) and value.startswith("https://"):
            return SourceRef(
                source_id=_CURATED_SOURCE_ID, upstream_id=override.key[:200], url=value
            )
    return SourceRef(
        source_id=_CURATED_SOURCE_ID, upstream_id=override.key[:200], url=_CURATED_SOURCE_URL
    )


def _with_curated_source(
    sources: Sequence[SourceRef], override: CuratedOverride
) -> list[JsonValue]:
    """The entry's sources with the curated file added, ranked and bounded."""
    refs = list(sources)
    if not any(ref.source_id == _CURATED_SOURCE_ID for ref in refs):
        refs.append(_curated_source_ref(override))
    refs.sort(key=lambda ref: (SOURCE_RANK.get(ref.source_id, len(SOURCE_RANK)), ref.upstream_id))
    return [ref.model_dump(mode="json") for ref in refs[:MAX_SOURCES]]


def _settable_fields(kind: str) -> frozenset[str]:
    """Every field name an override of this kind is allowed to name.

    Raises ``KeyError`` on a kind that is neither. The previous spelling —
    ``McpEntry if kind == "mcp" else SkillEntry`` — silently treated any
    unrecognised kind as a skill, which is how a typo becomes a validated
    entry of the wrong shape rather than a loud failure.
    """
    return frozenset(ENTRY_MODELS[kind].model_fields)


def _locate_index(entries: Sequence[CatalogEntry]) -> dict[str, str]:
    """Every canonical and alias key, pointing at the canonical key that owns it.

    The index maps to keys rather than to entries so that it stays correct as
    the overlay replaces entries, and entries are visited in canonical-key
    order so an alias two of them claim resolves to the same one every build.
    """
    index: dict[str, str] = {}
    for entry in sorted(entries, key=lambda item: item.canonical_key):
        for key in (entry.canonical_key, *entry.alias_keys):
            index.setdefault(key, entry.canonical_key)
    return index


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def _repo_signal_index(candidates: Sequence[Candidate]) -> dict[str, PopularitySignals]:
    """GitHub counts gathered per repository key, however weak that key is."""
    grouped: dict[str, PopularitySignals] = {}
    for candidate in candidates:
        if candidate.repo is None:
            continue
        if candidate.signals.github_stars is None and candidate.signals.github_forks is None:
            continue
        key = repo_key(candidate.repo, kind=candidate.kind)
        grouped[key] = _max_signals(grouped.get(key, PopularitySignals()), candidate.signals)
    return grouped


def _join_repo_signals(
    candidates: Sequence[Candidate], index: Mapping[str, PopularitySignals]
) -> list[Candidate]:
    """Every candidate lent the counts of the repository it names."""
    joined: list[Candidate] = []
    for candidate in candidates:
        if candidate.repo is None:
            joined.append(candidate)
            continue
        extra = index.get(repo_key(candidate.repo, kind=candidate.kind))
        if extra is None:
            joined.append(candidate)
            continue
        merged = _max_signals(candidate.signals, extra)
        if merged == candidate.signals:
            joined.append(candidate)
        else:
            joined.append(candidate.model_copy(update={"signals": merged}))
    return joined


def _max_signals(left: PopularitySignals, right: PopularitySignals) -> PopularitySignals:
    """Two signal sets folded together, taking the larger of each pair."""
    values: dict[str, int | None] = {}
    for name in _SIGNAL_FIELDS:
        values[name] = _larger(_signal(left, name), _signal(right, name))
    return PopularitySignals(
        github_stars=values["github_stars"],
        github_forks=values["github_forks"],
        npm_downloads_monthly=values["npm_downloads_monthly"],
        npm_dependents=values["npm_dependents"],
        smithery_use_count=values["smithery_use_count"],
        registry_version_count=values["registry_version_count"],
    )


def _signal(signals: PopularitySignals, name: str) -> int | None:
    """One signal by name, typed rather than fetched with ``getattr``."""
    match name:
        case "github_stars":
            return signals.github_stars
        case "github_forks":
            return signals.github_forks
        case "npm_downloads_monthly":
            return signals.npm_downloads_monthly
        case "npm_dependents":
            return signals.npm_dependents
        case "smithery_use_count":
            return signals.smithery_use_count
        case "registry_version_count":
            return signals.registry_version_count
        case _:
            return None


def _larger(left: int | None, right: int | None) -> int | None:
    """The larger of two counts, where ``None`` means nobody reported one."""
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


# ---------------------------------------------------------------------------
# Field merging
# ---------------------------------------------------------------------------


def _informative(value: JsonValue) -> bool:
    """Whether a value asserts something, as opposed to declining to."""
    if value is None or value is False:
        return False
    return not (isinstance(value, str | list | dict) and not value)


def _candidate_order(candidate: Candidate) -> tuple[int, str, str]:
    """The precedence order: source rank, then upstream id, then key."""
    return (
        SOURCE_RANK.get(candidate.source_id, len(SOURCE_RANK)),
        candidate.upstream_id,
        candidate.primary_key,
    )


def _merge_icon_url(candidates: Sequence[Candidate]) -> str:
    """The one icon URL a component keeps across its candidates.

    A Smithery icon route beats an owner avatar, because Smithery serves the
    mark the server's own publisher uploaded while an avatar is the owner's
    face for everything they have ever published. With no Smithery value the
    first non-empty URL in candidate order stands, which is the same
    precedence every other merged field follows.
    """
    values = [
        value for candidate in candidates if (value := _text(candidate.fields.get("icon_url")))
    ]
    for value in values:
        if "api.smithery.ai" in value:
            return value
    return values[0] if values else ""


def _merge_tags(candidates: Sequence[Candidate]) -> list[JsonValue]:
    """The union of every candidate's tags, sorted and bounded."""
    tags: set[str] = set()
    for candidate in candidates:
        value = candidate.fields.get("tags")
        if isinstance(value, list):
            tags.update(
                item.strip().lower()[:MAX_TAG_CHARS]
                for item in value
                if isinstance(item, str) and item.strip()
            )
    return _as_json_strings(sorted(tags)[:MAX_TAGS])


def _merge_keyed(
    candidates: Sequence[Candidate], field: str, identity: tuple[str, ...], limit: int
) -> list[JsonValue]:
    """A union of object lists keyed on the fields that identify each item."""
    seen: dict[tuple[str, ...], JsonObject] = {}
    for candidate in candidates:
        value = candidate.fields.get(field)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            key = tuple(_text(item.get(name)) for name in identity)
            seen.setdefault(key, item)
    ordered = [seen[key] for key in sorted(seen)]
    return list(ordered[:limit])


def _as_json_strings(values: Iterable[str]) -> list[JsonValue]:
    """Strings widened to JSON values so they can sit in an entry payload."""
    return list(values)


def _text(value: JsonValue) -> str:
    """A trimmed string, or ``""`` when the value is not a string.

    Trimmed, like the identically named helpers in ``normalize`` and
    ``build``. This one did not, and ``_merge_keyed`` builds package and
    remote identity tuples with it — so ``"npm"`` and ``" npm"`` were two
    distinct packages here while every other module treated them as one.
    """
    return value.strip() if isinstance(value, str) else ""


def _key_space(key: str) -> str:
    """The middle segment of a key, or ``""`` when the key is malformed."""
    parts = key.split(":", 2)
    return parts[1] if len(parts) == 3 else ""


class _DisjointSet:
    """Union-find over identity keys, with a deterministic representative.

    The smaller key always becomes the root, so the component a key lands in
    never depends on the order the unions happened to run in.
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, key: str) -> None:
        """Register a key as a component of its own."""
        self._parent.setdefault(key, key)

    def root(self, key: str) -> str:
        """The representative of the component holding ``key``."""
        self.add(key)
        current = key
        while self._parent[current] != current:
            self._parent[current] = self._parent[self._parent[current]]
            current = self._parent[current]
        return current

    def merge(self, left: str, right: str) -> None:
        """Join two components, keeping the lexicographically smaller root."""
        left_root, right_root = self.root(left), self.root(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self._parent[high] = low
