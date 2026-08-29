"""Identity: union by alias key, monorepo demotion, field precedence."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pytest

from jhin_catalog.dedupe import (
    ambiguous_repo_keys,
    apply_curated,
    elect_primary_key,
    elect_trust_tier,
    merge_candidates,
    merge_fields,
    merge_signals,
    repo_fanout,
    unresolved_denylist_keys,
)
from jhin_catalog.types import (
    Candidate,
    CatalogEntry,
    CuratedError,
    CuratedOverride,
    DedupeError,
    DenylistItem,
    JsonObject,
    McpEntry,
    PopularitySignals,
    RepoRef,
    SourceRef,
)

TAVILY_REPO = "mcp:repo:github.com/tavily-ai/tavily-mcp"
TAVILY_URL = "mcp:url:mcp.tavily.com/mcp"
TAVILY_REGISTRY = "mcp:registry:io.github.tavily-ai/tavily-mcp"
TAVILY_NPM = "mcp:npm:tavily-mcp"

SERVERS_REPO = "mcp:repo:github.com/modelcontextprotocol/servers"
FILESYSTEM = "mcp:registry:io.github.modelcontextprotocol/server-filesystem"
GIT = "mcp:registry:io.github.modelcontextprotocol/server-git"

BRAVE_REPO = "mcp:repo:github.com/brave/brave-search-mcp-server"
BRAVE_URL = "mcp:url:brave.run.tools"
BRAVE_REGISTRY = "mcp:registry:io.github.brave/brave-search"
BRAVE_SMITHERY = "mcp:smithery:brave"

ADLC_SKILL = "skill:skill:github.com/salesforceairesearch/agentforce-adlc/skills/design-review"
ADLC_NEW = "skill:plugin:anthropics-official/agentforce-adlc/design-review"
ADLC_OLD = "skill:plugin:anthropics-official/adlc/design-review"


def _candidate(
    *,
    source_id: str,
    upstream_id: str,
    keys: Iterable[str],
    kind: str = "mcp",
    repo: RepoRef | None = None,
    signals: PopularitySignals | None = None,
    trust_hint: str = "indexed",
    fields: JsonObject | None = None,
) -> Candidate:
    ordered = tuple(sorted(set(keys)))
    return Candidate(
        kind=kind,
        source_id=source_id,
        upstream_id=upstream_id,
        primary_key=elect_primary_key(ordered),
        alias_keys=ordered,
        repo=repo,
        signals=signals or PopularitySignals(),
        trust_hint=trust_hint,
        source_ref=SourceRef(
            source_id=source_id,
            upstream_id=upstream_id,
            url=f"https://{source_id}.example/{upstream_id}",
        ),
        fields=dict(fields or {}),
    )


def _entry(key: str, **overrides: Any) -> McpEntry:
    payload: dict[str, Any] = {
        "kind": "mcp",
        "canonical_key": key,
        "slug": "sample",
        "name": "Sample",
        "description": "An entry the curated overlay is applied on top of.",
        "trust_tier": "indexed",
        "sources": (
            SourceRef(source_id="registry", upstream_id=key, url="https://registry.example/x"),
        ),
        "category": "Developer tools",
        "icon": "terminal",
    }
    payload.update(overrides)
    return McpEntry.model_validate(payload)


def _by_key(merged: Sequence[Any]) -> dict[str, Any]:
    return {item.canonical_key: item for item in merged}


# --- worked example (a) ----------------------------------------------------


def test_worked_example_a_merges_registry_npm_and_topics_into_one_identity() -> None:
    """Three sources, one repository, one entry.

    Note for whoever implements ``ambiguous_repo_keys``: read literally, the
    rule in section 2.2 demotes the repo key here, because the topics
    candidate holds no second key and so the three do not "all share a second
    key in common". That would split this example into two components and
    strand the topics row with no key at all, which contradicts the stated
    outcome below. The worked example is taken as the normative one: a repo
    key is ambiguous only when the candidates claiming it carry conflicting
    independent identities, and a candidate whose only evidence is the
    repository is not evidence of a conflict.
    """
    registry = _candidate(
        source_id="registry",
        upstream_id="io.github.tavily-ai/tavily-mcp",
        keys=(TAVILY_REPO, TAVILY_URL, TAVILY_REGISTRY),
        repo=RepoRef(host="github.com", owner="tavily-ai", repo="tavily-mcp"),
        trust_hint="registry_verified",
    )
    npm = _candidate(
        source_id="npm",
        upstream_id="tavily-mcp",
        keys=(TAVILY_REPO, TAVILY_NPM),
        repo=RepoRef(host="github.com", owner="tavily-ai", repo="tavily-mcp"),
        signals=PopularitySignals(npm_downloads_monthly=50_000),
    )
    topics = _candidate(
        source_id="github_topics",
        upstream_id="tavily-ai/tavily-mcp",
        keys=(TAVILY_REPO,),
        repo=RepoRef(host="github.com", owner="tavily-ai", repo="tavily-mcp"),
        signals=PopularitySignals(github_stars=1_400),
    )

    merged = merge_candidates([registry, npm, topics])

    assert len(merged) == 1
    one = merged[0]
    assert one.canonical_key == TAVILY_REPO
    assert one.alias_keys == (TAVILY_NPM, TAVILY_REGISTRY, TAVILY_URL)
    assert one.trust_tier == "registry_verified"
    assert one.signals.github_stars == 1_400
    assert one.signals.npm_downloads_monthly == 50_000
    assert one.ambiguous_repo_keys == ()


# --- worked example (b) ----------------------------------------------------


def _monorepo_siblings() -> tuple[Candidate, Candidate]:
    repo = RepoRef(host="github.com", owner="modelcontextprotocol", repo="servers")
    filesystem = _candidate(
        source_id="registry",
        upstream_id="io.github.modelcontextprotocol/server-filesystem",
        keys=(SERVERS_REPO, FILESYSTEM),
        repo=repo,
        signals=PopularitySignals(github_stars=60_000),
        trust_hint="registry_verified",
    )
    git = _candidate(
        source_id="registry",
        upstream_id="io.github.modelcontextprotocol/server-git",
        keys=(SERVERS_REPO, GIT),
        repo=repo,
        trust_hint="registry_verified",
    )
    return filesystem, git


def test_worked_example_b_keeps_two_monorepo_servers_apart() -> None:
    merged = merge_candidates(list(_monorepo_siblings()))

    assert len(merged) == 2
    by_key = _by_key(merged)
    assert set(by_key) == {FILESYSTEM, GIT}
    for one in merged:
        assert one.ambiguous_repo_keys == (SERVERS_REPO,)
        assert SERVERS_REPO not in one.alias_keys


def test_worked_example_b_still_gives_both_siblings_the_repository_stars() -> None:
    """Identity and signal join are independent.

    Sixty thousand stars belong to the monorepo, not to either server, so
    stripping the shared key for identity must not also strip the evidence
    that both servers live somewhere popular.
    """
    merged = _by_key(merge_candidates(list(_monorepo_siblings())))
    assert merged[FILESYSTEM].signals.github_stars == 60_000
    assert merged[GIT].signals.github_stars == 60_000


def test_a_repo_key_claimed_once_is_never_ambiguous() -> None:
    lone = _candidate(
        source_id="registry",
        upstream_id="io.github.tavily-ai/tavily-mcp",
        keys=(TAVILY_REPO, TAVILY_REGISTRY),
        repo=RepoRef(host="github.com", owner="tavily-ai", repo="tavily-mcp"),
    )
    assert repo_fanout([lone])[TAVILY_REPO] == 1
    assert ambiguous_repo_keys([lone]) == frozenset()


def test_two_conflicting_claims_on_one_repo_key_make_it_ambiguous() -> None:
    assert ambiguous_repo_keys(list(_monorepo_siblings())) == frozenset({SERVERS_REPO})


def test_a_skill_key_is_never_demoted_however_many_share_the_repository() -> None:
    """A repository legitimately holds many skills, each at its own directory."""
    pack = RepoRef(host="github.com", owner="example-org", repo="pack")
    skills = [
        _candidate(
            kind="skill",
            source_id="marketplaces",
            upstream_id=f"example-org/pack#pack#skills/{name}",
            keys=(f"skill:skill:github.com/example-org/pack/skills/{name}",),
            repo=pack,
        )
        for name in ("alpha", "beta", "gamma")
    ]
    merged = merge_candidates(skills)
    assert len(merged) == 3
    assert all(one.ambiguous_repo_keys == () for one in merged)


# --- worked example (c) ----------------------------------------------------


def test_worked_example_c_unions_a_registry_and_a_smithery_row_on_one_endpoint() -> None:
    registry = _candidate(
        source_id="registry",
        upstream_id="io.github.brave/brave-search",
        keys=(BRAVE_REPO, BRAVE_URL, BRAVE_REGISTRY),
        repo=RepoRef(host="github.com", owner="brave", repo="brave-search-mcp-server"),
        trust_hint="registry_verified",
    )
    smithery = _candidate(
        source_id="smithery",
        upstream_id="brave",
        keys=(BRAVE_URL, BRAVE_SMITHERY),
        signals=PopularitySignals(smithery_use_count=87_579),
        trust_hint="smithery_verified",
    )

    merged = merge_candidates([registry, smithery])

    assert len(merged) == 1
    one = merged[0]
    assert one.canonical_key == BRAVE_REPO
    assert one.alias_keys == (BRAVE_REGISTRY, BRAVE_SMITHERY, BRAVE_URL)
    assert one.trust_tier == "registry_verified"
    assert one.signals.smithery_use_count == 87_579


def test_a_trailing_slash_does_not_make_a_second_endpoint() -> None:
    """The two rows of worked example (c) only meet because the key form is
    normalised; this pins that the key, not the raw URL, is what joins."""
    left = _candidate(source_id="registry", upstream_id="a", keys=(BRAVE_URL,))
    right = _candidate(source_id="smithery", upstream_id="b", keys=(BRAVE_URL,))
    assert len(merge_candidates([left, right])) == 1


# --- worked example (d) ----------------------------------------------------


def test_worked_example_d_keeps_a_renamed_plugin_reachable_through_its_old_name() -> None:
    """The repository path is the identity; both plugin names are aliases.

    Section 2.7(d) writes the primary key as
    ``skill:github.com/salesforceairesearch/…``, which the key grammar of
    section 2.1 cannot produce and ``CANONICAL_KEY_RE`` rejects — the ``skill``
    space segment is missing. The three-segment form is used here.
    """
    candidate = _candidate(
        kind="skill",
        source_id="marketplaces",
        upstream_id="anthropics/claude-plugins-official#agentforce-adlc#skills/design-review",
        keys=(ADLC_SKILL, ADLC_NEW, ADLC_OLD),
        repo=RepoRef(host="github.com", owner="salesforceairesearch", repo="agentforce-adlc"),
    )

    merged = merge_candidates([candidate])

    assert len(merged) == 1
    one = merged[0]
    assert one.canonical_key == ADLC_SKILL
    assert one.alias_keys == (ADLC_OLD, ADLC_NEW)
    assert one.trust_tier == "indexed"


def test_a_previous_build_keyed_on_the_old_plugin_name_reconciles_through_the_alias() -> None:
    stale = _candidate(
        kind="skill",
        source_id="marketplaces",
        upstream_id="anthropics/claude-plugins-official#adlc#skills/design-review",
        keys=(ADLC_OLD,),
    )
    fresh = _candidate(
        kind="skill",
        source_id="marketplaces",
        upstream_id="anthropics/claude-plugins-official#agentforce-adlc#skills/design-review",
        keys=(ADLC_SKILL, ADLC_NEW, ADLC_OLD),
    )
    merged = merge_candidates([stale, fresh])
    assert len(merged) == 1
    assert merged[0].canonical_key == ADLC_SKILL


# --- key election ----------------------------------------------------------


def test_elect_primary_key_prefers_the_strongest_space_then_the_lowest_key() -> None:
    assert elect_primary_key([TAVILY_REGISTRY, TAVILY_REPO, TAVILY_URL]) == TAVILY_REPO
    assert elect_primary_key([TAVILY_NPM, TAVILY_REGISTRY]) == TAVILY_REGISTRY
    assert elect_primary_key([ADLC_NEW, ADLC_SKILL]) == ADLC_SKILL


def test_elect_primary_key_on_nothing_is_an_error() -> None:
    with pytest.raises(DedupeError):
        elect_primary_key([])


def test_a_component_spanning_two_kinds_is_refused() -> None:
    """Structurally unreachable through the model, which is why it is asserted.

    ``Candidate`` will not accept a key whose prefix disagrees with its
    ``kind``, so this bypasses validation to prove the defensive branch is
    real rather than decorative.
    """
    server = _candidate(source_id="registry", upstream_id="a", keys=(TAVILY_REPO,))
    impostor = Candidate.model_construct(
        kind="skill",
        source_id="marketplaces",
        upstream_id="b",
        primary_key=TAVILY_REPO,
        alias_keys=(TAVILY_REPO,),
        repo=None,
        signals=PopularitySignals(),
        trust_hint="indexed",
        source_ref=server.source_ref,
        fields={},
    )
    with pytest.raises(DedupeError):
        merge_candidates([server, impostor])


# --- field precedence ------------------------------------------------------


def _named(source_id: str, upstream_id: str, **fields: Any) -> Candidate:
    return _candidate(
        source_id=source_id,
        upstream_id=upstream_id,
        keys=(TAVILY_REPO,),
        fields=dict(fields),
    )


def test_merge_fields_prefers_registry_then_smithery_then_npm() -> None:
    candidates = [
        _named("npm", "tavily-mcp", name="npm name", license="MIT"),
        _named("smithery", "tavily", name="smithery name", homepage="https://smithery.example"),
        _named("registry", "io.github.tavily-ai/tavily-mcp", name="registry name"),
    ]
    merged = merge_fields(candidates)
    assert merged["name"] == "registry name"
    assert merged["homepage"] == "https://smithery.example"
    assert merged["license"] == "MIT"


def test_an_empty_value_is_not_informative_and_loses_to_a_later_source() -> None:
    candidates = [
        _named("registry", "a", name="", tags=[]),
        _named("npm", "b", name="npm name", tags=["search"]),
    ]
    merged = merge_fields(candidates)
    assert merged["name"] == "npm name"


def test_false_never_beats_true_for_a_boolean_field() -> None:
    candidates = [
        _named("registry", "a", verified_upstream=False),
        _named("smithery", "b", verified_upstream=True),
    ]
    assert merge_fields(candidates)["verified_upstream"] is True


def test_deprecated_is_the_logical_or_across_every_source() -> None:
    candidates = [
        _named("registry", "a", deprecated=False),
        _named("smithery", "b", deprecated=True),
        _named("npm", "c", deprecated=False),
    ]
    assert merge_fields(candidates)["deprecated"] is True


def test_marketplace_reviewed_is_the_logical_or_across_every_source() -> None:
    """One reviewed sighting is enough; a candidate that says nothing cannot
    take the flag away from one that earned it."""
    candidates = [
        _named("marketplaces", "a", marketplace_reviewed=False),
        _named("marketplaces", "b", marketplace_reviewed=True),
    ]
    assert merge_fields(candidates)["marketplace_reviewed"] is True
    assert merge_fields([_named("registry", "a")])["marketplace_reviewed"] is False


def test_the_smithery_icon_route_beats_an_owner_avatar_in_the_merge() -> None:
    """Smithery serves the mark the publisher uploaded; an avatar is the
    owner's face for everything they ever published — even when the avatar
    candidate outranks the Smithery one."""
    candidates = [
        _named("registry", "a", icon_url="https://github.com/tavily-ai.png?size=128"),
        _named("smithery", "b", icon_url="https://api.smithery.ai/servers/tavily/icon"),
    ]
    assert merge_fields(candidates)["icon_url"] == "https://api.smithery.ai/servers/tavily/icon"


def test_icon_url_otherwise_follows_the_ordinary_candidate_precedence() -> None:
    candidates = [
        _named("npm", "b", icon_url="https://github.com/acme-example.png?size=128"),
        _named("registry", "a", icon_url="https://github.com/tavily-ai.png?size=128"),
    ]
    assert merge_fields(candidates)["icon_url"] == "https://github.com/tavily-ai.png?size=128"


def test_a_component_with_no_icon_url_merges_without_the_key() -> None:
    assert "icon_url" not in merge_fields([_named("registry", "a")])


def test_tags_are_unioned_sorted_and_truncated_to_twenty() -> None:
    candidates = [
        _named("registry", "a", tags=[f"tag-{index:02d}" for index in range(15)]),
        _named("npm", "b", tags=[f"tag-{index:02d}" for index in range(10, 30)]),
    ]
    merged = merge_fields(candidates)["tags"]
    assert isinstance(merged, list)
    tags = [tag for tag in merged if isinstance(tag, str)]
    assert len(tags) == 20
    assert tags == sorted(tags)
    assert len(set(tags)) == 20


def test_merge_signals_takes_the_maximum_rather_than_the_last_writer() -> None:
    candidates = [
        _candidate(
            source_id="registry",
            upstream_id="a",
            keys=(TAVILY_REPO,),
            signals=PopularitySignals(github_stars=1_400, npm_downloads_monthly=10),
        ),
        _candidate(
            source_id="npm",
            upstream_id="b",
            keys=(TAVILY_REPO,),
            signals=PopularitySignals(github_stars=900, npm_downloads_monthly=50_000),
        ),
    ]
    signals = merge_signals(candidates)
    assert signals.github_stars == 1_400
    assert signals.npm_downloads_monthly == 50_000
    assert signals.smithery_use_count is None


def test_a_signal_nobody_reported_stays_none_rather_than_becoming_zero() -> None:
    signals = merge_signals([_candidate(source_id="npm", upstream_id="a", keys=(TAVILY_REPO,))])
    assert signals.github_stars is None


# --- trust tier ------------------------------------------------------------


def test_curated_beats_every_crawled_tier() -> None:
    candidates = [
        _candidate(
            source_id="registry",
            upstream_id="a",
            keys=(TAVILY_REPO,),
            trust_hint="registry_verified",
        )
    ]
    assert elect_trust_tier(candidates, curated_keys={TAVILY_REPO}) == "curated"


def test_registry_verified_beats_smithery_verified() -> None:
    candidates = [
        _candidate(
            source_id="smithery",
            upstream_id="a",
            keys=(TAVILY_REPO,),
            trust_hint="smithery_verified",
        ),
        _candidate(
            source_id="registry",
            upstream_id="b",
            keys=(TAVILY_REPO,),
            trust_hint="registry_verified",
        ),
    ]
    assert elect_trust_tier(candidates, curated_keys=frozenset()) == "registry_verified"


def test_everything_else_is_merely_indexed() -> None:
    candidates = [_candidate(source_id="github_topics", upstream_id="a", keys=(TAVILY_REPO,))]
    assert elect_trust_tier(candidates, curated_keys=frozenset()) == "indexed"


# --- curated overlay -------------------------------------------------------


def test_apply_curated_overlays_one_field_at_a_time_and_records_which() -> None:
    entries: list[CatalogEntry] = [_entry(TAVILY_REPO, name="Crawled name", icon="terminal")]
    override = CuratedOverride(
        key=TAVILY_REPO,
        kind="mcp",
        fields={"name": "Tavily", "auth_note": "Use a Tavily API key."},
    )
    result = apply_curated(entries, overrides=[override], denylist=[])
    assert len(result) == 1
    curated = result[0]
    assert isinstance(curated, McpEntry)
    assert curated.name == "Tavily"
    assert curated.auth_note == "Use a Tavily API key."
    assert curated.icon == "terminal"
    assert curated.curated_fields == ("auth_note", "name")


def test_an_override_matching_an_alias_still_lands_on_the_entry() -> None:
    entries: list[CatalogEntry] = [_entry(TAVILY_REPO, alias_keys=(TAVILY_NPM,))]
    override = CuratedOverride(key=TAVILY_NPM, kind="mcp", fields={"name": "Tavily"})
    result = apply_curated(entries, overrides=[override], denylist=[])
    assert len(result) == 1
    assert result[0].name == "Tavily"


def test_an_override_matching_nothing_creates_the_entry_it_describes() -> None:
    override = CuratedOverride(
        key="mcp:registry:dev.jhin/slack",
        kind="mcp",
        fields={
            "slug": "slack",
            "name": "Slack",
            "description": "Channels, messages, and people.",
            "category": "Communication",
            "icon": "message-square",
            "url_unverified": True,
            "auth_hint": "bearer",
        },
    )
    result = apply_curated([], overrides=[override], denylist=[])
    assert len(result) == 1
    assert result[0].canonical_key == "mcp:registry:dev.jhin/slack"
    assert result[0].trust_tier == "curated"


def test_a_denylisted_entry_is_dropped_before_anything_is_written() -> None:
    entries: list[CatalogEntry] = [_entry(TAVILY_REPO), _entry(BRAVE_REPO)]
    denied = DenylistItem(key=BRAVE_REPO, reason="Endpoint was taken over on 2026-07-02.")
    result = apply_curated(entries, overrides=[], denylist=[denied])
    assert [one.canonical_key for one in result] == [TAVILY_REPO]


def test_a_denylist_key_that_matches_nothing_is_reported_and_not_fatal() -> None:
    """A stale denial is worth reporting; refusing to build over it is a trap.

    The packages this list names are typosquats, which is exactly the class an
    upstream registry later takes down; and anyone can force a repo key out of
    the corpus by publishing a second package claiming the same repository. If
    an unresolved key failed the build, either event would stop every nightly
    run until a human edited the file.
    """
    denied = DenylistItem(key=BRAVE_REPO, reason="Endpoint was taken over on 2026-07-02.")
    entries = [_entry(TAVILY_REPO)]
    assert unresolved_denylist_keys(entries, [denied]) == (BRAVE_REPO,)
    result = apply_curated(entries, overrides=[], denylist=[denied])
    assert [one.canonical_key for one in result] == [TAVILY_REPO]


def test_a_denial_removes_the_identity_not_only_the_key_that_named_it() -> None:
    """An override may not republish a denied entry under its canonical key.

    The denylist names one key; the entry it removes owns several. Recording
    only the named one let an override whose ``key`` was the entry's canonical
    key sail past the guard and re-create the denied record at ``curated`` —
    the strongest tier there is, and the one that reaches ``catalog.json``.
    """
    entry = _entry(TAVILY_REPO, alias_keys=(BRAVE_REPO,))
    denied = DenylistItem(key=BRAVE_REPO, reason="Endpoint was taken over on 2026-07-02.")
    override = CuratedOverride(key=TAVILY_REPO, kind="mcp", fields={"name": "Resurrected"})
    with pytest.raises(CuratedError, match="also denylisted"):
        apply_curated([entry], overrides=[override], denylist=[denied])


def test_an_override_may_not_reattach_a_denylisted_key_as_an_alias() -> None:
    """Resolving by a denied alias would resurrect the record it removed."""
    entries = [_entry(TAVILY_REPO), _entry(BRAVE_REPO)]
    denied = DenylistItem(key=BRAVE_REPO, reason="Endpoint was taken over on 2026-07-02.")
    override = CuratedOverride(key=TAVILY_REPO, kind="mcp", aliases=(BRAVE_REPO,), fields={})
    with pytest.raises(CuratedError, match="also denylisted"):
        apply_curated(entries, overrides=[override], denylist=[denied])


def test_an_override_naming_a_field_the_model_does_not_have_fails_the_build() -> None:
    override = CuratedOverride(key=TAVILY_REPO, kind="mcp", fields={"iconUrl": "https://x.example"})
    with pytest.raises(CuratedError):
        apply_curated([_entry(TAVILY_REPO)], overrides=[override], denylist=[])


def test_an_override_that_would_produce_an_invalid_entry_fails_the_build() -> None:
    override = CuratedOverride(key=TAVILY_REPO, kind="mcp", fields={"category": "Miscellaneous"})
    with pytest.raises(CuratedError):
        apply_curated([_entry(TAVILY_REPO)], overrides=[override], denylist=[])
