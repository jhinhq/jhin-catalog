"""Defects three reviews found, each pinned by the case that exposed it.

Every test here failed before its fix and passes after. They are grouped by
what breaks when the fix regresses rather than by which module holds it,
because in each case the module holding the bug is not the one that suffers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jhin_catalog.build import (
    assign_slugs,
    entries_from_fetches,
    is_publishable,
    load_marketplace_policy,
    merged_entries,
)
from jhin_catalog.sources.base import DEFAULT_LIMITS
from jhin_catalog.types import (
    McpEntry,
    NormalizeError,
    RawRecord,
    SourceFetch,
    SourceRef,
)


def _mcp(key: str, *, slug: str, tier: str, name: str = "Sample", **extra: object) -> McpEntry:
    payload: dict[str, object] = {
        "kind": "mcp",
        "canonical_key": key,
        "slug": slug,
        "name": name,
        "description": "A record two claimants would both like to be called by.",
        "trust_tier": tier,
        "sources": (
            SourceRef(source_id="registry", upstream_id=key, url="https://registry.example/x"),
        ),
        "category": "Project management",
        "icon": "linear",
    }
    payload.update(extra)
    return McpEntry.model_validate(payload)


def _fetch(*records: RawRecord) -> SourceFetch:
    return SourceFetch(
        source_id="registry",
        url="https://registry.modelcontextprotocol.io/v0.1/servers",
        sha256="0" * 64,
        entry_count=len(records),
        page_count=1,
        records=records,
    )


# --- slugs -----------------------------------------------------------------


def test_a_crawled_proxy_cannot_take_a_curated_records_slug() -> None:
    """Allocation used to run in ``canonical_key`` order alone.

    ``mcp:url:linear.run.tools`` sorts before ``mcp:url:mcp.linear.app/mcp``,
    so a third-party Smithery proxy took the ``linear`` slug and the
    hand-checked record — the one holding the real endpoint and the ``oauth``
    auth hint — was renamed to ``linear_08b7``. Jhin resolves stored
    connections by slug, so ``catalog_by_slug("linear")`` then returned the
    proxy, and nothing failed loudly.
    """
    proxy = _mcp("mcp:url:linear.run.tools", slug="linear", tier="smithery_verified")
    curated = _mcp("mcp:url:mcp.linear.app/mcp", slug="linear", tier="curated")

    by_key = {entry.canonical_key: entry.slug for entry in assign_slugs([proxy, curated])}
    assert by_key["mcp:url:mcp.linear.app/mcp"] == "linear"
    assert by_key["mcp:url:linear.run.tools"] != "linear"


def test_slug_allocation_does_not_depend_on_the_order_it_was_handed() -> None:
    proxy = _mcp("mcp:url:linear.run.tools", slug="linear", tier="smithery_verified")
    curated = _mcp("mcp:url:mcp.linear.app/mcp", slug="linear", tier="curated")

    forward = {e.canonical_key: e.slug for e in assign_slugs([proxy, curated])}
    backward = {e.canonical_key: e.slug for e in assign_slugs([curated, proxy])}
    assert forward == backward


def test_two_records_of_the_same_tier_are_still_decided_by_their_keys() -> None:
    """Trust breaks the tie; identity breaks the tie inside a tier."""
    first = _mcp("mcp:url:a.example/mcp", slug="shared", tier="curated")
    second = _mcp("mcp:url:b.example/mcp", slug="shared", tier="curated")
    by_key = {e.canonical_key: e.slug for e in assign_slugs([second, first])}
    assert by_key["mcp:url:a.example/mcp"] == "shared"
    assert by_key["mcp:url:b.example/mcp"] != "shared"


def test_assign_slugs_returns_entries_in_canonical_key_order() -> None:
    """Allocation order changed; the returned order is still shard order."""
    entries = [
        _mcp("mcp:url:c.example/mcp", slug="c", tier="indexed"),
        _mcp("mcp:url:a.example/mcp", slug="a", tier="curated"),
        _mcp("mcp:url:b.example/mcp", slug="b", tier="smithery_verified"),
    ]
    assigned = assign_slugs(entries)
    assert [e.canonical_key for e in assigned] == sorted(e.canonical_key for e in entries)


# --- publishability --------------------------------------------------------


def test_a_curated_self_host_record_reaches_the_catalog() -> None:
    """Fourteen of the fifty shipped connectors could never be exported.

    ``curated/mcp.yaml`` deliberately gives a self-hosted server a
    ``setup_note`` and ``url_unverified: true`` instead of an endpoint. The
    connectability test then dropped exactly those records from the library
    the file exists to populate.
    """
    entry = _mcp(
        "mcp:registry:dev.jhin/brave_search",
        slug="brave_search",
        tier="curated",
        name="Brave Search",
        url_unverified=True,
        setup_note="Self-host the Brave Search MCP server with your API key and paste its URL.",
    )
    assert entry.mcp_url is None
    assert entry.connector_type is None
    assert entry.stdio_only is False
    assert is_publishable(entry)


def test_a_crawled_record_with_nothing_to_dial_is_still_refused() -> None:
    """The exemption is for a human's judgement, not for every record."""
    entry = _mcp("mcp:repo:github.com/o/r", slug="r", tier="registry_verified")
    assert not is_publishable(entry)


# --- one malformed upstream row --------------------------------------------


def _registry_record(key: str, packages: list[dict[str, object]]) -> RawRecord:
    return RawRecord(
        source_id="registry",
        upstream_id=key,
        url="https://registry.modelcontextprotocol.io/v0.1/servers",
        payload={
            "server": {
                "name": key,
                "description": "A server row as the registry actually served it.",
                "status": "active",
                "packages": packages,
            }
        },
    )


def test_a_package_with_no_registry_type_does_not_abort_the_build() -> None:
    """``registryType`` is required by the registry's schema and is served missing.

    ``PackageRef.registry_type`` is ``min_length=1``, so one such row raised
    ``NormalizeError`` out of ``entries_from_fetches`` and the nightly sync
    exited 5. Anyone may publish to that registry.
    """
    record = _registry_record("io.github.a/b", [{"identifier": "pkg-a", "version": "1.0"}])
    entries = entries_from_fetches([_fetch(record)], overrides=[], denylist=[])
    assert len(entries) == 1
    packages = entries[0].model_dump()["packages"]
    assert packages[0]["registry_type"] == "unknown"


def _rows(count: int, prefix: str) -> list[RawRecord]:
    """Distinct rows: a shared package identifier would merge them into one."""
    return [
        _registry_record(
            f"io.github.a/{prefix}{index}",
            [{"identifier": f"pkg-{prefix}{index}", "registryType": "npm", "version": "1"}],
        )
        for index in range(count)
    ]


def _refusing(*keys: str) -> object:
    """A ``build_entry`` that rejects the named components and passes the rest.

    The realistic causes — a field an upstream served outside the model's
    range — are all clipped by the normaliser today, which is the point: the
    guard exists for the next one nobody has met yet. So the failure is
    injected rather than contrived, and what is under test is the guard's
    contract, not any particular malformed row.
    """
    from jhin_catalog import build as build_module

    real = build_module.build_entry  # type: ignore[attr-defined]
    refused = frozenset(keys)

    def fake(merged: object, **kwargs: object) -> object:
        key = getattr(merged, "canonical_key", "")
        if key in refused:
            raise NormalizeError(f"entry {key!r} does not validate")
        return real(merged, **kwargs)  # type: ignore[arg-type]

    return fake


def test_a_handful_of_unbuildable_components_are_dropped_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One row an upstream served wrong must not cost the whole nightly build.

    Anyone may publish to these registries, so a component that will not
    validate has to be survivable; before this, the first one exited 5 and
    nothing was written.
    """
    monkeypatch.setattr(
        "jhin_catalog.build.build_entry", _refusing("mcp:registry:io.github.a/bad0")
    )
    records = [*_rows(1, "good"), *_rows(1, "bad")]
    entries = entries_from_fetches([_fetch(*records)], overrides=[], denylist=[])
    assert [entry.canonical_key for entry in entries] == ["mcp:registry:io.github.a/good0"]


def test_enough_unbuildable_components_still_fail_the_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the bound it is a code fault again, which is what exit 5 means."""
    records = _rows(40, "bad")
    monkeypatch.setattr(
        "jhin_catalog.build.build_entry",
        _refusing(*(f"mcp:registry:io.github.a/bad{index}" for index in range(40))),
    )
    with pytest.raises(NormalizeError, match="would not validate"):
        entries_from_fetches([_fetch(*records)], overrides=[], denylist=[])


def test_merged_entries_and_entries_from_fetches_agree_before_curation() -> None:
    record = _registry_record(
        "io.github.a/b", [{"identifier": "pkg", "registryType": "npm", "version": "1"}]
    )
    built = merged_entries([_fetch(record)], overrides=[])
    entries = entries_from_fetches([_fetch(record)], overrides=[], denylist=[])
    assert [one.canonical_key for one in built] == [one.canonical_key for one in entries]


# --- the marketplace allowlist ---------------------------------------------


def test_the_shipped_curated_policy_confines_discovery_to_reviewed_repos() -> None:
    """``curated/skills.yaml`` declared this and no code read it.

    Topic search returned every repository carrying ``claude-code-plugin``,
    so one commit adding a topic put a stranger's free text into the index —
    the exact failure the file says it prevents.
    """
    policy = load_marketplace_policy(Path(__file__).resolve().parent.parent)
    assert policy.require_allowlist is True
    assert "anthropics/claude-plugins-official" in policy.allow
    assert "anthropics/claude-code" in policy.allow


def test_a_repository_root_with_no_curated_file_leaves_discovery_open(tmp_path: Path) -> None:
    policy = load_marketplace_policy(tmp_path)
    assert policy.require_allowlist is False
    assert policy.allow == ()


def test_the_policy_reaches_the_crawl_through_the_source_limits() -> None:
    limits = DEFAULT_LIMITS.model_copy(
        update={"marketplace_allowlist": ("owner/name",), "require_marketplace_allowlist": True}
    )
    assert limits.require_marketplace_allowlist
    assert limits.marketplace_allowlist == ("owner/name",)
