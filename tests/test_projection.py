"""The Jhin ``CatalogApp`` projection: 16 keys, omit-when-default, and the caps."""

from __future__ import annotations

import json
from typing import Any

from jhin_catalog.build import export_catalog_json, is_publishable, project_catalog_app
from jhin_catalog.score import rank_score
from jhin_catalog.types import (
    CATALOG_CATEGORIES,
    CATALOG_ICONS,
    MAX_SUMMARY_CHARS,
    CatalogEntry,
    McpEntry,
    RepoRef,
    SkillEntry,
    SourceRef,
)

CATALOG_APP_KEYS = frozenset(
    {
        "slug",
        "name",
        "category",
        "icon",
        "icon_url",
        "description",
        "connector_type",
        "mcp_url",
        "url_unverified",
        "transport",
        "auth_hint",
        "auth_note",
        "docs_url",
        "setup_note",
        "stdio_only",
        "connector_config",
    }
)


def _mcp(**overrides: Any) -> McpEntry:
    payload: dict[str, Any] = {
        "kind": "mcp",
        "canonical_key": "mcp:url:mcp.example/mcp",
        "slug": "example",
        "name": "Example",
        "description": "Everything a person needs to decide whether to connect it.",
        "trust_tier": "registry_verified",
        "sources": (SourceRef(source_id="registry", upstream_id="a", url="https://reg.example/a"),),
        "category": "Developer tools",
        "icon": "terminal",
        "mcp_url": "https://mcp.example/mcp",
        "transport": "streamable_http",
    }
    payload.update(overrides)
    return McpEntry.model_validate(payload)


def _skill() -> SkillEntry:
    return SkillEntry(
        kind="skill",
        canonical_key="skill:skill:github.com/example-org/pack/skills/review",
        slug="review",
        name="Review",
        description="A skill, which the server projection never emits.",
        trust_tier="curated",
        sources=(SourceRef(source_id="curated", upstream_id="a", url="https://example.com/a"),),
        repo=RepoRef(host="github.com", owner="example-org", repo="pack"),
        skill_name="review",
        category="General",
        source_ref="example-org/pack/skills/review",
        skill_path="skills/review/SKILL.md",
    )


# --- the sixteen keys ------------------------------------------------------


def test_the_projection_emits_no_key_outside_the_catalog_app_shape() -> None:
    projected = project_catalog_app(_mcp())
    assert set(projected) <= CATALOG_APP_KEYS


def test_auth_hint_is_emitted_even_at_its_default() -> None:
    """A connect dialog has to render an auth control; an absent hint is a bug."""
    projected = project_catalog_app(_mcp(auth_hint="bearer"))
    assert projected["auth_hint"] == "bearer"


def test_a_false_boolean_is_omitted_rather_than_written() -> None:
    projected = project_catalog_app(_mcp(url_unverified=False, stdio_only=False))
    assert "url_unverified" not in projected
    assert "stdio_only" not in projected


def test_a_true_boolean_is_written() -> None:
    projected = project_catalog_app(_mcp(url_unverified=True))
    assert projected["url_unverified"] is True


def test_an_empty_string_is_omitted_rather_than_written() -> None:
    projected = project_catalog_app(_mcp(auth_note="", setup_note="", docs_url=""))
    assert "auth_note" not in projected
    assert "setup_note" not in projected
    assert "docs_url" not in projected


def test_a_none_optional_is_omitted_rather_than_written_as_null() -> None:
    projected = project_catalog_app(_mcp(mcp_url=None, connector_type=None, stdio_only=True))
    assert "mcp_url" not in projected
    assert "connector_type" not in projected


def test_connector_config_is_omitted_when_empty() -> None:
    assert "connector_config" not in project_catalog_app(_mcp(connector_config={}))


def test_a_github_avatar_icon_url_is_projected() -> None:
    projected = project_catalog_app(_mcp(icon_url="https://github.com/acme-example.png?size=128"))
    assert projected["icon_url"] == "https://github.com/acme-example.png?size=128"


def test_a_smithery_icon_url_is_not_projected() -> None:
    """``CatalogApp`` on the consumer's side accepts only the avatar shape;
    a Smithery icon route reaches a deployment through the synced entry."""
    projected = project_catalog_app(_mcp(icon_url="https://api.smithery.ai/servers/exa/icon"))
    assert "icon_url" not in projected


def test_an_empty_icon_url_is_omitted_rather_than_written() -> None:
    assert "icon_url" not in project_catalog_app(_mcp(icon_url=""))


def test_connector_config_is_key_sorted_when_present() -> None:
    projected = project_catalog_app(
        _mcp(connector_config={"search_backend": "tavily", "base_url": "https://x.example"})
    )
    config = projected["connector_config"]
    assert isinstance(config, dict)
    assert list(config) == ["base_url", "search_backend"]


def test_the_projected_name_is_capped_at_sixty_characters() -> None:
    projected = project_catalog_app(_mcp(name="N" * 120))
    name = projected["name"]
    assert isinstance(name, str)
    assert len(name) == 60


def test_the_projected_description_is_summarised_to_the_cap() -> None:
    projected = project_catalog_app(_mcp(description="A long description. " * 25))
    description = projected["description"]
    assert isinstance(description, str)
    assert len(description) <= MAX_SUMMARY_CHARS


def test_every_projected_category_and_icon_is_a_token_jhin_can_render() -> None:
    for category in CATALOG_CATEGORIES:
        projected = project_catalog_app(_mcp(category=category, icon="terminal"))
        assert projected["category"] in CATALOG_CATEGORIES
        assert projected["icon"] in CATALOG_ICONS


def test_every_projected_endpoint_is_a_concrete_https_url() -> None:
    projected = project_catalog_app(_mcp())
    url = projected["mcp_url"]
    assert isinstance(url, str)
    assert url.startswith("https://")
    assert "{" not in url


# --- publishability --------------------------------------------------------


def test_a_merely_indexed_entry_is_not_published() -> None:
    """The long tail is stored and searchable; it is not offered as a connector."""
    assert not is_publishable(_mcp(trust_tier="indexed"))


def test_a_deprecated_entry_is_not_published() -> None:
    assert not is_publishable(_mcp(deprecated=True))


def test_an_entry_with_no_description_is_not_published() -> None:
    assert not is_publishable(_mcp(description=""))


def test_an_entry_with_no_endpoint_no_connector_and_no_stdio_note_is_not_published() -> None:
    assert not is_publishable(_mcp(mcp_url=None, connector_type=None, stdio_only=False))


def test_a_stdio_only_entry_is_published_so_the_setup_note_can_be_read() -> None:
    assert is_publishable(_mcp(mcp_url=None, stdio_only=True))


def test_a_native_connector_with_no_endpoint_is_published() -> None:
    assert is_publishable(_mcp(mcp_url=None, connector_type="github"))


def test_a_well_formed_hosted_server_is_published() -> None:
    assert is_publishable(_mcp())


# --- the exported file -----------------------------------------------------


def _ranked(count: int) -> list[CatalogEntry]:
    return [
        _mcp(
            canonical_key=f"mcp:url:server-{index:03d}.example/mcp",
            slug=f"srv_{index:03d}",
            mcp_url=f"https://server-{index:03d}.example/mcp",
            popularity=round(index / 100, 4),
        )
        for index in range(count)
    ]


def test_the_export_is_a_json_array_ending_in_one_newline() -> None:
    body = export_catalog_json(_ranked(5))
    assert body.endswith("\n")
    assert not body.endswith("\n\n")
    assert isinstance(json.loads(body), list)


def test_the_export_is_indented_for_a_human_to_read_in_a_diff() -> None:
    body = export_catalog_json(_ranked(3))
    assert body == json.dumps(json.loads(body), indent=2) + "\n"


def test_the_export_is_capped_at_its_limit() -> None:
    assert len(json.loads(export_catalog_json(_ranked(50), limit=7))) == 7


def test_the_export_carries_unique_slugs() -> None:
    exported = json.loads(export_catalog_json(_ranked(30)))
    slugs = [app["slug"] for app in exported]
    assert len(set(slugs)) == len(slugs)


def test_the_export_is_ordered_by_rank_and_keeps_that_order_after_the_cap() -> None:
    entries = _ranked(30)
    exported = json.loads(export_catalog_json(entries, limit=10))
    expected = [
        entry.slug for entry in sorted((e for e in entries if is_publishable(e)), key=rank_score)
    ][:10]
    assert [app["slug"] for app in exported] == expected


def test_the_export_drops_everything_unpublishable() -> None:
    entries: list[CatalogEntry] = [
        *_ranked(3),
        _mcp(canonical_key="mcp:url:hidden.example/mcp", slug="hidden", trust_tier="indexed"),
        _skill(),
    ]
    exported = json.loads(export_catalog_json(entries))
    assert "hidden" not in {app["slug"] for app in exported}
    assert "review" not in {app["slug"] for app in exported}


def test_every_exported_record_is_a_valid_catalog_app() -> None:
    for app in json.loads(export_catalog_json(_ranked(10))):
        assert set(app) <= CATALOG_APP_KEYS
        assert app["category"] in CATALOG_CATEGORIES
        assert app["icon"] in CATALOG_ICONS
        assert "auth_hint" in app
        assert len(app["description"]) <= MAX_SUMMARY_CHARS
