"""The shipped curated overlays load, resolve, and stay in bounds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from jhin_catalog.build import load_curated, load_denylist
from jhin_catalog.types import (
    CANONICAL_KEY_RE,
    CATALOG_CATEGORIES,
    CATALOG_ICONS,
    CONNECTOR_TYPES,
    MAX_CONNECTOR_CONFIG,
    MAX_CONNECTOR_CONFIG_KEY_CHARS,
    MAX_CONNECTOR_CONFIG_VALUE_CHARS,
    McpEntry,
    SourceRef,
)

CURATED_FILES = ("mcp.yaml", "skills.yaml", "denylist.yaml")


def _document(root: Path, name: str) -> dict[str, Any]:
    loaded = yaml.safe_load((root / "curated" / name).read_text("utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _entries(root: Path, name: str) -> list[dict[str, Any]]:
    raw = _document(root, name)["entries"]
    assert isinstance(raw, list)
    for item in raw:
        assert isinstance(item, dict)
    return list(raw)


def _fields(override: dict[str, Any]) -> dict[str, Any]:
    fields = override["fields"]
    assert isinstance(fields, dict)
    return fields


def _as_entry(override: dict[str, Any]) -> McpEntry:
    """Validate a partial override as a whole ``McpEntry``.

    The scaffolding below is exactly what ``apply_curated`` supplies for an
    override that matches no crawled entry, so every constraint the model
    places on a curated value is exercised here rather than at sync time.
    """
    return McpEntry.model_validate(
        {
            "kind": "mcp",
            "canonical_key": override["key"],
            "trust_tier": "curated",
            "sources": (
                SourceRef(
                    source_id="curated",
                    upstream_id=override["key"],
                    url="https://github.com/jhin-dev/jhin-catalog",
                ),
            ),
            **_fields(override),
        }
    )


@pytest.mark.parametrize("name", CURATED_FILES)
def test_every_curated_file_is_a_mapping_with_an_entry_list(repo_root: Path, name: str) -> None:
    document = _document(repo_root, name)
    assert document["version"] == 1
    assert isinstance(document["entries"], list)


def test_load_curated_reads_both_overlay_files(repo_root: Path) -> None:
    overrides = load_curated(repo_root / "curated" / "mcp.yaml")
    assert len(overrides) == 50
    assert all(override.kind == "mcp" for override in overrides)
    assert load_curated(repo_root / "curated" / "skills.yaml") == ()


def test_load_curated_on_a_missing_file_yields_no_overrides(tmp_path: Path) -> None:
    assert load_curated(tmp_path / "absent.yaml") == ()


def test_the_fifty_connector_records_survived_the_port(repo_root: Path) -> None:
    entries = _entries(repo_root, "mcp.yaml")
    assert len(entries) == 50
    slugs = [_fields(entry)["slug"] for entry in entries]
    assert len(set(slugs)) == 50
    assert "github" in slugs
    assert "fake_websearch" in slugs


def test_every_curated_key_and_alias_is_a_well_formed_mcp_key(repo_root: Path) -> None:
    for override in _entries(repo_root, "mcp.yaml"):
        key = override["key"]
        assert CANONICAL_KEY_RE.match(key)
        assert key.startswith("mcp:")
        aliases = override.get("aliases", [])
        assert len(set(aliases)) == len(aliases)
        for alias in aliases:
            assert CANONICAL_KEY_RE.match(alias)
            assert alias.startswith("mcp:")
            assert alias != key
    keys = [override["key"] for override in _entries(repo_root, "mcp.yaml")]
    assert len(set(keys)) == len(keys)


def test_every_curated_override_validates_as_a_partial_mcp_entry(repo_root: Path) -> None:
    for override in _entries(repo_root, "mcp.yaml"):
        entry = _as_entry(override)
        assert entry.trust_tier == "curated"
        assert entry.name
        assert entry.description


def test_curated_categories_icons_and_connector_types_are_all_known_tokens(
    repo_root: Path,
) -> None:
    for override in _entries(repo_root, "mcp.yaml"):
        fields = _fields(override)
        assert fields["category"] in CATALOG_CATEGORIES
        assert fields["icon"] in CATALOG_ICONS
        connector_type = fields.get("connector_type")
        if connector_type is not None:
            assert connector_type in CONNECTOR_TYPES


def test_every_curated_endpoint_is_https(repo_root: Path) -> None:
    for override in _entries(repo_root, "mcp.yaml"):
        url = _fields(override).get("mcp_url")
        if url is None:
            continue
        assert url.startswith("https://")
        assert "{" not in url
        assert "@" not in url.split("/", 3)[2]


def test_curated_connector_config_stays_within_the_ten_sixtyfour_fivehundred_bounds(
    repo_root: Path,
) -> None:
    for override in _entries(repo_root, "mcp.yaml"):
        config = _fields(override).get("connector_config")
        if config is None:
            continue
        assert len(config) <= MAX_CONNECTOR_CONFIG
        for key, value in config.items():
            assert 1 <= len(key) <= MAX_CONNECTOR_CONFIG_KEY_CHARS
            assert len(value) <= MAX_CONNECTOR_CONFIG_VALUE_CHARS


def test_an_endpointless_record_still_tells_a_person_what_to_do(repo_root: Path) -> None:
    """A record with no URL earns its place only by explaining the setup.

    Half the value of this file is the rows a crawler cannot produce: Slack,
    Gmail, and the rest have no hosted server, so the entry exists to say so
    and to say what to run instead.
    """
    for override in _entries(repo_root, "mcp.yaml"):
        fields = _fields(override)
        if fields.get("mcp_url") or fields.get("connector_type"):
            continue
        assert fields.get("setup_note") or fields.get("auth_note")


def test_a_missing_endpoint_is_always_marked_unverified(repo_root: Path) -> None:
    for override in _entries(repo_root, "mcp.yaml"):
        fields = _fields(override)
        if fields.get("mcp_url") or fields.get("connector_type"):
            continue
        assert fields["url_unverified"] is True


def test_the_marketplace_allowlist_is_seeded_with_both_first_party_repos(
    repo_root: Path,
) -> None:
    marketplaces = _document(repo_root, "skills.yaml")["marketplaces"]
    allowed = [item["repo"] for item in marketplaces["allow"]]
    assert allowed == ["anthropics/claude-plugins-official", "anthropics/claude-code"]
    assert marketplaces["discovery"]["require_allowlist"] is True
    for item in marketplaces["allow"]:
        assert item["maintainer"]
        assert len(item["notes"]) >= 40


def test_every_allowlisted_marketplace_is_named_owner_slash_repo(repo_root: Path) -> None:
    marketplaces = _document(repo_root, "skills.yaml")["marketplaces"]
    listed = list(marketplaces["allow"]) + list(marketplaces["community"])
    repos = [item["repo"] for item in listed]
    assert len(set(repos)) == len(repos)
    for repo in repos:
        owner, _, name = repo.partition("/")
        assert owner and name and "/" not in name


def test_the_denylist_is_empty_but_loads(repo_root: Path) -> None:
    assert load_denylist(repo_root / "curated" / "denylist.yaml") == ()
    assert _entries(repo_root, "denylist.yaml") == []


def test_every_denylist_entry_carries_a_reason_of_at_least_eight_characters(
    repo_root: Path,
) -> None:
    for item in _entries(repo_root, "denylist.yaml"):
        assert CANONICAL_KEY_RE.match(item["key"])
        assert 8 <= len(item["reason"]) <= 300


def test_no_curated_key_is_also_denylisted(repo_root: Path) -> None:
    denied = {item["key"] for item in _entries(repo_root, "denylist.yaml")}
    for name in ("mcp.yaml", "skills.yaml"):
        for override in _entries(repo_root, name):
            assert override["key"] not in denied
            assert not denied.intersection(override.get("aliases", []))


def test_the_denylist_documents_its_own_format(repo_root: Path) -> None:
    """A file that is empty forever has to explain itself, or it rots."""
    text = (repo_root / "curated" / "denylist.yaml").read_text("utf-8")
    assert "reason" in text
    assert "8 to 300 characters" in text
