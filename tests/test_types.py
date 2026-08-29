"""Canonical form: shard assignment, JSONL bytes, and model bounds."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from jhin_catalog.types import (
    SHARD_COUNT,
    SHARD_HEX_WIDTH,
    CatalogError,
    McpEntry,
    SkillEntry,
    SourceRef,
    all_shards,
    canonical_json,
    dumps_line,
    entry_sort_key,
    loads_line,
    payload_sha256,
    shard_for,
)

# Pinned at implementation time and re-derived inside the test below, so a
# change to the shard function shows up as a failure here rather than as 256
# silently rewritten files.
PINNED_SHARDS: dict[str, str] = {
    "mcp:repo:github.com/tavily-ai/tavily-mcp": "7e",
    "mcp:registry:io.github.modelcontextprotocol/server-filesystem": "50",
    "skill:plugin:anthropics-official/agentforce-adlc/design-review": "f6",
}


def _minimal(**overrides: Any) -> McpEntry:
    payload: dict[str, Any] = {
        "kind": "mcp",
        "canonical_key": "mcp:registry:example.test/minimal",
        "slug": "minimal",
        "name": "Minimal",
        "description": "The smallest entry the model will accept.",
        "trust_tier": "indexed",
        "sources": (
            SourceRef(
                source_id="registry",
                upstream_id="example.test/minimal",
                url="https://registry.example/v0.1/servers/minimal",
            ),
        ),
        "category": "Developer tools",
        "icon": "terminal",
    }
    payload.update(overrides)
    return McpEntry.model_validate(payload)


def _object_text(line: str, key: str) -> str:
    """The literal JSON text of one nested object inside a serialised record."""
    start = line.index(f'"{key}":{{') + len(key) + 3
    depth = 0
    for offset, character in enumerate(line[start:], start=start):
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return line[start : offset + 1]
    raise AssertionError(key)


def test_a_server_record_round_trips_through_the_line_form(sample_mcp_entry: McpEntry) -> None:
    assert loads_line(dumps_line(sample_mcp_entry)) == sample_mcp_entry


def test_a_skill_record_round_trips_through_the_line_form(sample_skill_entry: SkillEntry) -> None:
    assert loads_line(dumps_line(sample_skill_entry)) == sample_skill_entry


def test_the_discriminator_picks_the_right_model_back_off_the_line(
    sample_mcp_entry: McpEntry, sample_skill_entry: SkillEntry
) -> None:
    assert isinstance(loads_line(dumps_line(sample_mcp_entry)), McpEntry)
    assert isinstance(loads_line(dumps_line(sample_skill_entry)), SkillEntry)


def test_the_line_form_is_the_shortest_canonical_encoding(sample_mcp_entry: McpEntry) -> None:
    """One assertion that pins compactness, key order, and raw UTF-8 at once.

    Re-encoding the parsed record with the canonical settings has to give the
    line back byte for byte, which is only true if the writer sorted every
    level, used no separator spaces, and escaped nothing it did not have to.
    """
    line = dumps_line(sample_mcp_entry)
    reencoded = json.dumps(
        json.loads(line), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert reencoded + "\n" == line


def test_the_line_form_carries_no_separator_whitespace(sample_mcp_entry: McpEntry) -> None:
    line = dumps_line(sample_mcp_entry)
    assert '": "' not in line
    assert '", "' not in line
    assert '":[' in line or '":{' in line


def test_the_line_form_ends_in_exactly_one_newline(sample_mcp_entry: McpEntry) -> None:
    line = dumps_line(sample_mcp_entry)
    assert line.endswith("\n")
    assert not line.endswith("\n\n")
    assert line.count("\n") == 1


def test_the_line_form_is_byte_identical_across_two_calls(sample_mcp_entry: McpEntry) -> None:
    assert dumps_line(sample_mcp_entry) == dumps_line(sample_mcp_entry)


def test_nested_object_keys_are_ascii_sorted_too(sample_mcp_entry: McpEntry) -> None:
    signals = _object_text(dumps_line(sample_mcp_entry), "popularity_signals")
    keys = [key for key, _ in json.loads(signals, object_pairs_hook=list)]
    assert keys == sorted(keys)
    assert keys[0] == "github_forks"


def test_non_ascii_text_stays_utf8_instead_of_escaping(sample_mcp_entry: McpEntry) -> None:
    line = dumps_line(sample_mcp_entry)
    assert "ünicode" in line
    assert "\\u" not in line


def test_no_key_is_ever_omitted_from_a_stored_record() -> None:
    """``data/**`` materialises defaults; only the export omits them."""
    stored = json.loads(dumps_line(_minimal()))
    assert set(stored) == set(McpEntry.model_fields)
    assert stored["homepage"] == ""
    assert stored["mcp_url"] is None
    assert stored["alias_keys"] == []
    assert stored["connector_config"] == {}


def test_canonical_json_emits_no_trailing_newline() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_shard_names_are_two_lowercase_hex_characters() -> None:
    shard = shard_for("mcp:registry:example.test/minimal")
    assert len(shard) == SHARD_HEX_WIDTH
    assert shard == shard.lower()
    assert all(character in "0123456789abcdef" for character in shard)


def test_the_pinned_shard_assignments_still_hold() -> None:
    for key, expected in PINNED_SHARDS.items():
        independent = hashlib.sha256(key.encode("utf-8")).hexdigest()[:SHARD_HEX_WIDTH]
        assert shard_for(key) == expected
        assert independent == expected


def test_shard_assignment_is_stable_across_calls() -> None:
    key = "mcp:url:mcp.example/mcp"
    assert shard_for(key) == shard_for(key)


def test_all_shards_is_the_whole_ascending_two_hundred_and_fifty_six() -> None:
    shards = all_shards()
    assert len(shards) == SHARD_COUNT
    assert len(set(shards)) == SHARD_COUNT
    assert shards[0] == "00"
    assert shards[-1] == "ff"
    assert list(shards) == sorted(shards)


def test_every_key_lands_in_a_shard_that_exists() -> None:
    shards = set(all_shards())
    for index in range(500):
        assert shard_for(f"mcp:registry:example.test/server-{index}") in shards


def test_entry_sort_key_is_the_canonical_key(sample_mcp_entry: McpEntry) -> None:
    assert entry_sort_key(sample_mcp_entry) == sample_mcp_entry.canonical_key


def test_payload_sha256_is_lowercase_hex_of_the_raw_body() -> None:
    assert payload_sha256(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_a_popularity_above_one_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal(popularity=1.5)


def test_a_negative_popularity_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal(popularity=-0.1)


def test_a_popularity_with_too_many_decimals_is_rounded_rather_than_rejected() -> None:
    assert _minimal(popularity=0.123456).popularity == 0.1235


def test_a_slug_outside_the_pattern_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal(slug="Not A Slug")


def test_a_slug_longer_than_thirty_two_characters_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal(slug="a" * 33)


def test_an_eleventh_connector_config_pair_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal(connector_config={f"key_{index}": "v" for index in range(11)})


def test_a_sixty_five_character_connector_config_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal(connector_config={"k" * 65: "v"})


def test_a_five_hundred_and_one_character_connector_config_value_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal(connector_config={"k": "v" * 501})


def test_a_category_outside_the_twelve_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal(category="Miscellaneous")


def test_an_icon_outside_the_token_set_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal(icon="https://example.com/icon.png")


def test_an_unknown_field_is_rejected_rather_than_ignored() -> None:
    with pytest.raises(ValidationError):
        _minimal(iconUrl="https://example.com/icon.png")


def test_a_canonical_key_of_the_wrong_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal(canonical_key="skill:registry:example.test/minimal")


def test_a_canonical_key_with_no_space_segment_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal(canonical_key="mcp:github.com/example-org/example-mcp")


def test_an_entry_with_no_source_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal(sources=())


def test_a_templated_endpoint_is_rejected_as_an_mcp_url() -> None:
    with pytest.raises(ValidationError):
        _minimal(mcp_url="https://{region}.mcp.example/mcp")


def test_a_plain_http_endpoint_is_rejected_as_an_mcp_url() -> None:
    with pytest.raises(ValidationError):
        _minimal(mcp_url="http://mcp.example/mcp")


# The icon proxy on the consumer's side dials whatever this field names, so
# the accept set is exactly two URL shapes and the reject set is everything a
# hostile publisher might try — host spoofs by suffix and by path most of all.
ICON_URLS_ACCEPTED = (
    "",
    "https://api.smithery.ai/servers/exa/icon",
    "https://api.smithery.ai/servers/@acme-example/notes/icon",
    "https://github.com/anthropics.png?size=128",
    "https://github.com/a.png?size=128",
    "https://github.com/" + "a" * 39 + ".png?size=128",
)

ICON_URLS_REJECTED = (
    "https://api.smithery.ai.evil.com/servers/exa/icon",  # host spoofed by suffix
    "https://evil.example/api.smithery.ai/servers/exa/icon",  # host hidden in the path
    "http://api.smithery.ai/servers/exa/icon",  # not https
    "https://api.smithery.ai/servers//icon",  # empty path segment
    "https://api.smithery.ai/servers/exa/icon?width=64",  # query smuggled in
    "https://api.smithery.ai/servers/exa/icon#fragment",
    "https://api.smithery.ai/icon",  # not the servers route
    "https://github.com/anthropics.png?size=256",  # unpinned rendition
    "https://github.com/anthropics.png",  # no size at all
    "https://github.com/octo/cat.png?size=128",  # a repo path, not an owner
    "https://github.com.evil.example/anthropics.png?size=128",  # host spoofed by suffix
    "https://github.com/" + "a" * 40 + ".png?size=128",  # over GitHub's 39
    "https://github.com/an_thropics.png?size=128",  # outside GitHub's grammar
    "https://avatars.githubusercontent.com/u/1?v=4",  # the CDN, not the entry shape
    "javascript:alert(1)",
)


@pytest.mark.parametrize("url", ICON_URLS_ACCEPTED)
def test_a_well_shaped_icon_url_is_accepted(url: str) -> None:
    assert _minimal(icon_url=url).icon_url == url


@pytest.mark.parametrize("url", ICON_URLS_REJECTED)
def test_anything_else_offered_as_an_icon_url_is_rejected(url: str) -> None:
    with pytest.raises(ValidationError):
        _minimal(icon_url=url)


def test_an_icon_url_over_the_length_bound_is_rejected_even_when_well_shaped() -> None:
    overlong = "https://api.smithery.ai/servers/" + "a" * 500 + "/icon"
    with pytest.raises(ValidationError):
        _minimal(icon_url=overlong)


def test_marketplace_reviewed_defaults_to_false_and_round_trips() -> None:
    assert _minimal().marketplace_reviewed is False
    flagged = _minimal(marketplace_reviewed=True)
    assert loads_line(dumps_line(flagged)).marketplace_reviewed is True


def test_loads_line_rejects_a_line_that_is_not_json() -> None:
    with pytest.raises(CatalogError):
        loads_line("not json at all\n")


def test_loads_line_rejects_a_record_that_fails_the_schema() -> None:
    with pytest.raises(CatalogError):
        loads_line('{"kind":"mcp"}\n')
