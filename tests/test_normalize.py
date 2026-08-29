"""Slugs, summaries, key derivation, category and icon choice."""

from __future__ import annotations

from typing import Any

import pytest

from jhin_catalog.normalize import (
    build_entry,
    choose_category,
    choose_icon,
    collapse,
    name_key,
    parse_repo_url,
    repo_key,
    slugify,
    summarize,
    url_key,
)
from jhin_catalog.types import (
    MAX_SUMMARY_CHARS,
    Candidate,
    JsonObject,
    McpEntry,
    MergedCandidate,
    NormalizeError,
    PopularitySignals,
    RepoRef,
    SourceRef,
)


def _merged(
    *,
    key: str = "mcp:registry:example.test/server",
    kind: str = "mcp",
    trust_tier: str = "registry_verified",
    fields: JsonObject | None = None,
) -> MergedCandidate:
    source_ref = SourceRef(
        source_id="registry",
        upstream_id="example.test/server",
        url="https://registry.example/v0.1/servers/server",
    )
    candidate = Candidate(
        kind=kind,
        source_id="registry",
        upstream_id="example.test/server",
        primary_key=key,
        alias_keys=(key,),
        trust_hint=trust_tier,
        source_ref=source_ref,
        fields=dict(fields or {}),
    )
    return MergedCandidate(
        kind=kind,
        canonical_key=key,
        candidates=(candidate,),
        signals=PopularitySignals(),
        trust_tier=trust_tier,
        fields=dict(fields or {}),
    )


def _base_fields(**overrides: Any) -> JsonObject:
    fields: JsonObject = {
        "name": "Example Server",
        "description": "A server the normaliser has enough to work with.",
    }
    fields.update(overrides)
    return fields


# --- slugs -----------------------------------------------------------------


def test_slugify_lowercases_and_folds_every_run_of_punctuation() -> None:
    assert slugify("@scope/My Server!") == "scope_my_server"


def test_slugify_collapses_repeated_separators_into_one() -> None:
    assert slugify("a---b___c") == "a_b_c"


def test_slugify_strips_leading_and_trailing_separators() -> None:
    assert slugify("__github__") == "github"


def test_slugify_truncates_to_thirty_two_and_leaves_no_trailing_separator() -> None:
    slug = slugify("a" * 31 + " " + "b" * 20)
    assert len(slug) <= 32
    assert not slug.endswith("_")


def test_slugify_on_something_with_no_letters_or_digits_is_an_error() -> None:
    with pytest.raises(NormalizeError):
        slugify("---")


def test_slugify_on_the_empty_string_is_an_error() -> None:
    with pytest.raises(NormalizeError):
        slugify("")


# --- collapsing and summarising --------------------------------------------


def test_collapse_folds_whitespace_runs_and_drops_control_characters() -> None:
    assert collapse("a\t\t b\r\nc\x00", limit=50) == "a b c"


def test_collapse_truncates_to_its_limit() -> None:
    assert len(collapse("word " * 100, limit=20)) <= 20


def test_summarize_flattens_markdown_emphasis() -> None:
    assert "**" not in summarize("A **bold** claim about servers.")
    assert "bold" in summarize("A **bold** claim about servers.")


def test_summarize_keeps_link_text_and_drops_the_target() -> None:
    summary = summarize("See [the docs](https://example.com/docs) for more.")
    assert "the docs" in summary
    assert "https://example.com/docs" not in summary


def test_summarize_collapses_newlines_into_one_line() -> None:
    assert "\n" not in summarize("First line.\n\nSecond line.\n")


def test_summarize_never_exceeds_the_projection_cap() -> None:
    long_text = "A very long description of a server that keeps going. " * 20
    assert len(summarize(long_text)) <= MAX_SUMMARY_CHARS


def test_summarize_truncates_at_a_word_boundary_and_says_so() -> None:
    long_text = "supercalifragilistic " * 40
    summary = summarize(long_text)
    assert summary.endswith("…")
    assert not summary[:-1].endswith(" ")


def test_summarize_leaves_a_short_description_alone() -> None:
    assert summarize("Issues, projects, cycles, and comments.") == (
        "Issues, projects, cycles, and comments."
    )


# --- repository parsing ----------------------------------------------------


def test_parse_repo_url_accepts_github_and_strips_the_git_suffix() -> None:
    repo = parse_repo_url("https://github.com/Tavily-AI/tavily-mcp.git")
    assert repo == RepoRef(host="github.com", owner="tavily-ai", repo="tavily-mcp")


def test_parse_repo_url_accepts_gitlab_and_bitbucket() -> None:
    assert parse_repo_url("https://gitlab.com/o/r") is not None
    assert parse_repo_url("https://bitbucket.org/o/r") is not None


def test_parse_repo_url_refuses_a_host_that_is_not_one_of_the_three() -> None:
    assert parse_repo_url("https://evil.example/o/r") is None


def test_parse_repo_url_refuses_something_that_is_not_a_repository() -> None:
    assert parse_repo_url("https://github.com") is None
    assert parse_repo_url("") is None


def test_parse_repo_url_unwraps_the_git_plus_scheme_npm_publishes() -> None:
    repo = parse_repo_url("git+https://github.com/o/r.git")
    assert repo == RepoRef(host="github.com", owner="o", repo="r")


# --- key derivation --------------------------------------------------------


def test_repo_key_is_host_owner_repo_under_the_entry_kind() -> None:
    repo = RepoRef(host="github.com", owner="tavily-ai", repo="tavily-mcp")
    assert repo_key(repo, kind="mcp") == "mcp:repo:github.com/tavily-ai/tavily-mcp"


def test_a_repo_subpath_is_appended_after_a_hash() -> None:
    repo = RepoRef(
        host="github.com", owner="modelcontextprotocol", repo="servers", subpath="src/filesystem"
    )
    assert repo_key(repo, kind="mcp") == (
        "mcp:repo:github.com/modelcontextprotocol/servers#src/filesystem"
    )


def test_url_key_drops_the_trailing_slash_the_query_and_the_fragment() -> None:
    assert url_key("https://MCP.Tavily.com/mcp/?key=secret#frag", kind="mcp") == (
        "mcp:url:mcp.tavily.com/mcp"
    )


def test_url_key_lowercases_the_whole_key_not_only_the_host() -> None:
    """Section 2.1 opens by saying every key is "all lowercase ASCII" and then
    says of the ``url`` space only that the host is lowercased. The preamble is
    the general rule, and folding the path too costs nothing real — no observed
    MCP endpoint distinguishes two servers by path case — while buying a join
    between sources that disagree about capitalisation."""
    assert url_key("https://Example.COM/MCP", kind="mcp") == "mcp:url:example.com/mcp"


def test_url_key_never_carries_a_secret_from_the_query_string() -> None:
    """Tavily publishes its key in the URL; the identity must not remember it."""
    assert "secret" not in url_key("https://mcp.example/mcp?api_key=secret", kind="mcp")


def test_name_key_lowercases_the_value_under_the_named_space() -> None:
    assert name_key("registry", "IO.GitHub.Example/Server", kind="mcp") == (
        "mcp:registry:io.github.example/server"
    )
    assert name_key("smithery", "Exa", kind="mcp") == "mcp:smithery:exa"


# --- category and icon -----------------------------------------------------


def test_a_payments_description_wins_over_the_developer_word_it_also_contains() -> None:
    """Rule order is the tie-break, and payments is scanned before tooling."""
    category = choose_category(
        name="Stripe",
        description="A developer-friendly Stripe integration for payments.",
        tags=("api", "developer"),
    )
    assert category == "Payments & commerce"


def test_a_description_matching_nothing_falls_back_to_developer_tools() -> None:
    assert choose_category(name="Widget", description="Does a thing.", tags=()) == (
        "Developer tools"
    )


def test_keywords_are_matched_on_whole_words_only() -> None:
    """``cd`` must not fire on ``discard``, or every entry becomes tooling."""
    assert choose_category(name="Cards", description="Discard piles.", tags=()) == (
        "Developer tools"
    )


def test_tags_and_extra_text_both_feed_the_haystack() -> None:
    assert choose_category(name="Thing", description="", tags=("figma",)) == "Design"
    assert choose_category(name="Thing", description="", tags=(), extra="notion") == (
        "Documents & knowledge"
    )


def test_choose_icon_prefers_the_slug_map_over_the_category_map() -> None:
    assert choose_icon(slug="github", category="Developer tools") == "github"
    assert choose_icon(slug="figma", category="Design") == "pen-tool"


def test_choose_icon_falls_back_to_the_category_when_the_slug_is_unknown() -> None:
    assert choose_icon(slug="nobody_has_this_slug", category="Storage") == "folder"
    assert choose_icon(slug="nobody_has_this_slug", category="Automation") == "zap"


# --- building the final entry ----------------------------------------------


def test_build_entry_produces_a_valid_entry_with_a_derived_category_and_icon() -> None:
    merged = _merged(fields=_base_fields(name="Stripe", description="Invoices and payments."))
    entry = build_entry(merged, popularity=0.5, slug="stripe")
    assert isinstance(entry, McpEntry)
    assert entry.slug == "stripe"
    assert entry.popularity == 0.5
    assert entry.category == "Payments & commerce"
    assert entry.icon == "credit-card"
    assert McpEntry.model_validate(entry.model_dump()) == entry


def test_build_entry_recomputes_stdio_only_rather_than_trusting_the_merge() -> None:
    """A merged ``True`` from one source cannot survive a concrete endpoint."""
    merged = _merged(
        fields=_base_fields(
            stdio_only=True,
            mcp_url="https://mcp.example/mcp",
            transport="streamable_http",
        )
    )
    entry = build_entry(merged, popularity=0.0, slug="example")
    assert isinstance(entry, McpEntry)
    assert entry.stdio_only is False


def test_build_entry_sets_stdio_only_when_there_is_a_package_and_no_endpoint() -> None:
    merged = _merged(
        fields=_base_fields(
            packages=[
                {
                    "registry_type": "npm",
                    "identifier": "@example/server",
                    "version": "1.0.0",
                    "transport": "stdio",
                }
            ]
        )
    )
    entry = build_entry(merged, popularity=0.0, slug="example")
    assert isinstance(entry, McpEntry)
    assert entry.stdio_only is True
    assert entry.mcp_url is None


def test_build_entry_recomputes_url_unverified_from_the_elected_trust_tier() -> None:
    verified = build_entry(
        _merged(
            trust_tier="registry_verified",
            fields=_base_fields(
                url_unverified=True,
                mcp_url="https://mcp.example/mcp",
                transport="streamable_http",
                remotes=[{"transport": "streamable_http", "url": "https://mcp.example/mcp"}],
            ),
        ),
        popularity=0.0,
        slug="verified",
    )
    assert isinstance(verified, McpEntry)
    assert verified.url_unverified is False

    indexed = build_entry(
        _merged(
            trust_tier="indexed",
            fields=_base_fields(
                url_unverified=False,
                mcp_url="https://mcp.example/mcp",
                transport="streamable_http",
                remotes=[{"transport": "streamable_http", "url": "https://mcp.example/mcp"}],
            ),
        ),
        popularity=0.0,
        slug="indexed",
    )
    assert isinstance(indexed, McpEntry)
    assert indexed.url_unverified is True


def test_build_entry_rejects_a_field_the_model_does_not_have() -> None:
    merged = _merged(fields=_base_fields(iconUrl="https://example.com/icon.png"))
    with pytest.raises(NormalizeError):
        build_entry(merged, popularity=0.0, slug="example")
