"""Marketplace parsing: all eight source forms, renames, nested skills."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

import httpx
import pytest

from jhin_catalog.sources.base import DEFAULT_LIMITS, SourceError
from jhin_catalog.sources.marketplaces import (
    MARKETPLACE_PATH,
    MAX_FRONTMATTER_BYTES,
    MarketplacesSource,
    parse_marketplace,
    parse_plugin_source,
    parse_skill_frontmatter,
    skill_paths,
)
from jhin_catalog.types import JsonObject, JsonValue, RawRecord

type FixtureLoader = Callable[[str], JsonValue]
type ClientFactory = Callable[[Callable[[httpx.Request], httpx.Response]], httpx.AsyncClient]
type Sleeper = Callable[[float], Awaitable[None]]

OFFICIAL_URL = (
    f"https://raw.githubusercontent.com/anthropics/claude-plugins-official/HEAD/{MARKETPLACE_PATH}"
)
BODY_SENTINEL = "BODY-SENTINEL-DO-NOT-INDEX"


def _object(value: JsonValue) -> JsonObject:
    assert isinstance(value, dict)
    return value


def _text(value: JsonValue) -> str:
    assert isinstance(value, str)
    return value


def _plugins(load_fixture: FixtureLoader) -> dict[str, JsonObject]:
    records, _ = parse_marketplace(
        load_fixture("marketplace_official.json"),
        marketplace_repo="anthropics/claude-plugins-official",
        url=OFFICIAL_URL,
    )
    return {_text(record.payload["plugin"]): record.payload for record in records}


# --- the eight source forms -------------------------------------------------


def test_a_relative_string_source_keeps_its_path_without_the_dot_slash() -> None:
    source = parse_plugin_source("./sub/dir")
    assert source.source_kind == "relative"
    assert source.source_value == "sub/dir"


def test_a_bare_dot_slash_source_is_the_repository_root() -> None:
    source = parse_plugin_source("./")
    assert source.source_kind == "relative"
    assert source.source_value == ""


def test_a_url_source_with_a_sha_is_a_url_source() -> None:
    source = parse_plugin_source(
        {
            "source": "url",
            "url": "https://github.com/SalesforceAIResearch/agentforce-adlc.git",
            "sha": "d16d14ac1f4b3c2a5e6f708192a3b4c5d6e7f809",
        }
    )
    assert source.source_kind == "url"
    assert source.source_value == "https://github.com/SalesforceAIResearch/agentforce-adlc.git"
    assert source.sha == "d16d14ac1f4b3c2a5e6f708192a3b4c5d6e7f809"


def test_the_schema_invalid_url_plus_path_hybrid_is_read_as_a_git_subdir() -> None:
    """The plugin schema forbids this shape and marketplaces publish it anyway."""
    source = parse_plugin_source(
        {"source": "url", "url": "https://github.com/acme-example/monorepo.git", "path": "pkg/a"}
    )
    assert source.source_kind == "git-subdir"
    assert source.source_value == "https://github.com/acme-example/monorepo.git"
    assert source.path == "pkg/a"


def test_a_declared_git_subdir_source_keeps_its_url_path_and_ref() -> None:
    source = parse_plugin_source(
        {
            "source": "git-subdir",
            "url": "https://github.com/acme-example/monorepo.git",
            "path": "pkg/b",
            "ref": "main",
        }
    )
    assert source.source_kind == "git-subdir"
    assert source.path == "pkg/b"
    assert source.ref == "main"


def test_a_github_source_becomes_the_repository_url() -> None:
    source = parse_plugin_source({"source": "github", "repo": "acme-example/from-github"})
    assert source.source_kind == "github"
    assert source.source_value == "https://github.com/acme-example/from-github"


def test_an_npm_source_keeps_the_package_identifier() -> None:
    source = parse_plugin_source({"source": "npm", "package": "@acme-example/from-npm"})
    assert source.source_kind == "npm"
    assert source.source_value == "@acme-example/from-npm"


def test_a_shape_nobody_has_documented_is_recorded_verbatim_as_unknown() -> None:
    """An unreadable source is still evidence; it is written down, not guessed."""
    value: JsonObject = {"source": "carrier-pigeon", "coop": "west"}
    source = parse_plugin_source(value)
    assert source.source_kind == "unknown"
    assert source.source_value == json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_a_path_that_climbs_out_of_its_repository_is_refused() -> None:
    """``..`` is unresolvable rather than merely unrecognised, so it is refused.

    A leading ``/`` is a different thing: inside a manifest it means the
    repository root, not the filesystem root, so it is confined rather than
    rejected. Either way nothing addresses a path outside the repository.
    """
    assert parse_plugin_source("../../etc/passwd").source_kind == "unknown"
    confined = parse_plugin_source("/absolute/path")
    assert confined.source_value == "absolute/path"
    assert not confined.source_value.startswith("/")


# --- the manifest -----------------------------------------------------------


def test_the_official_manifest_yields_one_record_per_plugin(
    load_fixture: FixtureLoader,
) -> None:
    records, _ = parse_marketplace(
        load_fixture("marketplace_official.json"),
        marketplace_repo="anthropics/claude-plugins-official",
        url=OFFICIAL_URL,
    )
    assert len(records) == 8
    assert all(isinstance(record, RawRecord) for record in records)


def test_renames_come_back_separately_and_never_become_a_record(
    load_fixture: FixtureLoader,
) -> None:
    """A rename is continuity for an identity, not a thing that exists."""
    records, renames = parse_marketplace(
        load_fixture("marketplace_official.json"),
        marketplace_repo="anthropics/claude-plugins-official",
        url=OFFICIAL_URL,
    )
    assert renames == {"adlc": "agentforce-adlc"}
    assert "adlc" not in {_text(record.payload["plugin"]) for record in records}


def test_a_manifest_with_no_renames_reports_an_empty_map(
    load_fixture: FixtureLoader,
) -> None:
    _, renames = parse_marketplace(
        load_fixture("marketplace_community.json"),
        marketplace_repo="acme-example/community",
        url="https://raw.githubusercontent.com/acme-example/community/HEAD/x.json",
    )
    assert renames == {}


def test_the_plugin_root_prefixes_every_relative_source(
    load_fixture: FixtureLoader,
) -> None:
    """``metadata.pluginRoot`` is where the manifest says its own plugins live."""
    plugins = _plugins(load_fixture)
    assert plugins["design"]["plugin_root"] == "plugins/design"
    assert plugins["root-plugin"]["plugin_root"] == "plugins"


def test_the_plugin_root_has_no_authority_over_an_external_repository(
    load_fixture: FixtureLoader,
) -> None:
    plugins = _plugins(load_fixture)
    assert plugins["agentforce-adlc"]["plugin_root"] == ""
    assert plugins["subdir"]["plugin_root"] == "packages/subdir"


def test_a_manifest_missing_its_name_is_a_source_error() -> None:
    with pytest.raises(SourceError):
        parse_marketplace({"owner": {"name": "x"}, "plugins": []}, marketplace_repo="o/r", url="u")


def test_a_manifest_missing_its_owner_is_a_source_error() -> None:
    with pytest.raises(SourceError):
        parse_marketplace({"name": "x", "plugins": []}, marketplace_repo="o/r", url="u")


def test_a_manifest_missing_its_plugin_list_is_a_source_error() -> None:
    with pytest.raises(SourceError):
        parse_marketplace({"name": "x", "owner": {"name": "y"}}, marketplace_repo="o/r", url="u")


def test_a_declared_skill_list_is_carried_on_the_plugin_record(
    load_fixture: FixtureLoader,
) -> None:
    records, _ = parse_marketplace(
        load_fixture("marketplace_community.json"),
        marketplace_repo="acme-example/community",
        url="https://raw.githubusercontent.com/acme-example/community/HEAD/x.json",
    )
    declared = records[0].payload["declared_skills"]
    assert isinstance(declared, list)
    assert set(declared) == {"skills/code-review", "skills/eng/deep-review"}


# --- finding the skills -----------------------------------------------------


def test_skill_paths_matches_the_basename_at_any_depth(load_fixture: FixtureLoader) -> None:
    """A flat ``skills/*/SKILL.md`` glob would miss most of the corpus."""
    found = skill_paths(load_fixture("git_tree.json"), plugin_root="")
    assert found == (
        "plugins/x/skills/y/SKILL.md",
        "skills/eng/code-review/SKILL.md",
        "skills/foo/SKILL.md",
    )


def test_skill_paths_does_not_match_a_template_that_merely_starts_the_same_way(
    load_fixture: FixtureLoader,
) -> None:
    found = skill_paths(load_fixture("git_tree.json"), plugin_root="")
    assert "docs/SKILL.md.tmpl" not in found


def test_skill_paths_ignores_a_tree_entry_that_is_not_a_blob(
    load_fixture: FixtureLoader,
) -> None:
    found = skill_paths(load_fixture("git_tree.json"), plugin_root="")
    assert "docs/skills/SKILL.md" not in found


def test_a_bare_root_level_skill_file_names_no_directory_and_is_skipped(
    load_fixture: FixtureLoader,
) -> None:
    found = skill_paths(load_fixture("git_tree.json"), plugin_root="")
    assert "SKILL.md" not in found


def test_a_plugin_root_narrows_the_search_to_its_own_subtree(
    load_fixture: FixtureLoader,
) -> None:
    found = skill_paths(load_fixture("git_tree.json"), plugin_root="plugins/x")
    assert found == ("plugins/x/skills/y/SKILL.md",)


def test_a_tree_response_with_no_tree_list_is_a_source_error() -> None:
    with pytest.raises(SourceError):
        skill_paths({"sha": "abc"}, plugin_root="")


# --- frontmatter ------------------------------------------------------------


def test_a_block_scalar_description_is_parsed_as_yaml_not_by_regex(
    load_fixture: FixtureLoader,
) -> None:
    """The ``description: >`` form is why this uses a real YAML parser.

    A line-based regex reads the fold marker as the value and produces a skill
    whose description is ``>``, which is worse than no skill at all.
    """
    frontmatter = parse_skill_frontmatter(_text(load_fixture("skill_frontmatter_block.md")))
    assert frontmatter.name == "design-review"
    assert frontmatter.description.startswith("Reads a diff")
    assert ">" not in frontmatter.description
    assert "\n" not in frontmatter.description


def test_a_comma_separated_allowed_tools_string_becomes_a_tuple(
    load_fixture: FixtureLoader,
) -> None:
    frontmatter = parse_skill_frontmatter(_text(load_fixture("skill_frontmatter_block.md")))
    assert frontmatter.allowed_tools == ("Bash", "Grep", "Read")


def test_an_allowed_tools_list_becomes_the_same_tuple(load_fixture: FixtureLoader) -> None:
    frontmatter = parse_skill_frontmatter(_text(load_fixture("skill_frontmatter_plain.md")))
    assert frontmatter.allowed_tools == ("Read", "Write")


def test_a_quoted_value_that_runs_onto_a_second_line_is_folded(
    load_fixture: FixtureLoader,
) -> None:
    frontmatter = parse_skill_frontmatter(_text(load_fixture("skill_frontmatter_plain.md")))
    assert frontmatter.description == (
        "A short skill whose description is quoted and runs onto a second line."
    )


def test_the_version_and_licence_are_kept_when_the_author_declared_them(
    load_fixture: FixtureLoader,
) -> None:
    frontmatter = parse_skill_frontmatter(_text(load_fixture("skill_frontmatter_block.md")))
    assert frontmatter.version == "1.2.0"
    assert frontmatter.license == "Apache-2.0"


def test_disabling_model_invocation_keeps_the_skill_and_marks_it(
    load_fixture: FixtureLoader,
) -> None:
    """The skill still exists and is still findable; a model just will not
    reach for it on its own, so it is excluded at projection rather than here."""
    frontmatter = parse_skill_frontmatter(_text(load_fixture("skill_frontmatter_plain.md")))
    assert frontmatter.disable_model_invocation is True
    assert frontmatter.name == "quick-note"


def test_the_measured_frontmatter_size_is_recorded(load_fixture: FixtureLoader) -> None:
    frontmatter = parse_skill_frontmatter(_text(load_fixture("skill_frontmatter_block.md")))
    assert 0 < frontmatter.frontmatter_bytes <= MAX_FRONTMATTER_BYTES


def test_a_file_with_no_frontmatter_block_is_a_source_error() -> None:
    with pytest.raises(SourceError):
        parse_skill_frontmatter("# Just a heading\n\nAnd some prose.\n")


def test_a_frontmatter_block_that_is_not_a_mapping_is_a_source_error() -> None:
    with pytest.raises(SourceError):
        parse_skill_frontmatter("---\n- one\n- two\n---\n\nbody\n")


def test_a_frontmatter_block_with_no_name_is_a_source_error() -> None:
    with pytest.raises(SourceError):
        parse_skill_frontmatter("---\ndescription: No name here.\n---\n\nbody\n")


def test_a_frontmatter_block_over_eight_kilobytes_is_a_source_error() -> None:
    padded = "---\nname: huge\ndescription: x\npad: " + ("y" * MAX_FRONTMATTER_BYTES) + "\n---\n"
    with pytest.raises(SourceError):
        parse_skill_frontmatter(padded)


# --- the crawl --------------------------------------------------------------


def _crawl_handler(
    load_fixture: FixtureLoader, requested: list[str]
) -> Callable[[httpx.Request], httpx.Response]:
    """Serves the two seed repos and refuses every discovery result."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        requested.append(url)
        path = request.url.path
        if "/search/repositories" in path:
            return httpx.Response(200, json={"total_count": 0, "items": []})
        if "/git/trees/" in path:
            return httpx.Response(200, json=load_fixture("git_tree.json"))
        if path.endswith(MARKETPLACE_PATH):
            if "claude-plugins-official" in path:
                return httpx.Response(200, json=load_fixture("marketplace_official.json"))
            return httpx.Response(200, json=load_fixture("marketplace_community.json"))
        if path.endswith("marketplace.json"):
            return httpx.Response(404, text="Not Found")
        if path.endswith("SKILL.md"):
            return httpx.Response(200, text=_text(load_fixture("skill_frontmatter_block.md")))
        if path.endswith("plugin.json"):
            return httpx.Response(404, text="Not Found")
        return httpx.Response(404, text="Not Found")

    return handler


async def test_the_skill_body_never_reaches_a_raw_record(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: Sleeper
) -> None:
    """Pointer, not payload: the prompt text is read, parsed, and dropped.

    This is the invariant the whole repository is built around. A skill index
    that stored skill text would be republishing other people's prompts, and
    a Jhin deployment syncing it would be importing them.
    """
    requested: list[str] = []
    async with mock_client(_crawl_handler(load_fixture, requested)) as client:
        fetched = await MarketplacesSource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)

    assert fetched.records
    for record in fetched.records:
        assert BODY_SENTINEL not in json.dumps(record.payload)
    assert BODY_SENTINEL not in json.dumps([record.payload for record in fetched.records])


async def test_a_missing_plugin_json_is_not_an_error(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: Sleeper
) -> None:
    """``strict: false`` is the documented default and the common case."""
    requested: list[str] = []
    async with mock_client(_crawl_handler(load_fixture, requested)) as client:
        fetched = await MarketplacesSource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)
    assert fetched.entry_count > 0


async def test_the_seed_repositories_are_crawled_first_and_in_order(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: Sleeper
) -> None:
    requested: list[str] = []
    async with mock_client(_crawl_handler(load_fixture, requested)) as client:
        await MarketplacesSource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)

    manifests = [url for url in requested if url.endswith(MARKETPLACE_PATH)]
    assert manifests
    assert "anthropics/claude-plugins-official" in manifests[0]


async def test_every_record_points_at_a_blob_a_person_can_open(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: Sleeper
) -> None:
    requested: list[str] = []
    async with mock_client(_crawl_handler(load_fixture, requested)) as client:
        fetched = await MarketplacesSource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)

    for record in fetched.records:
        assert record.source_id == "marketplaces"
        assert record.url.startswith("https://github.com/")


def _discovering_handler(
    load_fixture: FixtureLoader, requested: list[str]
) -> Callable[[httpx.Request], httpx.Response]:
    """Like ``_crawl_handler``, but topic search returns one stranger's repo."""

    inner = _crawl_handler(load_fixture, requested)

    def handler(request: httpx.Request) -> httpx.Response:
        if "/search/repositories" in request.url.path:
            requested.append(str(request.url))
            return httpx.Response(
                200,
                json={"total_count": 1, "items": [{"full_name": "stranger/plugins"}]},
            )
        return inner(request)

    return handler


async def test_an_unreviewed_repository_is_crawled_when_no_allowlist_is_required(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: Sleeper
) -> None:
    """The pre-review default, kept so a bare checkout behaves as it always did."""
    requested: list[str] = []
    async with mock_client(_discovering_handler(load_fixture, requested)) as client:
        await MarketplacesSource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)
    assert any("stranger/plugins" in url for url in requested)


async def test_an_allowlisted_repository_without_the_topic_is_visited_directly(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: Sleeper
) -> None:
    """Most of the allowlist never applied ``topic:claude-code-plugin``.

    Until the crawl visited the allowlist directly, a repository a person had
    reviewed stayed invisible unless its maintainer happened to add the
    discovery label — the topic search here returns nothing, and the
    repository must be read anyway.
    """
    limits = DEFAULT_LIMITS.model_copy(
        update={
            "marketplace_allowlist": ("acme-example/community",),
            "require_marketplace_allowlist": True,
        }
    )
    requested: list[str] = []
    async with mock_client(_crawl_handler(load_fixture, requested)) as client:
        fetched = await MarketplacesSource().fetch(client, limits=limits, sleep=no_sleep)

    assert any("acme-example/community" in url for url in requested)
    repos = {record.payload["marketplace_repo"] for record in fetched.records}
    assert "acme-example/community" in repos


async def test_an_allowlisted_repository_that_fails_is_skipped_not_fatal(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: Sleeper
) -> None:
    """Only the two seeds are load-bearing; a flaky allowlisted repo costs
    itself, exactly as a discovered one does."""
    inner = _crawl_handler(load_fixture, [])

    def handler(request: httpx.Request) -> httpx.Response:
        if "broken-example" in request.url.path:
            return httpx.Response(500, text="upstream broke")
        return inner(request)

    limits = DEFAULT_LIMITS.model_copy(
        update={
            "marketplace_allowlist": ("broken-example/marketplace",),
            "require_marketplace_allowlist": True,
        }
    )
    async with mock_client(handler) as client:
        fetched = await MarketplacesSource().fetch(client, limits=limits, sleep=no_sleep)
    assert fetched.records


async def test_skills_from_a_reviewed_marketplace_carry_the_flag_and_others_do_not(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: Sleeper
) -> None:
    """``trust: reviewed`` travels as a boolean stamped where the marketplace
    is known — a fact about the crawl's configuration, never about anything
    the repository says for itself."""
    limits = DEFAULT_LIMITS.model_copy(
        update={
            "marketplace_allowlist": ("acme-example/community",),
            "require_marketplace_allowlist": True,
            "marketplace_reviewed": ("acme-example/community",),
        }
    )
    requested: list[str] = []
    async with mock_client(_crawl_handler(load_fixture, requested)) as client:
        fetched = await MarketplacesSource().fetch(client, limits=limits, sleep=no_sleep)

    flags = {
        _text(record.payload["marketplace_repo"]): record.payload["marketplace_reviewed"]
        for record in fetched.records
    }
    assert flags["acme-example/community"] is True
    assert flags["anthropics/claude-plugins-official"] is False


async def test_every_record_names_its_repository_owners_avatar_as_icon_url(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: Sleeper
) -> None:
    requested: list[str] = []
    async with mock_client(_crawl_handler(load_fixture, requested)) as client:
        fetched = await MarketplacesSource().fetch(client, limits=DEFAULT_LIMITS, sleep=no_sleep)

    assert fetched.records
    for record in fetched.records:
        repo = _object(record.payload["repo"])
        icon_url = _text(record.payload["icon_url"])
        assert icon_url == f"https://github.com/{_text(repo['owner'])}.png?size=128"


async def test_an_unreviewed_repository_is_never_fetched_under_the_allowlist(
    load_fixture: FixtureLoader, mock_client: ClientFactory, no_sleep: Sleeper
) -> None:
    """``curated/skills.yaml`` says this happens; until now nothing made it happen.

    A skill's ``description`` is attacker-authored free text that reaches
    Jhin's agents, and the diff gate deliberately never blocks additions — so
    one commit adding ``topic:claude-code-plugin`` was enough to put a
    stranger's prose in the index. The allowlist is the control that stops it.
    """
    limits = DEFAULT_LIMITS.model_copy(
        update={
            "marketplace_allowlist": ("anthropics/claude-plugins-official",),
            "require_marketplace_allowlist": True,
        }
    )
    requested: list[str] = []
    async with mock_client(_discovering_handler(load_fixture, requested)) as client:
        fetched = await MarketplacesSource().fetch(client, limits=limits, sleep=no_sleep)

    assert not any("stranger/plugins" in url for url in requested)
    # The seeds are always allowed: they are the reviewed core, and their
    # absence would read as a mass deletion at the diff gate.
    assert any("claude-plugins-official" in url for url in requested)
    assert fetched.records
