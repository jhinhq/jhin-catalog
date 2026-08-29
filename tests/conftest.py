"""Repo-root paths, recorded upstream payloads, and a mock-transport client —
the same shape every source sees, with no socket ever opened."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import httpx
import pytest

from jhin_catalog.http import build_client
from jhin_catalog.types import (
    JsonValue,
    McpEntry,
    PackageRef,
    PluginRef,
    PopularitySignals,
    RemoteRef,
    RepoRef,
    SkillEntry,
    SourceRef,
)


class FixtureLoader(Protocol):
    def __call__(self, name: str) -> JsonValue: ...


class ClientFactory(Protocol):
    def __call__(self, handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient: ...


class RecordingSleep:
    """A sleep that never sleeps and remembers what it was asked for.

    Retry policy is worth testing and waiting for is not, so every coroutine
    that would pause takes its sleep as a parameter and the tests assert on
    ``delays`` instead of on elapsed time.
    """

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture
def load_fixture(data_dir: Path) -> FixtureLoader:
    """Read one recorded upstream payload out of ``tests/data``."""

    def load(name: str) -> JsonValue:
        text = (data_dir / name).read_text("utf-8")
        if name.endswith(".json"):
            loaded: JsonValue = json.loads(text)
            return loaded
        return text

    return load


@pytest.fixture
def no_sleep() -> RecordingSleep:
    return RecordingSleep()


@pytest.fixture
def mock_client() -> ClientFactory:
    """A real ``AsyncClient`` — real agent, real timeouts, no redirects — over
    a transport that answers from the test's own handler."""

    def factory(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
        return build_client(transport=httpx.MockTransport(handler))

    return factory


@pytest.fixture
def tmp_catalog(tmp_path: Path) -> Path:
    root = tmp_path / "catalog"
    for relative in ("data/mcp", "data/skills", "curated"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def sample_mcp_entry() -> McpEntry:
    """A server entry with every optional field carrying a real value.

    Nothing here contradicts anything else: the alias keys are the keys this
    identity would really produce, the signals are the ones that really score
    ``0.6684``, and the elected remote is really the one ``mcp_url`` names. A
    golden serialisation subject that told a small lie would teach the reader
    the wrong shape.
    """
    return McpEntry(
        kind="mcp",
        canonical_key="mcp:repo:github.com/example-org/example-mcp#servers/example",
        alias_keys=(
            "mcp:npm:example-mcp",
            "mcp:registry:io.github.example-org/example-mcp",
            "mcp:url:mcp.example/mcp",
        ),
        slug="example_mcp",
        name="Example MCP — kept as ünicode",
        description="A server that carries one of everything, for the serialisation tests.",
        homepage="https://example.com/",
        docs_url="https://docs.example.com/mcp",
        repo=RepoRef(
            host="github.com", owner="example-org", repo="example-mcp", subpath="servers/example"
        ),
        trust_tier="registry_verified",
        popularity=0.6684,
        popularity_signals=PopularitySignals(
            github_stars=1200,
            github_forks=97,
            npm_downloads_monthly=50_000,
            npm_dependents=42,
            smithery_use_count=8_100,
            registry_version_count=7,
        ),
        sources=(
            SourceRef(
                source_id="registry",
                upstream_id="io.github.example-org/example-mcp",
                url="https://registry.example/v0.1/servers/example-mcp/versions",
            ),
            SourceRef(
                source_id="smithery",
                upstream_id="example-org/example-mcp",
                url="https://smithery.ai/server/example-org/example-mcp",
            ),
            SourceRef(
                source_id="npm",
                upstream_id="example-mcp",
                url="https://www.npmjs.com/package/example-mcp",
            ),
            SourceRef(
                source_id="github_topics",
                upstream_id="example-org/example-mcp",
                url="https://github.com/example-org/example-mcp",
            ),
        ),
        tags=("example", "mcp-server", "search"),
        license="MIT",
        curated_fields=("auth_note", "setup_note"),
        category="Search & web",
        icon="search",
        connector_type="mcp",
        mcp_url="https://mcp.example/mcp",
        transport="streamable_http",
        auth_hint="header",
        auth_note="Set the X-Api-Key header from the provider's dashboard.",
        setup_note="Create a key in the dashboard before you connect.",
        connector_config={"region": "eu", "search_backend": "example"},
        packages=(
            PackageRef(
                registry_type="npm",
                identifier="example-mcp",
                version="1.4.0",
                runtime_hint="node",
                transport="stdio",
            ),
        ),
        remotes=(
            RemoteRef(
                transport="streamable_http",
                url="https://mcp.example/mcp",
                header_names=("X-Api-Key",),
            ),
        ),
        tool_count=12,
        registry_name="io.github.example-org/example-mcp",
        smithery_qualified_name="example-org/example-mcp",
        npm_package="example-mcp",
        verified_upstream=True,
    )


@pytest.fixture
def sample_skill_entry() -> SkillEntry:
    """A skill entry with every optional field carrying a real value."""
    return SkillEntry(
        kind="skill",
        canonical_key="skill:skill:github.com/example-org/example-pack/skills/design-review",
        alias_keys=("skill:plugin:example-marketplace/example-pack/design-review",),
        slug="design_review",
        name="Design review",
        description="Reads a diff and reports on layout, hierarchy, and consistency.",
        homepage="https://example.com/skills",
        docs_url="https://github.com/example-org/example-pack",
        repo=RepoRef(host="github.com", owner="example-org", repo="example-pack"),
        trust_tier="curated",
        popularity=0.5506,
        popularity_signals=PopularitySignals(github_stars=1000, github_forks=31),
        sources=(
            SourceRef(
                source_id="curated",
                upstream_id="skill:skill:github.com/example-org/example-pack/skills/design-review",
                url="https://github.com/jhin-dev/jhin-catalog/blob/main/curated/skills.yaml",
            ),
            SourceRef(
                source_id="marketplaces",
                upstream_id="example-org/example-pack#example-pack#skills/design-review",
                url=(
                    "https://github.com/example-org/example-pack"
                    "/blob/HEAD/skills/design-review/SKILL.md"
                ),
            ),
        ),
        tags=("design", "review"),
        license="Apache-2.0",
        curated_fields=("description",),
        skill_name="design-review",
        category="Design",
        source_ref="example-org/example-pack/skills/design-review",
        skill_path="skills/design-review/SKILL.md",
        plugin=PluginRef(
            marketplace="example-marketplace",
            marketplace_repo="example-org/example-marketplace",
            plugin="example-pack",
            source_kind="git-subdir",
            source_value="https://github.com/example-org/example-pack.git",
            sha="d16d14ac1f4b3c2a5e6f708192a3b4c5d6e7f809",
        ),
        commit_sha="d16d14ac1f4b3c2a5e6f708192a3b4c5d6e7f809",
        model_invocable=True,
        allowed_tools=("Bash", "Read"),
        skill_version="1.2.0",
        frontmatter_bytes=214,
    )
