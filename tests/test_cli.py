"""Command surface and the exit-code contract."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

import jhin_catalog.build
import jhin_catalog.cli
from jhin_catalog.build import plan_shards, render_schema, write_shards
from jhin_catalog.cli import main
from jhin_catalog.diffgate import DiffGateFailed
from jhin_catalog.sources.base import SourceError
from jhin_catalog.types import (
    CatalogEntry,
    CuratedError,
    DiffReport,
    McpEntry,
    RepoRef,
    SkillEntry,
    SourceRef,
)


def _mcp(index: int) -> McpEntry:
    """A publishable synthetic server, distinct in key, slug, and rank."""
    return McpEntry(
        kind="mcp",
        canonical_key=f"mcp:registry:example.test/server-{index:03d}",
        slug=f"srv_{index:03d}",
        name=f"Server {index:03d}",
        description="A synthetic server, present so the gate has something to measure.",
        trust_tier="registry_verified",
        sources=(
            SourceRef(
                source_id="registry",
                upstream_id=f"example.test/server-{index:03d}",
                url=f"https://registry.example/v0.1/servers/server-{index:03d}",
            ),
        ),
        category="Developer tools",
        icon="terminal",
        mcp_url=f"https://server-{index:03d}.example/mcp",
        transport="streamable_http",
        popularity=round(1.0 - index / 1000, 4),
    )


def _skill(index: int) -> SkillEntry:
    return SkillEntry(
        kind="skill",
        canonical_key=f"skill:skill:github.com/example-org/pack/skills/s{index:03d}",
        slug=f"skill_{index:03d}",
        name=f"Skill {index:03d}",
        description="A synthetic skill, present so the gate has something to measure.",
        trust_tier="indexed",
        sources=(
            SourceRef(
                source_id="marketplaces",
                upstream_id=f"example-org/pack#pack#skills/s{index:03d}",
                url=f"https://github.com/example-org/pack/blob/HEAD/skills/s{index:03d}/SKILL.md",
            ),
        ),
        repo=RepoRef(host="github.com", owner="example-org", repo="pack"),
        skill_name=f"skill-{index:03d}",
        category="General",
        source_ref=f"example-org/pack/skills/s{index:03d}",
        skill_path=f"skills/s{index:03d}/SKILL.md",
    )


def _populate(root: Path, mcp: Sequence[CatalogEntry], skills: Sequence[CatalogEntry]) -> None:
    """A root `verify` accepts: 512 shards and a generated schema."""
    write_shards(root, "mcp", plan_shards(mcp))
    write_shards(root, "skill", plan_shards(skills))
    schema = root / "schema"
    schema.mkdir(parents=True, exist_ok=True)
    (schema / "catalog.schema.json").write_text(render_schema(), encoding="utf-8")


@pytest.fixture
def clean_root(tmp_path: Path) -> Path:
    root = tmp_path / "clean"
    root.mkdir()
    _populate(root, [_mcp(index) for index in range(5)], [_skill(0)])
    return root


def _run(argv: list[str]) -> int:
    """The process exit code, whether ``main`` returned it or argparse raised it."""
    try:
        return main(argv)
    except SystemExit as exit_signal:
        return 0 if exit_signal.code is None else int(exit_signal.code)


def _patch_run_sync(monkeypatch: pytest.MonkeyPatch, fake: Callable[..., Any]) -> None:
    monkeypatch.setattr(jhin_catalog.build, "run_sync", fake)
    if hasattr(jhin_catalog.cli, "run_sync"):
        monkeypatch.setattr(jhin_catalog.cli, "run_sync", fake)


def _raiser(exc: BaseException) -> Callable[..., Any]:
    async def fake(*_args: object, **_kwargs: object) -> None:
        raise exc

    return fake


def test_help_exits_zero() -> None:
    assert _run(["--help"]) == 0


def test_an_unknown_command_is_a_usage_error() -> None:
    assert _run(["nope"]) == 2


def test_verify_accepts_a_root_the_builder_wrote(clean_root: Path) -> None:
    assert _run(["verify", "--root", str(clean_root)]) == 0


def test_verify_rejects_a_shard_someone_edited_by_hand(clean_root: Path) -> None:
    shard = next(
        path
        for path in sorted((clean_root / "data" / "mcp").glob("*.jsonl"))
        if path.stat().st_size > 0
    )
    body = bytearray(shard.read_bytes())
    body[0] = body[0] + 1
    shard.write_bytes(bytes(body))
    assert _run(["verify", "--root", str(clean_root)]) == 5


def test_verify_rejects_an_entry_filed_in_the_wrong_shard(clean_root: Path) -> None:
    shards = sorted((clean_root / "data" / "mcp").glob("*.jsonl"))
    occupied = next(path for path in shards if path.stat().st_size > 0)
    empty = next(path for path in shards if path.stat().st_size == 0)
    empty.write_bytes(occupied.read_bytes())
    occupied.write_bytes(b"")
    assert _run(["verify", "--root", str(clean_root)]) == 5


def test_verify_rejects_a_schema_that_drifted_from_the_generator(clean_root: Path) -> None:
    path = clean_root / "schema" / "catalog.schema.json"
    document = json.loads(path.read_text("utf-8"))
    document["title"] = "hand edited"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert _run(["verify", "--root", str(clean_root)]) == 5


def test_verify_json_prints_one_object_and_nothing_else(
    clean_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(["verify", "--root", str(clean_root), "--json"]) == 0
    stdout = capsys.readouterr().out
    assert json.loads(stdout) is not None
    assert stdout.count("\n") <= 1


def test_diff_between_identical_roots_passes(clean_root: Path, tmp_path: Path) -> None:
    other = tmp_path / "same"
    other.mkdir()
    _populate(other, [_mcp(index) for index in range(5)], [_skill(0)])
    assert _run(["diff", "--root", str(clean_root), "--against", str(other)]) == 0


def test_diff_exits_three_when_a_threshold_would_fail(tmp_path: Path) -> None:
    """Twenty-five of a hundred entries differ, which breaches 0.20 either way.

    ``--against`` does not say which side is the baseline, so the fixture is
    built to fail in both directions: the change fraction is 0.25 whichever
    root is read first.
    """
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    baseline = [_mcp(index) for index in range(100)]
    changed: list[CatalogEntry] = [
        entry.model_copy(update={"name": f"Renamed {index:03d}"}) if index < 25 else entry
        for index, entry in enumerate(baseline)
    ]
    _populate(left, baseline, [_skill(0)])
    _populate(right, changed, [_skill(0)])
    assert _run(["diff", "--root", str(left), "--against", str(right)]) == 3


def test_diff_json_prints_one_object(
    clean_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = tmp_path / "same"
    other.mkdir()
    _populate(other, [_mcp(index) for index in range(5)], [_skill(0)])
    assert _run(["diff", "--root", str(clean_root), "--against", str(other), "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert isinstance(document, dict)


def test_export_limit_three_prints_three_array_elements(
    clean_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(["export", "--root", str(clean_root), "--limit", "3"]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert isinstance(exported, list)
    assert len(exported) == 3
    assert len({app["slug"] for app in exported}) == 3


def test_export_writes_to_a_file_when_out_is_given(clean_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "catalog.json"
    assert _run(["export", "--root", str(clean_root), "--out", str(out), "--limit", "2"]) == 0
    body = out.read_text("utf-8")
    assert body.endswith("\n")
    assert len(json.loads(body)) == 2


def test_stats_json_prints_one_object(clean_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["stats", "--root", str(clean_root), "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert isinstance(document, dict)


def test_a_tripped_gate_exits_three(clean_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = DiffReport(
        kind="mcp",
        baseline_count=100,
        candidate_count=94,
        added=(),
        dropped=tuple(f"mcp:registry:example.test/server-{index:03d}" for index in range(6)),
        changed=(),
        drop_fraction=0.06,
        change_fraction=0.0,
    )
    _patch_run_sync(monkeypatch, _raiser(DiffGateFailed("too much dropped", report=report)))
    assert _run(["sync", "--root", str(clean_root)]) == 3


def test_a_fetch_fault_exits_four(clean_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run_sync(
        monkeypatch,
        _raiser(SourceError("smithery served an empty page", source_id="smithery")),
    )
    assert _run(["sync", "--root", str(clean_root)]) == 4


def test_bad_curated_input_exits_six(clean_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run_sync(monkeypatch, _raiser(CuratedError("denylist key matches nothing")))
    assert _run(["sync", "--root", str(clean_root)]) == 6


def test_an_unexpected_exception_exits_one_and_prints_the_traceback(
    clean_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_run_sync(monkeypatch, _raiser(ValueError("something nobody predicted")))
    assert _run(["sync", "--root", str(clean_root)]) == 1
    assert "Traceback" in capsys.readouterr().err


def test_a_known_failure_names_its_own_exception_type(
    clean_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_run_sync(monkeypatch, _raiser(CuratedError("denylist key matches nothing")))
    _run(["sync", "--root", str(clean_root)])
    assert "CuratedError: denylist key matches nothing" in capsys.readouterr().err
