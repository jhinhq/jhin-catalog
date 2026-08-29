"""``schema/catalog.schema.json`` is generated, not hand-written."""

from __future__ import annotations

import json
from pathlib import Path

from jhin_catalog.build import SCHEMA_PATH, render_schema


def test_the_committed_schema_is_exactly_what_the_generator_emits(repo_root: Path) -> None:
    """Regenerate with ``jhin-catalog verify`` in mind: the schema is written by
    ``build.render_schema`` and committed, so a model change that is not
    regenerated fails here rather than downstream in a Jhin deployment."""
    assert (repo_root / SCHEMA_PATH).read_text("utf-8") == render_schema()


def test_the_rendered_schema_is_indented_sorted_and_newline_terminated() -> None:
    """Indented for review, sorted for stability, raw UTF-8 like every other
    file this repository writes."""
    body = render_schema()
    document = json.loads(body)
    assert body == json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert body.endswith("\n")
    assert not body.endswith("\n\n")


def test_the_schema_describes_both_record_kinds() -> None:
    body = render_schema()
    assert '"mcp"' in body
    assert '"skill"' in body
