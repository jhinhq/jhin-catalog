"""The seam between the marketplaces crawl and the skill normaliser.

Every other test in this suite exercises one side of it. Nothing fed a real
``_skill_record`` into ``normalize_marketplace_skill``, and while nobody was
looking the two stopped agreeing about the shape of a payload: the crawl wrote
the frontmatter scalars flat, the normaliser read them from a ``frontmatter``
block that was never written, and ten fields were silently lost on every
skill. One of them inverted — a skill whose author set
``disable-model-invocation`` was published as model-invocable.

It went unnoticed because ``data/skills/`` is empty in the committed tree, so
the loss showed up in no shard and no count. These tests hold the two halves
together at the only place their disagreement is visible.
"""

from __future__ import annotations

import pytest

from jhin_catalog.normalize import normalize_marketplace_skill
from jhin_catalog.sources.marketplaces import (
    SkillFrontmatter,
    _docs_url,
    _Plugin,
    _skill_record,
)
from jhin_catalog.types import MAX_URL_CHARS, Candidate

COMMIT = "a" * 40


def _plugin(**overrides: object) -> _Plugin:
    payload: dict[str, object] = {
        "marketplace": "official",
        "marketplace_repo": "anthropics/claude-plugins-official",
        "marketplace_owner": "anthropics",
        "plugin": "notes",
        "description": "The plugin's own description, used only as a fallback.",
        "category": "productivity",
        "license": "Apache-2.0",
        "keywords": ("notes", "writing"),
        "declared_skills": ("skills/quick-note",),
        "source_kind": "github",
        "source_value": "https://github.com/anthropics/skills",
        "source_git_ref": "",
        "source_sha": "",
        "plugin_root": "",
    }
    payload.update(overrides)
    return _Plugin.model_validate(payload)


def _frontmatter(**overrides: object) -> SkillFrontmatter:
    payload: dict[str, object] = {
        "name": "quick-note",
        "description": "Capture a thought without leaving the terminal.",
        "version": "1.2.3",
        "license": "MIT",
        "allowed_tools": ("Read", "Write"),
        "disable_model_invocation": False,
        "frontmatter_bytes": 169,
    }
    payload.update(overrides)
    return SkillFrontmatter.model_validate(payload)


def _candidate(
    *,
    plugin: _Plugin | None = None,
    frontmatter: SkillFrontmatter | None = None,
    skill_path: str = "skills/quick-note/SKILL.md",
    renamed_from: tuple[str, ...] = (),
    marketplace_reviewed: bool = False,
) -> Candidate:
    """One crawled skill, all the way through to a normalised candidate."""
    record = _skill_record(
        plugin or _plugin(),
        owner="anthropics",
        repo="skills",
        skill_path=skill_path,
        commit_sha=COMMIT,
        frontmatter=frontmatter or _frontmatter(),
        renamed_from=renamed_from,
        marketplace_reviewed=marketplace_reviewed,
    )
    candidate = normalize_marketplace_skill(record)
    assert candidate is not None
    return candidate


def test_every_frontmatter_field_the_crawl_read_survives_normalisation() -> None:
    """The regression itself: ten fields used to arrive empty."""
    fields = _candidate().fields
    assert fields["name"] == "quick-note"
    assert fields["skill_name"] == "quick-note"
    assert fields["description"] == "Capture a thought without leaving the terminal."
    assert fields["license"] == "MIT"
    assert fields["skill_version"] == "1.2.3"
    assert fields["allowed_tools"] == ["Read", "Write"]
    assert fields["frontmatter_bytes"] == 169
    assert fields["tags"] == ["notes", "writing"]
    assert fields["category"] == "Productivity"
    assert fields["commit_sha"] == COMMIT


def test_a_skill_that_opted_out_of_model_invocation_is_not_published_as_invocable() -> None:
    """The inversion. ``model_invocable`` is stated; ``disable-...`` is its source.

    Reading only the frontmatter spelling of the flag on a payload that states
    the resolved one gave every skill the default ``True``, which reverses the
    author's decision rather than merely losing it.
    """
    opted_out = _candidate(frontmatter=_frontmatter(disable_model_invocation=True))
    assert opted_out.fields["model_invocable"] is False
    assert _candidate().fields["model_invocable"] is True


def test_the_plugin_reference_is_populated_from_the_nested_plugin_object() -> None:
    """``plugin`` is a dict in the payload; it was read as a string, so always empty."""
    plugin = _candidate().fields["plugin"]
    assert plugin == {
        "marketplace": "official",
        "marketplace_repo": "anthropics/claude-plugins-official",
        "plugin": "notes",
        "source_kind": "github",
        "source_value": "https://github.com/anthropics/skills",
        "sha": COMMIT,
    }


def test_a_rename_reconciles_to_the_row_the_previous_build_wrote() -> None:
    """The whole point of ``renamed_from``, and it was unreachable.

    Without a plugin name there was no ``skill:plugin:`` key at all, so a
    renamed plugin's skills were deleted and reinserted under new identities
    on the build after the rename.
    """
    candidate = _candidate(renamed_from=("scratchpad",))
    assert candidate.primary_key == ("skill:skill:github.com/anthropics/skills/skills/quick-note")
    assert "skill:plugin:official/notes/quick-note" in candidate.alias_keys
    assert "skill:plugin:official/scratchpad/quick-note" in candidate.alias_keys


def test_the_plugin_description_stands_in_when_the_frontmatter_has_none() -> None:
    fields = _candidate(frontmatter=_frontmatter(description="")).fields
    assert fields["description"] == "The plugin's own description, used only as a fallback."


def test_a_frontmatter_block_is_still_read_when_a_record_nests_one() -> None:
    """Flat first, nested second — neither shape loses a field."""
    record = _skill_record(
        _plugin(),
        owner="anthropics",
        repo="skills",
        skill_path="skills/quick-note/SKILL.md",
        commit_sha=COMMIT,
        frontmatter=_frontmatter(license=""),
        renamed_from=(),
        marketplace_reviewed=False,
    )
    payload = dict(record.payload)
    payload.pop("license")
    payload["frontmatter"] = {"license": "BSD-3-Clause"}
    nested = record.model_copy(update={"payload": payload})
    candidate = normalize_marketplace_skill(nested)
    assert candidate is not None
    assert candidate.fields["license"] == "BSD-3-Clause"


def test_the_whole_candidate_builds_into_a_validated_entry() -> None:
    """The fields are not merely present; the model accepts all of them."""
    from jhin_catalog.build import _provisional_slug
    from jhin_catalog.dedupe import merge_candidates
    from jhin_catalog.normalize import build_entry

    merged = merge_candidates([_candidate()])
    assert len(merged) == 1
    entry = build_entry(merged[0], popularity=0.0, slug=_provisional_slug(merged[0]))
    assert entry.kind == "skill"
    assert entry.description == "Capture a thought without leaving the terminal."
    assert entry.tags == ("notes", "writing")


def test_a_reviewed_marketplace_stamps_its_flag_all_the_way_onto_the_entry() -> None:
    """The whole reviewed pipeline: crawl flag → candidate → merged → entry.

    The flag is what a consumer elects its own "reviewed" tier from, so a
    loss anywhere along this seam silently demotes every reviewed skill back
    to the community pile.
    """
    from jhin_catalog.build import _provisional_slug
    from jhin_catalog.dedupe import merge_candidates
    from jhin_catalog.normalize import build_entry

    merged = merge_candidates([_candidate(marketplace_reviewed=True)])
    entry = build_entry(merged[0], popularity=0.0, slug=_provisional_slug(merged[0]))
    assert entry.marketplace_reviewed is True
    assert entry.trust_tier == "indexed"  # the wire tier itself never moves


def test_an_unreviewed_marketplace_leaves_the_flag_off_the_entry() -> None:
    from jhin_catalog.build import _provisional_slug
    from jhin_catalog.dedupe import merge_candidates
    from jhin_catalog.normalize import build_entry

    merged = merge_candidates([_candidate(marketplace_reviewed=False)])
    entry = build_entry(merged[0], popularity=0.0, slug=_provisional_slug(merged[0]))
    assert entry.marketplace_reviewed is False


def test_a_skill_carries_its_repository_owners_avatar_as_its_icon_url() -> None:
    """Rule 2 of the icon election, end to end from the crawl to the entry."""
    from jhin_catalog.build import _provisional_slug
    from jhin_catalog.dedupe import merge_candidates
    from jhin_catalog.normalize import build_entry

    candidate = _candidate()
    assert candidate.fields["icon_url"] == "https://github.com/anthropics.png?size=128"
    merged = merge_candidates([candidate])
    entry = build_entry(merged[0], popularity=0.0, slug=_provisional_slug(merged[0]))
    assert entry.icon_url == "https://github.com/anthropics.png?size=128"


# --- the crawl's own bounds ------------------------------------------------


def test_a_declared_skill_path_too_long_for_its_url_never_reaches_a_record() -> None:
    """``_skill_record`` used to raise ``SourceError`` out of the entire crawl.

    ``plugin_root`` is two 255-character values concatenated, so a manifest
    can name a path that overruns ``MAX_URL_CHARS`` in the derived
    ``docs_url``. Any anonymous GitHub user could commit one and fail every
    nightly build; the candidate is dropped before it is fetched instead.
    """
    overlong = "d" * (MAX_URL_CHARS + 40)
    with pytest.raises(Exception):  # noqa: B017 - the point is that it raised at all
        _skill_record(
            _plugin(),
            owner="anthropics",
            repo="skills",
            skill_path=f"{overlong}/SKILL.md",
            commit_sha=COMMIT,
            frontmatter=_frontmatter(),
            renamed_from=(),
            marketplace_reviewed=False,
        )


def test_the_docs_url_budget_matches_the_url_the_record_actually_writes() -> None:
    """The filter and the record must agree, or the filter is decorative."""
    prefix = _docs_url(owner="anthropics", repo="skills", skill_path="")
    room = MAX_URL_CHARS - len(prefix)
    longest = "a" * (room - len("/SKILL.md")) + "/SKILL.md"
    assert len(_docs_url(owner="anthropics", repo="skills", skill_path=longest)) <= MAX_URL_CHARS
