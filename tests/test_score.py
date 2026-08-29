"""Popularity: fixed log anchors, coverage weighting, worked values."""

from __future__ import annotations

from jhin_catalog.score import (
    CEILINGS,
    COVERAGE_FLOOR,
    POPULARITY_DECIMALS,
    WEIGHTS,
    normalize_signal,
    popularity,
    rank_score,
)
from jhin_catalog.types import McpEntry, PopularitySignals, SourceRef


def _entry(*, key: str, tier: str, score: float) -> McpEntry:
    return McpEntry(
        kind="mcp",
        canonical_key=key,
        slug=key.rsplit("/", 1)[-1].replace("-", "_"),
        name="Ranked",
        description="An entry that exists so the rank order has something to order.",
        trust_tier=tier,
        sources=(
            SourceRef(source_id="registry", upstream_id=key, url="https://registry.example/x"),
        ),
        category="Developer tools",
        icon="terminal",
        popularity=score,
    )


def test_no_signal_at_all_scores_zero() -> None:
    assert popularity(PopularitySignals()) == 0.0


def test_a_thousand_stars_alone_scores_the_worked_value() -> None:
    assert popularity(PopularitySignals(github_stars=1000)) == 0.5506


def test_stars_and_downloads_together_score_the_worked_value() -> None:
    signals = PopularitySignals(github_stars=1200, npm_downloads_monthly=50_000)
    assert popularity(signals) == 0.6209


def test_every_signal_at_its_ceiling_scores_exactly_one() -> None:
    signals = PopularitySignals(
        github_stars=100_000,
        npm_downloads_monthly=10_000_000,
        smithery_use_count=100_000,
    )
    assert popularity(signals) == 1.0


def test_a_lone_smithery_use_count_is_discounted_for_coverage() -> None:
    """One signal near its ceiling, weighted down for saying only one thing.

    The build specification's worked-value table gives ``0.8560`` for this
    input, which its own formula does not produce: ``log1p(87579) /
    log1p(100_000)`` is ``0.98848``, and ``0.98848 * (0.85 + 0.15 * 0.20)`` is
    ``0.86986``. The other four rows of that table agree with the formula to
    the last digit, so the formula is taken as the normative statement and the
    single table row as the typo. Resolve it in the specification before
    changing this literal.
    """
    assert popularity(PopularitySignals(smithery_use_count=87_579)) == 0.8699


def test_a_missing_signal_normalises_to_none_rather_than_zero() -> None:
    assert normalize_signal(None, CEILINGS["github_stars"]) is None


def test_a_zero_signal_normalises_to_zero_rather_than_none() -> None:
    assert normalize_signal(0, CEILINGS["github_stars"]) == 0.0


def test_a_signal_above_its_ceiling_clamps_to_one() -> None:
    assert normalize_signal(10_000_000, CEILINGS["github_stars"]) == 1.0


def test_a_negative_signal_clamps_to_zero() -> None:
    assert normalize_signal(-5, CEILINGS["github_stars"]) == 0.0


def test_one_strong_signal_never_outscores_three_strong_signals() -> None:
    """Coverage is the whole reason the floor exists.

    Without it a server with a single saturated signal would tie a server
    that is popular on every axis, and the catalog would rank a one-registry
    curiosity above a genuinely widely used server.
    """
    lone = popularity(PopularitySignals(github_stars=100_000))
    broad = popularity(
        PopularitySignals(
            github_stars=100_000,
            npm_downloads_monthly=10_000_000,
            smithery_use_count=100_000,
        )
    )
    assert lone < broad


def test_the_coverage_floor_never_discounts_below_its_own_value() -> None:
    saturated = popularity(PopularitySignals(smithery_use_count=100_000))
    weight = WEIGHTS["smithery_use_count"]
    assert saturated == round(COVERAGE_FLOOR + (1 - COVERAGE_FLOOR) * weight, POPULARITY_DECIMALS)


def test_every_score_is_rounded_to_four_decimals() -> None:
    for stars in (1, 7, 999, 54_321):
        value = popularity(PopularitySignals(github_stars=stars))
        assert value == round(value, POPULARITY_DECIMALS)
        assert 0.0 <= value <= 1.0


def test_forks_dependents_and_version_counts_are_stored_but_not_scored() -> None:
    bare = PopularitySignals(github_stars=1000)
    padded = PopularitySignals(
        github_stars=1000, github_forks=9_000, npm_dependents=800, registry_version_count=40
    )
    assert popularity(bare) == popularity(padded)


def test_rank_score_puts_curated_ahead_of_registry_verified_at_equal_popularity() -> None:
    curated = _entry(key="mcp:registry:a/one", tier="curated", score=0.5)
    verified = _entry(key="mcp:registry:a/two", tier="registry_verified", score=0.5)
    assert sorted([verified, curated], key=rank_score) == [curated, verified]


def test_rank_score_puts_the_more_popular_entry_first_inside_a_tier() -> None:
    quiet = _entry(key="mcp:registry:a/quiet", tier="indexed", score=0.1)
    loud = _entry(key="mcp:registry:a/loud", tier="indexed", score=0.9)
    assert sorted([quiet, loud], key=rank_score) == [loud, quiet]


def test_rank_score_breaks_a_tie_on_the_canonical_key() -> None:
    first = _entry(key="mcp:registry:a/aaa", tier="indexed", score=0.4)
    second = _entry(key="mcp:registry:a/bbb", tier="indexed", score=0.4)
    assert sorted([second, first], key=rank_score) == [first, second]


def test_rank_score_is_a_plain_tuple_of_tier_negated_popularity_and_key() -> None:
    entry = _entry(key="mcp:registry:a/one", tier="smithery_verified", score=0.25)
    assert rank_score(entry) == (2, -0.25, "mcp:registry:a/one")
