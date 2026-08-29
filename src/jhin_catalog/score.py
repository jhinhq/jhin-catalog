"""Popularity, normalised against fixed anchors rather than the corpus.

Three counts are scored — GitHub stars, monthly npm downloads, and the
Smithery use count — each divided by ``log1p`` of a constant ceiling and
combined under fixed weights. The anchors are constants on purpose: a
percentile would move every entry's score whenever the corpus moved, and the
diff gate would then fire on noise rather than on change. A signal nobody
reported is not read as a zero; it shrinks a coverage multiplier instead, so
a well-observed server is not overtaken by a barely-observed one. Forks,
dependents, and version counts are stored upstream but never scored.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

from jhin_catalog.types import TRUST_RANK, CatalogEntry, PopularitySignals

CEILINGS: Final[Mapping[str, int]] = {
    "github_stars": 100_000,
    "npm_downloads_monthly": 10_000_000,
    "smithery_use_count": 100_000,
}
WEIGHTS: Final[Mapping[str, float]] = {
    "github_stars": 0.45,
    "npm_downloads_monthly": 0.35,
    "smithery_use_count": 0.20,
}
COVERAGE_FLOOR: Final[float] = 0.85
POPULARITY_DECIMALS: Final[int] = 4

# The scored signals in a fixed order. Floating-point addition is not
# associative, so the sum is taken over a tuple rather than over a mapping:
# the score must not depend on how a dict happens to be laid out.
_SCORED: Final[tuple[str, ...]] = (
    "github_stars",
    "npm_downloads_monthly",
    "smithery_use_count",
)
_TOTAL_WEIGHT: Final[float] = sum(WEIGHTS[name] for name in _SCORED)


def normalize_signal(value: int | None, ceiling: int) -> float | None:
    """One raw count between ``0.0`` and ``1.0``, or ``None`` when unseen.

    The curve is ``log1p(value) / log1p(ceiling)``, clamped at both ends, so
    the first thousand stars move a server much further than the ninety
    thousandth does. A negative count is read as zero and a count above the
    ceiling saturates rather than overflowing the scale.
    """
    if value is None:
        return None
    if ceiling <= 0:
        return 0.0
    scaled = math.log1p(max(value, 0)) / math.log1p(ceiling)
    return min(1.0, max(0.0, scaled))


def popularity(signals: PopularitySignals) -> float:
    """The blended score for one entry, rounded to ``POPULARITY_DECIMALS``.

    Present signals are averaged under their own weights, so an entry seen
    only on npm is compared against the npm scale rather than against a
    corpus it never appeared in. The average is then multiplied by
    ``COVERAGE_FLOOR + (1 - COVERAGE_FLOOR) * coverage``, where ``coverage``
    is the share of total weight that was actually observed: a server three
    sources agree on outranks one that only npm has heard of, without the
    silence being scored as unpopularity. An entry with no signals at all
    scores ``0.0``.
    """
    observed = [
        (name, scaled)
        for name, scaled in (
            (name, normalize_signal(_signal(signals, name), CEILINGS[name])) for name in _SCORED
        )
        if scaled is not None
    ]
    if not observed:
        return 0.0
    present_weight = sum(WEIGHTS[name] for name, _ in observed)
    base = sum(WEIGHTS[name] * scaled for name, scaled in observed) / present_weight
    coverage = present_weight / _TOTAL_WEIGHT
    return round(base * (COVERAGE_FLOOR + (1.0 - COVERAGE_FLOOR) * coverage), POPULARITY_DECIMALS)


def rank_score(entry: CatalogEntry) -> tuple[int, float, str]:
    """The publication order key: trust first, then popularity, then key.

    Sorting ascending gives the strongest trust tier first, the most popular
    entry first inside a tier, and the canonical key as a total tie-break so
    two builds of the same corpus publish the same list. The value is derived
    on demand and never stored, because storing it would make every entry's
    bytes depend on how the rest of the corpus scored.
    """
    return (TRUST_RANK[entry.trust_tier], -entry.popularity, entry.canonical_key)


def _signal(signals: PopularitySignals, name: str) -> int | None:
    """One scored count by name, typed rather than fetched with ``getattr``."""
    match name:
        case "github_stars":
            return signals.github_stars
        case "npm_downloads_monthly":
            return signals.npm_downloads_monthly
        case "smithery_use_count":
            return signals.smithery_use_count
        case _:
            return None
