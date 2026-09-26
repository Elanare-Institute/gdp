"""The orthogonal population must actually be orthogonal.

Phase 3 §3 had to retract an fx gradient because `fx_share` is confounded
with every other structural parameter in the archetype population: both come
from the same archetype, so a pooled regression on fx measures the group.
The orthogonal population exists to break that. These tests state what
"broken" means, so a future change to the sampler cannot quietly restore the
confounding.
"""

from __future__ import annotations

import numpy as np
import pytest

from legacy.v7.population import (
    ORTHOGONAL_FX, _ORTHOGONAL_RANGE, make_orthogonal_population, make_population,
)

SEEDS = range(1, 21)
#: Correlations this small cannot carry a gradient across 31 blocs.
MAX_ABS_CORR = 0.15


def _pooled(fn, field: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [b for s in SEEDS for b in fn(s)]
    return (np.array([b.fx_share for b in rows]),
            np.array([getattr(b, field) for b in rows]))


def test_population_is_reproducible() -> None:
    a = make_orthogonal_population(7)
    b = make_orthogonal_population(7)
    assert [x.fx_share for x in a] == [x.fx_share for x in b]
    assert [x.name for x in a] == [x.name for x in b]


def test_size_and_single_reserve_issuer() -> None:
    pop = make_orthogonal_population(3)
    assert len(pop) == 31
    assert sum(b.reserve for b in pop) == 1


@pytest.mark.parametrize("field", sorted(_ORTHOGONAL_RANGE))
def test_fx_is_uncorrelated_with_every_structural_parameter(field: str) -> None:
    fx, v = _pooled(make_orthogonal_population, field)
    r = float(np.corrcoef(fx, v)[0, 1])
    assert abs(r) < MAX_ABS_CORR, f"fx correlates {r:+.3f} with {field}"


def test_reserve_status_is_uncorrelated_with_fx() -> None:
    """In the archetypes the reserve issuer IS the fx=0 bloc; here it is not."""
    rows = [b for s in SEEDS for b in make_orthogonal_population(s)]
    fx = np.array([b.fx_share for b in rows])
    res = np.array([float(b.reserve) for b in rows])
    r = float(np.corrcoef(fx, res)[0, 1])
    assert abs(r) < MAX_ABS_CORR, f"reserve status correlates {r:+.3f} with fx"


def test_fx_spans_the_requested_range() -> None:
    fx = np.array([b.fx_share for s in SEEDS for b in make_orthogonal_population(s)])
    lo, hi = ORTHOGONAL_FX
    assert fx.min() < lo + 0.05 and fx.max() > hi - 0.05
    assert lo <= fx.min() and fx.max() <= hi


def test_low_fx_region_is_better_populated_than_the_archetypes() -> None:
    """The point of the sensitivity: group A holds a single bloc."""
    ortho = np.array([b.fx_share for b in make_orthogonal_population(1)])
    arch = np.array([b.fx_share for b in make_population(1)])
    assert (ortho < 0.1).sum() > (arch < 0.1).sum()


def test_archetype_population_is_confounded() -> None:
    """The baseline this sensitivity is run against. Not a defect — a fact."""
    fx, v = _pooled(make_population, "m_H")
    assert abs(float(np.corrcoef(fx, v)[0, 1])) > 0.5
