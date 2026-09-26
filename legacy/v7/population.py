"""A population of blocs drawn around the four v6 archetypes.

Phase 3 needs a continuum rather than four points: the question is whether
structure predicts which provisioning form a bloc converges to, and that
cannot be asked of four observations. Each group is sampled around its
archetype, with ``fx_share`` spread continuously within groups so it can be
used as a regressor rather than a label.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Sequence

from .params import Bloc, archetypes

#: Group sizes from TASKS.md: one reserve issuer, five advanced, ten emerging,
#: fifteen developing.
GROUP_SIZES: dict[str, int] = {"A": 1, "B": 5, "C": 10, "D": 15}

#: Multiplicative jitter applied to continuous structural parameters.
JITTER: float = 0.25

#: fx_share is spread wider than the rest: it is the regressor the phase's
#: main question is posed in, so within-group variation has to be real.
FX_SPREAD: dict[str, tuple[float, float]] = {
    "A": (0.0, 0.05),
    "B": (0.05, 0.35),
    "C": (0.25, 0.65),
    "D": (0.45, 0.85),
}

_JITTERED = ("c_income_share", "c_pop_share", "m_F", "m_G", "m_H",
             "eps_supply", "land_share", "omega0", "openness")


def make_population(seed: int, sizes: dict[str, int] | None = None) -> list[Bloc]:
    """Draw a population of blocs around the archetypes.

    Args:
        seed: Random seed; the same seed always yields the same population.
        sizes: Override the group sizes (used by tests).

    Returns:
        Blocs named ``<group><index>``, e.g. ``C3``.
    """
    rng = random.Random(seed)
    base = archetypes()
    sizes = sizes or GROUP_SIZES
    out: list[Bloc] = []

    for group, count in sizes.items():
        archetype = base[group]
        lo, hi = FX_SPREAD[group]
        for i in range(count):
            kwargs = {}
            for field in _JITTERED:
                value = getattr(archetype, field)
                kwargs[field] = max(1e-6, value * (1.0 + JITTER * (2 * rng.random() - 1)))
            # Shares must stay in their natural range after jittering.
            for field in ("c_income_share", "c_pop_share", "m_F", "m_G", "m_H",
                          "land_share", "omega0", "openness"):
                kwargs[field] = min(0.95, kwargs[field])

            kwargs["fx_share"] = lo + (hi - lo) * rng.random()
            kwargs["gdp"] = archetype.gdp * (1.0 + JITTER * (2 * rng.random() - 1))
            kwargs["debt"] = max(0.05, archetype.debt * (1.0 + JITTER * (2 * rng.random() - 1)))
            kwargs["rate"] = max(0.001, archetype.rate * (1.0 + JITTER * (2 * rng.random() - 1)))
            kwargs["name"] = f"{group}{i + 1}"
            out.append(replace(archetype, **kwargs))

    return out


def structural_distance(a: Bloc, b: Bloc) -> float:
    """Distance used when a bloc looks for someone to imitate.

    Defined on ``fx_share`` and reserve status per TASKS.md: a bloc compares
    itself with others facing a similar external-finance position, not with
    whoever happens to be doing best overall.
    """
    reserve_gap = 0.0 if a.reserve == b.reserve else 1.0
    return abs(a.fx_share - b.fx_share) + reserve_gap


def group_of(bloc: Bloc) -> str:
    """Archetype group a bloc was drawn from, read off its name."""
    return bloc.name[0]


#: Structural parameters sampled independently of ``fx_share`` in the
#: orthogonal population, with the range spanned by the four archetypes.
_ORTHOGONAL_RANGE: dict[str, tuple[float, float]] = {
    "gdp": (0.2, 1.2), "debt": (0.35, 0.85), "rate": (0.02, 0.07),
    "base_growth": (0.018, 0.038), "decay": (0.0003, 0.0008),
    "openness": (0.5, 1.0), "capital": (0.2, 1.3),
    "c_income_share": (0.15, 0.25), "c_pop_share": (0.3, 0.5),
    "m_F": (0.1, 0.3), "m_G": (0.15, 0.35), "m_H": (0.05, 0.25),
    "eps_supply": (0.3, 0.5), "land_share": (0.3, 0.45),
    "omega0": (0.05, 0.35),
}

#: fx_share range for the orthogonal population, per the review instruction.
ORTHOGONAL_FX: tuple[float, float] = (0.0, 0.85)


def make_orthogonal_population(
    seed: int, n: int = 31, fx_range: tuple[float, float] = ORTHOGONAL_FX,
) -> list[Bloc]:
    """Draw a population in which ``fx_share`` is independent of everything else.

    The archetype population confounds ``fx_share`` with group structure: a
    bloc with high fx also has high ``m_H``, low ``land_share`` and so on,
    because both come from the same archetype. A pooled regression on fx then
    measures the group, which is what Phase 3 §3 had to retract.

    Here group labels are dropped. Every structural parameter is drawn
    uniformly over the range the four archetypes span, independently of
    ``fx_share``, which is itself drawn uniformly over `fx_range`. Any fx
    gradient estimated on this population is not confounded with structure,
    and no group fixed effect is needed — or available.

    ``reserve`` cannot be drawn independently and still mean anything: it is
    assigned to exactly one bloc, chosen at random and so uncorrelated with
    that bloc's fx. The result is a population in which reserve status and fx
    are orthogonal too, unlike the archetypes where the reserve issuer is by
    construction the fx=0 bloc.

    Args:
        seed: Random seed; the same seed always yields the same population.
        n: Number of blocs (31 to match the archetype population).
        fx_range: Bounds of the uniform ``fx_share`` draw.

    Returns:
        Blocs named ``X<index>``; :func:`group_of` returns ``X`` for all of
        them, so any code that groups by label sees a single group.
    """
    rng = random.Random(seed)
    template = archetypes()["C"]  # only supplies the non-sampled fields
    lo_fx, hi_fx = fx_range
    reserve_index = rng.randrange(n)
    out: list[Bloc] = []
    for i in range(n):
        kwargs = {field: lo + (hi - lo) * rng.random()
                  for field, (lo, hi) in _ORTHOGONAL_RANGE.items()}
        kwargs["fx_share"] = lo_fx + (hi_fx - lo_fx) * rng.random()
        kwargs["reserve"] = 1 if i == reserve_index else 0
        kwargs["name"] = f"X{i + 1}"
        out.append(replace(template, **kwargs))
    return out
