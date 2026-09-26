"""Blocs generated on two independent axes.

The four archetypes bundle everything together: bloc D has a high interest
rate, a high foreign-currency debt share, a high food import share and a high
construction import share, all at once. Nothing in the model can then say
which of those does the work. Phase 3 ran into exactly this — an fx gradient
that turned out to be group structure — and phase D would run into it again,
since "cannot borrow" and "must import food" arrive together in the archetypes
and would be credited to whichever one the analysis happened to name.

So blocs are drawn on two axes that are sampled independently:

  credit            what a bloc can borrow abroad, what lenders charge it, and
                    what reserves it starts with
  self_sufficiency  how much of its food, goods and construction it has to buy
                    from abroad

Reserve-currency status is a third, binary attribute, independent of both.

Both axes run 0 (worst) to 1 (best), so a bloc at (0.2, 0.8) can borrow little
but feeds itself, and one at (0.8, 0.2) borrows freely but imports its food.
Those two are indistinguishable in the archetype population and are expected
to fail in different ways.

See `specs/V8_TWOAXIS.md`.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Sequence

from .params import Bloc, archetypes

#: Endpoints of each structural parameter, from worst to best on its axis.
#: The ranges are those the four archetypes already span, so the plane
#: contains them rather than extrapolating past them.
CREDIT_RANGE: dict[str, tuple[float, float]] = {
    # worst credit -> best credit
    "rate": (0.070, 0.020),
    "omega0": (0.35, 0.05),
    # What lenders will advance, and what the bloc starts with in the vault.
    # These are what "credit" means for a settlement constraint; the opening
    # debt level is deliberately *not* on this axis. Putting it here — a
    # low-credit bloc starting with low debt — makes its risk premium small
    # and its borrowing window wide, which inverts the axis.
    "credit_standing": (0.15, 1.6),
    "reserve_standing": (0.25, 1.75),
}

SELF_SUFFICIENCY_RANGE: dict[str, tuple[float, float]] = {
    # least self-sufficient -> most self-sufficient
    "m_F": (0.30, 0.10),
    "m_G": (0.35, 0.15),
    "m_H": (0.25, 0.05),
}

#: Parameters that belong to neither axis. Held at the middle of the archetype
#: range so they cannot smuggle a third dimension in.
NEUTRAL: dict[str, float] = {
    "gdp": 0.7, "base_growth": 0.028, "decay": 0.00055, "openness": 0.8,
    "capital": 0.74, "c_income_share": 0.20, "c_pop_share": 0.40,
    "eps_supply": 0.4, "land_share": 0.375,
    # Held off both axes so that the credit axis is about what a bloc can
    # raise, not about how indebted it happens to start.
    "debt": 0.6,
}


def _interp(lo: float, hi: float, position: float) -> float:
    return lo + (hi - lo) * position


def make_bloc(name: str, credit: float, self_sufficiency: float,
              reserve: bool = False, fx_share: float | None = None) -> Bloc:
    """One bloc at a point on the plane.

    Args:
        credit: 0 = cannot borrow and pays dearly for what it does; 1 = borrows
            freely and cheaply.
        self_sufficiency: 0 = imports its food, goods and building materials;
            1 = largely feeds and builds for itself.
        reserve: Issues the reserve currency. Independent of both axes.
        fx_share: Share of debt denominated in foreign currency. Defaults to
            following the credit axis inversely — a bloc lenders distrust
            borrows in their money, not its own — but can be set to break even
            that link.
    """
    kwargs: dict[str, float | bool | str] = dict(NEUTRAL)
    for field, (lo, hi) in CREDIT_RANGE.items():
        kwargs[field] = _interp(lo, hi, credit)
    for field, (lo, hi) in SELF_SUFFICIENCY_RANGE.items():
        kwargs[field] = _interp(lo, hi, self_sufficiency)
    kwargs["fx_share"] = (1.0 - credit) * 0.85 if fx_share is None else fx_share
    kwargs["reserve"] = reserve
    kwargs["name"] = name
    return Bloc(**kwargs)  # type: ignore[arg-type]


def grid(steps: int = 5, reserve_at: tuple[float, float] | None = (1.0, 1.0),
         ) -> list[Bloc]:
    """A regular lattice over the plane.

    Used for the phase diagram: every combination of credit and
    self-sufficiency, so the two can be read apart.

    Args:
        reserve_at: Where to place the reserve issuer, or None for a world
            without one. Defaults to the corner that both axes favour, which
            is where the actual reserve issuer sits.
    """
    out: list[Bloc] = []
    for i in range(steps):
        for j in range(steps):
            credit = i / (steps - 1) if steps > 1 else 0.5
            selfsuf = j / (steps - 1) if steps > 1 else 0.5
            is_reserve = (reserve_at is not None
                          and abs(credit - reserve_at[0]) < 1e-9
                          and abs(selfsuf - reserve_at[1]) < 1e-9)
            out.append(make_bloc(f"c{credit:.2f}_s{selfsuf:.2f}",
                                 credit, selfsuf, reserve=is_reserve))
    return out


def sample(seed: int, n: int = 20, reserve_count: int = 1) -> list[Bloc]:
    """Draw `n` blocs with the two axes independent of each other.

    Independence is the point: in the archetypes, credit and self-sufficiency
    move together, so no estimate can separate them.
    """
    rng = random.Random(seed)
    reserve_index = rng.randrange(n) if reserve_count else -1
    return [make_bloc(f"X{i + 1}", rng.random(), rng.random(),
                      reserve=(i == reserve_index))
            for i in range(n)]


def locate_archetypes() -> dict[str, tuple[float, float]]:
    """Where the four archetypes sit on the plane.

    Reported so the two-axis results can be read against the earlier phases,
    and so it is visible how tightly the archetypes are bunched along the
    diagonal — which is what made the two effects impossible to separate.
    """
    out: dict[str, tuple[float, float]] = {}
    for key, b in archetypes().items():
        lo, hi = CREDIT_RANGE["rate"]
        credit = (b.rate - lo) / (hi - lo)
        lo_f, hi_f = SELF_SUFFICIENCY_RANGE["m_F"]
        selfsuf = (b.m_F - lo_f) / (hi_f - lo_f)
        out[key] = (round(credit, 4), round(selfsuf, 4))
    return out
