"""The world price level and world tradable capacity.

The paper's second claim is that cash escapes a country but not the world: if
everyone hands out cash, world demand for tradables rises against a capacity
that is fixed in the short run, and the world price level rises. Leakage then
does not dispose of the problem, it relocates it.

v7 had no such object. Each bloc's inflation was its own, driven by its own
domestic demand, and one bloc's imports were nobody's exports. This module
supplies the missing piece: a world price `P_w` that every bloc faces through
its exchange rate.

See `specs/V8_A_B.md` §3.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Sequence

from .params import DT, Bloc, G
from .trade import export_capacity


#: Largest world price the model will report. A configuration that reaches it
#: has a runaway loop; the number itself carries no meaning beyond that.
PRICE_CEILING: float = 1e12

#: Largest log price change allowed in one quarter, so the exponential cannot
#: overflow before the ceiling is reached.
MAX_LOG_STEP: float = 20.0


@dataclass
class WorldState:
    """The world's price level and productive capacity.

    ``price`` is the world-currency price of the tradable good, normalised to
    1 at the start. ``capacity`` is world supply of tradables in real terms,
    normalised so that the no-transfer baseline starts balanced.

    ``anchor`` is the price the reversion term pulls towards. It is the
    starting price, not a moving target: the reversion exists to stop a
    transitory gap from shifting the level permanently at full force, and an
    anchor that drifted with the price would not do that. A *sustained* gap
    still moves the level without limit, which is the behaviour the phase is
    meant to exhibit.
    """

    price: float = 1.0
    capacity: float = 1.0
    anchor: float = 1.0

    def copy(self) -> "WorldState":
        return WorldState(self.price, self.capacity, self.anchor)


def world_growth(blocs: Sequence[Bloc], g: G, t: float,
                 demand_weights: Sequence[float] | None = None) -> float:
    """Growth of world tradable capacity, per year, from assumed bloc growth.

    Used only to seed the baseline run that produces the capacity path every
    other configuration then shares (see `baseline_capacity_path`). Weighted by
    where demand comes from rather than by who supplies it: supply capacity
    sits in the large, slow-growing blocs while import demand sits in the
    small, fast-growing ones, so a supply-weighted trend would drift away from
    demand with no programme running anywhere.

    Args:
        demand_weights: Each bloc's share of world import demand. Falls back to
            export capacity before any bloc has computed its imports.
    """
    if g.world_supply_growth is not None:
        return g.world_supply_growth
    w = list(demand_weights) if demand_weights is not None else export_capacity(blocs)
    total = sum(w)
    if total <= 0:
        return 0.0
    return sum(wi * max(0.005, b.base_growth - b.decay * t)
               for wi, b in zip(w, blocs)) / total


def initial_capacity(imports: Sequence[float]) -> float:
    """World capacity that exactly meets a given world demand.

    Calling this with the no-transfer demand of the first period normalises the
    output gap to zero at the start, so any later gap is caused by what the
    simulation does rather than by an arbitrary initial level.
    """
    total = sum(imports)
    return total if total > 0 else 1.0


def output_gap(demand: float, capacity: float) -> float:
    """Excess of world demand over world capacity, as a fraction of capacity.

    Positive means the world is trying to buy more tradables than it can make.
    """
    if capacity <= 0:
        return 0.0
    return demand / capacity - 1.0


def step_world(state: WorldState, demand: float, blocs: Sequence[Bloc], g: G,
               t: float, dt: float = DT,
               demand_weights: Sequence[float] | None = None,
               capacity: float | None = None) -> WorldState:
    """Advance the world state one quarter.

    World inflation has two terms: the output gap pushes the price level up,
    and a slow reversion pulls it back towards the anchor. Both are taken in
    logs and the price is advanced by an exponential, so the update cannot
    drive the level through zero however wide the gap gets.

    Args:
        capacity: Next period's world capacity. Supplied from the shared
            baseline path (`baseline_capacity_path`) so that capacity does not
            respond to the demand a programme creates; without it, capacity
            grows at the assumed rate, which is how the baseline path itself is
            produced.
    """
    gap = output_gap(demand, state.capacity)
    # The pull towards the anchor is taken in logs. Written on the level —
    # lambda*(P/anchor - 1) — the restoring term grows without bound in the
    # price: at P = 500 it asks for an annual deflation of 2500%, which drives
    # the level negative. In logs the parameter means "close this fraction of
    # the log gap per year" at every level.
    infl = g.kappa_w * gap - g.lambda_w * math.log(state.price / state.anchor)
    if capacity is None:
        growth = world_growth(blocs, g, t, demand_weights)
        capacity = state.capacity * math.exp(growth * dt)
    # The indexation loop of phase C can drive inflation high enough to
    # overflow the exponential. That is a real runaway, not a numerical one,
    # but a run that raises OverflowError cannot be reported at all — so the
    # price is capped and the run goes on. A configuration sitting at the cap
    # is one whose loop ran away; `PRICE_CEILING` is a reporting limit, not a
    # claim about where the world stops.
    grown = state.price * math.exp(min(infl * dt, MAX_LOG_STEP))
    return WorldState(
        price=min(PRICE_CEILING, max(1e-6, grown)),
        capacity=max(1e-6, capacity),
        anchor=state.anchor,
    )


def import_price(world: WorldState, e: float) -> float:
    """Price of the tradable good in one bloc's own currency.

    The world price converted at that bloc's exchange rate. A bloc that
    depreciates pays more for the same world good — which is how a world-wide
    demand expansion reaches a bloc that did not expand its own.
    """
    return world.price * e
