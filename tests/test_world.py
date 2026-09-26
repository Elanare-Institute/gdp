"""The world price level responds to world demand, and only to world demand.

The paper's second claim needs a world price that rises when world demand for
tradables outruns world capacity. These tests fix the properties that claim
depends on, including the one that is easy to get wrong: a world in balance
must have a *constant* price, so that any movement can be attributed to demand
rather than to the arithmetic of the update rule.

Specification: `specs/V8_A_B.md` §3 and §4.1.
"""

from __future__ import annotations

import pytest

from model.params import DT, G, archetypes
from model.world import (
    WorldState, import_price, initial_capacity, output_gap, step_world,
    world_growth,
)

BLOCS = list(archetypes().values())
BASE_IMPORTS = [0.10, 0.08, 0.06, 0.03]


def _run(demand_path, g: G, steps: int = 120) -> WorldState:
    """Advance the world `steps` quarters along a given demand path."""
    state = WorldState(capacity=initial_capacity(BASE_IMPORTS))
    for step in range(steps):
        state = step_world(state, demand_path(step, state), BLOCS, g, step * DT)
    return state


def _growing(multiple: float):
    """Demand that tracks world capacity, scaled by `multiple`."""
    return lambda step, state: multiple * state.capacity


# --- the balanced world ---

def test_initial_capacity_zeroes_the_gap() -> None:
    cap = initial_capacity(BASE_IMPORTS)
    assert abs(output_gap(sum(BASE_IMPORTS), cap)) < 1e-12


def test_a_balanced_world_has_a_constant_price() -> None:
    """Zero gap at the anchor must mean zero inflation, exactly."""
    final = _run(_growing(1.0), G())
    assert abs(final.price - 1.0) < 1e-9


def test_capacity_grows_at_the_weighted_bloc_growth_rate() -> None:
    g = G()
    rate = world_growth(BLOCS, g, 0.0)
    lo = min(b.base_growth for b in BLOCS)
    hi = max(b.base_growth for b in BLOCS)
    assert lo <= rate <= hi


def test_capacity_growth_can_be_overridden() -> None:
    assert world_growth(BLOCS, G(world_supply_growth=0.0), 0.0) == 0.0


# --- demand moves the price ---

def test_excess_demand_raises_the_world_price() -> None:
    assert _run(_growing(1.10), G()).price > 1.0


def test_deficient_demand_lowers_the_world_price() -> None:
    assert _run(_growing(0.90), G()).price < 1.0


@pytest.mark.parametrize("multiple", [1.02, 1.05, 1.10, 1.20])
def test_the_price_rises_monotonically_in_the_size_of_the_excess(multiple: float) -> None:
    """A bigger world-wide demand expansion must raise the price by more."""
    smaller = _run(_growing(multiple - 0.01), G()).price
    larger = _run(_growing(multiple), G()).price
    assert larger > smaller


def test_a_transitory_gap_decays_but_does_not_fully_unwind() -> None:
    """Reversion damps a one-off shock without erasing the level change."""
    g = G()
    state = WorldState(capacity=initial_capacity(BASE_IMPORTS))
    for step in range(4):                      # one year of excess demand
        state = step_world(state, 1.10 * state.capacity, BLOCS, g, step * DT)
    peak = state.price
    for step in range(4, 120):                 # then back to balance
        state = step_world(state, state.capacity, BLOCS, g, step * DT)
    assert 1.0 < state.price < peak


def test_a_sustained_gap_keeps_moving_the_level() -> None:
    """Reversion must not turn a permanent gap into a bounded price path."""
    g = G()
    short = _run(_growing(1.10), g, steps=40).price
    long = _run(_growing(1.10), g, steps=120).price
    assert long > short


# --- pass-through to a bloc ---

def test_import_price_is_the_world_price_at_the_blocs_exchange_rate() -> None:
    w = WorldState(price=1.3)
    assert import_price(w, 1.0) == pytest.approx(1.3)
    assert import_price(w, 2.0) == pytest.approx(2.6)   # depreciation costs more
    assert import_price(w, 0.5) == pytest.approx(0.65)  # appreciation costs less


def test_a_bloc_that_did_not_expand_still_pays_the_higher_world_price() -> None:
    """This is how the world channel reaches a bloc with no programme."""
    quiet_rate = 1.0
    calm = import_price(WorldState(price=1.0), quiet_rate)
    inflated = import_price(_run(_growing(1.10), G()), quiet_rate)
    assert inflated > calm


# --- determinism ---

def test_the_world_is_reproducible() -> None:
    a = _run(_growing(1.05), G())
    b = _run(_growing(1.05), G())
    assert (a.price, a.capacity) == (b.price, b.capacity)
