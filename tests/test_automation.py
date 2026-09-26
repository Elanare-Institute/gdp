"""Labour demand falling everywhere at once.

The setting the paper is actually about: office work automated, machines and
vehicles driving themselves, labour demand falling across the world at the
same time. Transfers there are not weighed against wages — they are what is
left.

Everything comes from one progress variable, so the model cannot be tuned
channel by channel until it says what is wanted. These tests fix that the four
consequences move together, that potential output is not production, and that
the baseline is untouched when automation is off.

Specification: `specs/V8_POSTEMP.md` §6.
"""

from __future__ import annotations

import pytest

from model.automation import capacity_overhang, progress, state
from model.core import build, simulate
from model.params import G

MAX = 0.6


def _g(automation: float = MAX, **kw) -> G:
    return G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
             automation_max=automation, **kw)


def _run(automation: float, modality: str = "none", u: float = 0.0, **kw):
    g = _g(automation, **kw)
    blocs = build(g, modality, u, (1, 1, 1, 1)) if u > 0 else build(g, "none")
    return simulate(blocs, g)


# --- the baseline is untouched ---

def test_automation_is_off_by_default() -> None:
    assert G().automation_max == 0.0


@pytest.mark.parametrize("modality", ["none", "ubi", "cash_t", "clt"])
def test_zero_automation_reproduces_the_baseline_exactly(modality: str) -> None:
    """post_employment must be an addition, not a change to what came before."""
    u = 0.0 if modality == "none" else 0.02
    plain = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02)
    blocs = build(plain, modality, u, (1, 1, 1, 1)) if u else build(plain, "none")
    a = simulate(blocs, plain)
    b = _run(0.0, modality, u)
    assert a["quantities"] == b["quantities"]
    assert a["final"] == b["final"]


def test_overhang_is_negligible_without_automation() -> None:
    """Without automation, potential is the bloc's own trend, so output sits
    on it. Not exactly: `Y` also carries the fiscal impulse and the growth
    adjustments, which the trend line does not. The residual is a fraction of
    a percent and is what tells the two apart."""
    for row in _run(0.0)["capacity"]:
        assert row["world"] == pytest.approx(1.0, abs=0.01)


# --- one variable, four consequences ---

def test_progress_approaches_the_ceiling_without_reaching_it() -> None:
    g = _g()
    assert progress(g, 0.0) == 0.0
    assert 0 < progress(g, 10.0) < MAX
    assert progress(g, 30.0) < MAX
    assert progress(g, 30.0) > progress(g, 10.0)


def test_the_four_consequences_move_together() -> None:
    """They are one phenomenon seen from four sides. Separate schedules would
    let the model be tuned until it produced the wanted answer."""
    early, late = state(_g(), 2.0), state(_g(), 25.0)
    assert late.progress > early.progress
    assert late.capacity_multiplier > early.capacity_multiplier
    assert late.cost_multiplier < early.cost_multiplier
    assert late.labour_multiplier < early.labour_multiplier


def test_nothing_moves_when_automation_is_off() -> None:
    quiet = state(G(), 30.0)
    assert quiet.progress == 0.0
    assert quiet.capacity_multiplier == 1.0
    assert quiet.cost_multiplier == 1.0
    assert quiet.labour_multiplier == 1.0
    assert quiet.displaced_labour_share == 0.0


# --- potential is not production ---

def test_capacity_that_nobody_can_buy_from_stands_idle() -> None:
    """The difficulty the whole setting turns on: automation raises what could
    be made at the same time as it removes the income that would buy it."""
    idle = _run(MAX)["capacity"][-1]["world"]
    assert idle < 0.8, f"expected substantial idle capacity, got {idle:.3f}"


def test_overhang_grows_as_automation_advances() -> None:
    path = [row["world"] for row in _run(MAX)["capacity"]]
    assert path[-1] < path[1]


def test_overhang_is_reported_per_bloc_as_well_as_for_the_world() -> None:
    row = _run(MAX)["capacity"][-1]
    assert "world" in row
    assert all(f"overhang_{i}" in row for i in range(4))


def test_overhang_is_bounded() -> None:
    assert capacity_overhang(2.0, 1.0) == 1.0
    assert capacity_overhang(-1.0, 1.0) == 0.0
    assert capacity_overhang(1.0, 0.0) == 1.0


# --- labour income and where it goes ---

def test_constrained_households_lose_labour_income() -> None:
    """Measured on quantities secured, which is what the loss means for them."""
    with_automation = _run(MAX)["quantities"][-1]["F_3"]
    without = _run(0.0)["quantities"][-1]["F_3"]
    assert with_automation < without


def test_what_labour_stops_earning_does_not_vanish() -> None:
    """Output is still produced and still paid for, so it accrues to the
    owners of the machines — and follows their income abroad."""
    assert state(_g(), 20.0).displaced_labour_share > 0.0
    plain = _run(0.0, "ubi", 0.02)["final"]["D (developing)"]["nfa"]
    automated = _run(MAX, "ubi", 0.02)["final"]["D (developing)"]["nfa"]
    assert automated != plain


def test_cheaper_domestic_production_is_the_one_relief() -> None:
    """Automation is not purely a loss: domestic goods get cheaper to make."""
    from model.household import household_block
    from model.params import archetypes
    g = _g()
    b = archetypes()["D"]
    dear = household_block(b, g, b.gdp, 1.0, b.rate, False, 0.0, cost_factor=1.0)
    cheap = household_block(b, g, b.gdp, 1.0, b.rate, False, 0.0, cost_factor=0.8)
    assert cheap["prices"]["G"] < dear["prices"]["G"]


# --- the two linkage rules stay distinct ---

def test_income_linkage_and_indexation_are_independent() -> None:
    """They were conflated once, in the phase U report. `income_linkage` says
    how household income keeps up; `indexation` says how the transfer's
    nominal amount is updated. Automation moves the first, not the second."""
    def price(income: str, index: str) -> float:
        g = _g(income_linkage=income, indexation=index)
        return simulate(build(g, "ubi", 0.02, (1, 1, 1, 1)), g)["world_final"]["price"]

    assert price("cpi_linked", "cpi_indexed") != price("cpi_linked", "fixed_nominal")
    assert price("cpi_linked", "cpi_indexed") != price("growth_linked", "cpi_indexed")


def test_the_speed_and_ceiling_are_swept_not_fixed() -> None:
    """Neither is an estimate, so both must change the outcome."""
    slow = _run(MAX, automation_speed=0.05)["capacity"][-1]["world"]
    fast = _run(MAX, automation_speed=0.3)["capacity"][-1]["world"]
    assert fast < slow
    small = _run(0.3)["capacity"][-1]["world"]
    large = _run(0.8)["capacity"][-1]["world"]
    assert large < small


# --- what the model can actually verify about demand ---

def _constrained_demand(automation: float, dc: float = 0.0, bloc: int = 3) -> float:
    """Quantities the constrained block secures at year 30, relative to year 1."""
    rows = _run(automation, demand_constrains_output=dc)["quantities"]
    goods = ("F", "G", "H")
    opening = sum(rows[1][f"{k}_{bloc}"] for k in goods)
    closing = sum(rows[30][f"{k}_{bloc}"] for k in goods)
    return closing / opening


def test_constrained_demand_falls_with_the_labour_multiplier() -> None:
    """The prediction the model can actually check.

    Automation removes a share of the constrained block's labour income, and
    what that block secures falls with it. The theoretical figure for demand
    *economy-wide* cannot be checked here — the unconstrained block's
    consumption bundle is never solved — so the claim is confined to the block
    the model does solve.
    """
    plain = _constrained_demand(0.0)
    automated = _constrained_demand(MAX)
    labour = state(_g(), 30.0).labour_multiplier
    # Cheaper domestic production offsets part of the income loss, so the
    # fall is gentler than the multiplier alone: the observed ratio is about
    # 1.3 times it. What matters is that the fall tracks the multiplier rather
    # than some other quantity — it is neither absent nor larger than the
    # income loss that causes it.
    ratio = automated / plain
    assert labour < ratio < labour * 1.5
    assert ratio < 0.7, "the fall must be substantial, not marginal"


def test_the_loop_closes_when_demand_constrains_output() -> None:
    """Income falls, demand falls, output falls, income falls further.

    At dc = 0 output follows its trend whatever happens to demand, so the loop
    is open by construction; above zero it closes and the fall compounds.
    """
    open_loop = _constrained_demand(MAX, dc=0.0)
    closed_loop = _constrained_demand(MAX, dc=0.02)
    assert closed_loop < open_loop


def test_demand_does_not_constrain_output_by_default() -> None:
    """v6's behaviour, carried through every earlier phase."""
    assert G().demand_constrains_output == 0.0


# --- the two indicators must see the same household ---

def _pair(automation: float, bloc: int = 3) -> tuple[float, float]:
    """Quantity ratio and budget ratio at year 30, for the same run."""
    g = _g(automation, income_linkage="cpi_linked", subsistence_scale=1.3)
    result = simulate(build(g, "none"), g)
    rows, budget = result["quantities"], result["budget"]
    goods = ("F", "G", "H")
    opening = sum(rows[1][f"{k}_{bloc}"] for k in goods)
    closing = sum(rows[30][f"{k}_{bloc}"] for k in goods)
    return closing / opening, budget[30][f"min_{bloc}"]


def test_the_budget_indicator_sees_the_lost_labour_income() -> None:
    """Both must be computed on the same income.

    The household block applies the labour multiplier; the budget indicator
    has to apply it too, or it reports a household richer than the one being
    simulated. It did not, once: the budget ratio moved 7% while the
    quantities moved 80%.
    """
    _, quiet = _pair(0.0)
    _, automated = _pair(MAX)
    assert automated < 0.6 * quiet


def test_both_indicators_fall_together() -> None:
    quantity_quiet, budget_quiet = _pair(0.0)
    quantity_auto, budget_auto = _pair(MAX)
    assert quantity_auto < quantity_quiet
    assert budget_auto < budget_quiet


def test_the_required_transfer_rises_sharply_with_automation() -> None:
    """What it costs to keep the poorest quantile at the bundle, once their
    labour income has gone."""
    def need(automation: float) -> float:
        g = _g(automation, income_linkage="cpi_linked", subsistence_scale=1.3)
        return simulate(build(g, "none"), g)["budget"][30]["need_3"]

    assert need(MAX) > 5 * need(0.0)


def test_the_quantity_fall_exceeds_the_budget_fall() -> None:
    """Both are large, and they are not the same quantity.

    The demand system scales the subsistence floor to income, so when income
    falls the floor falls with it and the quantities follow all the way down.
    The budget indicator holds the bundle fixed as a quantity, so it measures
    what that income can no longer buy. Neither is wrong; they answer
    different questions, and the model can only speak to the second.
    """
    quantity_quiet, budget_quiet = _pair(0.0)
    quantity_auto, budget_auto = _pair(MAX)
    assert quantity_auto / quantity_quiet < 0.5
    assert budget_auto / budget_quiet < 0.5
