"""World-level pressure indicators.

These report levels and changes. They deliberately do **not** judge whether
the world survived: that would depend on routes the model does not contain
(trade blocs forming, migration, conflict, policy reversal), so a verdict
computed here would be decided by where a threshold was placed rather than by
anything the model knows.

The tests therefore fix two kinds of property: that the quantities respond in
the right direction to the thing that drives them, and that a run which goes
quiet after its blocs default cannot be read as a calm one.

Specification: `specs/V8_COLLAPSE.md`.
"""

from __future__ import annotations

import pytest

from model.collapse import (
    WorldPressure, annual_inflation, first_default_year, measure, output_gaps,
    solvent_demand_share,
)
from model.core import build, simulate
from model.params import G


def _g() -> G:
    return G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02)


def _run(u: float):
    g = _g()
    blocs = build(g, "ubi", u, (1, 1, 1, 1)) if u > 0 else build(g, "none")
    return simulate(blocs, g)


# --- no verdict is computed ---

def test_the_measurement_carries_no_survival_verdict() -> None:
    """Whether the world holds is not something this model can decide."""
    fields = set(WorldPressure.__dataclass_fields__)
    for forbidden in ("survives", "verdict", "passed", "collapsed"):
        assert forbidden not in fields
    assert "survives" not in measure(_run(0.01)).as_dict()


# --- the quantities respond to what drives them ---

@pytest.mark.parametrize("u", [0.005, 0.01, 0.02])
def test_a_bigger_transfer_opens_a_bigger_gap(u: float) -> None:
    assert measure(_run(u)).gap_max > measure(_run(u / 2)).gap_max


@pytest.mark.parametrize("u", [0.005, 0.01, 0.02])
def test_a_bigger_transfer_raises_peak_inflation(u: float) -> None:
    """Peak, not close: the close can fall once blocs default."""
    assert measure(_run(u)).inflation_peak > measure(_run(u / 2)).inflation_peak


def test_the_quiet_baseline_is_quiet() -> None:
    """With no programme anywhere, pressure is small but not asserted zero."""
    m = measure(_run(0.0))
    assert m.inflation_peak < 0.01
    assert m.gap_max < 0.05
    assert m.insolvent_blocs == 0


# --- a world that went quiet after defaults must not read as a calm one ---

def test_a_run_that_subsides_after_default_reports_its_defaults() -> None:
    """The context that makes a low closing figure readable."""
    m = measure(_run(0.05))
    assert m.insolvent_blocs > 0
    assert m.first_default_year is not None
    assert m.demand_share_solvent < 1.0


def test_closing_inflation_understates_what_a_large_transfer_did() -> None:
    """The reason the peak and the default count travel with the close.

    A large transfer's inflation peaks early and is well down by year thirty,
    so the closing figure is a small fraction of what the configuration
    actually did: at u=0.05 the world peaks above 8% a year and closes near
    0.4%. Read alone, the closing number would put that configuration in the
    same bracket as one that never went above 1.6%.
    """
    calm = measure(_run(0.01))
    broken = measure(_run(0.05))
    assert broken.inflation_peak > 4 * calm.inflation_peak
    assert broken.inflation_final < 0.25 * broken.inflation_peak
    assert broken.gap_max > calm.gap_max
    assert broken.insolvent_blocs > calm.insolvent_blocs


def test_solvent_demand_share_is_one_when_nobody_defaults() -> None:
    assert measure(_run(0.01)).demand_share_solvent == pytest.approx(1.0)


# --- the derived series ---

def test_annual_inflation_has_one_fewer_point_than_the_path() -> None:
    path = _run(0.01)["world"]
    assert len(annual_inflation(path)) == len(path) - 1


def test_output_gap_starts_at_zero() -> None:
    """World capacity is normalised on the opening demand."""
    assert output_gaps(_run(0.0)["world"])[0][1] == pytest.approx(0.0, abs=1e-9)


def test_first_default_year_is_none_without_defaults() -> None:
    assert first_default_year(_run(0.0)["world"]) is None


def test_solvent_share_handles_an_empty_path() -> None:
    assert solvent_demand_share([]) == 1.0


# --- phases C and D are not silently assumed to have passed ---

def test_unmeasured_quantities_are_none_not_zero() -> None:
    """Reporting 0 would claim a measurement that has not been made."""
    m = measure(_run(0.01))
    assert m.real_value is None
    assert m.settlement_bound is None


def test_phase_c_and_d_values_pass_through_when_supplied() -> None:
    m = measure(_run(0.01), real_value={"D (developing)": 0.4}, settlement_bound=2)
    assert m.real_value == {"D (developing)": 0.4}
    assert m.settlement_bound == 2


def test_the_measurement_is_reproducible() -> None:
    assert measure(_run(0.01)).as_dict() == measure(_run(0.01)).as_dict()
