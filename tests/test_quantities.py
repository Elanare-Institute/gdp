"""What a recipient actually ends up with.

The transfer's value in CPI terms and its value in housing terms each answer a
different question, and neither compares a cash programme with an in-kind one
on the same footing: a cash transfer's purchasing power and a housing
programme's tenancy are not the same kind of object. The quantity a
constrained household secures — food eaten, housing lived in — is, and it is
the measure the paper's claim has to be settled on.

Because the programmes are calibrated to equal welfare, they start level. Any
divergence is what the macro path does to them over thirty years, which is the
point.

Specification: `specs/V8_C.md`.
"""

from __future__ import annotations

import pytest

from model.core import build, simulate
from model.params import G

U = 0.02
MODALITIES = ("ubi", "cash_t", "voucher", "clt")
BLOC = 3  # bloc D, where the external constraint binds hardest
RULES = ("gdp_linked", "fixed_nominal", "cpi_indexed")


def _series(rule: str, modality: str, good: str, bloc: int = BLOC) -> list[float]:
    """Quantity path, indexed to 100 at the programme's start."""
    g = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02, indexation=rule)
    blocs = build(g, modality, U, (1, 1, 1, 1)) if modality != "none" else build(g, "none")
    rows = simulate(blocs, g)["quantities"]
    opening = rows[1][f"{good}_{bloc}"]
    return [100.0 * row[f"{good}_{bloc}"] / opening for row in rows[1:]]


# --- they start level, or the calibration is wrong ---

@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("good", ["F", "H"])
def test_the_modalities_start_from_the_same_quantities(rule: str, good: str) -> None:
    """Welfare-equivalent programmes must deliver near-identical bundles on
    day one. If they did not, every later difference would be reading the
    calibration rather than the dynamics."""
    opening = {}
    for modality in MODALITIES:
        g = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02, indexation=rule)
        rows = simulate(build(g, modality, U, (1, 1, 1, 1)), g)["quantities"]
        opening[modality] = rows[1][f"{good}_{BLOC}"]
    lo, hi = min(opening.values()), max(opening.values())
    assert (hi - lo) / lo < 0.05, opening


# --- and then they diverge ---

@pytest.mark.parametrize("rule", RULES)
def test_in_kind_housing_outlasts_cash(rule: str) -> None:
    """The paper's claim, on the measure that can settle it.

    Housing already built goes on housing the same households whatever happens
    to prices. Cash handed out is worth what it is worth when it is spent.
    """
    assert _series(rule, "clt", "H")[-1] > _series(rule, "ubi", "H")[-1]


@pytest.mark.parametrize("rule", RULES)
def test_in_kind_housing_also_leaves_more_for_food(rule: str) -> None:
    """A housing programme relieves rent, which frees budget for everything
    else — so it beats cash on food too, without handing out any food."""
    assert _series(rule, "clt", "F")[-1] > _series(rule, "ubi", "F")[-1]


def test_indexation_widens_the_gap_rather_than_closing_it() -> None:
    """Indexing cash holds its CPI value but not what it buys in housing.

    The rent is a price the transfer itself pushes up, so indexation to the
    CPI does not protect a cash recipient's housing — and the extra spending
    it funds pushes the macro path further, which costs them more.
    """
    def gap(rule: str) -> float:
        return _series(rule, "clt", "H")[-1] - _series(rule, "ubi", "H")[-1]
    assert gap("cpi_indexed") > gap("fixed_nominal")


def test_the_paths_are_level_before_they_separate() -> None:
    """Divergence must build over the run, not appear at the start."""
    clt = _series("cpi_indexed", "clt", "H")
    ubi = _series("cpi_indexed", "ubi", "H")
    early = abs(clt[9] - ubi[9])
    late = abs(clt[-1] - ubi[-1])
    assert early < 3.0, "the first decade must stay close"
    assert late > 10 * early, "and the gap must then open"


# --- the uncomfortable one ---

def test_a_universal_cash_transfer_can_leave_recipients_worse_off() -> None:
    """Reported because it is what the model says, not because it is wanted.

    In bloc D a universal indexed transfer ends with recipients holding less
    housing than if nothing had been handed out at all: the programme's own
    macro consequences — inflation, depreciation, dearer imports — cost them
    more than the transfer gives them.
    """
    assert _series("cpi_indexed", "ubi", "H")[-1] < _series("cpi_indexed", "none", "H")[-1]


def test_an_in_kind_programme_does_not_have_that_problem() -> None:
    nothing = _series("cpi_indexed", "none", "H")[-1]
    housing = _series("cpi_indexed", "clt", "H")[-1]
    assert housing >= nothing * 0.99


# --- is the result a mechanism or an artefact of the reduced form? ---

def _gap_to_baseline(penalties: tuple[str, ...], modality: str,
                     good: str = "H") -> float:
    """Year-30 quantity index, minus the no-transfer baseline's."""
    def index(m: str) -> float:
        g = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
              indexation="cpi_indexed", growth_penalties=penalties)
        blocs = build(g, m, U, (1, 1, 1, 1)) if m != "none" else build(g, "none")
        rows = simulate(blocs, g)["quantities"]
        return 100.0 * rows[30][f"{good}_{BLOC}"] / rows[1][f"{good}_{BLOC}"]
    return index(modality) - index("none")


ALL_PENALTIES = ("debt", "inflation", "fx", "capital")


def test_the_default_keeps_every_reduced_form_penalty() -> None:
    """v6's growth adjustments stay on unless a run switches them off."""
    assert set(G().growth_penalties) == set(ALL_PENALTIES)


@pytest.mark.parametrize("modality", ["ubi", "cash_t", "voucher"])
def test_cash_still_trails_the_baseline_with_every_penalty_off(modality: str) -> None:
    """The result must not rest on the reduced-form coefficients.

    Growth falling when debt is high, when inflation is high, or when the
    exchange rate moves are assumptions, not mechanisms — exactly the kind of
    thing that should not be carrying a finding. Switching all four off leaves
    most of the gap in place, so what remains comes from the household block
    and the world market.
    """
    full = _gap_to_baseline(ALL_PENALTIES, modality)
    bare = _gap_to_baseline((), modality)
    assert full < 0 and bare < 0
    assert bare / full > 0.75, f"{modality}: only {bare / full:.0%} survives"


def test_in_kind_housing_matches_the_baseline_either_way() -> None:
    for penalties in (ALL_PENALTIES, ()):
        assert _gap_to_baseline(penalties, "clt") > -2.0


def test_switching_penalties_off_changes_something() -> None:
    """A flag that changed nothing would make the test above vacuous."""
    assert _gap_to_baseline(ALL_PENALTIES, "ubi") != _gap_to_baseline((), "ubi")
