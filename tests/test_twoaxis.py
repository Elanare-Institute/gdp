"""Blocs on two independent axes.

The four archetypes bundle credit and self-sufficiency together — bloc D
borrows dearly *and* imports its food — so no estimate on them can say which
does the work. These tests fix that the two axes are separable here, and that
each produces the failure mode it should.

Specification: `specs/V8_TWOAXIS.md`.
"""

from __future__ import annotations

import pytest

from model.params import G
from model.twoaxis import (
    CREDIT_RANGE, NEUTRAL, SELF_SUFFICIENCY_RANGE, grid, locate_archetypes,
    make_bloc, sample,
)


# --- the axes are separable ---

def test_the_archetypes_all_sit_on_the_diagonal() -> None:
    """The reason the two axes could not be told apart before.

    Recorded as a test so that the confounding is a documented property of the
    old population rather than a claim in a report.
    """
    for key, (credit, self_suff) in locate_archetypes().items():
        assert abs(credit - self_suff) < 0.3, f"{key}: {credit} vs {self_suff}"


def test_the_two_corners_are_genuinely_different_blocs() -> None:
    """(low credit, high self-sufficiency) and its mirror must not coincide."""
    a = make_bloc("a", credit=0.2, self_sufficiency=0.8)
    b = make_bloc("b", credit=0.8, self_sufficiency=0.2)
    assert a.credit_standing < b.credit_standing
    assert a.m_F < b.m_F           # a feeds itself; b imports food
    assert a.rate > b.rate


def test_credit_moves_only_the_credit_parameters() -> None:
    low = make_bloc("l", credit=0.0, self_sufficiency=0.5)
    high = make_bloc("h", credit=1.0, self_sufficiency=0.5)
    for field in SELF_SUFFICIENCY_RANGE:
        assert getattr(low, field) == getattr(high, field), field
    for field in CREDIT_RANGE:
        assert getattr(low, field) != getattr(high, field), field


def test_self_sufficiency_moves_only_the_import_shares() -> None:
    low = make_bloc("l", credit=0.5, self_sufficiency=0.0)
    high = make_bloc("h", credit=0.5, self_sufficiency=1.0)
    for field in CREDIT_RANGE:
        assert getattr(low, field) == getattr(high, field), field
    for field in SELF_SUFFICIENCY_RANGE:
        assert getattr(low, field) != getattr(high, field), field


def test_opening_debt_is_off_both_axes() -> None:
    """Putting it on the credit axis inverts that axis.

    A low-credit bloc given a low opening debt has a small risk premium and
    therefore a *wide* borrowing window — the opposite of low credit. What
    lenders will advance is set directly instead.
    """
    assert "debt" in NEUTRAL
    assert "debt" not in CREDIT_RANGE
    assert (make_bloc("l", 0.0, 0.5).debt
            == make_bloc("h", 1.0, 0.5).debt == NEUTRAL["debt"])


def test_lower_credit_means_a_narrower_borrowing_window() -> None:
    assert make_bloc("l", 0.0, 0.5).credit_standing < \
           make_bloc("h", 1.0, 0.5).credit_standing


def test_reserve_status_is_independent_of_both_axes() -> None:
    plain = make_bloc("p", 0.3, 0.7, reserve=False)
    issuer = make_bloc("i", 0.3, 0.7, reserve=True)
    assert issuer.reserve and not plain.reserve
    assert issuer.credit_standing == plain.credit_standing
    assert issuer.m_F == plain.m_F


# --- generated populations ---

def test_a_sample_is_reproducible_and_has_one_issuer() -> None:
    a, b = sample(5, n=12), sample(5, n=12)
    assert [x.name for x in a] == [x.name for x in b]
    assert [x.credit_standing for x in a] == [x.credit_standing for x in b]
    assert sum(x.reserve for x in a) == 1


def test_the_grid_covers_the_plane() -> None:
    blocs = grid(4)
    assert len(blocs) == 16
    assert len({b.credit_standing for b in blocs}) == 4
    assert len({b.m_F for b in blocs}) == 4


@pytest.mark.parametrize("seed", range(5))
def test_the_axes_are_uncorrelated_in_a_sample(seed: int) -> None:
    """Independence is the point: correlated axes reproduce the archetypes."""
    import statistics as st
    blocs = sample(seed, n=60)
    credit = [b.credit_standing for b in blocs]
    imports = [b.m_F for b in blocs]
    r = st.correlation(credit, imports)
    assert abs(r) < 0.4, f"axes correlate at {r:+.2f}"


# --- each axis produces its own failure mode ---

def _run_pair():
    from dataclasses import replace
    from model.core import simulate
    from model.household import calibrate
    g = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
          indexation="cpi_indexed", settlement=True, trade_elasticity=1.0,
          policy_response=1.0)
    blocs = [make_bloc("poor_credit", 0.0, 0.9),
             make_bloc("poor_supply", 0.9, 0.0),
             make_bloc("both_poor", 0.0, 0.0),
             make_bloc("comfortable", 1.0, 1.0, reserve=True)]
    sized = [replace(b, modality="ubi", size=calibrate(b, g, 0.02)["ubi"], start=1.0)
             for b in blocs]
    return simulate(sized, g)


def test_low_self_sufficiency_drains_the_reserves() -> None:
    """A bloc that must import runs its vault down and gets rationed."""
    result = _run_pair()
    final = result["settlement"][-1]
    assert final["share_1"] > 0.0, "the import-dependent bloc should be rationed"


def test_poor_credit_alone_is_survivable_if_you_feed_yourself() -> None:
    """The finding the two axes exist to make visible.

    A bloc lenders distrust but which imports little has nothing much to
    settle, so a narrow borrowing window costs it nothing: it ends with *less*
    debt than the comfortable bloc, not more. Bundled together in the
    archetypes, "cannot borrow" and "must import" were indistinguishable and
    this case could not arise.
    """
    result = _run_pair()
    assert result["final"]["poor_credit"]["debt"] < \
           result["final"]["comfortable"]["debt"]
    assert result["settlement"][-1]["share_0"] == 0.0


def test_both_low_is_far_worse_than_either_alone() -> None:
    """Needing to import and being unable to borrow is the combination that
    breaks: reserves drain and the debt runs away at the same time."""
    result = _run_pair()
    both = result["final"]["both_poor"]["debt"]
    assert both > 2.0
    assert both > 3 * result["final"]["poor_credit"]["debt"]
    assert both > 3 * result["final"]["comfortable"]["debt"]
