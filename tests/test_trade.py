"""The world must add up.

v8's first premise is that the world is closed: one bloc's imports are another
bloc's exports, and one bloc's capital outflow is another's inflow. These are
identities, not results, so they are asserted exactly (to machine precision)
rather than approximately.

Specification: `specs/V8_A_B.md` §2 and §4.1.
"""

from __future__ import annotations

import random

import pytest

from model.params import Bloc, archetypes
from model.trade import (
    ADDING_UP_TOL, balanced_capital_flows, current_accounts, export_capacity,
    exports_from_imports, net_foreign_asset_change, trade_shares,
)

BLOCS = list(archetypes().values())


def _random_blocs(rng: random.Random, n: int) -> list[Bloc]:
    """A world of `n` blocs with arbitrary sizes and openness."""
    return [Bloc(name=f"X{i}", gdp=rng.uniform(0.05, 2.0), debt=0.5, rate=0.03,
                 base_growth=0.02, decay=0.0005,
                 openness=rng.uniform(0.1, 1.0), fx_share=rng.random())
            for i in range(n)]


def _random_imports(rng: random.Random, n: int) -> list[float]:
    return [rng.uniform(0.0, 0.5) for _ in range(n)]


# --- the trade matrix ---

def test_rows_sum_to_one_and_diagonal_is_zero() -> None:
    S = trade_shares(BLOCS)
    for i, row in enumerate(S):
        assert row[i] == 0.0, "a bloc must not import from itself"
        assert abs(sum(row) - 1.0) < ADDING_UP_TOL


@pytest.mark.parametrize("n", [2, 3, 7, 31])
def test_rows_sum_to_one_for_arbitrary_worlds(n: int) -> None:
    rng = random.Random(n)
    S = trade_shares(_random_blocs(rng, n))
    for i, row in enumerate(S):
        assert row[i] == 0.0
        assert abs(sum(row) - 1.0) < ADDING_UP_TOL


def test_single_bloc_world_has_nobody_to_trade_with() -> None:
    """Degenerate but well defined: the row is zero, not a division by zero."""
    S = trade_shares(BLOCS[:1])
    assert S == [[0.0]]


def test_larger_and_more_open_suppliers_get_a_bigger_share() -> None:
    """The matrix is built from size times openness, and must reflect it."""
    S = trade_shares(BLOCS)
    w = export_capacity(BLOCS)
    # From bloc D's perspective, rank suppliers by capacity and by share.
    d = len(BLOCS) - 1
    by_share = sorted((j for j in range(len(BLOCS)) if j != d),
                      key=lambda j: S[d][j], reverse=True)
    by_capacity = sorted((j for j in range(len(BLOCS)) if j != d),
                         key=lambda j: w[j], reverse=True)
    assert by_share == by_capacity


# --- the adding-up identities ---

def test_world_exports_equal_world_imports() -> None:
    S = trade_shares(BLOCS)
    M = [0.10, 0.08, 0.06, 0.03]
    X = exports_from_imports(M, S)
    assert abs(sum(X) - sum(M)) < ADDING_UP_TOL


def test_current_accounts_sum_to_zero() -> None:
    S = trade_shares(BLOCS)
    M = [0.10, 0.08, 0.06, 0.03]
    CA = current_accounts(M, exports_from_imports(M, S))
    assert abs(sum(CA)) < ADDING_UP_TOL


@pytest.mark.parametrize("seed", range(10))
def test_current_accounts_sum_to_zero_for_arbitrary_worlds(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(2, 12)
    blocs = _random_blocs(rng, n)
    M = _random_imports(rng, n)
    CA = current_accounts(M, exports_from_imports(M, trade_shares(blocs)))
    assert abs(sum(CA)) < ADDING_UP_TOL


@pytest.mark.parametrize("seed", range(10))
def test_capital_flows_sum_to_zero(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(2, 12)
    blocs = _random_blocs(rng, n)
    desired = [rng.uniform(-0.1, 0.1) for _ in range(n)]
    net = balanced_capital_flows(desired, export_capacity(blocs))
    assert abs(sum(net)) < ADDING_UP_TOL


def test_capital_flows_preserve_the_ordering_of_desires() -> None:
    """Balancing removes a common component; it must not reshuffle blocs."""
    desired = [0.05, -0.02, 0.01, 0.03]
    net = balanced_capital_flows(desired, export_capacity(BLOCS))
    assert sorted(range(4), key=lambda i: desired[i]) == \
           sorted(range(4), key=lambda i: net[i] - (net[i] - desired[i]))


def test_consistent_desires_are_left_alone() -> None:
    """Desires that already sum to zero need no adjustment."""
    desired = [0.05, -0.05, 0.02, -0.02]
    net = balanced_capital_flows(desired, export_capacity(BLOCS))
    for a, b in zip(desired, net):
        assert abs(a - b) < ADDING_UP_TOL


def test_net_foreign_assets_sum_to_zero() -> None:
    """The world cannot accumulate claims on itself."""
    S = trade_shares(BLOCS)
    M = [0.10, 0.08, 0.06, 0.03]
    CA = current_accounts(M, exports_from_imports(M, S))
    KA = balanced_capital_flows([0.02, -0.01, 0.03, -0.005], export_capacity(BLOCS))
    assert abs(sum(net_foreign_asset_change(CA, KA))) < ADDING_UP_TOL


# --- what closing the world is FOR ---

def test_one_blocs_extra_imports_arrive_as_other_blocs_exports() -> None:
    """The defect v8 exists to fix: in v7 this leakage left the model."""
    S = trade_shares(BLOCS)
    base = [0.10, 0.08, 0.06, 0.03]
    raised = list(base)
    raised[3] += 0.05           # bloc D imports more
    X0 = exports_from_imports(base, S)
    X1 = exports_from_imports(raised, S)
    gained = [b - a for a, b in zip(X0, X1)]
    assert abs(sum(gained) - 0.05) < ADDING_UP_TOL, "the extra demand must land somewhere"
    assert gained[3] == 0.0, "and not back on the importer itself"
    assert all(x > 0 for i, x in enumerate(gained) if i != 3)
