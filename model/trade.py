"""Closing the world: trade shares, adding-up, and capital-account balance.

v7 let one bloc's leakage leave the model. A bloc that imported more simply
lost reserves to nowhere, and no other bloc's exports rose to match. That made
the central claim of the paper — cash escapes an individual country but cannot
escape the world — impossible to state, let alone test.

Here every import is somebody's export and every capital outflow is somebody's
inflow. The functions are pure: they take quantities and return quantities,
and the identities they satisfy are asserted in `tests/test_trade.py`.

See `specs/V8_A_B.md` §2.
"""

from __future__ import annotations

from typing import Sequence

from .params import Bloc

#: Absolute tolerance on the adding-up identities. The quantities are O(1)
#: shares of GDP, so this is far tighter than anything the dynamics resolve.
ADDING_UP_TOL: float = 1e-10


def export_capacity(blocs: Sequence[Bloc]) -> list[float]:
    """Each bloc's weight as a supplier of tradables.

    Size times openness: a large closed economy and a small open one can be
    comparable suppliers. These are the initial values and do not move, so the
    matrix below is time-invariant.
    """
    return [b.gdp * b.openness for b in blocs]


def trade_shares(blocs: Sequence[Bloc],
                 exchange_rates: Sequence[float] | None = None,
                 elasticity: float = 0.0) -> list[list[float]]:
    """Row-stochastic matrix `S` with `S[i][j]` = share of i's imports bought from j.

    A bloc does not import from itself, so the diagonal is zero and each row is
    normalised over the others. With a single bloc there is nobody to trade
    with and the row is all zeros — the caller must then treat imports as
    unsatisfiable rather than as satisfied by nobody.

    Args:
        exchange_rates: Current rates, so that a bloc which has depreciated
            takes a larger share of everyone's imports. Omit for the fixed
            matrix of phases A-C.
        elasticity: How strongly a depreciation wins market share. Zero
            reproduces the fixed matrix exactly.

    Without this channel a bloc that cannot pay for its imports has no way out:
    depreciation does not earn it more foreign exchange, so the settlement
    constraint of phase D tightens without limit. That is not a finding about
    peripheral economies, it is an omission — the adjustment every open economy
    relies on was missing from the matrix.
    """
    w = export_capacity(blocs)
    if exchange_rates is not None and elasticity > 0:
        # A depreciated currency makes that bloc's goods cheaper abroad, so it
        # supplies a larger share of world demand.
        w = [wi * max(1e-9, e) ** elasticity for wi, e in zip(w, exchange_rates)]
    n = len(blocs)
    out: list[list[float]] = []
    for i in range(n):
        total = sum(w[j] for j in range(n) if j != i)
        if total <= 0:
            out.append([0.0] * n)
            continue
        out.append([0.0 if j == i else w[j] / total for j in range(n)])
    return out


def exports_from_imports(
    imports: Sequence[float], shares: Sequence[Sequence[float]],
) -> list[float]:
    """Distribute every bloc's imports across its suppliers.

    ``X[j] = sum_i S[i][j] * M[i]``. World exports equal world imports by
    construction; that is the point of routing demand through `S` rather than
    letting it leave.
    """
    n = len(imports)
    return [sum(shares[i][j] * imports[i] for i in range(n)) for j in range(n)]


def current_accounts(
    imports: Sequence[float], exports: Sequence[float],
) -> list[float]:
    """Current account of each bloc, `X - M`. Sums to zero across the world."""
    return [x - m for x, m in zip(exports, imports)]


def balanced_capital_flows(
    desired: Sequence[float], weights: Sequence[float],
) -> list[float]:
    """Net capital flows that sum to zero across the world.

    Each bloc has a desired inflow driven by its own return differential. Those
    desires need not be mutually consistent: the world cannot receive net
    capital from outside itself. The excess is removed in proportion to size,
    so a large bloc absorbs more of the adjustment than a small one.

    With zero total weight (a degenerate world) the desires are returned
    demeaned, which still sums to zero.
    """
    n = len(desired)
    if n == 0:
        return []
    excess = sum(desired)
    total_w = sum(weights)
    if total_w <= 0:
        mean = excess / n
        return [d - mean for d in desired]
    return [d - excess * (w / total_w) for d, w in zip(desired, weights)]


def net_foreign_asset_change(
    current: Sequence[float], capital: Sequence[float],
) -> list[float]:
    """Change in each bloc's net foreign asset position.

    ``CA + KA``. Both components sum to zero across the world, so this does
    too: the world as a whole cannot accumulate claims on itself.
    """
    return [c + k for c, k in zip(current, capital)]
