"""Nominal transfers and what they are worth.

Phases A/B size a transfer as `size * Y`, and `Y` is real GDP, so the transfer
is a real quantity that inflation cannot touch. The paper's first claim is
about precisely the loop that absence rules out: inflation erodes the
transfer, the transfer is raised, the raise feeds inflation. Phase C carries
the transfer as a nominal amount so the loop can run.

Specification: `specs/V8_C.md` §6.
"""

from __future__ import annotations

import pytest

from model.indexation import RULES, TransferState, advance


def _path(rule: str, inflation: float, steps: int = 8,
          size: float = 0.05, output: float = 1.0):
    """Run one transfer forward at a constant inflation rate."""
    state = TransferState()
    price = 1.0
    out = []
    for _ in range(steps):
        amount = advance(state, rule, size=size, output=output, price=price,
                         inflation=inflation)
        out.append((price, amount, state.real_value_ratio(price)))
        price *= 1.0 + inflation
    return out


# --- fixed_nominal: the transfer is eroded ---

def test_a_fixed_nominal_transfer_keeps_its_nominal_amount() -> None:
    amounts = [a for _, a, _ in _path("fixed_nominal", 0.05)]
    assert all(a == pytest.approx(amounts[0]) for a in amounts)


def test_a_fixed_nominal_transfer_loses_real_value_monotonically() -> None:
    ratios = [r for _, _, r in _path("fixed_nominal", 0.05)]
    assert ratios[0] == pytest.approx(1.0)
    assert all(b < a for a, b in zip(ratios, ratios[1:]))


def test_a_fixed_nominal_transfer_is_untouched_by_zero_inflation() -> None:
    ratios = [r for _, _, r in _path("fixed_nominal", 0.0)]
    assert all(r == pytest.approx(1.0) for r in ratios)


# --- cpi_indexed: the transfer keeps its value, and costs more ---

def test_an_indexed_transfer_holds_its_real_value() -> None:
    ratios = [r for _, _, r in _path("cpi_indexed", 0.05)]
    assert all(r == pytest.approx(1.0, rel=1e-9) for r in ratios)


def test_an_indexed_transfer_costs_more_in_nominal_terms() -> None:
    """The fiscal side of holding real value constant."""
    fixed = [a for _, a, _ in _path("fixed_nominal", 0.05)]
    indexed = [a for _, a, _ in _path("cpi_indexed", 0.05)]
    assert indexed[-1] > fixed[-1]
    assert all(b >= a for a, b in zip(indexed, indexed[1:]))


def test_indexation_holds_more_value_than_no_indexation() -> None:
    fixed = [r for _, _, r in _path("fixed_nominal", 0.05)]
    indexed = [r for _, _, r in _path("cpi_indexed", 0.05)]
    assert all(i >= f for i, f in zip(indexed, fixed))
    assert indexed[-1] > fixed[-1]


def test_indexation_uses_last_periods_inflation_not_this_periods() -> None:
    """Indexing to current inflation would be a simultaneous equation.

    The first period's amount must be the unindexed opening size: a transfer
    cannot be adjusted for inflation that has not happened yet.
    """
    first = _path("cpi_indexed", 0.20)[0][1]
    assert first == pytest.approx(0.05)


# --- gdp_linked: the A/B behaviour ---

def test_gdp_linked_tracks_output_and_ignores_prices() -> None:
    """Why phase C exists: under this rule inflation never reaches the transfer."""
    hot = _path("gdp_linked", 0.20)
    cold = _path("gdp_linked", 0.0)
    assert [a for _, a, _ in hot] == [a for _, a, _ in cold]


def test_gdp_linked_grows_with_real_output() -> None:
    state = TransferState()
    small = advance(state, "gdp_linked", size=0.05, output=1.0, price=1.0,
                    inflation=0.0)
    large = advance(state, "gdp_linked", size=0.05, output=2.0, price=1.0,
                    inflation=0.0)
    assert large == pytest.approx(2 * small)


# --- shared behaviour ---

@pytest.mark.parametrize("rule", RULES)
def test_every_rule_opens_at_the_same_amount(rule: str) -> None:
    """The rules differ in how a transfer evolves, not in how it starts."""
    assert _path(rule, 0.05)[0][1] == pytest.approx(0.05)


@pytest.mark.parametrize("rule", RULES)
def test_real_value_ratio_is_one_before_the_programme_starts(rule: str) -> None:
    """A transfer that has not begun has not been eroded."""
    assert TransferState().real_value_ratio(2.0) == 1.0


def test_an_unknown_rule_is_refused() -> None:
    with pytest.raises(ValueError):
        advance(TransferState(), "whatever", size=0.05, output=1.0, price=1.0,
                inflation=0.0)
