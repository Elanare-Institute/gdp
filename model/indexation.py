"""How a transfer's nominal amount moves over time, and what it is worth.

v7 and phases A/B size every transfer as ``size * Y_t``. ``Y`` is real GDP, so
the transfer is a real quantity: it cannot lose value to inflation, and the
loop the paper's first claim is about — inflation erodes the transfer, the
transfer is raised to compensate, the raise feeds inflation — has no way to
run.

Here the transfer is carried as a nominal amount whose path is a policy
choice:

  gdp_linked      T(t) = size * Y(t)          the A/B behaviour, kept for
                                              regression and as a sensitivity
  fixed_nominal   T(t) = T(start)             no adjustment; real value falls
                                              at the rate of inflation
  cpi_indexed     T(t) = T(t-1) * (1+pi)      adjusted to last period's
                                              inflation

Indexation uses the **previous** period's inflation. Indexing to the current
period would make the transfer and the inflation it causes simultaneous
equations, and real indexation arrangements lag too.

See `specs/V8_C.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

IndexationRule = Literal["gdp_linked", "fixed_nominal", "cpi_indexed"]

RULES: tuple[IndexationRule, ...] = ("gdp_linked", "fixed_nominal", "cpi_indexed")


@dataclass
class TransferState:
    """One bloc's transfer, carried as a nominal amount.

    ``nominal`` is the amount handed out this period, in the bloc's own money.
    ``price_at_start`` is the bloc's price level when the programme began, so
    the real value can be stated relative to what it was worth on day one.
    ``started`` is False until the programme's start date passes.
    """

    nominal: float = 0.0
    price_at_start: float = 1.0
    rent_at_start: float = 1.0
    nominal_at_start: float = 0.0
    started: bool = False

    def begin(self, nominal: float, price: float, rent: float = 1.0) -> None:
        """Record the opening size and the prices it faced.

        Both numeraires are recorded at the start, so the transfer's value can
        be stated in consumption goods and in housing without re-running
        anything.
        """
        self.nominal = nominal
        self.nominal_at_start = nominal
        self.price_at_start = price if price > 0 else 1.0
        self.rent_at_start = rent if rent > 0 else 1.0
        self.started = True

    def real_value(self, price: float) -> float:
        """What the transfer is worth now, in the goods it could buy."""
        return self.nominal / price if price > 0 else 0.0

    def real_value_ratio(self, price: float) -> float:
        """Real value relative to what it was worth at introduction.

        1.0 means the transfer still buys what it bought on the first day;
        below 1.0 it has been eroded. Undefined before the programme starts,
        and reported as 1.0 there rather than as a fall from nothing.

        `price` decides what "worth" means, and the answer differs by numeraire
        — the same nominal transfer buys a steady basket of goods and a
        shrinking amount of housing. Pass the CPI for one reading and the rent
        for the other; see `housing_at_start` and `specs/V8_C.md`.
        """
        if not self.started or self.nominal_at_start <= 0:
            return 1.0
        opening = self.nominal_at_start / self.price_at_start
        now = self.real_value(price)
        return now / opening if opening > 0 else 1.0

    def housing_value_ratio(self, rent: float) -> float:
        """Real value in housing, relative to introduction.

        The paper's companion measure to the CPI reading. A transfer that holds
        its value in consumption goods can still lose it in housing, because
        rent is a price the transfer itself pushes up: cash handed to tenants
        in an inelastic market is bid into rents. An in-kind housing programme
        has no such problem, which is the asymmetry this ratio exists to show.
        """
        if not self.started or self.nominal_at_start <= 0:
            return 1.0
        if self.rent_at_start <= 0 or rent <= 0:
            return 1.0
        opening = self.nominal_at_start / self.rent_at_start
        return (self.nominal / rent) / opening if opening > 0 else 1.0


def advance(state: TransferState, rule: IndexationRule, *, size: float,
            output: float, price: float, inflation: float,
            rent: float = 1.0) -> float:
    """The nominal transfer for this period under `rule`.

    Args:
        size: The programme's size parameter. Under ``gdp_linked`` this and
            `output` set the amount directly; under the other rules it only
            fixes the opening amount.
        output: Real GDP this period.
        price: The bloc's price level this period, used to record what the
            transfer was worth when it began.
        rent: The clearing rent this period, recorded at the start so the
            transfer's value in housing can be stated alongside its value in
            goods.
        inflation: **Last** period's inflation, used by ``cpi_indexed``.

    Returns:
        The nominal amount to hand out, which is also written into `state`.
    """
    if rule not in RULES:
        raise ValueError(f"unknown indexation rule: {rule}")

    if rule == "gdp_linked":
        # The A/B behaviour: a fixed share of real output, so the transfer is
        # real and inflation never touches it.
        amount = size * output
        if not state.started:
            state.begin(amount, price, rent)
        else:
            state.nominal = amount
        return amount

    if not state.started:
        state.begin(size * output, price, rent)
        return state.nominal

    if rule == "fixed_nominal":
        return state.nominal

    # cpi_indexed
    state.nominal *= 1.0 + inflation
    return state.nominal
