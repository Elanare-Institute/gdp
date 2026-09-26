"""Labour demand falling everywhere at once.

The paper's setting is not a redistribution debate inside a working economy.
It is an economy where office work has been automated and machines and
vehicles drive themselves, so labour demand falls across the world at the same
time. Transfers there are not a policy option weighed against wages; they are
what is left.

Everything is derived from a single progress variable, so the model cannot be
tuned channel by channel to produce an answer:

  potential capacity   rises — machines can make more
  unit cost            falls — domestic goods get cheaper to produce
  labour income        falls — the constrained block earns less
  capital income       rises — what labour no longer earns accrues to owners

**Potential is not production.** Capacity that nobody has the money to buy from
stands idle, which is the whole difficulty: automation raises what could be
made at the same time as it removes the income that would buy it. Utilisation
is reported for that reason.

See `specs/V8_POSTEMP.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .params import G


@dataclass(frozen=True)
class AutomationState:
    """How far automation has gone, and what follows from it."""

    progress: float
    """Between 0 (nothing has changed) and `automation_max`."""

    capacity_multiplier: float
    """What potential output has been multiplied by."""

    cost_multiplier: float
    """What the unit cost of domestic production has been multiplied by."""

    labour_multiplier: float
    """What the constrained block's labour income has been multiplied by."""

    @property
    def displaced_labour_share(self) -> float:
        """Share of the constrained block's original income no longer earned.

        It does not vanish: output still gets produced and still gets paid
        for, so what labour stops receiving accrues to the owners of the
        machines — the unconstrained block.
        """
        return 1.0 - self.labour_multiplier


def progress(g: G, t: float) -> float:
    """Automation at time `t`, approaching `automation_max` from zero.

    Exponential approach rather than a straight line: the early years move
    fastest and the last of the work is the hardest to automate. Neither the
    speed nor the ceiling is an estimate, so both are swept.
    """
    if g.automation_speed <= 0 or g.automation_max <= 0:
        return 0.0
    import math
    return g.automation_max * (1.0 - math.exp(-g.automation_speed * t))


def state(g: G, t: float) -> AutomationState:
    """The four consequences of automation at time `t`.

    All four move together because they are one phenomenon seen from four
    sides. Giving each its own schedule would let the model be tuned until it
    said what was wanted.
    """
    a = progress(g, t)
    return AutomationState(
        progress=a,
        capacity_multiplier=1.0 + g.automation_capacity * a,
        cost_multiplier=max(0.05, 1.0 - g.automation_cost * a),
        labour_multiplier=max(0.0, 1.0 - g.automation_labour * a),
    )


def capacity_overhang(output: float, potential: float) -> float:
    """Output as a share of what automation made possible.

    **Not a utilisation rate for the economy.** The numerator is the model's
    `Y`, which is driven by trend growth and a fiscal impulse, with demand
    entering only through the programme's domestic component. The unconstrained
    block's spending reaches it as `mpc_u` times its income and nothing more:
    that block's consumption bundle is never solved, so its demand is not in
    the numerator in any full sense.

    What this measures, then, is how far the capacity automation adds runs
    ahead of the output the model actually produces — capacity overhang
    relative to the demand the model can see, which is mostly the constrained
    block's. Reading it as "this share of the world's plant stands idle" would
    claim an aggregate the model cannot compute.

    Named for what it is, so the reading cannot drift.
    """
    if potential <= 0:
        return 1.0
    return min(1.0, max(0.0, output / potential))
