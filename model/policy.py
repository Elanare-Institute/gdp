"""What authorities do when their economy falls behind the rest.

The first version of this module triggered on deflation — inflation below a
floor — and did almost nothing, because a world whose imports are rationed
sees its *world* price fall while every bloc's own CPI keeps rising. That was
the wrong trigger. Capital does not leave because prices are falling in
absolute terms; it leaves because growth here is lower than growth there, and
authorities watch the same relative position.

So the trigger is relative throughout: a bloc eases in proportion to how far
its growth has fallen below the world's, and how far its real rate has fallen
below the world's. A bloc growing at 2% in a world growing at 4% is losing
capital and will ease, whatever its price level is doing.

Every bloc gets the same rule. The periphery's predicament has to come from
the settlement constraint and from debt denominated in someone else's money —
never from handing its authorities a weaker reaction function. Which is what
puts a peripheral bloc in a vice: easing depreciates the currency and makes
imports dearer against a settlement constraint, while not easing lets growth
fall further behind and the capital leave. Both routes end in less foreign
exchange.

See `specs/V8_D.md`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyImpulse:
    """One period's easing, split between the two instruments."""

    rate_cut: float
    """Subtracted from the policy rate target, in annual points."""

    fiscal_boost: float
    """Extra general government spending, as a share of output."""

    @property
    def total(self) -> float:
        return self.rate_cut + self.fiscal_boost


def relative_shortfall(own: float, world: float) -> float:
    """How far a bloc sits below the world on some measure. Zero if above."""
    return max(0.0, world - own)


def respond(growth: float, world_growth: float,
            real_rate: float, world_real_rate: float,
            strength: float, fiscal_share: float) -> PolicyImpulse:
    """Easing in response to falling behind the rest of the world.

    Both triggers are gaps, not levels. A bloc growing more slowly than the
    world is losing capital to it; a bloc whose real rate is below the world's
    is losing it faster still. Authorities lean against both, in proportion.

    Nothing here names a target. An authority told to hold inflation at two per
    cent would be told the answer; one that leans in proportion to how far it
    has fallen behind is only told the direction.

    Args:
        growth: The bloc's own real growth, annual.
        world_growth: Size-weighted world real growth, annual.
        real_rate: The bloc's own real policy rate.
        world_real_rate: The world's average real rate.
        strength: How hard authorities lean. Zero disables the response.
        fiscal_share: Share of the impulse carried by spending, in [0, 1].
    """
    if strength <= 0:
        return PolicyImpulse(0.0, 0.0)
    gap = (relative_shortfall(growth, world_growth)
           + relative_shortfall(real_rate, world_real_rate))
    impulse = strength * gap
    share = min(1.0, max(0.0, fiscal_share))
    return PolicyImpulse(rate_cut=impulse * (1.0 - share),
                         fiscal_boost=impulse * share)
