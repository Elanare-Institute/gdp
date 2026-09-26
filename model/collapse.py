"""World-level pressure indicators.

These describe how much pressure a configuration puts on the world, and in
which direction. They do **not** judge whether the world survives.

That restraint is deliberate. Whether a real world comes apart depends on
routes this model does not contain — trade blocs forming, migration, conflict,
policy reversal — so a pass/fail verdict computed from thresholds inside the
model would be decided by where those thresholds were placed, not by anything
the model knows. What can honestly be shown is the size and direction of the
pressure a configuration generates, and where that pressure starts rising
faster than the configuration that caused it.

Four quantities, of which two can be computed from phases A/B alone:

  world inflation, annual, and its path               (here)
  world demand against capacity, and its path         (here)
  real value of transfers, by bloc                    (needs phase C)
  blocs bound by the settlement constraint            (needs phase D)

Everything here is a level or a change. Nothing is compared against a cut-off.

See `specs/V8_COLLAPSE.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class WorldPressure:
    """What one configuration did to the world.

    No verdict field, by design: a reader comparing configurations should be
    comparing inflation against inflation and gap against gap, not two booleans
    whose meaning is set by a threshold chosen here.

    ``inflation_final`` can be low for opposite reasons — a world that stayed
    calm, and a world that went quiet after its blocs defaulted — so the
    default count and the peak are carried alongside it rather than left for
    the reader to look up. They are the context that makes the closing number
    readable, not a test it has to pass.
    """

    inflation_final: float
    inflation_peak: float
    inflation_mean: float
    inflation_path: tuple[tuple[float, float], ...]
    gap_final: float
    gap_max: float
    gap_path: tuple[tuple[float, float], ...]
    insolvent_blocs: int
    first_default_year: float | None
    demand_share_solvent: float
    real_value: dict[str, float] | None = None       # phase C
    settlement_bound: int | None = None              # phase D

    def as_dict(self) -> dict[str, Any]:
        return {
            "inflation_final": self.inflation_final,
            "inflation_peak": self.inflation_peak,
            "inflation_mean": self.inflation_mean,
            "gap_final": self.gap_final,
            "gap_max": self.gap_max,
            "insolvent_blocs": self.insolvent_blocs,
            "first_default_year": self.first_default_year,
            "demand_share_solvent": self.demand_share_solvent,
            "real_value": self.real_value,
            "settlement_bound": self.settlement_bound,
            "inflation_path": list(self.inflation_path),
            "gap_path": list(self.gap_path),
        }


def annual_inflation(world_path: Sequence[dict[str, float]]) -> list[tuple[float, float]]:
    """Year-on-year world inflation from the annual price samples."""
    out: list[tuple[float, float]] = []
    for prev, now in zip(world_path, world_path[1:]):
        p0, p1 = prev["price"], now["price"]
        out.append((now["year"], p1 / p0 - 1.0 if p0 > 0 else 0.0))
    return out


def output_gaps(world_path: Sequence[dict[str, float]]) -> list[tuple[float, float]]:
    """World demand against world capacity, per sampled year."""
    return [(row["year"], row["demand"] / row["capacity"] - 1.0 if row["capacity"] > 0 else 0.0)
            for row in world_path]


def first_default_year(world_path: Sequence[dict[str, float]]) -> float | None:
    """When the first bloc became insolvent, or None if none did."""
    for row in world_path:
        if int(row.get("insolvent", 0)) > 0:
            return float(row["year"])
    return None


def solvent_demand_share(world_path: Sequence[dict[str, float]]) -> float:
    """Fraction of closing world demand coming from blocs still solvent.

    Below one, part of the world's demand is being placed by blocs that have
    passed the default threshold. Any inflation figure for such a run is about
    a world that is no longer whole, and this number says how much of it is.
    """
    if not world_path:
        return 1.0
    row = world_path[-1]
    total = row.get("demand", 0.0)
    if total <= 0:
        return 1.0
    return row.get("demand_solvent", total) / total


def measure(result: dict[str, Any],
            real_value: dict[str, float] | None = None,
            settlement_bound: int | None = None) -> WorldPressure:
    """Summarise what one run did to the world.

    Args:
        result: what `model.core.simulate` returned.
        real_value: per-bloc real value of the transfer at the end, relative to
            its value at introduction. Available from phase C.
        settlement_bound: how many blocs were rationed by the settlement
            constraint. Available from phase D.
    """
    path = result["world"]
    infl = annual_inflation(path)
    gaps = output_gaps(path)
    values = [v for _, v in infl]
    return WorldPressure(
        inflation_final=values[-1] if values else 0.0,
        inflation_peak=max(values, default=0.0),
        inflation_mean=sum(values) / len(values) if values else 0.0,
        inflation_path=tuple(infl),
        gap_final=gaps[-1][1] if gaps else 0.0,
        gap_max=max((v for _, v in gaps), default=0.0),
        gap_path=tuple(gaps),
        insolvent_blocs=int(path[-1].get("insolvent", 0)) if path else 0,
        first_default_year=first_default_year(path),
        demand_share_solvent=solvent_demand_share(path),
        real_value=real_value,
        settlement_bound=settlement_bound,
    )
