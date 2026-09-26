"""Phase 1 parameter space: definitions shared by the LHS and Sobol runs.

Ranges are multipliers on the baseline value unless marked absolute. The
parameters are swept independently rather than in blocks, so that each one's
contribution is separable in the Sobol decomposition.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Callable, NamedTuple

from legacy.v7.params import Bloc, G, archetypes


class Param(NamedTuple):
    """One swept parameter."""

    name: str
    low: float
    high: float
    kind: str           # "mult" | "abs" | "logmult"
    target: str         # "bloc" | "global" | "run"
    note: str = ""


#: Prior expectation, recorded before the run (TASKS.md).
#: Kept verbatim so it can be scored honestly against the result.
PRIOR_EXPECTATION = (
    "eps_supply and land_share dominate ΔLeak."
)

#: Added after Phase 0, which showed CLT leaking more than cash in bloc D on
#: the per-fiscal metric, where the import content of construction is highest.
#: Recorded separately so it is not confused with the pre-registered prior.
POST_HOC_CANDIDATE = (
    "m_H (and the new clt_domestic_sourcing lever) may dominate the sign of "
    "ΔLeak. Added after Phase 0; not part of the prior expectation."
)

PARAMS: tuple[Param, ...] = (
    Param("m_F", 0.5, 1.5, "mult", "bloc", "food import content"),
    Param("m_G", 0.5, 1.5, "mult", "bloc", "other-goods import content"),
    Param("m_H", 0.5, 1.5, "mult", "bloc", "construction import content"),
    Param("eps_supply", 0.1, 50.0, "logmult", "bloc", "housing supply elasticity"),
    Param("land_share", 0.5, 1.5, "mult", "bloc", "land rent share, capped at 0.8"),
    Param("land_discount", 0.0, 1.0, "abs", "global", "CLT land acquisition discount"),
    Param("omega0", 0.5, 2.0, "mult", "bloc", "baseline foreign-asset share"),
    Param("omega_sens", 0.5, 2.0, "mult", "global", "risk sensitivity of that share"),
    Param("mpc_u", 0.1, 0.6, "abs", "global", "unconstrained MPC"),
    Param("bop_coeff", 0.5, 2.0, "mult", "global", "leakage to FX pressure"),
    Param("export_switch", 0.5, 2.0, "mult", "global", "expenditure switching"),
    Param("sub_H", 0.7, 1.3, "mult", "global", "housing subsistence"),
    Param("beta_H", 0.7, 1.3, "mult", "global", "housing marginal budget share"),
    Param("u", 0.02, 0.15, "abs", "run", "UBI level, share of GDP"),
    Param("clt_domestic_sourcing", 0.0, 1.0, "abs", "bloc",
          "policy effort to source CLT construction domestically"),
    Param("sourcing_cost_kappa", 0.0, 0.5, "abs", "global",
          "cost markup per unit of domestic sourcing"),
)

#: Matched-range variant. The headline sweep gives the policy lever a full
#: [0,1] span while m_H moves only +/-50% around its baseline, so the lever has
#: far more room to act. This variant equalises the *effective* span of the two
#: so the policy-vs-structure asymmetry can be tested rather than assumed.
#: m_H is swept over a multiplier range whose effect on realised import content
#: matches what the sourcing lever can achieve.
MATCHED_OVERRIDES: dict[str, tuple[float, float, str]] = {
    "m_H": (0.0, 2.0, "mult"),                 # realised imports span 0 .. 2x
    "clt_domestic_sourcing": (0.0, 1.0, "abs"),  # realised imports span 1x .. 0
}


def matched_params() -> tuple[Param, ...]:
    """PARAMS with m_H and the sourcing lever given comparable spans."""
    out = []
    for p in PARAMS:
        if p.name in MATCHED_OVERRIDES:
            low, high, kind = MATCHED_OVERRIDES[p.name]
            out.append(p._replace(low=low, high=high, kind=kind,
                                  note=p.note + " [matched range]"))
        else:
            out.append(p)
    return tuple(out)

PARAM_NAMES: tuple[str, ...] = tuple(p.name for p in PARAMS)

#: Introduction pattern is sampled as a discrete choice.
START_PATTERNS: dict[str, tuple[float, ...]] = {
    "staggered": (1, 5, 10, 15),
    "simultaneous": (1, 1, 1, 1),
}


def unit_to_value(p: Param, unit: float, base: float) -> float:
    """Map a unit-cube coordinate to a parameter value."""
    if p.kind == "abs":
        return p.low + unit * (p.high - p.low)
    if p.kind == "logmult":
        lo, hi = math.log(p.low), math.log(p.high)
        return base * math.exp(lo + unit * (hi - lo))
    return base * (p.low + unit * (p.high - p.low))


def apply_point(unit: dict[str, float],
                params: tuple[Param, ...] | None = None) -> tuple[list[Bloc], G, float]:
    """Turn a unit-cube point into (blocs, globals, u).

    Budget feasibility is enforced by construction: the Stone-Geary
    subsistence shares must leave a positive supernumerary budget, so `sub_H`
    is clipped if the draw would violate it.
    """
    params = params or PARAMS
    g = G(reserve_flight_mode="capped")
    base_g = G()

    for p in params:
        if p.target != "global":
            continue
        setattr(g, p.name, unit_to_value(p, unit[p.name], getattr(base_g, p.name)))

    # Feasibility: subsistence must not exhaust the constrained household's budget.
    total_sub = g.sub_F + g.sub_G + g.sub_H
    if total_sub >= 0.95:
        g.sub_H = max(0.01, 0.95 - g.sub_F - g.sub_G)

    blocs: list[Bloc] = []
    for key, b in archetypes().items():
        kwargs: dict[str, Any] = {}
        for p in params:
            if p.target != "bloc":
                continue
            value = unit_to_value(p, unit[p.name], getattr(b, p.name, 0.0))
            if p.name == "land_share":
                value = min(value, 0.8)
            if p.name in ("m_F", "m_G", "m_H", "omega0"):
                value = min(value, 0.95)
            if p.name == "clt_domestic_sourcing":
                value = min(max(value, 0.0), 1.0)
            kwargs[p.name] = value
        blocs.append(replace(b, **kwargs))

    u = unit_to_value(params[PARAM_NAMES.index("u")], unit["u"], 0.0)
    return blocs, g, u
