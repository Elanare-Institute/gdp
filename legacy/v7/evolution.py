"""Proportional-imitation dynamics over provisioning mixes.

Each bloc holds a mix over the four modalities and revises it every five years
by sampling a structurally similar bloc and moving towards it when that bloc
is under less stress. The question is whether structure — principally the
foreign-currency share of debt — predicts where blocs end up.

Nothing here is solved for an equilibrium. The output is the distribution of
mixes a stochastic process happens to visit, and it is only called selection
when it separates from a neutral-drift control run with the same noise.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any, Sequence

from .heterogeneous import calibrate_hetero
from .mix import compare_with_cash, transition_path
from .params import Bloc, G
from .population import structural_distance

MODALITIES: tuple[str, ...] = ("cash_universal", "cash_targeted", "voucher", "clt")
CLT_INDEX = MODALITIES.index("clt")

#: Three-way control set. `cash_cheap` matches CLT's fiscal cost while paying
#: out like cash, so a bloc that picks CLT over it is picking non-fugitivity
#: rather than cheapness. Welfare is not held fixed here — that is the point
#: of the control, and it is stated wherever its results are used.
CONTROL_MODALITIES: tuple[str, ...] = ("cash_targeted", "clt", "cash_cheap")
CONTROL_CLT_INDEX = CONTROL_MODALITIES.index("clt")


@dataclass
class BlocState:
    """A bloc's mix and the stress it has accumulated."""

    bloc: Bloc
    mix: list[float]
    stress: float = 0.0
    switch_paid: float = 0.0
    is_invader: bool = False
    seeded_with: str = ""

    clt_index: int = CLT_INDEX

    @property
    def clt_share(self) -> float:
        return self.mix[self.clt_index]


def uniform_mix() -> list[float]:
    return [0.25, 0.25, 0.25, 0.25]


def all_cash_mix(modality_set: Sequence[str] = MODALITIES) -> list[float]:
    """Everything in targeted cash — the status quo the phase starts from."""
    mix = [0.0] * len(modality_set)
    mix[list(modality_set).index("cash_targeted")] = 1.0
    return mix


def random_mix(rng: random.Random,
               modality_set: Sequence[str] = MODALITIES) -> list[float]:
    draws = [rng.random() for _ in modality_set]
    total = sum(draws)
    return [d / total for d in draws]


def normalise(mix: Sequence[float]) -> list[float]:
    total = sum(max(0.0, m) for m in mix)
    if total <= 0:
        return all_cash_mix()
    return [max(0.0, m) / total for m in mix]


def move_towards(
    mix: Sequence[float], target: Sequence[float], step: float,
    max_clt_increase: float, clt_index: int = CLT_INDEX,
) -> list[float]:
    """Move `mix` a fraction `step` towards `target`.

    The CLT component cannot rise faster than the construction sector allows,
    so a bloc that wants to imitate a CLT-heavy neighbour still has to build
    its way there. The remaining mass is redistributed over the other
    modalities in proportion to the move they wanted to make.
    """
    proposed = [m + step * (t - m) for m, t in zip(mix, target)]

    clt_gain = proposed[clt_index] - mix[clt_index]
    if clt_gain > max_clt_increase:
        excess = clt_gain - max_clt_increase
        proposed[clt_index] = mix[clt_index] + max_clt_increase
        others = [i for i in range(len(proposed)) if i != clt_index]
        weight = sum(max(0.0, proposed[i]) for i in others)
        for i in others:
            share = (max(0.0, proposed[i]) / weight) if weight > 0 else 1.0 / len(others)
            proposed[i] += excess * share

    return normalise(proposed)


def mutate(mix: Sequence[float], rng: random.Random, rate: float) -> list[float]:
    """Perturb a mix slightly, so the process can leave any corner it reaches."""
    if rate <= 0:
        return list(mix)
    noise = [rng.gauss(0.0, rate) for _ in mix]
    return normalise([m + n for m, n in zip(mix, noise)])


def mix_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / 2.0


_CALIBRATION_CACHE: dict[tuple[str, int, float], dict[str, Any]] = {}
_CORNER_CACHE: dict[tuple[str, int, float, float], dict[str, tuple[float, float]]] = {}


#: Maps a mix component to the household-block modality that implements it.
_NAME_MAP: dict[str, str] = {
    "cash_universal": "ubi", "cash_targeted": "cash_t",
    "voucher": "voucher", "clt": "clt", "cash_cheap": "cash_cheap",
}


def _modality_size(cal: dict[str, Any], key: str, u: float) -> float:
    """Programme size for one mix component.

    Every component is sized to the common welfare target except
    ``cash_cheap``, which is deliberately sized to CLT's fiscal cost instead.
    """
    if key == "cash_cheap":
        return cal.get("_fiscal_clt", u)
    return cal.get(_NAME_MAP.get(key, key), u)


def _bloc_key(b: Bloc) -> tuple:
    """Identity of a bloc for caching.

    The name alone is not enough: counterfactual sweeps hold the name fixed
    while moving a structural parameter, and keying on the name would then
    return a stale result for a different economy.
    """
    return (b.name, b.fx_share, b.clt_domestic_sourcing, b.m_H, b.land_share,
            b.eps_supply, b.gdp, b.rate, b.c_income_share, b.c_pop_share,
            b.m_F, b.m_G, b.omega0, b.openness, b.debt)


def _cached_calibration(b: Bloc, g: G, u: float) -> dict[str, Any]:
    """Per-bloc calibration, which does not depend on the mix.

    Re-solving it for every mix evaluation dominated the runtime; the sizes
    depend only on the bloc and the welfare target.
    """
    key = (_bloc_key(b), id(g), u)
    if key not in _CALIBRATION_CACHE:
        _CALIBRATION_CACHE[key] = calibrate_hetero(b, g, u, "utilitarian")
    return _CALIBRATION_CACHE[key]


def mix_outcome(
    b: Bloc, g: G, mix: Sequence[float], u: float = 0.05,
    years_since_start: float = 1e9,
    modality_set: Sequence[str] | None = None,
) -> dict[str, float]:
    """Fiscal cost and external leakage of one mix, at a fixed welfare target.

    Each modality is sized to deliver the same constrained-household welfare
    as the reference UBI, then the mix is the weighted combination. Holding
    welfare fixed is what makes the comparison about *how* provision is made
    rather than how much of it there is.
    """
    from dataclasses import replace as _replace

    from .heterogeneous import hetero_household_block

    cal = _cached_calibration(b, g, u)
    base = hetero_household_block(_replace(b, modality="none"), g, b.gdp, 1.0,
                                  b.rate, False, 0.0)

    # Each modality's cost is linear in its weight, so evaluate the four
    # corners once per bloc and combine. This turns an O(mixes) problem into
    # an O(blocs) one.
    modalities = tuple(modality_set or MODALITIES)
    corner_key = (_bloc_key(b), id(g), u, years_since_start, modalities)
    if corner_key not in _CORNER_CACHE:
        corners = {}
        for key in modalities:
            modality = _NAME_MAP[key]
            size = _modality_size(cal, key, u)
            out = hetero_household_block(
                _replace(b, modality=modality), g, b.gdp, 1.0, b.rate, True,
                size, years_since_start=years_since_start)
            corners[key] = (100 * out.fiscal / b.gdp,
                            100 * (out.imp_c - base.imp_c + out.imp_gov) / b.gdp)
        _CORNER_CACHE[corner_key] = corners

    corners = _CORNER_CACHE[corner_key]
    fiscal = sum(w * corners[k][0] for w, k in zip(mix, modalities))
    leak = sum(w * corners[k][1] for w, k in zip(mix, modalities))
    return {"fiscal_pct_gdp": fiscal, "leak_pct_gdp": leak}


def stress_of(
    b: Bloc, g: G, mix: Sequence[float], u: float = 0.05,
    endogenous: bool = True, years: int = 30, discount: float = 0.03,
    modality_set: Sequence[str] | None = None,
) -> float:
    """Stress a bloc carries under its current mix.

    The main series runs the macro block and prices the mix by what it does to
    the bloc's own state: the present value of the extra debt ratio, the extra
    risk premium, and the extra depreciation it causes, relative to making no
    transfer at all. ``fx_share`` enters through the model's own revaluation
    channel — foreign-currency debt is marked up when the currency falls — so
    no external weight is imposed on leakage.

    ``endogenous=False`` restores the earlier reduced form, in which leakage is
    weighted by ``1 + 2 * fx_share``. It is kept for sensitivity analysis
    because that weighting is an assumption about the hierarchy rather than a
    result of it.
    """
    if not endogenous:
        out = mix_outcome(b, g, mix, u, modality_set=modality_set)
        return out["fiscal_pct_gdp"] + (1.0 + 2.0 * b.fx_share) * out["leak_pct_gdp"]

    components = stress_components(b, g, mix, u, years, discount, modality_set)
    return sum(components.values())


def stress_components(
    b: Bloc, g: G, mix: Sequence[float], u: float = 0.05,
    years: int = 30, discount: float = 0.03,
    modality_set: Sequence[str] | None = None,
) -> dict[str, float]:
    """The three present-value terms that make up the endogenous stress.

    Reported separately so the channel carrying any fx effect can be named
    rather than inferred.
    """
    paths = _macro_paths(b, g, mix, u, years, modality_set)
    weights = [(1.0 + discount) ** -t for t in range(len(paths["debt"]))]
    pv = lambda xs: sum(w * x for w, x in zip(weights, xs))
    return {"debt": pv(paths["debt"]),
            "risk_premium": pv(paths["risk_premium"]),
            "depreciation": pv(paths["depreciation"])}


_MACRO_CACHE: dict[tuple, dict[str, list[float]]] = {}


def _size_for_modality(cal: dict[str, Any], modality: str, u: float) -> float:
    """Size of a household-block modality under the common welfare target."""
    if modality == "cash_cheap":
        return cal.get("_fiscal_clt", u)
    return cal.get(modality, u)


def _macro_paths(
    b: Bloc, g: G, mix: Sequence[float], u: float, years: int,
    modality_set: Sequence[str] | None = None,
) -> dict[str, list[float]]:
    """Run the macro block under a mix and return deviations from no transfer.

    Each modality is simulated on its own and the paths are combined by the
    mix weights. The modalities enter the macro block through different flows
    (imports, rent, foreign assets), so their debt and exchange-rate paths
    differ even when their fiscal cost is matched.
    """
    from dataclasses import replace as _replace

    from .core import build, simulate

    cal = _cached_calibration(b, g, u)

    def run(modality: str | None) -> dict[str, list[float]]:
        """Macro path for one modality, cached per (bloc, modality).

        Caching per modality rather than per mix is what makes the endogenous
        stress affordable: a mix is a weighted sum of four corner paths, so
        the expensive simulate() calls are amortised across every mix the
        dynamics ever visits.
        """
        cache_key = (_bloc_key(b), id(g), u, years, modality)
        if cache_key in _MACRO_CACHE:
            return _MACRO_CACHE[cache_key]

        if modality is None:
            blocs = [_replace(b, modality="none", start=999.0)]
        else:
            size = _size_for_modality(cal, modality, u)
            blocs = [_replace(b, modality=modality, size=size, start=1.0)]
        result = simulate(blocs, g)
        history = result["history"][:years]
        path = {
            "debt": [h["d_0"] for h in history],
            "risk_premium": [max(0.0, h["r_0"] - b.rate) for h in history],
            "depreciation": [max(0.0, h["e_0"] - 1.0) for h in history],
        }
        _MACRO_CACHE[cache_key] = path
        return path

    baseline = run(None)
    combined = {k: [0.0] * len(v) for k, v in baseline.items()}
    for weight, key_name in zip(mix, modality_set or MODALITIES):
        if weight <= 1e-12:
            continue
        treated = run(_NAME_MAP[key_name])
        for field in combined:
            for t in range(min(len(combined[field]), len(treated[field]))):
                combined[field][t] += weight * (treated[field][t] - baseline[field][t])
    return combined


@dataclass(frozen=True)
class EvolutionConfig:
    """Settings for one evolutionary run."""

    years: int = 80
    revision_interval: int = 5
    step: float = 0.3
    """How far a bloc moves towards the bloc it imitates."""

    mutation_rate: float = 0.02
    switch_cost: float = 0.1
    """Fiscal cost of changing mix, as a share of GDP, charged for one year."""

    u: float = 0.05
    neutral: bool = False
    """Control run: imitate without regard to stress, so any structure in the
    outcome comes from drift and the build cap alone."""

    initial: str = "invasion"
    """Initial mixes.

    ``invasion`` (main series) starts everyone on targeted cash except a few
    randomly chosen blocs seeded at a CLT share of 0.5. Proportional imitation
    cannot spread a modality nobody holds, so a pure all-cash start measures
    the mutation rate rather than selection. Seeding a minority asks the
    question the phase is about: does CLT spread from where it appears, and
    does structure decide where it takes hold?

    ``all_cash`` and ``random`` are kept as sensitivities.
    """

    n_invaders: int = 3
    invader_clt_share: float = 0.5

    symmetric_invasion: bool = False
    """Seed every non-cash modality, not just CLT.

    The asymmetric default gives CLT a source to be imitated from while the
    control has none, so CLT can win the three-way arena on availability
    alone. Seeding each alternative in its own blocs makes the comparison
    about which one survives rather than which one was planted.
    """

    endogenous_stress: bool = True
    """Main series prices a mix by its macro consequences; see `stress_of`."""

    modality_set: tuple[str, ...] = MODALITIES
    """Which modalities the mix ranges over.

    Set to :data:`CONTROL_MODALITIES` for the three-way control that separates
    cheapness from non-fugitivity.
    """


def _initial_states(
    population: Sequence[Bloc], config: EvolutionConfig, rng: random.Random,
) -> list[BlocState]:
    """Build the starting mixes for one run."""
    modalities = list(config.modality_set)
    clt_index = modalities.index("clt")
    if config.initial == "random":
        return [BlocState(bloc=b, mix=random_mix(rng, modalities), clt_index=clt_index)
                for b in population]

    states = [BlocState(bloc=b, mix=all_cash_mix(modalities), clt_index=clt_index)
              for b in population]
    if config.initial == "invasion":
        seeded = ([m for m in modalities if m != "cash_targeted"]
                  if config.symmetric_invasion else ["clt"])
        chosen = rng.sample(range(len(states)),
                            min(config.n_invaders * len(seeded), len(states)))
        for slot, index in enumerate(chosen):
            modality = seeded[slot % len(seeded)]
            share = config.invader_clt_share
            mix = [0.0] * len(modalities)
            mix[modalities.index(modality)] = share
            mix[modalities.index("cash_targeted")] = 1.0 - share
            states[index].mix = mix
            states[index].is_invader = True
            states[index].seeded_with = modality
    return states


def _imitation_target(
    states: list[BlocState], index: int, rng: random.Random,
) -> BlocState | None:
    """Sample one structurally similar bloc to compare against."""
    others = [s for i, s in enumerate(states) if i != index]
    if not others:
        return None
    me = states[index].bloc
    weights = [1.0 / (0.05 + structural_distance(me, s.bloc)) for s in others]
    total = sum(weights)
    draw = rng.random() * total
    running = 0.0
    for state, weight in zip(others, weights):
        running += weight
        if draw <= running:
            return state
    return others[-1]


def run_evolution(
    population: Sequence[Bloc], g: G, config: EvolutionConfig, seed: int,
) -> dict[str, Any]:
    """Run the imitation process and return the path and final mixes."""
    rng = random.Random(seed)
    states = _initial_states(population, config, rng)

    # Stress is expensive to evaluate, so cache it per (bloc, rounded mix).
    cache: dict[tuple[str, tuple[int, ...]], float] = {}

    def stress(state: BlocState) -> float:
        key = (_bloc_key(state.bloc), tuple(round(m, 2) for m in state.mix))
        if key not in cache:
            cache[key] = stress_of(state.bloc, g, state.mix, config.u,
                                   endogenous=config.endogenous_stress,
                                   modality_set=config.modality_set)
        return cache[key]

    max_clt_step = 1.0
    if g.clt_build_rate is not None:
        # Share of the housing stock the sector can add between revisions.
        max_clt_step = min(1.0, g.clt_build_rate * config.revision_interval * 20.0)

    history: list[dict[str, Any]] = []
    for year in range(0, config.years + 1, config.revision_interval):
        for state in states:
            state.stress = stress(state)
        adopters = sum(1 for s in states if s.clt_share > 0.1)
        history.append({
            "year": year,
            "mean_clt": sum(s.clt_share for s in states) / len(states),
            "mean_stress": sum(s.stress for s in states) / len(states),
            "adopter_share": adopters / len(states),
        })
        if year == config.years:
            break

        for i, state in enumerate(states):
            other = _imitation_target(states, i, rng)
            if other is None:
                continue
            if config.neutral:
                # Control: copy regardless of whether it helps.
                probability = 0.5
            else:
                gap = state.stress - other.stress
                probability = max(0.0, min(1.0, gap / 5.0))
            if rng.random() < probability:
                before = list(state.mix)
                state.mix = move_towards(state.mix, other.mix, config.step,
                                         max_clt_step, state.clt_index)
                if mix_distance(before, state.mix) > 1e-6:
                    state.switch_paid += config.switch_cost
            state.mix = mutate(state.mix, rng, config.mutation_rate)

    return {
        "seed": seed,
        "neutral": config.neutral,
        "initial": config.initial,
        "history": history,
        "final": [
            {"name": s.bloc.name, "group": s.bloc.name[0], "fx_share": s.bloc.fx_share,
             "reserve": s.bloc.reserve, "clt_share": s.clt_share,
             "mix": list(s.mix), "stress": s.stress, "switch_paid": s.switch_paid,
             "is_invader": s.is_invader, "seeded_with": s.seeded_with,
             "mix_labels": list(config.modality_set)}
            for s in states
        ],
    }
