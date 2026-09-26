"""Experiments 1-3 plus the sensitivity analysis.

Run as a module to regenerate every figure and ``output/results.json``::

    uv run python -m src.experiments
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import (
    CHANNELS,
    PHI_HIGH,
    PHI_LOW,
    SCENARIOS,
    CalibrationConfig,
    ModelParams,
    ScenarioConfig,
    SensitivityConfig,
    WageConfig,
    scaled_params,
)
from .decomposition import Decomposition, decompose_all
from .model import ChannelPhis, EquilibriumError, calibrated_params, labor_share, sweep_phi

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
PHI_GRID = np.linspace(PHI_LOW, PHI_HIGH, 71)


@dataclass(frozen=True)
class ScenarioResult:
    """Everything computed for one parameter regime."""

    scenario: ScenarioConfig
    params: ModelParams
    kappa: float
    phi_grid: list[float]
    theta_path: list[float]
    channel_paths: dict[str, list[float]]
    joint_path: list[float]
    decompositions: list[Decomposition]

    @property
    def theta_start(self) -> float:
        return self.theta_path[0]

    @property
    def theta_end(self) -> float:
        return self.theta_path[-1]

    @property
    def total_decline(self) -> float:
        return self.theta_start - self.theta_end


def build_scenario(scenario: ScenarioConfig, base: ModelParams | None = None) -> ScenarioResult:
    """Calibrate a regime and run experiments 1 and 2 within it."""
    base = base or ModelParams()
    params, kappa = calibrated_params(scaled_params(base, scenario))

    theta_path = [eq.theta_L for eq in sweep_phi(params, PHI_GRID)]

    # Experiment 1: each channel alone, others frozen at the baseline fugacity.
    channel_paths = {
        channel: [eq.theta_L for eq in sweep_phi(params, PHI_GRID, active=[channel], baseline=PHI_LOW)]
        for channel in CHANNELS
    }
    joint_path = [
        eq.theta_L for eq in sweep_phi(params, PHI_GRID, active=CHANNELS, baseline=PHI_LOW)
    ]

    return ScenarioResult(
        scenario=scenario,
        params=params,
        kappa=kappa,
        phi_grid=[float(p) for p in PHI_GRID],
        theta_path=theta_path,
        channel_paths=channel_paths,
        joint_path=joint_path,
        decompositions=decompose_all(params, PHI_LOW, PHI_HIGH),
    )


# --- experiment 3: wage representations -------------------------------------

def linear_phi_path(cfg: WageConfig) -> np.ndarray:
    """Fugacity rising linearly from PHI_LOW to PHI_HIGH."""
    return np.linspace(PHI_LOW, PHI_HIGH, cfg.years + 1)


def logistic_phi_path(cfg: WageConfig, sens: SensitivityConfig) -> np.ndarray:
    """S-shaped fugacity path, with the steep transition in the 1980s."""
    years = np.arange(cfg.years + 1) + cfg.start_year
    raw = 1.0 / (1.0 + np.exp(-sens.logistic_steepness * (years - sens.logistic_midpoint_year)))
    # Rescale so the endpoints match the linear path exactly.
    return PHI_LOW + (PHI_HIGH - PHI_LOW) * (raw - raw[0]) / (raw[-1] - raw[0])


@dataclass(frozen=True)
class WageSeries:
    """Three representations of the same wage path, indexed to 100 at t=0."""

    label: str
    years: list[int]
    phi_path: list[float]
    nominal: list[float]
    cpi_real: list[float]
    gold: list[float]


def wage_experiment(
    params: ModelParams, phi_path: np.ndarray, cfg: WageConfig, label: str
) -> WageSeries:
    """Compute nominal, CPI-real, and gold-denominated wage indices.

    A pure post-processing layer over the equilibrium wage: the currency
    dilution and gold price never enter the equilibrium solve.
    """
    equilibria = sweep_phi(params, [float(p) for p in phi_path])
    # Aggregate wage: labor-weighted across sectors.
    share_wage = np.array(
        [(eq.w_h * eq.L_h + eq.w_l * eq.L_l) / (eq.L_h + eq.L_l) for eq in equilibria]
    )

    t = np.arange(len(phi_path))
    # Productivity growth offsets the share decline; without it the real wage
    # would carry the full fall, which the data do not show.
    real_wage = share_wage * (1.0 + cfg.productivity_growth) ** t
    nominal = real_wage * (1.0 + cfg.pi) ** t
    gold_price = cfg.gold_price_0 * (1.0 + cfg.pi + cfg.epsilon) ** t
    gold = nominal / gold_price

    index = lambda a: list(100.0 * a / a[0])
    return WageSeries(
        label=label,
        years=[int(cfg.start_year + i) for i in t],
        phi_path=[float(p) for p in phi_path],
        nominal=index(nominal),
        cpi_real=index(real_wage),
        gold=index(gold),
    )


# --- sensitivity analysis ---------------------------------------------------

@dataclass(frozen=True)
class SensitivityResult:
    """Sensitivity of the labor-share path to the structural parameters."""

    sigma_paths: dict[str, list[float]]
    perturbation_paths: dict[str, list[float]]
    phi_grid: list[float]
    sigma_note: str


def run_sensitivity(
    base: ModelParams, scenario: ScenarioConfig, cfg: SensitivityConfig
) -> SensitivityResult:
    """Vary sigma, then perturb gamma/delta/eta by +/-50%."""
    scaled = scaled_params(base, scenario)

    sigma_paths: dict[str, list[float]] = {}
    for sigma in cfg.sigma_values:
        params, _ = calibrated_params(replace(scaled, sigma=sigma))
        sigma_paths[f"{sigma:g}"] = [eq.theta_L for eq in sweep_phi(params, PHI_GRID)]

    perturbation_paths: dict[str, list[float]] = {}
    for name in ("gamma", "delta", "eta"):
        for factor, tag in ((1.0 - cfg.perturbation, "-50%"), (1.0 + cfg.perturbation, "+50%")):
            candidate = replace(scaled, **{name: getattr(scaled, name) * factor})
            try:
                params, _ = calibrated_params(candidate)
                perturbation_paths[f"{name} {tag}"] = [
                    eq.theta_L for eq in sweep_phi(params, PHI_GRID)
                ]
            except (EquilibriumError, ValueError) as exc:
                print(f"  [skip] {name} {tag}: {exc}")

    # Each sigma is re-calibrated to the same target, so the paths coincide at
    # phi_low by construction; the informative spread is at the far end.
    spread = max(p[-1] for p in sigma_paths.values()) - min(p[-1] for p in sigma_paths.values())
    note = (
        f"Across sigma in {cfg.sigma_values}, theta_L(phi=0.8) varies by only {spread:.4f} "
        f"(all paths are calibrated to the same theta_L at phi=0.1, so they coincide there "
        f"by construction). The elasticity of substitution is nearly irrelevant here: the "
        "four wedges act directly on alpha, mu, and the wage, not through substitution. The "
        "flat sigma lines are the expected result, not a bug."
    )
    return SensitivityResult(
        sigma_paths=sigma_paths,
        perturbation_paths=perturbation_paths,
        phi_grid=[float(p) for p in PHI_GRID],
        sigma_note=note,
    )


# --- serialization ----------------------------------------------------------

CAPTIONS = {
    "fig1": (
        "Aggregate labor share against fugacity. The illustrative regime uses the "
        "specification's sensitivities as written; the empirical regime scales them so the "
        "total decline matches the ~8pp US decline. Both are calibrated to theta_L = 0.66 "
        "at phi = 0.1."
    ),
    "fig2": (
        "Channel decomposition. Each thin line switches on one channel while the other "
        "three are frozen at phi = 0.1; the heavy line moves all four together. The "
        "individual lines lie above the joint line, which is the over-counting the "
        "Grossman-Oberfield critique describes."
    ),
    "fig3": (
        "Over-counting is an artifact of the decomposition convention. On identical "
        "parameters, switching channels on one at a time over-counts, leaving one out "
        "under-counts, and the Shapley decomposition is exact by construction. Since each "
        "of the five literatures estimates its own channel under the first convention, the "
        "convention is itself a source of the over-counting they diagnose."
    ),
    "fig4": (
        "Three representations of one wage path, each indexed to 100 in 1970 and drawn on a "
        "single logarithmic axis. Nominal wages rise, CPI-deflated wages are roughly flat, "
        "and gold-denominated wages collapse. A single indexed axis is used because the "
        "three series differ by orders of magnitude; a dual axis would imply a spurious "
        "correspondence between scales."
    ),
    "fig5": (
        "Sensitivity analysis. The elasticity of substitution barely moves the result: the "
        "sigma lines nearly coincide. The channel sensitivities gamma, delta, and eta "
        "dominate, which locates the model's leverage in the wedges rather than in "
        "substitution."
    ),
}


def _decomposition_json(d: Decomposition) -> dict:
    return {
        "convention": d.convention,
        "contributions": {k: float(v) for k, v in d.contributions.items()},
        "sum_of_individual": float(d.total),
        "joint_effect": float(d.joint),
        "ratio": float(d.ratio),
    }


def _scenario_json(r: ScenarioResult) -> dict:
    return {
        "label": r.scenario.label,
        "sensitivity_scale": r.scenario.sensitivity_scale,
        "calibration_kappa": float(r.kappa),
        "parameters": {k: float(v) for k, v in asdict(r.params).items()},
        "theta_at_phi_low": float(r.theta_start),
        "theta_at_phi_high": float(r.theta_end),
        "total_decline": float(r.total_decline),
        "decline_as_pct_of_initial": float(100.0 * r.total_decline / r.theta_start),
        "phi_grid": r.phi_grid,
        "theta_path": [float(v) for v in r.theta_path],
        "channel_paths": {k: [float(x) for x in v] for k, v in r.channel_paths.items()},
        "joint_path": [float(v) for v in r.joint_path],
        "decompositions": [_decomposition_json(d) for d in r.decompositions],
    }


def _wage_json(w: WageSeries) -> dict:
    return {
        "label": w.label,
        "years": w.years,
        "phi_path": w.phi_path,
        "nominal_index": [float(v) for v in w.nominal],
        "cpi_real_index": [float(v) for v in w.cpi_real],
        "gold_index": [float(v) for v in w.gold],
    }


def run_all(output_dir: Path = OUTPUT_DIR) -> dict:
    """Run every experiment, write the figures, and return the results payload."""
    from . import plotting

    output_dir.mkdir(parents=True, exist_ok=True)
    base = ModelParams()

    print("Calibrating scenarios and running experiments 1-2...")
    scenarios = [build_scenario(s, base) for s in SCENARIOS]
    for r in scenarios:
        print(
            f"  {r.scenario.name:13s} kappa={r.kappa:+.4f}  "
            f"theta {r.theta_start:.4f} -> {r.theta_end:.4f}  "
            f"decline {r.total_decline:.4f} ({100*r.total_decline/r.theta_start:.1f}%)"
        )
        for d in r.decompositions:
            print(f"      {d.convention:15s} ratio = {d.ratio:.3f}x")

    print("Running experiment 3 (wage representations)...")
    wage_cfg, sens_cfg = WageConfig(), SensitivityConfig()
    main = scenarios[0]
    wages = [
        wage_experiment(main.params, linear_phi_path(wage_cfg), wage_cfg, "Linear $\\varphi(t)$"),
        wage_experiment(
            main.params, logistic_phi_path(wage_cfg, sens_cfg), wage_cfg, "S-shaped $\\varphi(t)$"
        ),
    ]

    print("Running sensitivity analysis...")
    sensitivity = run_sensitivity(base, SCENARIOS[0], sens_cfg)
    print(f"  {sensitivity.sigma_note}")

    print("Generating figures...")
    plotting.make_all_figures(scenarios, wages, sensitivity, output_dir)

    results = {
        "description": (
            "Numerical demonstration: fugacity (phi) as a single fundamental cause "
            "generating four proximate channels of labor-share decline."
        ),
        "phi_low": PHI_LOW,
        "phi_high": PHI_HIGH,
        "calibration_target": CalibrationConfig().target_theta,
        "scenarios": {r.scenario.name: _scenario_json(r) for r in scenarios},
        "wage_experiment": [_wage_json(w) for w in wages],
        "wage_config": {k: float(v) for k, v in asdict(wage_cfg).items()},
        "sensitivity": {
            "phi_grid": sensitivity.phi_grid,
            "sigma_paths": {k: [float(x) for x in v] for k, v in sensitivity.sigma_paths.items()},
            "perturbation_paths": {
                k: [float(x) for x in v] for k, v in sensitivity.perturbation_paths.items()
            },
            "sigma_note": sensitivity.sigma_note,
        },
        "figure_captions": CAPTIONS,
        "key_findings": _key_findings(scenarios),
    }

    path = output_dir / "results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    return results


def _key_findings(scenarios: Sequence[ScenarioResult]) -> list[str]:
    """The findings a reader should take away, stated plainly."""
    main = scenarios[0]
    emp = scenarios[1]
    one, leave, shapley = main.decompositions
    return [
        (
            "Total factor productivity cannot calibrate this model: under constant-returns "
            "CES, A_i cancels exactly out of the labor share. Calibration runs through a "
            "common shift kappa in capital intensity instead "
            f"(kappa = {main.kappa:.4f}), which preserves the alpha_H - alpha_L gap."
        ),
        (
            f"Over-counting is convention-dependent. On identical parameters: "
            f"one-at-a-time = {one.ratio:.3f}x (over-counts), "
            f"leave-one-out = {leave.ratio:.3f}x (under-counts), "
            f"Shapley = {shapley.ratio:.3f}x (exact). The labor share is multiplicatively "
            "separable in the four channels, so it is additive in logs and the interaction "
            "terms are second-order."
        ),
        (
            f"The specification's parameters imply a decline far larger than observed: "
            f"{main.theta_start:.3f} -> {main.theta_end:.3f} "
            f"({100*main.total_decline/main.theta_start:.0f}% of the initial share) against "
            f"roughly 8pp in US data. Scaling the sensitivities to match the data "
            f"({emp.theta_start:.3f} -> {emp.theta_end:.3f}) drops over-counting to "
            f"{emp.decompositions[0].ratio:.3f}x."
        ),
        (
            "Realistic decline magnitude and visible over-counting are in direct conflict "
            "in this model; they cannot be obtained simultaneously."
        ),
        (
            f"Even at full specification parameters, over-counting reaches only "
            f"{one.ratio:.2f}x, short of the 3-4x that Grossman & Oberfield report for the "
            "literature. This is reported as measured rather than tuned toward the target."
        ),
    ]


if __name__ == "__main__":
    run_all()
