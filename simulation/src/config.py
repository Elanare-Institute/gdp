"""Configuration for the fugacity model.

All parameters live here as frozen dataclasses; nothing is hard-coded in the
model, experiment, or plotting layers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

# Endpoints of the fugacity sweep: 1970s (Bretton Woods, high unionization)
# through 2020s (free capital mobility, low unionization).
PHI_LOW: Final[float] = 0.1
PHI_HIGH: Final[float] = 0.8

# The four channels through which fugacity acts.
CHANNELS: Final[tuple[str, ...]] = ("bargaining", "technology", "concentration", "markup")

CHANNEL_LABELS: Final[dict[str, str]] = {
    "bargaining": "Bargaining (a)",
    "technology": "Technology (b)",
    "concentration": "Concentration (c)",
    "markup": "Markup (d)",
}


@dataclass(frozen=True)
class ModelParams:
    """Structural parameters of the two-sector CES general equilibrium model.

    Notes:
        `a_h0`/`a_l0` are the *pre-calibration* capital intensities. The
        calibration routine shifts both by a common kappa, preserving the
        `a_h0 - a_l0` gap that identifies the sectoral spread.

        `A_h`/`A_l` are NOT free parameters: under constant-returns CES the
        total factor productivity terms cancel exactly out of the labor share.
        They are fixed at 1.0 and retained only for explicitness.
    """

    sigma: float = 0.7  # elasticity of substitution; sigma < 1 => complements
    a_h0: float = 0.5  # initial capital intensity, sector H
    a_l0: float = 0.3  # initial capital intensity, sector L
    s_h0: float = 0.3  # initial output share of sector H
    beta: float = 0.5  # bargaining-power sensitivity
    gamma: float = 0.15  # technology-choice sensitivity
    delta: float = 0.3  # concentration sensitivity
    eta: float = 0.2  # markup sensitivity
    A_h: float = 1.0  # TFP, sector H (unidentified — see class docstring)
    A_l: float = 1.0  # TFP, sector L (unidentified)
    K_total: float = 1.0  # capital endowment (normalization)
    L_total: float = 1.0  # labor endowment (normalization)

    @property
    def rho(self) -> float:
        """CES substitution parameter. Zero at the Cobb-Douglas limit."""
        return 1.0 - 1.0 / self.sigma

    def shifted(self, kappa: float) -> "ModelParams":
        """Return a copy with both capital intensities shifted by `kappa`."""
        return replace(self, a_h0=self.a_h0 + kappa, a_l0=self.a_l0 + kappa)


@dataclass(frozen=True)
class SolverConfig:
    """Numerical settings for the equilibrium solve."""

    tol: float = 1e-15  # xtol/ftol/gtol handed to least_squares
    residual_tol: float = 1e-9  # acceptance threshold on max|F|
    max_nfev: int = 2000
    # Cold-start guess in raw (untransformed) space: (K_H, L_H, p).
    cold_start: tuple[float, float, float] = (0.35, 0.25, 1.0)


@dataclass(frozen=True)
class CalibrationConfig:
    """Target for the initial labor share."""

    target_theta: float = 0.66  # US labor share, ~1970s
    phi_at_target: float = PHI_LOW
    kappa_bracket: tuple[float, float] = (-0.28, 0.28)
    tol: float = 1e-12


@dataclass(frozen=True)
class WageConfig:
    """Parameters for experiment 3 (nominal / CPI-real / gold-denominated wages)."""

    years: int = 50  # 1970 -> 2020
    start_year: int = 1970
    pi: float = 0.05  # annual currency dilution rate
    epsilon: float = 0.03  # gold's lead over dilution (asset front-running)
    gold_price_0: float = 1.0
    productivity_growth: float = 0.015
    """Annual labor-productivity growth.

    Without this term the model's real wage carries the entire labor-share
    decline, and the CPI-deflated series falls by more than half — which is not
    what the data show. In the US, real wages were roughly flat because
    productivity growth largely offset the falling share. This parameter
    supplies that offset; it enters only the wage post-processing layer and
    never the equilibrium solve. Set it to 0.0 to see the unoffset decline.
    """


@dataclass(frozen=True)
class SensitivityConfig:
    """Ranges for the sensitivity analysis."""

    sigma_values: tuple[float, ...] = (0.5, 0.7, 0.9, 1.0)
    perturbation: float = 0.5  # +/-50% on gamma, delta, eta
    # Logistic phi(t) path: midpoint in the 1980s, steep transition.
    logistic_midpoint_year: float = 1985.0
    logistic_steepness: float = 0.18


@dataclass(frozen=True)
class ScenarioConfig:
    """A named parameter regime.

    Two regimes are reported. `illustrative` uses the specification's
    sensitivities as written and amplifies the mechanism; `empirical` scales
    them down so the total decline matches the ~8pp US decline. The two cannot
    be reconciled inside this model — see README.
    """

    name: str
    label: str
    sensitivity_scale: float


ILLUSTRATIVE: Final[ScenarioConfig] = ScenarioConfig(
    name="illustrative",
    label="Illustrative (spec parameters)",
    sensitivity_scale=1.0,
)

EMPIRICAL: Final[ScenarioConfig] = ScenarioConfig(
    name="empirical",
    label="Empirical (calibrated to ~8pp decline)",
    sensitivity_scale=0.178,
)

SCENARIOS: Final[tuple[ScenarioConfig, ...]] = (ILLUSTRATIVE, EMPIRICAL)


def scaled_params(base: ModelParams, scenario: ScenarioConfig) -> ModelParams:
    """Apply a scenario's sensitivity scaling to the four channel parameters."""
    s = scenario.sensitivity_scale
    return replace(
        base,
        beta=base.beta * s,
        gamma=base.gamma * s,
        delta=base.delta * s,
        eta=base.eta * s,
    )


# Colorblind-safe categorical palette (validated: all checks pass in light mode).
PALETTE: Final[dict[str, str]] = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
}

CHANNEL_COLORS: Final[dict[str, str]] = {
    "bargaining": PALETTE["orange"],
    "technology": PALETTE["aqua"],
    "concentration": PALETTE["yellow"],
    "markup": PALETTE["magenta"],
}

INK: Final[dict[str, str]] = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}

FIGURE_DPI: Final[int] = 300
