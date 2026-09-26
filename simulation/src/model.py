"""Two-sector CES general equilibrium model with a fugacity parameter.

The model is written as pure functions: every routine maps inputs to outputs
with no hidden state. The equilibrium solve is the only place numerics enter.

Equilibrium system (3 unknowns, 3 residuals)
--------------------------------------------
Unknowns are ``(K_H, L_H, p)`` with sector L's price as numeraire. Endowment
constraints are substituted, not imposed: ``K_L = K - K_H``, ``L_L = L - L_H``.

    F1 = p*MPK_H/mu_H - MPK_L/mu_L          capital factor-price equalization
    F2 = p*MPL_H/mu_H - MPL_L/mu_L          labor factor-price equalization
    F3 = p*Y_H/(p*Y_H + Y_L) - s_H(phi)     administered *value* share

Wedges
------
Two wedges compound. The markup ``mu_i`` is a product-market wedge and applies
symmetrically to both factors. The bargaining term ``(1 - phi_a*beta)`` is a
labor-market wedge and applies to labor only::

    w_i = (1 - phi_a*beta) * p_i * MPL_i / mu_i
    r_i =                    p_i * MPK_i / mu_i

Capital is the residual claimant: it collects pure profit ``(1 - 1/mu_i)`` plus
the bargaining rent extracted from labor, so ``theta_K = 1 - theta_L`` does not
hold and is never imposed.

Because ``(1 - phi_a*beta)`` is a scalar common to both sectors it cancels
identically from F2. It is therefore excluded from the residuals and applied
only when computing the labor share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import brentq, least_squares

from .config import (
    CalibrationConfig,
    ModelParams,
    SolverConfig,
)

# Below this distance from sigma = 1, use the Cobb-Douglas closed form.
_CD_TOLERANCE = 1e-6


class EquilibriumError(RuntimeError):
    """Raised when the equilibrium system cannot be solved to tolerance."""


@dataclass(frozen=True)
class ChannelPhis:
    """Fugacity applied to each channel independently.

    Freezing a channel means passing the baseline ``phi`` for that channel
    only; the equilibrium quantities are always re-solved.
    """

    bargaining: float
    technology: float
    concentration: float
    markup: float

    @classmethod
    def uniform(cls, phi: float) -> "ChannelPhis":
        """All four channels at the same fugacity."""
        return cls(phi, phi, phi, phi)

    @classmethod
    def selective(cls, active: Sequence[str], phi: float, baseline: float) -> "ChannelPhis":
        """`phi` for channels named in `active`; `baseline` for the rest."""
        pick = lambda name: phi if name in active else baseline
        return cls(
            bargaining=pick("bargaining"),
            technology=pick("technology"),
            concentration=pick("concentration"),
            markup=pick("markup"),
        )


@dataclass(frozen=True)
class Equilibrium:
    """Solved allocation and the derived labor share."""

    K_h: float
    L_h: float
    K_l: float
    L_l: float
    p: float
    Y_h: float
    Y_l: float
    mu_h: float
    mu_l: float
    w_h: float
    w_l: float
    alpha_h: float
    alpha_l: float
    s_h: float
    theta_L: float
    max_residual: float


def ces_output(A: float, alpha: float, K: float, L: float, rho: float) -> tuple[float, float, float]:
    """CES output and marginal products.

    Returns:
        ``(Y, MPK, MPL)``. Branches to the Cobb-Douglas closed form near
        ``rho == 0``, where the CES expression is singular.
    """
    if abs(rho) < _CD_TOLERANCE:
        Y = A * K**alpha * L ** (1.0 - alpha)
        return Y, alpha * Y / K, (1.0 - alpha) * Y / L

    bracket = alpha * K**rho + (1.0 - alpha) * L**rho
    Y = A * bracket ** (1.0 / rho)
    common = A * bracket ** (1.0 / rho - 1.0)
    return Y, common * alpha * K ** (rho - 1.0), common * (1.0 - alpha) * L ** (rho - 1.0)


def capital_intensity(alpha0: float, gamma: float, phi_tech: float) -> float:
    """Technology channel (b): capital intensity rises with fugacity."""
    return alpha0 + gamma * phi_tech


def high_sector_share(s_h0: float, delta: float, phi_conc: float) -> float:
    """Concentration channel (c): sector H's value share rises with fugacity."""
    return s_h0 + delta * phi_conc


def markup(eta: float, phi_markup: float, k_over_l: float, aggregate_k_over_l: float) -> float:
    """Markup channel (d), specified as a ratio to the aggregate K/L.

    The specification writes ``1 + eta*phi*(K_i/L_i)``, which depends on the
    *level* of K/L and so is not invariant to the endowment normalization.
    Dividing by the aggregate K/L makes it scale-free; at ``K = L`` the two
    forms coincide numerically.
    """
    return max(1.0, 1.0 + eta * phi_markup * (k_over_l / aggregate_k_over_l))


def wage_wedge(beta: float, phi_barg: float) -> float:
    """Bargaining channel (a): labor is paid below its markup-adjusted MPL."""
    return 1.0 - phi_barg * beta


def validate_preconditions(params: ModelParams, phis: ChannelPhis) -> None:
    """Check that the parameterization yields a feasible equilibrium problem."""
    s_h = high_sector_share(params.s_h0, params.delta, phis.concentration)
    if not 0.0 < s_h < 0.95:
        raise EquilibriumError(
            f"s_H(phi)={s_h:.4f} outside (0, 0.95); the value-share constraint is infeasible"
        )

    alpha_h = capital_intensity(params.a_h0, params.gamma, phis.technology)
    alpha_l = capital_intensity(params.a_l0, params.gamma, phis.technology)
    for name, alpha in (("alpha_H", alpha_h), ("alpha_L", alpha_l)):
        if not 0.0 < alpha < 1.0:
            raise EquilibriumError(f"{name}={alpha:.4f} outside (0, 1)")

    wedge = wage_wedge(params.beta, phis.bargaining)
    if not 0.0 < wedge <= 1.0:
        raise EquilibriumError(f"wage wedge (1-phi*beta)={wedge:.4f} outside (0, 1]")


def _unpack(u: np.ndarray, params: ModelParams) -> tuple[float, float, float, float, float]:
    """Map unconstrained variables to ``(K_H, L_H, K_L, L_L, p)``.

    The sigmoid keeps the sectoral allocations inside ``(0, endowment)`` and
    the exponential keeps the relative price positive, so the solver never
    steps outside the feasible region.
    """
    K_h = params.K_total / (1.0 + np.exp(-u[0]))
    L_h = params.L_total / (1.0 + np.exp(-u[1]))
    return K_h, L_h, params.K_total - K_h, params.L_total - L_h, float(np.exp(u[2]))


def _pack(K_h: float, L_h: float, p: float, params: ModelParams) -> np.ndarray:
    """Inverse of :func:`_unpack`."""
    frac_k = np.clip(K_h / params.K_total, 1e-12, 1.0 - 1e-12)
    frac_l = np.clip(L_h / params.L_total, 1e-12, 1.0 - 1e-12)
    return np.array(
        [np.log(frac_k / (1.0 - frac_k)), np.log(frac_l / (1.0 - frac_l)), np.log(max(p, 1e-12))]
    )


def _residual_fn(params: ModelParams, phis: ChannelPhis) -> Callable[[np.ndarray], list[float]]:
    """Build the residual vector for the given parameters and channel fugacities."""
    rho = params.rho
    alpha_h = capital_intensity(params.a_h0, params.gamma, phis.technology)
    alpha_l = capital_intensity(params.a_l0, params.gamma, phis.technology)
    s_h = high_sector_share(params.s_h0, params.delta, phis.concentration)
    agg_kl = params.K_total / params.L_total

    def residuals(u: np.ndarray) -> list[float]:
        K_h, L_h, K_l, L_l, p = _unpack(u, params)
        Y_h, mpk_h, mpl_h = ces_output(params.A_h, alpha_h, K_h, L_h, rho)
        Y_l, mpk_l, mpl_l = ces_output(params.A_l, alpha_l, K_l, L_l, rho)
        mu_h = markup(params.eta, phis.markup, K_h / L_h, agg_kl)
        mu_l = markup(params.eta, phis.markup, K_l / L_l, agg_kl)
        return [
            p * mpk_h / mu_h - mpk_l / mu_l,
            p * mpl_h / mu_h - mpl_l / mu_l,
            p * Y_h / (p * Y_h + Y_l) - s_h,
        ]

    return residuals


def solve_equilibrium(
    params: ModelParams,
    phis: ChannelPhis,
    solver: SolverConfig | None = None,
    guess: np.ndarray | None = None,
) -> Equilibrium:
    """Solve the equilibrium system and compute the aggregate labor share.

    Args:
        params: Structural parameters.
        phis: Per-channel fugacity.
        solver: Numerical settings; defaults to :class:`SolverConfig`.
        guess: Optional warm start in unconstrained space (see `_pack`).

    Raises:
        EquilibriumError: If preconditions fail or the residual tolerance is
            not met, or if the resulting labor share leaves ``(0, 1)``.
    """
    solver = solver or SolverConfig()
    validate_preconditions(params, phis)

    residuals = _residual_fn(params, phis)
    if guess is None:
        guess = _pack(*solver.cold_start, params)

    solution = least_squares(
        residuals,
        guess,
        method="trf",
        xtol=solver.tol,
        ftol=solver.tol,
        gtol=solver.tol,
        max_nfev=solver.max_nfev,
    )

    max_residual = float(np.max(np.abs(residuals(solution.x))))
    if max_residual > solver.residual_tol:
        raise EquilibriumError(
            f"equilibrium did not converge: max|F|={max_residual:.3e} > {solver.residual_tol:.1e}"
        )

    return _build_equilibrium(solution.x, params, phis, max_residual)


def _build_equilibrium(
    u: np.ndarray, params: ModelParams, phis: ChannelPhis, max_residual: float
) -> Equilibrium:
    """Assemble the solved allocation and derive the labor share."""
    K_h, L_h, K_l, L_l, p = _unpack(u, params)
    alpha_h = capital_intensity(params.a_h0, params.gamma, phis.technology)
    alpha_l = capital_intensity(params.a_l0, params.gamma, phis.technology)
    s_h = high_sector_share(params.s_h0, params.delta, phis.concentration)
    agg_kl = params.K_total / params.L_total

    Y_h, _, mpl_h = ces_output(params.A_h, alpha_h, K_h, L_h, params.rho)
    Y_l, _, mpl_l = ces_output(params.A_l, alpha_l, K_l, L_l, params.rho)
    mu_h = markup(params.eta, phis.markup, K_h / L_h, agg_kl)
    mu_l = markup(params.eta, phis.markup, K_l / L_l, agg_kl)

    # The bargaining wedge enters here only — it cancels from the residuals.
    wedge = wage_wedge(params.beta, phis.bargaining)
    w_h = wedge * p * mpl_h / mu_h
    w_l = wedge * mpl_l / mu_l

    theta_L = s_h * (w_h * L_h) / (p * Y_h) + (1.0 - s_h) * (w_l * L_l) / Y_l
    if not 0.0 < theta_L < 1.0:
        raise EquilibriumError(f"theta_L={theta_L:.4f} outside (0, 1)")

    return Equilibrium(
        K_h=K_h, L_h=L_h, K_l=K_l, L_l=L_l, p=p,
        Y_h=Y_h, Y_l=Y_l, mu_h=mu_h, mu_l=mu_l,
        w_h=w_h, w_l=w_l,
        alpha_h=alpha_h, alpha_l=alpha_l, s_h=s_h,
        theta_L=theta_L, max_residual=max_residual,
    )


def labor_share(params: ModelParams, phis: ChannelPhis, guess: np.ndarray | None = None) -> float:
    """Convenience wrapper returning only the aggregate labor share."""
    return solve_equilibrium(params, phis, guess=guess).theta_L


def sweep_phi(
    params: ModelParams,
    phi_values: Sequence[float],
    active: Sequence[str] | None = None,
    baseline: float | None = None,
    solver: SolverConfig | None = None,
) -> list[Equilibrium]:
    """Solve across a fugacity path using continuation.

    Each solve is seeded with the previous solution. On failure the seed falls
    back to a cold start, then to a bisected continuation step.

    Args:
        active: Channel names that vary with phi. ``None`` means all four.
        baseline: Fugacity held by frozen channels. Required if `active` is given.
    """
    solver = solver or SolverConfig()
    results: list[Equilibrium] = []
    warm: np.ndarray | None = None

    for phi in phi_values:
        phis = (
            ChannelPhis.uniform(phi)
            if active is None
            else ChannelPhis.selective(active, phi, baseline if baseline is not None else phi)
        )
        eq = _solve_with_fallback(params, phis, solver, warm, phi, active, baseline)
        results.append(eq)
        warm = _pack(eq.K_h, eq.L_h, eq.p, params)

    return results


def _solve_with_fallback(
    params: ModelParams,
    phis: ChannelPhis,
    solver: SolverConfig,
    warm: np.ndarray | None,
    phi: float,
    active: Sequence[str] | None,
    baseline: float | None,
) -> Equilibrium:
    """Try warm start, then cold start, then a bisected continuation step."""
    for guess in (warm, None):
        try:
            return solve_equilibrium(params, phis, solver=solver, guess=guess)
        except EquilibriumError:
            continue

    # Final fallback: approach this phi through an intermediate step.
    if warm is not None:
        midpoint = 0.5 * phi
        mid_phis = (
            ChannelPhis.uniform(midpoint)
            if active is None
            else ChannelPhis.selective(active, midpoint, baseline if baseline is not None else midpoint)
        )
        mid = solve_equilibrium(params, mid_phis, solver=solver, guess=warm)
        return solve_equilibrium(
            params, phis, solver=solver, guess=_pack(mid.K_h, mid.L_h, mid.p, params)
        )

    raise EquilibriumError(f"all fallbacks exhausted at phi={phi}")


def calibrate_kappa(
    params: ModelParams,
    calib: CalibrationConfig | None = None,
    solver: SolverConfig | None = None,
) -> float:
    """Find the common shift in capital intensity that hits the target share.

    Total factor productivity cannot serve this role: under constant-returns
    CES, ``A_i`` cancels exactly out of the labor share. The bargaining
    parameter cannot either (it would require beta < 0). A common shift to
    ``alpha_H^0`` and ``alpha_L^0`` moves the level while preserving the
    ``alpha_H - alpha_L`` gap that identifies the sectoral spread.
    """
    calib = calib or CalibrationConfig()
    phis = ChannelPhis.uniform(calib.phi_at_target)

    def gap(kappa: float) -> float:
        return labor_share(params.shifted(kappa), phis) - calib.target_theta

    lo, hi = calib.kappa_bracket
    return float(brentq(gap, lo, hi, xtol=calib.tol))


def calibrated_params(
    params: ModelParams,
    calib: CalibrationConfig | None = None,
    solver: SolverConfig | None = None,
) -> tuple[ModelParams, float]:
    """Return the calibrated parameters alongside the solved shift."""
    kappa = calibrate_kappa(params, calib, solver)
    return params.shifted(kappa), kappa
