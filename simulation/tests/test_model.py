"""Consistency tests for the fugacity model.

These test the *specification*, not the implementation: each case encodes a
property the model must satisfy for its economic claims to hold.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from src.config import (
    CHANNELS,
    PHI_HIGH,
    PHI_LOW,
    CalibrationConfig,
    ModelParams,
    SolverConfig,
)
from src.decomposition import decompose_all, decompose_shapley, joint_effect
from src.model import (
    ChannelPhis,
    EquilibriumError,
    calibrated_params,
    ces_output,
    labor_share,
    solve_equilibrium,
    sweep_phi,
)


@pytest.fixture(scope="module")
def calibrated() -> ModelParams:
    """Baseline parameters calibrated to theta_L(phi=0.1) = 0.66."""
    params, _ = calibrated_params(ModelParams())
    return params


PHI_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


# --- 1. equilibrium validity ------------------------------------------------

@pytest.mark.parametrize("phi", PHI_GRID)
def test_equilibrium_solves_to_tolerance(calibrated: ModelParams, phi: float) -> None:
    """The residual system is satisfied and the allocation is interior."""
    eq = solve_equilibrium(calibrated, ChannelPhis.uniform(phi))

    assert eq.max_residual < SolverConfig().residual_tol
    assert 0.0 < eq.K_h < calibrated.K_total
    assert 0.0 < eq.L_h < calibrated.L_total
    assert eq.p > 0.0
    assert 0.0 < eq.theta_L < 1.0


def test_factor_prices_equalize_net_of_markup(calibrated: ModelParams) -> None:
    """Factor prices equalize across sectors *net of the markup wedge*.

    Gross marginal products diverge — that divergence is what breaks
    factor-price equalization in this model.
    """
    eq = solve_equilibrium(calibrated, ChannelPhis.uniform(0.5))
    rho = calibrated.rho

    _, mpk_h, mpl_h = ces_output(calibrated.A_h, eq.alpha_h, eq.K_h, eq.L_h, rho)
    _, mpk_l, mpl_l = ces_output(calibrated.A_l, eq.alpha_l, eq.K_l, eq.L_l, rho)

    assert eq.p * mpk_h / eq.mu_h == pytest.approx(mpk_l / eq.mu_l, rel=1e-9)
    assert eq.p * mpl_h / eq.mu_h == pytest.approx(mpl_l / eq.mu_l, rel=1e-9)
    # Gross products genuinely differ, so the equalization above is not vacuous.
    assert eq.p * mpk_h != pytest.approx(mpk_l, rel=1e-3)


# --- 2. TFP is unidentified -------------------------------------------------

@pytest.mark.parametrize("A_h,A_l", [(1.0, 1.0), (3.7, 3.7), (2.5, 0.4), (0.1, 8.0)])
def test_labor_share_invariant_to_tfp(calibrated: ModelParams, A_h: float, A_l: float) -> None:
    """A_i cancels exactly out of the labor share.

    This is why the specification's implied calibration via A is impossible:
    under constant-returns CES the share is homogeneous of degree zero in A.
    """
    reference = labor_share(calibrated, ChannelPhis.uniform(0.5))
    perturbed = labor_share(replace(calibrated, A_h=A_h, A_l=A_l), ChannelPhis.uniform(0.5))
    assert perturbed == pytest.approx(reference, abs=1e-12)


# --- 3. Cobb-Douglas limit --------------------------------------------------

def test_cobb_douglas_branch_is_continuous(calibrated: ModelParams) -> None:
    """sigma -> 1 matches the closed form; the CES expression is singular there."""
    near = labor_share(replace(calibrated, sigma=0.999999), ChannelPhis.uniform(0.5))
    at = labor_share(replace(calibrated, sigma=1.0), ChannelPhis.uniform(0.5))
    assert near == pytest.approx(at, abs=1e-4)


def test_ces_reduces_to_cobb_douglas_at_rho_zero() -> None:
    """The rho == 0 branch reproduces the Cobb-Douglas forms."""
    Y, mpk, mpl = ces_output(A=1.0, alpha=0.4, K=1.3, L=0.8, rho=0.0)
    assert Y == pytest.approx(1.3**0.4 * 0.8**0.6)
    assert mpk == pytest.approx(0.4 * Y / 1.3)
    assert mpl == pytest.approx(0.6 * Y / 0.8)


# --- 4. calibration ---------------------------------------------------------

def test_calibration_hits_target(calibrated: ModelParams) -> None:
    """theta_L(phi=0.1) equals the 1970s US labor share."""
    calib = CalibrationConfig()
    theta = labor_share(calibrated, ChannelPhis.uniform(calib.phi_at_target))
    assert theta == pytest.approx(calib.target_theta, abs=1e-6)


def test_calibration_preserves_sector_gap(calibrated: ModelParams) -> None:
    """The alpha_H - alpha_L spread that identifies the sectors is untouched."""
    base = ModelParams()
    assert calibrated.a_h0 - calibrated.a_l0 == pytest.approx(base.a_h0 - base.a_l0)


# --- 5. monotonicity --------------------------------------------------------

def test_labor_share_declines_monotonically(calibrated: ModelParams) -> None:
    """Rising fugacity lowers the labor share at every step."""
    thetas = [eq.theta_L for eq in sweep_phi(calibrated, PHI_GRID)]
    assert all(b < a for a, b in zip(thetas, thetas[1:])), thetas


@pytest.mark.parametrize("channel", CHANNELS)
def test_each_channel_lowers_labor_share(calibrated: ModelParams, channel: str) -> None:
    """Every channel pushes the labor share down on its own."""
    base = labor_share(calibrated, ChannelPhis.selective(CHANNELS, PHI_LOW, PHI_LOW))
    single = labor_share(calibrated, ChannelPhis.selective([channel], PHI_HIGH, PHI_LOW))
    assert single < base


# --- 6. decomposition identities -------------------------------------------

def test_shapley_sums_to_joint_effect(calibrated: ModelParams) -> None:
    """The Shapley decomposition is exact by construction."""
    d = decompose_shapley(calibrated, PHI_LOW, PHI_HIGH)
    assert d.total == pytest.approx(d.joint, abs=1e-10)
    assert d.ratio == pytest.approx(1.0, abs=1e-10)


def test_decomposition_convention_changes_the_conclusion(calibrated: ModelParams) -> None:
    """The headline reverses sign with the convention — the core finding.

    One-at-a-time over-counts, leave-one-out under-counts, Shapley is exact.
    Reporting only the first would select the convention that yields the
    desired answer.
    """
    one, leave, shapley = decompose_all(calibrated, PHI_LOW, PHI_HIGH)

    assert one.ratio > 1.0, "one-at-a-time should over-count"
    assert leave.ratio < 1.0, "leave-one-out should under-count"
    assert shapley.ratio == pytest.approx(1.0, abs=1e-10)
    assert one.ratio > shapley.ratio > leave.ratio


def test_joint_effect_is_positive(calibrated: ModelParams) -> None:
    """Fugacity lowers the labor share overall."""
    assert joint_effect(calibrated, PHI_LOW, PHI_HIGH) > 0.0


# --- 7. scale invariance ----------------------------------------------------

def test_proportional_scaling_leaves_labor_share_unchanged(calibrated: ModelParams) -> None:
    """Doubling both endowments changes nothing.

    Note this does NOT discriminate between the level and ratio forms of the
    markup: proportional scaling leaves every K_i/L_i fixed, so both forms pass.
    The non-proportional case below is the one that bites.
    """
    a = labor_share(calibrated, ChannelPhis.uniform(0.5))
    doubled = replace(calibrated, K_total=2.0, L_total=2.0)
    assert labor_share(doubled, ChannelPhis.uniform(0.5)) == pytest.approx(a, abs=1e-12)


def test_markup_is_scale_free_under_nonproportional_endowments() -> None:
    """The ratio form of the markup is invariant to the units of K and L.

    With the specification's level form, mu depends on the level of K_i/L_i, so
    raising K relative to L inflates every markup. The ratio form removes that.
    This is the test the proportional case cannot perform.
    """
    from src.model import markup

    eta, phi = 0.2, 0.8
    # Same allocation, endowments expressed in different units.
    at_unit = markup(eta, phi, k_over_l=1.5, aggregate_k_over_l=1.0)
    at_double = markup(eta, phi, k_over_l=3.0, aggregate_k_over_l=2.0)
    assert at_unit == pytest.approx(at_double)


def test_nonproportional_endowments_do_change_the_allocation(calibrated: ModelParams) -> None:
    """Sanity check: K=2, L=1 is a genuinely different economy."""
    a = labor_share(calibrated, ChannelPhis.uniform(0.5))
    skewed = replace(calibrated, K_total=2.0, L_total=1.0)
    assert labor_share(skewed, ChannelPhis.uniform(0.5)) != pytest.approx(a, abs=1e-3)


# --- 8. precondition guards -------------------------------------------------

def test_infeasible_share_is_rejected(calibrated: ModelParams) -> None:
    """s_H(phi) >= 0.95 makes the value-share constraint infeasible."""
    bad = replace(calibrated, delta=1.2)
    with pytest.raises(EquilibriumError, match="s_H"):
        labor_share(bad, ChannelPhis.uniform(PHI_HIGH))


def test_out_of_range_capital_intensity_is_rejected(calibrated: ModelParams) -> None:
    """alpha_i must stay inside (0, 1)."""
    bad = replace(calibrated, gamma=1.5)
    with pytest.raises(EquilibriumError, match="alpha"):
        labor_share(bad, ChannelPhis.uniform(PHI_HIGH))


def test_degenerate_wage_wedge_is_rejected(calibrated: ModelParams) -> None:
    """beta*phi >= 1 would wipe out the wage entirely."""
    bad = replace(calibrated, beta=1.5)
    with pytest.raises(EquilibriumError, match="wedge"):
        labor_share(bad, ChannelPhis.uniform(PHI_HIGH))


# --- robustness across the sensitivity ranges -------------------------------

@pytest.mark.parametrize("sigma", [0.5, 0.7, 0.9, 1.0])
@pytest.mark.parametrize("phi", [PHI_LOW, PHI_HIGH])
def test_converges_across_sigma_range(calibrated: ModelParams, sigma: float, phi: float) -> None:
    """The solver holds up across the elasticity range used in the sensitivity run."""
    eq = solve_equilibrium(replace(calibrated, sigma=sigma), ChannelPhis.uniform(phi))
    assert eq.max_residual < SolverConfig().residual_tol


@pytest.mark.parametrize("name", ["gamma", "delta", "eta"])
@pytest.mark.parametrize("factor", [0.5, 1.5])
def test_converges_under_parameter_perturbation(
    calibrated: ModelParams, name: str, factor: float
) -> None:
    """+/-50% perturbations of the channel sensitivities still solve."""
    perturbed = replace(calibrated, **{name: getattr(calibrated, name) * factor})
    eq = solve_equilibrium(perturbed, ChannelPhis.uniform(PHI_HIGH))
    assert eq.max_residual < SolverConfig().residual_tol
