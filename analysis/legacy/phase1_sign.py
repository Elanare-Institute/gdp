"""What decides the SIGN of the CLT-vs-cash comparison.

Two composite quantities are the candidates, each being what the CLT itself
has to buy:

    effective land rent  = land_share * (1 - land_discount)
    effective build imports = m_H * (1 - clt_domestic_sourcing)

The primary instrument is a multivariate logistic regression on standardised
regressors, so each coefficient is the effect holding the others fixed. The
stratified t-statistics reported earlier are marginal comparisons and are kept
only as a cross-check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis.legacy.phase1_spec import PARAMS, unit_to_value  # noqa: E402
from legacy.v7.params import archetypes  # noqa: E402

RESULTS = ROOT / "results" / "phase1"
SPEC = {p.name: p for p in PARAMS}


def to_real(rows: list[dict], name: str, bloc: str) -> np.ndarray:
    """Map unit-cube draws to the parameter's real scale for one bloc."""
    base = getattr(archetypes()[bloc], name, 0.0)
    return np.asarray([unit_to_value(SPEC[name], r[name], base) for r in rows],
                      dtype=float)


def design(rows: list[dict], bloc: str) -> tuple[np.ndarray, list[str]]:
    """Composite regressors plus the controls that could confound them."""
    land_share = to_real(rows, "land_share", bloc)
    land_disc = to_real(rows, "land_discount", bloc)
    m_h = to_real(rows, "m_H", bloc)
    sourcing = to_real(rows, "clt_domestic_sourcing", bloc)

    cols = {
        "eff_land_rent": land_share * (1.0 - land_disc),
        "eff_build_imports": m_h * (1.0 - sourcing),
        "eps_supply": np.log(to_real(rows, "eps_supply", bloc)),
        "u": to_real(rows, "u", bloc),
        "m_F": to_real(rows, "m_F", bloc),
        "m_G": to_real(rows, "m_G", bloc),
        "omega0": to_real(rows, "omega0", bloc),
        "mpc_u": to_real(rows, "mpc_u", bloc),
        "sub_H": to_real(rows, "sub_H", bloc),
        "beta_H": to_real(rows, "beta_H", bloc),
    }
    if "sourcing_cost_kappa" in rows[0]:
        cols["sourcing_cost_kappa"] = to_real(rows, "sourcing_cost_kappa", bloc)

    # sub_H and beta_H are global parameters read off the bloc record, so they
    # are constant within a bloc and carry no information. Drop any such column
    # rather than standardising a zero-variance regressor.
    names = [n for n in cols if float(np.std(cols[n])) > 1e-12]
    X = np.column_stack([cols[n] for n in names])
    X = (X - X.mean(axis=0)) / X.std(axis=0)   # standardise
    return X, names


def logistic(X: np.ndarray, y: np.ndarray, iters: int = 200) -> np.ndarray:
    """Newton-Raphson logistic regression with a small ridge penalty."""
    Xb = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(Xb.shape[1])
    ridge = 1e-6 * np.eye(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ beta, -30, 30)))
        W = p * (1 - p) + 1e-9
        grad = Xb.T @ (y - p) - ridge @ beta
        H = Xb.T @ (Xb * W[:, None]) + ridge
        step = np.linalg.solve(H, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def analyse(rows: list[dict], bloc: str, basis: str = "cost") -> dict | None:
    key = f"{basis}.{bloc}.d_leak_abs"
    vals = [r for r in rows if r.get(key) is not None]
    y = np.asarray([1.0 if r[key] > 0 else 0.0 for r in vals])  # 1 = reversal
    if y.sum() < 50 or (1 - y).sum() < 50:
        return None

    X, names = design(vals, bloc)
    beta = logistic(X, y)
    coef = dict(zip(names, beta[1:]))
    ranked = sorted(coef.items(), key=lambda kv: -abs(kv[1]))
    return {"bloc": bloc, "basis": basis, "n": len(vals),
            "n_reversals": int(y.sum()), "coefficients": coef,
            "ranked": ranked}


def main() -> None:
    rows = [json.loads(l) for l in (RESULTS / "lhs.jsonl").read_text().splitlines() if l.strip()]
    print(f"LHS rows: {len(rows)}")
    print("Logistic regression on standardised regressors.")
    print("Outcome = 1 when CLT leaks MORE than cash (a sign reversal).\n")

    out = {}
    for bloc in ("A", "B", "C", "D"):
        res = analyse(rows, bloc)
        if res is None:
            print(f"bloc {bloc}: too few reversals to fit")
            continue
        out[bloc] = res
        print(f"bloc {bloc}  ({res['n_reversals']}/{res['n']} reversals)")
        for name, value in res["ranked"][:6]:
            print(f"    {name:24s} {value:+7.3f}")
        print()

    (RESULTS / "sign_logistic.json").write_text(
        json.dumps({k: {"n": v["n"], "n_reversals": v["n_reversals"],
                        "coefficients": v["coefficients"]}
                    for k, v in out.items()}, indent=1), encoding="utf-8")
    print(f"wrote {RESULTS / 'sign_logistic.json'}")


if __name__ == "__main__":
    main()
