"""Phase 1 analysis: sign shares, Sobol indices, and phase diagrams.

Reporting follows CLAUDE.md: signs, orderings, and shares of parameter space.
No thresholds are presented as structural boundaries, and no optimality
language is used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis.legacy.phase1_spec import (  # noqa: E402
    PARAM_NAMES, POST_HOC_CANDIDATE, PRIOR_EXPECTATION,
)

RESULTS = ROOT / "results" / "phase1"
METRICS = ("leak_abs", "leak_per_fiscal", "leak_per_welfare")
BLOCS = ("A", "B", "C", "D")
BLOC_LABEL = {"A": "A (reserve)", "B": "B (non-reserve adv.)",
              "C": "C (emerging)", "D": "D (developing)"}


def load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sign_share(rows: list[dict], basis: str, bloc: str, metric: str) -> dict[str, Any]:
    """Share of sampled space where CLT leaks less than cash (Δ < 0)."""
    key = f"{basis}.{bloc}.d_{metric}"
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return {"n": 0, "share_negative": None, "median": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(arr.size),
        "share_negative": float((arr < 0).mean()),
        "median": float(np.median(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q90": float(np.quantile(arr, 0.90)),
    }


def sign_share_table(rows: list[dict]) -> dict[str, Any]:
    """Share-of-space table across bloc x basis x metric."""
    out: dict[str, Any] = {}
    for metric in METRICS:
        for basis in ("welfare", "cost"):
            for bloc in BLOCS:
                out[f"{metric}|{basis}|{bloc}"] = sign_share(rows, basis, bloc, metric)
    return out




def sign_drivers(rows: list[dict], basis: str, bloc: str,
                 metric: str = "leak_abs") -> dict[str, Any] | None:
    """Rank what separates sign reversals from non-reversals, via LHS strata.

    This is a more powerful instrument than the Sobol decomposition for the
    sign question. The sign is a binary with little variance, so its Sobol
    indices come out statistically indistinguishable across the leading
    parameters. Comparing the parameter distributions of reversing and
    non-reversing draws instead yields t-statistics that separate cleanly.
    """
    key = f"{basis}.{bloc}.d_{metric}"
    pos = [r for r in rows if r.get(key) is not None and r[key] > 0]
    neg = [r for r in rows if r.get(key) is not None and r[key] <= 0]
    if len(pos) < 50 or len(neg) < 50:
        return None

    names = [n for n in PARAM_NAMES if n in rows[0]]
    out = []
    for name in names:
        a = np.asarray([r[name] for r in pos], dtype=float)
        b = np.asarray([r[name] for r in neg], dtype=float)
        se = float(np.sqrt(a.var() / a.size + b.var() / b.size))
        gap = float(a.mean() - b.mean())
        out.append({"param": name, "gap": gap,
                    "t": abs(gap) / se if se > 0 else 0.0})
    out.sort(key=lambda d: -d["t"])
    return {"n_reversals": len(pos), "n_total": len(pos) + len(neg),
            "ranked": out}


def sobol_indices(rows: list[dict], problem: dict, basis: str, bloc: str,
                  metric: str, n: int, prefix: str = "d") -> dict[str, Any] | None:
    """First-order and total Sobol indices for one output.

    The Saltelli design must be complete, so this returns ``None`` if any
    sample point in the design was infeasible and dropped.
    """
    from SALib.analyze import sobol as sobol_analyze

    key = f"{basis}.{bloc}.{prefix}_{metric}"
    # With calc_second_order=False the Saltelli design is N*(k+2) rows, not
    # N*(2k+2); the latter applies only when second-order terms are requested.
    expected = n * (problem["num_vars"] + 2)
    if len(rows) != expected:
        return None
    y = np.asarray([r.get(key) for r in rows], dtype=float)
    if np.isnan(y).any():
        return None
    # A constant output has no variance to attribute; Sobol indices would be
    # meaningless (and numerically unstable) there.
    if float(np.var(y)) < 1e-12:
        return None
    res = sobol_analyze.analyze(problem, y, calc_second_order=False,
                                print_to_console=False)
    return {
        "S1": {name: float(v) for name, v in zip(problem["names"], res["S1"])},
        "ST": {name: float(v) for name, v in zip(problem["names"], res["ST"])},
        "S1_conf": {name: float(v) for name, v in zip(problem["names"], res["S1_conf"])},
        "ST_conf": {name: float(v) for name, v in zip(problem["names"], res["ST_conf"])},
    }


def _rank_of(ranked: list[tuple[str, float]], name: str) -> int | None:
    names = [n for n, _ in ranked]
    return names.index(name) + 1 if name in names else None


def _judge(sobol: dict[str, Any] | None, question: str) -> dict[str, Any]:
    """Score the prior against one output."""
    if not sobol:
        return {"scored": False, "question": question}
    ranked = sorted(sobol["ST"].items(), key=lambda kv: kv[1], reverse=True)
    top2 = [n for n, _ in ranked[:2]]
    return {
        "scored": True,
        "question": question,
        "top2": top2,
        "eps_supply_rank": _rank_of(ranked, "eps_supply"),
        "land_share_rank": _rank_of(ranked, "land_share"),
        "m_H_rank": _rank_of(ranked, "m_H"),
        "sourcing_rank": _rank_of(ranked, "clt_domestic_sourcing"),
        "eps_supply_upheld": _rank_of(ranked, "eps_supply") in (1, 2),
        "land_share_upheld": _rank_of(ranked, "land_share") in (1, 2),
        "ranked_total_order": ranked,
    }


def _judge_sign(drivers: dict[str, Any] | None) -> dict[str, Any]:
    """Score the prior on the sign question using the t-statistic ranking."""
    if not drivers:
        return {"scored": False, "question": "what drives the DIRECTION"}
    ranked = [d["param"] for d in drivers["ranked"]]
    by_name = {d["param"]: d for d in drivers["ranked"]}
    return {
        "scored": True,
        "question": "what drives the DIRECTION of the gap (sign of ΔLeak)",
        "instrument": "LHS stratification (t-statistic), not Sobol",
        "top2": ranked[:2],
        "eps_supply_rank": ranked.index("eps_supply") + 1,
        "land_share_rank": ranked.index("land_share") + 1,
        "eps_supply_t": by_name["eps_supply"]["t"],
        "land_share_t": by_name["land_share"]["t"],
        "eps_supply_upheld": ranked.index("eps_supply") < 2,
        "land_share_upheld": ranked.index("land_share") < 2,
        "ranked_total_order": [(d["param"], d["t"]) for d in drivers["ranked"]],
    }


def score_prior(magnitude: dict[str, Any] | None,
                sign: dict[str, Any] | None) -> dict[str, Any]:
    """Score the pre-registered prior on two distinct questions.

    The prior names eps_supply and land_share. Those can be right about *which
    way* the comparison goes without being right about *how large* the gap is,
    so the two questions are judged separately and both are reported. The
    prior text itself is never rewritten.
    """
    return {
        "prior": PRIOR_EXPECTATION,
        "post_hoc_candidate": POST_HOC_CANDIDATE,
        "magnitude": _judge(magnitude, "what drives the SIZE of the gap (d_leak_abs)"),
        "sign": _judge_sign(sign),
    }


def main() -> None:
    lhs_rows = load(RESULTS / "lhs.jsonl")
    sobol_rows = load(RESULTS / "sobol.jsonl")
    print(f"loaded {len(lhs_rows)} LHS rows, {len(sobol_rows)} Sobol rows")

    report: dict[str, Any] = {
        "n_lhs": len(lhs_rows),
        "n_sobol": len(sobol_rows),
        "prior_expectation": PRIOR_EXPECTATION,
        "post_hoc_candidate": POST_HOC_CANDIDATE,
        "sign_shares": sign_share_table(lhs_rows),
    }

    # Pattern split (staggered vs simultaneous introduction).
    for pattern in ("staggered", "simultaneous"):
        subset = [r for r in lhs_rows if r.get("pattern") == pattern]
        report[f"sign_shares_{pattern}"] = sign_share_table(subset)

    meta_path = RESULTS / "sobol_problem.json"
    if sobol_rows and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        problem, n = meta["problem"], meta["N"]
        report["sobol"] = {}
        for metric, basis, bloc in (("leak_abs", "welfare", "D"),
                                    ("leak_per_fiscal", "cost", "D"),
                                    ("leak_abs", "welfare", "C"),
                                    ("leak_per_fiscal", "cost", "C")):
            key = f"{metric}|{basis}|{bloc}"
            idx = sobol_indices(sobol_rows, problem, basis, bloc, metric, n)
            if idx:
                report["sobol"][key] = idx

        # Scale-free and sign outputs. `ratio` removes the units in which the
        # leak is measured, so a parameter that only rescales both modalities
        # (such as the transfer level u) cannot dominate it by construction.
        # `sign` isolates what decides the direction rather than the size.
        for basis, bloc in (("cost", "A"), ("cost", "B"), ("cost", "D"),
                            ("welfare", "D")):
            for prefix in ("ratio", "sign"):
                idx = sobol_indices(sobol_rows, problem, basis, bloc,
                                    "leak_abs", n, prefix=prefix)
                if idx:
                    report["sobol"][f"{prefix}_leak_abs|{basis}|{bloc}"] = idx
        for basis, bloc in (("welfare", "D"), ("welfare", "C")):
            idx = sobol_indices(sobol_rows, problem, basis, bloc, "severity_noapp", n)
            if idx:
                report["sobol"][f"severity_noapp|{basis}|{bloc}"] = idx
        report["sign_drivers"] = {
            f"{basis}|{bloc}": sign_drivers(lhs_rows, basis, bloc)
            for basis, bloc in (("cost", "A"), ("cost", "B"), ("cost", "D"))
        }
        report["prior_score"] = score_prior(
            report["sobol"].get("leak_abs|welfare|D"),
            report["sign_drivers"].get("cost|A"),
        )

    out = RESULTS / "analysis.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {out}")

    print("\nShare of sampled space with CLT leaking LESS than cash (Δ<0):")
    print(f"  {'metric':18s} {'basis':8s} " + " ".join(f"{b:>7s}" for b in BLOCS))
    for metric in METRICS:
        for basis in ("welfare", "cost"):
            cells = []
            for bloc in BLOCS:
                share = report["sign_shares"][f"{metric}|{basis}|{bloc}"]["share_negative"]
                cells.append("    n/a" if share is None else f"{100 * share:6.1f}%")
            print(f"  {metric:18s} {basis:8s} " + " ".join(cells))


if __name__ == "__main__":
    main()
