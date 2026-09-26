"""Phase 1: global sensitivity of the sign of ΔLeak.

Each sample point is evaluated under both comparison bases (welfare-equivalent
and equal fiscal cost) for cash_t and clt, and the difference CLT − cash is
recorded on all three leakage metrics.

Results are reported as shares of the parameter space in which a sign holds,
never as thresholds.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis.legacy.phase1_spec import (  # noqa: E402
    PARAM_NAMES, PARAMS, START_PATTERNS, apply_point, matched_params,
)
from legacy.v7.core import build, simulate  # noqa: E402
from legacy.v7.household import calibrate, equal_cost_size  # noqa: E402
from legacy.v7.params import Bloc, G  # noqa: E402

RESULTS = ROOT / "results" / "phase1"
METRICS = ("leak_abs", "leak_per_fiscal", "leak_per_welfare")
BLOC_KEYS = ("A", "B", "C", "D")


def _scenario(blocs: list[Bloc], g: G, modality: str, u: float,
              starts: tuple[float, ...], basis: str) -> dict[str, Any]:
    """Run one modality on a sampled parameter point."""
    from dataclasses import replace

    out: list[Bloc] = []
    for b, st in zip(blocs, starts):
        cal = calibrate(b, g, u)
        if basis == "welfare":
            size = cal[modality]
        else:
            size = equal_cost_size(b, g, modality, cal["_fiscal_ubi"])
        out.append(replace(b, modality=modality, size=size, start=st))
    return simulate(out, g)


def evaluate(unit: dict[str, float], pattern: str,
             matched: bool = False) -> dict[str, Any] | None:
    """Evaluate one sample point; ``None`` if the point is infeasible.

    Args:
        matched: use the matched-range parameter spans (see phase1_spec).
    """
    try:
        blocs, g, u = apply_point(unit, matched_params() if matched else None)
        starts = START_PATTERNS[pattern]
        row: dict[str, Any] = {"pattern": pattern, **{k: unit[k] for k in PARAM_NAMES}}

        for basis in ("welfare", "cost"):
            runs = {mod: _scenario(blocs, g, mod, u, starts, basis)
                    for mod in ("cash_t", "clt")}
            for key, bloc in zip(BLOC_KEYS, blocs):
                cash = runs["cash_t"]["leakage"].get(bloc.name)
                clt = runs["clt"]["leakage"].get(bloc.name)
                if not cash or not clt:
                    return None
                for metric in METRICS:
                    a, b_ = cash.get(metric), clt.get(metric)
                    row[f"{basis}.{key}.d_{metric}"] = (
                        None if a is None or b_ is None else b_ - a
                    )
                    # Levels are kept so scale-free outputs (ratios) can be
                    # derived later without re-running the sweep.
                    row[f"{basis}.{key}.cash_{metric}"] = a
                    row[f"{basis}.{key}.clt_{metric}"] = b_
                    if a not in (None, 0.0) and b_ is not None:
                        row[f"{basis}.{key}.ratio_{metric}"] = b_ / a
                    else:
                        row[f"{basis}.{key}.ratio_{metric}"] = None
                    row[f"{basis}.{key}.sign_{metric}"] = (
                        None if a is None or b_ is None else float(b_ - a < 0)
                    )
                cash_c = runs["cash_t"]["crisis"][bloc.name]
                clt_c = runs["clt"]["crisis"][bloc.name]
                row[f"{basis}.{key}.d_severity"] = clt_c["severity"] - cash_c["severity"]
                row[f"{basis}.{key}.d_severity_noapp"] = (
                    clt_c["severity_no_appreciation"] - cash_c["severity_no_appreciation"]
                )
                row[f"{basis}.{key}.d_final_debt"] = (
                    runs["clt"]["final"][bloc.name]["debt"]
                    - runs["cash_t"]["final"][bloc.name]["debt"]
                )
                cash_ins = cash_c["insolvent_year"]
                clt_ins = clt_c["insolvent_year"]
                row[f"{basis}.{key}.d_insolvency"] = (
                    None if cash_ins is None and clt_ins is None
                    else (clt_ins or 31.0) - (cash_ins or 31.0)
                )
                row[f"{basis}.{key}.d_fiscal"] = (
                    clt["fiscal_pct_gdp"] - cash["fiscal_pct_gdp"]
                )
        return row
    except (ValueError, ZeroDivisionError, OverflowError, KeyError):
        return None


def _worker(task: tuple[dict[str, float], str, bool]) -> dict[str, Any] | None:
    return evaluate(*task)


def run_batch(tasks: list[tuple[dict[str, float], str, bool]], workers: int,
              label: str, partial: Path | None = None) -> list[dict[str, Any]]:
    """Evaluate a batch of sample points in parallel."""
    print(f"  {label}: {len(tasks)} points on {workers} workers", flush=True)
    rows: list[dict[str, Any]] = []
    done = 0
    # Run in bounded chunks with a fresh pool each time. A single long-lived
    # pool proved able to wedge on this workload (workers alive but consuming
    # no CPU); recycling it bounds the damage and keeps progress observable.
    chunk = max(workers * 32, 256)

    # Checkpoint after every chunk. A long sweep that is interrupted keeps the
    # work already done, and a rerun with the same partial file resumes from
    # where it stopped instead of starting over.
    start_at = 0
    if partial is not None and partial.exists():
        cached = [json.loads(line) for line in
                  partial.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows.extend(cached)
        start_at = len(cached)
        done = start_at
        print(f"    resuming from checkpoint: {start_at} points already done",
              flush=True)

    for start in range(start_at, len(tasks), chunk):
        block = tasks[start:start + chunk]
        with mp.Pool(workers, maxtasksperchild=64) as pool:
            for row in pool.imap_unordered(_worker, block, chunksize=4):
                if row is not None:
                    rows.append(row)
                done += 1
        if partial is not None:
            write_jsonl(partial, rows, quiet=True)
        print(f"    {done}/{len(tasks)} ({100 * done // len(tasks)}%)", flush=True)
    print(f"  {label}: {len(rows)} feasible of {len(tasks)}", flush=True)
    return rows


def lhs(n: int, seed: int) -> list[dict[str, float]]:
    """Latin hypercube sample on the unit cube (stdlib only, seeded)."""
    import random

    rng = random.Random(seed)
    k = len(PARAM_NAMES)
    columns = []
    for _ in range(k):
        cuts = [(i + rng.random()) / n for i in range(n)]
        rng.shuffle(cuts)
        columns.append(cuts)
    return [{PARAM_NAMES[j]: columns[j][i] for j in range(k)} for i in range(n)]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], quiet: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    if not quiet:
        print(f"  wrote {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-lhs", type=int, default=4000)
    ap.add_argument("--n-sobol", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--skip-sobol", action="store_true")
    ap.add_argument("--skip-lhs", action="store_true")
    ap.add_argument("--matched", action="store_true",
                    help="use matched m_H / sourcing ranges (asymmetry check)")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    if not args.skip_lhs:
        print("Phase 1 / LHS")
        points = lhs(args.n_lhs, args.seed)
        half = len(points) // 2
        tasks = [(p, "staggered" if i < half else "simultaneous", False)
                 for i, p in enumerate(points)]
        write_jsonl(RESULTS / "lhs.jsonl", run_batch(tasks, args.workers, "LHS"))

    if not args.skip_sobol:
        print("Phase 1 / Sobol (Saltelli)")
        from SALib.sample import sobol as sobol_sample

        problem = {"num_vars": len(PARAM_NAMES), "names": list(PARAM_NAMES),
                   "bounds": [[0.0, 1.0]] * len(PARAM_NAMES)}
        design = sobol_sample.sample(problem, args.n_sobol, calc_second_order=False,
                                     seed=args.seed)
        tasks = [({PARAM_NAMES[j]: float(row[j]) for j in range(len(PARAM_NAMES))},
                  "staggered", args.matched) for row in design]
        suffix = "_matched" if args.matched else ""
        rows = run_batch(tasks, args.workers, "Sobol",
                         partial=RESULTS / f"sobol{suffix}.partial.jsonl")
        write_jsonl(RESULTS / f"sobol{suffix}.jsonl", rows)
        (RESULTS / f"sobol{suffix}.partial.jsonl").unlink(missing_ok=True)
        (RESULTS / f"sobol_problem{suffix}.json").write_text(
            json.dumps({"problem": problem, "N": args.n_sobol,
                        "calc_second_order": False, "n_design": len(design),
                        "n_feasible": len(rows)}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
