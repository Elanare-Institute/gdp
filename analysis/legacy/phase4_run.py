"""Phase 4: stochastic shocks and Monte Carlo.

Every modality is run on the *same* shock paths (common random numbers), so a
modality difference is a per-path difference, not a difference between two
independent samples. Each scenario also runs with shocks switched off, which
gives the deterministic v6 answer as the zero-variance reference.

Reported quantities are deliberately of two kinds:
  - threshold-free: severity, final debt, final e, time to insolvency
  - threshold-crossing: crisis onsets, insolvency (kept because TASKS.md asks
    for them, but the thresholds are model constants and are not treated as
    structural boundaries)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from legacy.v7.core import build, simulate  # noqa: E402
from legacy.v7.params import G  # noqa: E402
from legacy.v7.shocks import ShockConfig, draw_path, summarise, zero_path  # noqa: E402

RESULTS = ROOT / "results" / "phase4"
MODALITIES = ("none", "ubi", "cash_t", "voucher", "clt")
U_VALUES = (0.03, 0.05, 0.10)
# Two introduction patterns, as in v6: staggered (A first, D last) and
# simultaneous (every bloc at year 1).
PATTERNS = {"staggered": (1.0, 5.0, 10.0, 15.0), "simultaneous": (1.0, 1.0, 1.0, 1.0)}
# Shock regimes. "all" is the main series; the single-channel ones identify
# which process each result comes from.
REGIMES = ("all", "rate_only", "food_only", "stop_only", "none")
#: Group letter -> bloc name as `archetypes()` spells it.
BLOCS = {"A": "A (reserve)", "B": "B (non-reserve adv.)",
         "C": "C (emerging)", "D": "D (developing)"}
STEPS_Q = 121


def globals_for_phase4() -> G:
    """Phase 2/3 main series, carried forward."""
    return G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
             clt_build_rate=0.02, reserve_flight_mode="capped")


def regime_config(regime: str) -> ShockConfig | None:
    """A ShockConfig with only the named channel active."""
    if regime == "none":
        return None
    off_rate = dict(rate_sigma=0.0, rate_jump_prob=0.0)
    off_food = dict(food_sigma=0.0)
    off_stop = dict(stop_rate=0.0)
    if regime == "all":
        return ShockConfig()
    if regime == "rate_only":
        return ShockConfig(**off_food, **off_stop)
    if regime == "food_only":
        return ShockConfig(**off_rate, **off_stop)
    if regime == "stop_only":
        return ShockConfig(**off_rate, **off_food)
    raise ValueError(regime)


def _one(task: tuple[str, float, str, str, int]) -> dict[str, Any]:
    modality, u, pattern, regime, seed = task
    g = globals_for_phase4()
    blocs = build(g, modality, u, PATTERNS[pattern], "welfare")
    cfg = regime_config(regime)
    path = zero_path(STEPS_Q) if cfg is None else draw_path(seed, STEPS_Q, config=cfg)
    res = simulate(blocs, g, shocks=path)
    out: dict[str, Any] = {"modality": modality, "u": u, "pattern": pattern,
                           "regime": regime, "seed": seed}
    for letter, name in BLOCS.items():
        c = res["crisis"][name]
        f = res["final"][name]
        out[letter] = {
            "onsets": c["onsets"],
            "severity": c["severity"],
            "severity_na": c["severity_no_appreciation"],
            "duration_q": c["duration_q"],
            "insolvent_year": c["insolvent_year"],
            "debt": f["debt"], "e": f["e"], "pi": f["pi"], "Y": f["Y"],
        }
    out["_system"] = res["crisis"]["_system"]
    out["_shock"] = summarise(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--regimes", nargs="*", default=list(REGIMES))
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, float, str, str, int]] = []
    for regime in args.regimes:
        # The deterministic regime has no sampling variation: one path each.
        n = 1 if regime == "none" else args.paths
        for modality in MODALITIES:
            for u in U_VALUES:
                for pattern in PATTERNS:
                    # "none" (no transfer) does not depend on u; run it once.
                    if modality == "none" and u != U_VALUES[0]:
                        continue
                    tasks += [(modality, u, pattern, regime, s) for s in range(1, n + 1)]

    # Shard by index across independent processes rather than sharing a pool.
    # multiprocessing.Pool wedged repeatedly on the Phase 3 workload (parent
    # alive, workers gone, no CPU); separate processes remove the shared state
    # that wedges, at the cost of each rebuilding its own calibration cache.
    shard, n_shards = args.shard, args.n_shards
    mine = [t for i, t in enumerate(tasks) if i % n_shards == shard]
    suffix = "" if n_shards == 1 else f".{shard}"
    out_path = RESULTS / f"montecarlo{suffix}.jsonl"
    print(f"shard {shard}/{n_shards}: {len(mine)} of {len(tasks)} tasks", flush=True)
    with out_path.open("w") as fh:
        for i, task in enumerate(mine, 1):
            fh.write(json.dumps(_one(task)) + "\n")
            if i % 500 == 0:
                fh.flush()
                print(f"  {i}/{len(mine)}", flush=True)
    print(f"wrote {len(mine)} -> {out_path}")


if __name__ == "__main__":
    main()
