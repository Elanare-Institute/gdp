"""Phase 3: does structure predict which provisioning mix a bloc converges to?

Every configuration is run against a neutral-drift control with the same seeds,
because a mix distribution that looks structured can be produced by drift and
the build cap alone. Only differences from that control are reported as
selection.
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

from legacy.v7.evolution import (  # noqa: E402
    CONTROL_MODALITIES, EvolutionConfig, MODALITIES, run_evolution,
)
from legacy.v7.params import G  # noqa: E402
from legacy.v7.population import make_population  # noqa: E402

RESULTS = ROOT / "results" / "phase3"
N_SEEDS = 50
YEARS = 80


def globals_for_phase3() -> G:
    """Phase 2's main series, carried forward."""
    return G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
             clt_build_rate=0.02)


def _one(task: tuple[int, str, bool, bool, str]) -> dict[str, Any]:
    seed, initial, neutral, endogenous, arena = task
    population = make_population(seed)
    three_way = arena.startswith("three_way")
    modality_set = CONTROL_MODALITIES if three_way else MODALITIES
    config = EvolutionConfig(years=YEARS, initial=initial, neutral=neutral,
                             endogenous_stress=endogenous,
                             modality_set=modality_set,
                             symmetric_invasion=arena.endswith("symmetric"))
    result = run_evolution(population, globals_for_phase3(), config, seed)
    result["population_seed"] = seed
    result["endogenous"] = endogenous
    result["arena"] = arena
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    # Main series: invasion start with endogenous stress. The others are
    # sensitivities — the reduced-form stress and the two older initial
    # conditions — each with its own drift control.
    # The four-way arena is run under both seedings. The asymmetric one plants
    # only CLT, which gives it an imitation source the alternatives lack; the
    # symmetric one plants each non-cash modality in its own blocs. Table 1,
    # table 2 and section 6 are read off the symmetric arm.
    tasks = [(seed, "invasion", neutral, True, arena)
             for seed in range(1, args.seeds + 1) for neutral in (False, True)
             for arena in ("four_way", "four_way_symmetric")]
    tasks += [(seed, initial, neutral, True, "four_way")
              for seed in range(1, args.seeds + 1)
              for initial in ("all_cash", "random")
              for neutral in (False, True)]
    # Reduced-form stress, as a sensitivity. Run under both seedings so that
    # "the gradient disappears under the reduced form" is not itself a
    # statement about which modality was planted.
    tasks += [(seed, "invasion", neutral, False, arena)
              for seed in range(1, args.seeds + 1) for neutral in (False, True)
              for arena in ("four_way", "four_way_symmetric")]
    # Three-way control: CLT against a cash transfer of identical fiscal cost.
    # Run both seedings — asymmetric (only CLT planted) and symmetric (each
    # alternative planted) — because planting only one gives it a source to be
    # imitated from that the other lacks.
    tasks += [(seed, "invasion", neutral, True, arena)
              for seed in range(1, args.seeds + 1)
              for neutral in (False, True)
              for arena in ("three_way", "three_way_symmetric")]

    # Shard by index so several independent processes can cover the task list
    # without sharing a pool. multiprocessing.Pool has wedged twice on this
    # workload (parent alive, workers gone, no CPU); separate processes each
    # writing their own file removes the shared state that wedges.
    shard, n_shards = args.shard, args.n_shards
    mine = [t for i, t in enumerate(tasks) if i % n_shards == shard]
    suffix = "" if n_shards == 1 else f".{shard}"
    path = RESULTS / f"evolution{suffix}.jsonl"
    print(f"shard {shard}/{n_shards}: {len(mine)} evolutions", flush=True)
    with path.open("w", encoding="utf-8") as fh:
        for i, task in enumerate(mine, 1):
            fh.write(json.dumps(_one(task)) + "\n")
            fh.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(mine)}", flush=True)
    print(f"wrote {path.name} ({len(mine)} runs)")


if __name__ == "__main__":
    main()
