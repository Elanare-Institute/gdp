"""Sensitivity: does the Phase 3 conclusion depend on the population?

The archetype population confounds `fx_share` with every other structural
parameter (corr with gdp -0.84, with m_H +0.79), and group A holds a single
bloc, so the fx<0.1 region is identified by one observation per run. Phase 3
§3 had to retract the fx gradient for exactly this reason.

Here the group labels are dropped: `fx_share` is drawn uniformly on [0, 0.85]
and every other structural parameter is drawn independently over the range
the archetypes span. fx is then orthogonal to structure by construction, so a
pooled regression needs no fixed effect — and none is available, since there
are no groups.

Two questions, both answered against the matched neutral-drift control:
  1. does the level effect (CLT above drift) survive?
  2. does an fx gradient appear once the confounding is removed?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from legacy.v7.evolution import EvolutionConfig, MODALITIES, run_evolution  # noqa: E402
from legacy.v7.population import make_orthogonal_population  # noqa: E402

from analysis.legacy.phase3_run import globals_for_phase3  # noqa: E402

RESULTS = ROOT / "results" / "phase3"
N_SEEDS = 50
YEARS = 80


def _one(task: tuple[int, bool]) -> dict[str, Any]:
    """One evolution on the orthogonal population.

    The arena is the symmetric one throughout: Phase 3's revised main series.
    """
    seed, neutral = task
    population = make_orthogonal_population(seed)
    config = EvolutionConfig(years=YEARS, initial="invasion", neutral=neutral,
                             endogenous_stress=True, modality_set=MODALITIES,
                             symmetric_invasion=True)
    result = run_evolution(population, globals_for_phase3(), config, seed)
    result["population_seed"] = seed
    result["endogenous"] = True
    result["arena"] = "four_way_symmetric"
    result["population"] = "orthogonal"
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    tasks = [(seed, neutral) for seed in range(1, args.seeds + 1)
             for neutral in (False, True)]
    shard, n_shards = args.shard, args.n_shards
    mine = [t for i, t in enumerate(tasks) if i % n_shards == shard]
    suffix = "" if n_shards == 1 else f".{shard}"
    path = RESULTS / f"orthogonal{suffix}.jsonl"
    print(f"shard {shard}/{n_shards}: {len(mine)} evolutions", flush=True)
    with path.open("w", encoding="utf-8") as fh:
        for i, task in enumerate(mine, 1):
            fh.write(json.dumps(_one(task)) + "\n")
            fh.flush()
            if i % 5 == 0:
                print(f"  {i}/{len(mine)}", flush=True)
    print(f"wrote {path.name} ({len(mine)} runs)")


if __name__ == "__main__":
    main()
