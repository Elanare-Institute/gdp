"""Compare Sobol results across sample sizes.

Reports whether the parameter ranking is stable between N=256 and N=512, and
how much the confidence intervals tightened. A ranking that moves a lot
between sample sizes is not yet resolved, whatever the point estimates say.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

SCRATCH = Path("/private/tmp/claude-501/-Users-rysh-src-gdp/"
               "1edf7c70-b320-4b79-968f-cbba30bb2ac6/scratchpad")
RESULTS = ROOT / "results" / "phase1"

KEYS = ("leak_abs|welfare|D", "ratio_leak_abs|welfare|D",
        "sign_leak_abs|cost|A", "sign_leak_abs|cost|B")


def ranking(idx: dict) -> list[str]:
    return [n for n, _ in sorted(idx["ST"].items(), key=lambda kv: -kv[1])]


def spearman(a: list[str], b: list[str]) -> float:
    """Rank correlation between two orderings of the same names."""
    pos_b = {name: i for i, name in enumerate(b)}
    n = len(a)
    d2 = sum((i - pos_b[name]) ** 2 for i, name in enumerate(a))
    return 1 - 6 * d2 / (n * (n * n - 1))


def main() -> None:
    old_path = SCRATCH / "analysis_n256.json"
    new_path = RESULTS / "analysis.json"
    if not old_path.exists():
        raise SystemExit("N=256 archive not found")

    old = json.loads(old_path.read_text()).get("sobol", {})
    new = json.loads(new_path.read_text()).get("sobol", {})

    print(f"{'output':30s} {'rank corr':>10s} {'top3 same':>10s} "
          f"{'CI 256':>8s} {'CI 512':>8s}")
    for key in KEYS:
        if key not in old or key not in new:
            print(f"{key:30s} {'n/a':>10s}")
            continue
        ra, rb = ranking(old[key]), ranking(new[key])
        rho = spearman(ra, rb)
        same_top3 = set(ra[:3]) == set(rb[:3])
        ci_old = sum(old[key]["ST_conf"].values()) / len(old[key]["ST_conf"])
        ci_new = sum(new[key]["ST_conf"].values()) / len(new[key]["ST_conf"])
        print(f"{key:30s} {rho:10.3f} {str(same_top3):>10s} "
              f"{ci_old:8.3f} {ci_new:8.3f}")

    print("\nprior-relevant ranks (N=256 -> N=512):")
    for key in KEYS:
        if key not in old or key not in new:
            continue
        ra, rb = ranking(old[key]), ranking(new[key])
        for name in ("eps_supply", "land_share", "m_H", "clt_domestic_sourcing"):
            if name in ra and name in rb:
                print(f"  {key:30s} {name:24s} "
                      f"{ra.index(name) + 1:2d} -> {rb.index(name) + 1:2d}")


if __name__ == "__main__":
    main()
