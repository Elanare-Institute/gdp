"""Matched-range check: is 'policy beats structure' an artifact of the ranges?

The headline sweep lets ``clt_domestic_sourcing`` span all of [0,1] while
``m_H`` moves only +/-50% around its baseline, so the policy lever has more
room to act by construction. This script re-runs the decomposition on the
matched design and reports both side by side.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis.legacy.phase1_analyze import load, sobol_indices  # noqa: E402

RESULTS = ROOT / "results" / "phase1"
TARGETS = (("leak_abs", "welfare", "D", "d"),
           ("leak_abs", "welfare", "C", "d"),
           ("leak_abs", "welfare", "D", "ratio"))


def main() -> None:
    out: dict[str, dict] = {"sobol": {}}
    rows = load(RESULTS / "sobol_matched.jsonl")
    meta_path = RESULTS / "sobol_problem_matched.json"
    if not rows or not meta_path.exists():
        raise SystemExit("matched Sobol output not found")

    meta = json.loads(meta_path.read_text())
    problem, n = meta["problem"], meta["N"]
    for metric, basis, bloc, prefix in TARGETS:
        idx = sobol_indices(rows, problem, basis, bloc, metric, n, prefix=prefix)
        if idx:
            out["sobol"][f"{prefix}_{metric}|{basis}|{bloc}"
                         if prefix != "d" else f"{metric}|{basis}|{bloc}"] = idx

    (RESULTS / "analysis_matched.json").write_text(json.dumps(out, indent=1),
                                                   encoding="utf-8")

    base = json.loads((RESULTS / "analysis.json").read_text()).get("sobol", {})
    print(f"{'output':28s} {'param':24s} {'headline':>9s} {'matched':>9s}")
    for key in out["sobol"]:
        for name in ("m_H", "clt_domestic_sourcing"):
            h = base.get(key, {}).get("ST", {}).get(name)
            m = out["sobol"][key]["ST"].get(name)
            if h is None or m is None:
                continue
            print(f"{key:28s} {name:24s} {h:9.3f} {m:9.3f}")
    print("\nranks under matched ranges:")
    for key, idx in out["sobol"].items():
        ranked = [n for n, _ in sorted(idx["ST"].items(), key=lambda kv: -kv[1])]
        print(f"  {key:28s} m_H={ranked.index('m_H') + 1:2d}  "
              f"sourcing={ranked.index('clt_domestic_sourcing') + 1:2d}  (of {len(ranked)})")


if __name__ == "__main__":
    main()
