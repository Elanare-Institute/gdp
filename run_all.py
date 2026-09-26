#!/usr/bin/env python3
"""Batch runner for the v7 phases.

Usage::

    python run_all.py --phase 0                    # baseline, v6-identical
    python run_all.py --phase 0 --config configs/phase0_robustness.yaml
    python run_all.py --phase 0 --compare-flight   # all reserve_flight_mode values

Results are written to ``results/phase<N>/`` as JSON Lines plus a pretty JSON
copy of the full run. Only the standard library is used here so the runner
works without the analysis extras installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.core import run_scenarios  # noqa: E402
from model.params import G  # noqa: E402

RESULTS = ROOT / "results"


def read_config(path: Path) -> dict[str, Any]:
    """Read a flat ``key: value`` YAML subset.

    Only the scalar forms this project uses are supported, which keeps the
    runner free of a YAML dependency.
    """
    cfg: dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if value in ("null", "~", ""):
            cfg[key] = None
        elif value in ("true", "false"):
            cfg[key] = value == "true"
        else:
            try:
                cfg[key] = float(value) if ("." in value or "e" in value.lower()) else int(value)
            except ValueError:
                cfg[key] = value.strip("'\"")
    return cfg


def globals_from(cfg: dict[str, Any]) -> G:
    """Build a :class:`G` from a config dict, ignoring run-level keys."""
    skip = {"u", "simultaneous"}
    return G(**{k: v for k, v in cfg.items() if k not in skip and hasattr(G, k)})


def summarize(res: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a run into one record per (scenario, bloc) for JSON Lines."""
    rows: list[dict[str, Any]] = []
    for scenario, run in res["runs"].items():
        basis, modality = scenario.split(":")
        for bloc, fin in run["final"].items():
            crisis = run["crisis"].get(bloc, {})
            leak = run["leakage"].get(bloc, {})
            rows.append({
                "basis": basis, "modality": modality, "bloc": bloc,
                "final_debt": fin["debt"], "final_e": fin["e"],
                "final_pi": fin["pi"], "final_Y": fin["Y"],
                "onsets": crisis.get("onsets"), "duration_q": crisis.get("duration_q"),
                "severity": crisis.get("severity"), "insolvent_year": crisis.get("insolvent_year"),
                "external_leak_per_fiscal": leak.get("external_leak_per_fiscal"),
                "rent_capture_per_fiscal": leak.get("rent_capture_per_fiscal"),
                "fiscal_pct_gdp": leak.get("fiscal_pct_gdp"),
                "external_leak_pct_gdp": leak.get("external_leak_pct_gdp"),
            })
    return rows


def write(out_dir: Path, tag: str, res: dict[str, Any]) -> None:
    """Write both the full JSON and the flattened JSON Lines."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with (out_dir / f"{tag}.jsonl").open("w", encoding="utf-8") as fh:
        for row in summarize(res):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {out_dir.name}/{tag}.json and .jsonl")


def leak_table(res: dict[str, Any]) -> str:
    """Render the external-leakage table that Phase 0 must reproduce."""
    lines = ["", "External leakage per unit of fiscal cost (welfare-equivalent basis)",
             f"  {'bloc':22s} {'cash_t':>9s} {'clt':>9s} {'clt-cash':>10s}"]
    cash = res["runs"]["welfare:cash_t"]["leakage"]
    clt = res["runs"]["welfare:clt"]["leakage"]
    for bloc in cash:
        c, k = cash[bloc]["external_leak_per_fiscal"], clt[bloc]["external_leak_per_fiscal"]
        lines.append(f"  {bloc:22s} {c:9.3f} {k:9.3f} {k - c:+10.3f}")
    return "\n".join(lines)


def phase0(args: argparse.Namespace) -> None:
    """Phase 0: reproduce v6 and run the reserve-flight robustness check."""
    cfg = read_config(Path(args.config))
    u = float(cfg.get("u", 0.05))
    simultaneous = bool(cfg.get("simultaneous", False))
    out = RESULTS / "phase0"

    modes = ("v6", "off", "capped") if args.compare_flight else (cfg.get("reserve_flight_mode", "v6"),)
    for mode in modes:
        g = globals_from({**cfg, "reserve_flight_mode": mode})
        print(f"running phase 0: u={u} simultaneous={simultaneous} reserve_flight_mode={mode}")
        res = run_scenarios(g, u, simultaneous=simultaneous)
        res["_config"] = {**cfg, "reserve_flight_mode": mode}
        write(out, f"u{u:g}_{mode}" + ("_sim" if simultaneous else ""), res)
        if mode == "v6":
            print(leak_table(res))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", type=int, required=True, help="phase number to run")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--compare-flight", action="store_true",
                    help="run all reserve_flight_mode values side by side")
    args = ap.parse_args()

    if args.phase == 0:
        phase0(args)
    else:
        raise SystemExit(f"phase {args.phase} is not implemented yet")


if __name__ == "__main__":
    main()
