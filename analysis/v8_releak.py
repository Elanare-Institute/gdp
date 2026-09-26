"""Re-read the Phase 1-3 leakage results in levels rather than in rates.

Phase A/B established that in a closed world it is the *amount* of demand a
programme sends to the world market that moves the world price, not the amount
per unit of fiscal cost. A modality can leak a large fraction of a small
programme and still press on the world market less than one that leaks a small
fraction of a large programme. Phases 1-3 reported rates, so their orderings
have to be re-read before they are carried into v8.

Two corrections are applied here, and they pull in opposite directions:

  levels    a rate is divided by fiscal cost; the level is not. Welfare-matched
            CLT costs about 0.4x what targeted cash costs, so its rate is
            inflated relative to its level by about 2.5x.

  goods     `leak` in Phase 1 is imports **plus** foreign-asset purchases.
            Only the import half buys goods on the world market; money that
            leaves as a claim on a foreign asset does not bid for tradables.
            Phase 3's `cheap_levels` already counted imports only, so the two
            phases were never measuring the same thing.

Phase 3's saved output is re-read directly. Phase 1's does not carry the
import/asset split, so the split is recomputed for the archetypes — the
smallest run that answers the question.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legacy.v7.core import build, simulate  # noqa: E402
from legacy.v7.params import G as V7G  # noqa: E402

RESULTS = ROOT / "results" / "v8a"
PHASE3 = ROOT / "results" / "phase3"
MODALITIES = ("ubi", "cash_t", "voucher", "clt")


def globals_for() -> V7G:
    """Phase 2/3 main series, so the re-read matches what was reported."""
    return V7G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
               clt_build_rate=0.02)


# --- Phase 3: re-read the saved three-way control -------------------------

def phase3_levels() -> dict[str, Any]:
    """CLT against targeted cash, in levels and in rates, from saved output.

    `cheap_levels.jsonl` stores fiscal cost and leakage as percentages of each
    bloc's own GDP, so the level comparison needs no new run.
    """
    path = PHASE3 / "cheap_levels.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out: dict[str, Any] = {"n_blocs": len(rows), "source": path.name}

    for m in ("cash_t", "clt", "cash_cheap"):
        level = [r[f"{m}_leak"] for r in rows]
        rate = [r[f"{m}_leak"] / r[f"{m}_fiscal"] for r in rows]
        fiscal = [r[f"{m}_fiscal"] for r in rows]
        out[m] = {
            "leak_level_mean": st.mean(level),
            "leak_level_median": st.median(level),
            "leak_rate_mean": st.mean(rate),
            "fiscal_mean": st.mean(fiscal),
        }

    # The comparison Phase 3 §5 rests on, stated both ways.
    for other in ("cash_t", "cash_cheap"):
        d_level = [r["clt_leak"] - r[f"{other}_leak"] for r in rows]
        d_rate = [r["clt_leak"] / r["clt_fiscal"] - r[f"{other}_leak"] / r[f"{other}_fiscal"]
                  for r in rows]
        out[f"clt_vs_{other}"] = {
            "level_mean_diff": st.mean(d_level),
            "level_share_positive": sum(1 for x in d_level if x > 0) / len(d_level),
            "rate_mean_diff": st.mean(d_rate),
            "rate_share_positive": sum(1 for x in d_rate if x > 0) / len(d_rate),
        }
    return out


# --- Phase 1: the import / foreign-asset split ----------------------------

def import_share_of_leakage(u: float = 0.05) -> dict[str, Any]:
    """How much of each modality's leakage actually buys goods.

    Phase 1's `leak_abs` counts imports and foreign-asset purchases together.
    Only imports reach the world goods market, so a modality whose leakage is
    mostly financial presses on the world price less than its leak_abs
    suggests. The split is not in the saved file; this recomputes it for the
    four archetypes, which is enough to establish the ordering.
    """
    g = globals_for()
    out: dict[str, Any] = {"u": u}
    for modality in MODALITIES:
        result = simulate(build(g, modality, u), g)
        per_bloc = {}
        for name, cell in result["leakage"].items():
            imp = cell["import_per_fiscal"]
            fa = cell["foreign_assets_per_fiscal"]
            total = cell["external_leak_per_fiscal"]
            per_bloc[name] = {
                "import_per_fiscal": imp,
                "foreign_assets_per_fiscal": fa,
                "goods_share_of_leak": imp / total if total else None,
                "leak_abs": cell["leak_abs"],
                "fiscal_pct_gdp": cell["fiscal_pct_gdp"],
                # The level that matters in a closed world: the programme's
                # own import demand as a share of the bloc's GDP.
                "import_level_pct_gdp": cell["leak_abs"] * (imp / total) if total else 0.0,
            }
        out[modality] = per_bloc
    return out


# --- what changes, stated as an ordering ----------------------------------

def orderings(imports: dict[str, Any], bloc: str = "D (developing)") -> dict[str, Any]:
    """Rank the modalities three ways and report where the ranks disagree."""
    keys = [m for m in MODALITIES if bloc in imports[m]]

    def rank(metric):
        return sorted(keys, key=lambda m: imports[m][bloc][metric])

    return {
        "bloc": bloc,
        "by_leak_per_fiscal": rank("import_per_fiscal"),
        "by_leak_abs": rank("leak_abs"),
        "by_import_level": rank("import_level_pct_gdp"),
        "values": {m: {k: imports[m][bloc][k] for k in
                       ("import_per_fiscal", "leak_abs", "import_level_pct_gdp",
                        "goods_share_of_leak", "fiscal_pct_gdp")}
                   for m in keys},
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {
        "phase3": phase3_levels(),
        "phase1_split": import_share_of_leakage(),
    }
    report["orderings"] = {
        b: orderings(report["phase1_split"], b)
        for b in report["phase1_split"]["cash_t"]
    }
    path = RESULTS / "releak.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {path}\n")

    p3 = report["phase3"]
    print(f"=== Phase 3 three-way control ({p3['n_blocs']} blocs, {p3['source']}) ===")
    print(f"  {'modality':11s} {'fiscal':>8s} {'leak level':>11s} {'leak rate':>10s}")
    for m in ("cash_t", "clt", "cash_cheap"):
        c = p3[m]
        print(f"  {m:11s} {c['fiscal_mean']:8.4f} {c['leak_level_mean']:11.4f} "
              f"{c['leak_rate_mean']:10.4f}")
    for other in ("cash_t", "cash_cheap"):
        c = p3[f"clt_vs_{other}"]
        print(f"\n  CLT vs {other}:")
        print(f"    level: {c['level_mean_diff']:+.4f}  "
              f"CLT higher in {c['level_share_positive']:.0%} of blocs")
        print(f"    rate:  {c['rate_mean_diff']:+.4f}  "
              f"CLT higher in {c['rate_share_positive']:.0%} of blocs")

    print("\n=== Which half of 'leakage' buys goods (archetypes, u=0.05) ===")
    print(f"  {'bloc':16s} {'modality':9s} {'imp/fisc':>9s} {'fa/fisc':>8s} "
          f"{'goods%':>7s} {'leak_abs':>9s} {'import lvl':>11s}")
    for bloc in report["phase1_split"]["cash_t"]:
        for m in MODALITIES:
            c = report["phase1_split"][m][bloc]
            gs = c["goods_share_of_leak"]
            print(f"  {bloc[:15]:16s} {m:9s} {c['import_per_fiscal']:9.3f} "
                  f"{c['foreign_assets_per_fiscal']:8.3f} "
                  f"{(f'{gs:.0%}' if gs is not None else '—'):>7s} "
                  f"{c['leak_abs']:9.4f} {c['import_level_pct_gdp']:11.4f}")

    print("\n=== Do the rankings change? ===")
    for bloc, o in report["orderings"].items():
        same = o["by_leak_per_fiscal"] == o["by_import_level"]
        print(f"  {bloc[:15]:16s} rate: {'>'.join(o['by_leak_per_fiscal']):32s} "
              f"level: {'>'.join(o['by_import_level']):32s} {'same' if same else 'DIFFERENT'}")


if __name__ == "__main__":
    main()
