"""Regression: the modular model must reproduce ``legacy/sim_v6.py`` exactly.

The legacy script is imported and run in-process, so the comparison is against
the original code rather than a stored snapshot that could drift.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
LEGACY = ROOT / "legacy" / "sim_v6.py"

from legacy.v7.core import run_scenarios, simulate, build  # noqa: E402
from legacy.v7.params import G, archetypes  # noqa: E402


@pytest.fixture(scope="module")
def legacy():
    """Import legacy/sim_v6.py as a module."""
    spec = importlib.util.spec_from_file_location("sim_v6_legacy", LEGACY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



# v6's output schema. v7 adds fields (extra leakage normalisations, a crisis
# variant that drops the appreciation trigger); the contract is that every
# field v6 produced still carries the identical value, not that the dicts are
# equal. `project` strips the additions so the comparison is like-for-like.
V6_LEAK_KEYS = ("import_per_fiscal", "foreign_assets_per_fiscal",
                "external_leak_per_fiscal", "rent_capture_per_fiscal",
                "fiscal_pct_gdp", "external_leak_pct_gdp")
V6_CRISIS_KEYS = ("onsets", "duration_q", "severity", "insolvent_year")


def project(result: dict) -> dict:
    """Restrict a v7 result to the fields v6 also produced."""
    out = {"final": result["final"], "crisis": {}, "leakage": {}}
    for bloc, value in result["crisis"].items():
        out["crisis"][bloc] = (value if bloc == "_system"
                               else {k: value[k] for k in V6_CRISIS_KEYS})
    for bloc, value in result["leakage"].items():
        out["leakage"][bloc] = {k: value[k] for k in V6_LEAK_KEYS}
    return out


def project_scenarios(res: dict) -> dict:
    """Apply :func:`project` across a full ``run_scenarios`` result."""
    return {"calibration": res["calibration"],
            "runs": {k: project(v) for k, v in res["runs"].items()}}


def _legacy_scenarios(legacy, u: float, simultaneous: bool = False) -> dict:
    """Reproduce ``sim_v6.main``'s result dict without going through argparse."""
    g = legacy.G()
    starts = (1, 1, 1, 1) if simultaneous else (1, 5, 10, 15)
    res = {"calibration": {k: legacy.calibrate(b, g, u)
                           for k, b in legacy.archetypes().items()},
           "runs": {}}
    for basis in ("welfare", "cost"):
        for mod in ("none", "ubi", "cash_t", "voucher", "clt"):
            if basis == "cost" and mod in ("none", "ubi"):
                continue
            r = legacy.simulate(legacy.build(g, mod, u, starts, basis), g)
            r.pop("history")
            res["runs"][f"{basis}:{mod}"] = r
    return res


@pytest.mark.parametrize("u", [0.03, 0.05, 0.10])
def test_full_scenario_set_matches_legacy(legacy, u: float) -> None:
    """Every scenario's final / crisis / leakage block matches, bit for bit."""
    new = project_scenarios(run_scenarios(G(), u))
    old = _legacy_scenarios(legacy, u)
    assert json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True)


def test_simultaneous_start_matches_legacy(legacy) -> None:
    """The simultaneous-introduction variant also matches."""
    new = project_scenarios(run_scenarios(G(), 0.05, simultaneous=True))
    old = _legacy_scenarios(legacy, 0.05, simultaneous=True)
    assert json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True)


@pytest.mark.parametrize("basis", ["welfare", "cost"])
@pytest.mark.parametrize("mod", ["none", "ubi", "cash_t", "voucher", "clt"])
def test_each_scenario_blockwise(legacy, basis: str, mod: str) -> None:
    """Per-scenario comparison, so a failure names the scenario that broke."""
    if basis == "cost" and mod in ("none", "ubi"):
        pytest.skip("equal-cost basis is defined only against UBI")

    g_new, g_old = G(), legacy.G()
    new = simulate(build(g_new, mod, 0.05, (1, 5, 10, 15), basis), g_new)
    old = legacy.simulate(legacy.build(g_old, mod, 0.05, (1, 5, 10, 15), basis), g_old)
    new.pop("history")
    old.pop("history")
    projected = project(new)

    assert projected["final"] == old["final"]
    assert projected["crisis"] == old["crisis"]
    assert projected["leakage"] == old["leakage"]


def test_calibration_matches_legacy(legacy) -> None:
    """Welfare-equivalent sizes agree for every archetype."""
    g_new, g_old = G(), legacy.G()
    for key, b_new in archetypes().items():
        b_old = legacy.archetypes()[key]
        assert calibration_equal(
            __import__("legacy.v7.household", fromlist=["calibrate"]).calibrate(b_new, g_new, 0.05),
            legacy.calibrate(b_old, g_old, 0.05),
        ), key


def calibration_equal(a: dict, b: dict) -> bool:
    """Compare two calibration dicts exactly."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_new_flags_default_to_v6(legacy) -> None:
    """The v7 flags must be inert at their defaults.

    This is the guarantee CLAUDE.md requires: new features are switchable and
    reproduce v6 when switched off.
    """
    g = G()
    assert g.reserve_flight_mode == "v6"
    assert g.universal_split is None

    explicit = G(reserve_flight_mode="v6", universal_split=None)
    assert json.dumps(project_scenarios(run_scenarios(explicit, 0.05)), sort_keys=True) == \
           json.dumps(_legacy_scenarios(legacy, 0.05), sort_keys=True)


# --- legacy archive completeness -------------------------------------------

LEGACY_FILES = ("sim.py", "sim_robustness_2.py", "sim_v6.py")


@pytest.mark.parametrize("name", LEGACY_FILES)
def test_legacy_file_is_present(name: str) -> None:
    """All three archived implementations are in place and non-empty.

    CLAUDE.md requires these be kept unchanged as the provenance record for
    the v6 rewrite.
    """
    path = ROOT / "legacy" / name
    assert path.exists(), f"legacy/{name} is missing"
    assert path.stat().st_size > 0


@pytest.mark.parametrize("name", LEGACY_FILES)
def test_legacy_file_imports(name: str) -> None:
    """Each archived script still loads, so it can be run for comparison."""
    spec = importlib.util.spec_from_file_location(
        f"legacy_check_{name[:-3]}", ROOT / "legacy" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module is not None


def test_v1_v5_carried_a_clt_specific_cost_coefficient() -> None:
    """The pre-v6 code hard-coded a CLT-only cost multiplier of 0.3.

    This is the coefficient CLAUDE.md now forbids ("no modality-specific
    dampening or dummy coefficients"). sim.py fixes it at 0.3; the robustness
    variant only parameterises it. Pinned here as the provenance of what v6
    removed, and to make it obvious if it ever reappears in model/.
    """
    original = (ROOT / "legacy" / "sim.py").read_text(encoding="utf-8")
    variant = (ROOT / "legacy" / "sim_robustness_2.py").read_text(encoding="utf-8")

    assert "ubi_cost = s.ubi_level * 0.3" in original
    assert "clt_cost_ratio" in variant
    assert "ubi_cost = s.ubi_level * 0.3" not in variant


def test_current_model_has_no_modality_cost_coefficient() -> None:
    """The v6+ model must not reintroduce a CLT-specific cost multiplier."""
    for module in ("core.py", "household.py", "params.py"):
        text = (ROOT / "legacy" / "v7" / module).read_text(encoding="utf-8")
        assert "clt_cost_ratio" not in text, module
        assert "ubi_level * 0.3" not in text, module
