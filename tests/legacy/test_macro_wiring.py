"""Every modality must actually reach the macro block.

`cash_cheap` was added to `model/heterogeneous.py` but not to
`model/household.py`, so macro runs silently treated it as a zero transfer:
its debt and exchange-rate paths were identical to no transfer at all, and a
Phase 3 conclusion was drawn from that. These tests close that hole for every
modality the model can run, including the ones only the mix layer uses.

The specification being tested is: a positive programme, whatever its
modality, moves at least one macro state variable away from the no-transfer
path. The test asserts a difference, not its sign or size.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from legacy.v7.core import simulate
from legacy.v7.evolution import _NAME_MAP
from legacy.v7.household import calibrate
from legacy.v7.params import G, archetypes

# Every modality the household block is expected to implement.
MACRO_MODALITIES = ("ubi", "cash_t", "voucher", "clt", "clt_cash", "cash_cheap")
U = 0.05


def _paths(modality: str, size: float) -> tuple[list[float], list[float]]:
    """Debt and exchange-rate paths for bloc D under one modality."""
    g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
          clt_build_rate=0.02)
    b = archetypes()["D"]
    bloc = replace(b, modality=modality, size=size, start=1.0)
    hist = simulate([bloc], g)["history"]
    return [h["d_0"] for h in hist], [h["e_0"] for h in hist]


def _size(modality: str) -> float:
    """Welfare-matched size, or CLT's fiscal cost for the cheap control."""
    g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
          clt_build_rate=0.02)
    cal = calibrate(archetypes()["D"], g, U)
    if modality == "cash_cheap":
        return cal.get("_fiscal_clt", U)
    if modality == "clt_cash":
        return cal.get("clt", U)
    return cal.get(modality, U)


@pytest.mark.parametrize("modality", MACRO_MODALITIES)
def test_modality_moves_the_macro_path(modality: str) -> None:
    """A positive programme must not leave the macro path at no-transfer."""
    size = _size(modality)
    assert size > 0, f"{modality} calibrated to a zero programme"
    base_d, base_e = _paths("none", 0.0)
    d, e = _paths(modality, size)
    assert (d, e) != (base_d, base_e), (
        f"{modality} produced the no-transfer debt and FX paths exactly: "
        "it is not wired into the macro household block"
    )


@pytest.mark.parametrize("modality", MACRO_MODALITIES)
def test_modality_is_fiscally_costly(modality: str) -> None:
    """A programme that moves no fiscal cost cannot be a transfer."""
    from legacy.v7.household import household_block

    g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
          clt_build_rate=0.02)
    b = replace(archetypes()["D"], modality=modality, size=_size(modality))
    hb = household_block(b, g, b.gdp, 1.0, b.rate, True, b.size)
    assert hb["fiscal"] > 0, f"{modality} records no fiscal cost in the macro block"


def test_name_map_covers_every_mix_component() -> None:
    """Every mix component must map to a modality the macro block implements."""
    for key, macro in _NAME_MAP.items():
        assert macro in MACRO_MODALITIES, f"{key} maps to unknown modality {macro}"
