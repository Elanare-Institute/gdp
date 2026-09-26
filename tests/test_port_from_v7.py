"""The pieces carried over from v7 must still compute what they computed.

v8 drops bit-for-bit parity with v6 as a *system* requirement: closing the
world and nominalising transfers are incompatible with v6's premises. It does
not drop parity for the parts that were merely moved. The household block, the
heterogeneous block and the composite variables were all ported, and at
``world_price = 1.0`` they must agree with v7 exactly — that is what makes
them a port rather than a rewrite.

Any later divergence must therefore come from the world price moving, which is
the intended channel, not from an accident of transcription.

Specification: `specs/V8_A_B.md` §4.3.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import legacy.v7.heterogeneous as v7_hetero
import legacy.v7.household as v7_household
from legacy.v7.params import G as V7G, archetypes as v7_archetypes
from model.heterogeneous import hetero_household_block
from model.household import household_block
from model.params import G, archetypes

EXCHANGE_RATES = (0.7, 0.9, 1.0, 1.3, 2.0)
MODALITIES = ("none", "ubi", "cash_t", "voucher", "clt")
#: Ported code must agree to the last bit, not merely to plotting precision.
EXACT = 1e-12


def _pair(**kwargs):
    """Matched v8 and v7 globals."""
    return G(**kwargs), V7G(**kwargs)


def _blocs(key: str):
    return archetypes()[key], v7_archetypes()[key]


@pytest.mark.parametrize("key", ["A", "B", "C", "D"])
@pytest.mark.parametrize("e", EXCHANGE_RATES)
def test_household_block_matches_v7_at_unit_world_price(key: str, e: float) -> None:
    b8, b7 = _blocs(key)
    g8, g7 = _pair()
    x8 = household_block(b8, g8, b8.gdp, e, b8.rate, False, 0.0, world_price=1.0)
    x7 = v7_household.household_block(b7, g7, b7.gdp, e, b7.rate, False, 0.0)
    for field in ("pH", "rent", "imp_c", "dom_c", "fiscal", "U"):
        assert x8[field] == pytest.approx(x7[field], abs=EXACT), field


@pytest.mark.parametrize("modality", MODALITIES)
@pytest.mark.parametrize("e", EXCHANGE_RATES)
def test_household_block_matches_v7_for_every_modality(modality: str, e: float) -> None:
    b8, b7 = _blocs("D")
    g8, g7 = _pair()
    size = 0.0 if modality == "none" else 0.05
    on = modality != "none"
    x8 = household_block(replace(b8, modality=modality, size=size), g8, b8.gdp,
                         e, b8.rate, on, size, world_price=1.0)
    x7 = v7_household.household_block(replace(b7, modality=modality, size=size),
                                      g7, b7.gdp, e, b7.rate, on, size)
    for field in ("pH", "rent", "imp_c", "dom_c", "imp_gov", "dom_gov", "fiscal", "U"):
        assert x8[field] == pytest.approx(x7[field], abs=EXACT), f"{modality}/{field}"


@pytest.mark.parametrize("modality", MODALITIES)
def test_heterogeneous_block_matches_v7(modality: str) -> None:
    """Phase 2's household heterogeneity and build cap, ported intact."""
    b8, b7 = _blocs("D")
    g8, g7 = _pair(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
                   clt_build_rate=0.02)
    size = 0.0 if modality == "none" else 0.05
    on = modality != "none"
    x8 = hetero_household_block(replace(b8, modality=modality, size=size), g8,
                                b8.gdp, 1.0, b8.rate, on, size, world_price=1.0)
    x7 = v7_hetero.hetero_household_block(replace(b7, modality=modality, size=size),
                                          g7, b7.gdp, 1.0, b7.rate, on, size)
    assert x8.pH == pytest.approx(x7.pH, abs=EXACT)
    assert x8.fiscal == pytest.approx(x7.fiscal, abs=EXACT)
    assert [t.name for t in x8.types] == [t.name for t in x7.types]
    for t8, t7 in zip(x8.types, x7.types):
        assert t8.utility == pytest.approx(t7.utility, abs=EXACT), t8.name
        assert t8.is_tenant == t7.is_tenant, t8.name
    assert x8.tenant_share == pytest.approx(x7.tenant_share, abs=EXACT)


def test_build_cap_is_ported() -> None:
    """A finite build rate must still phase the programme in."""
    b8, _ = _blocs("D")
    g8, _ = _pair(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
                  clt_build_rate=0.02)
    bloc = replace(b8, modality="clt", size=0.05)
    early = hetero_household_block(bloc, g8, b8.gdp, 1.0, b8.rate, True, 0.05,
                                   years_since_start=0.5)
    late = hetero_household_block(bloc, g8, b8.gdp, 1.0, b8.rate, True, 0.05,
                                  years_since_start=40.0)
    assert early.pH > late.pH, "a part-built programme must relieve rent less"


# --- what the port is allowed to change ---

def test_a_higher_world_price_raises_traded_goods_prices() -> None:
    """The one intended divergence: v7 had no such channel."""
    b8, _ = _blocs("D")
    g8, _ = _pair()
    calm = household_block(b8, g8, b8.gdp, 1.0, b8.rate, False, 0.0, world_price=1.0)
    dear = household_block(b8, g8, b8.gdp, 1.0, b8.rate, False, 0.0, world_price=1.5)
    assert dear["U"] < calm["U"], "dearer imports must leave the household worse off"


def test_a_closed_bloc_is_insulated_from_the_world_price() -> None:
    """With no import content, the world price cannot reach the household."""
    b8, _ = _blocs("D")
    g8, _ = _pair()
    closed = replace(b8, m_F=0.0, m_G=0.0, m_H=0.0)
    calm = household_block(closed, g8, closed.gdp, 1.0, closed.rate, False, 0.0,
                           world_price=1.0)
    dear = household_block(closed, g8, closed.gdp, 1.0, closed.rate, False, 0.0,
                           world_price=2.0)
    assert dear["U"] == pytest.approx(calm["U"], abs=EXACT)
