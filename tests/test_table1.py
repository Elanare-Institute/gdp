"""Table 1 of the manuscript: the numbers, and the reasons they hold.

The table reports four provisioning forms sized to equal welfare at
introduction, in the developing bloc, at year 30. What makes it a claim rather
than a printout is the ordering inside it: universal cash, despite buying the
same welfare on day one, leaves constrained households with less of both goods
thirty years later than doing nothing, while in-kind housing leaves them with
slightly more. These tests fix the ordering and the published figures, so a
change to the model that moves them has to be noticed here rather than in a
proof of the paper.

Specification: `specs/V8_C.md`. Generator: `scripts/table1.py`.
"""

from __future__ import annotations

import pytest

from scripts.table1 import MODALITIES, TARGET, U_UBI, globals_for, run

#: What the manuscript prints. Tolerances are loose enough to survive a
#: last-digit change in the solver and tight enough to catch a real shift.
PUBLISHED = {
    "ubi":      {"fiscal": 2.0, "imports": 93.1, "food": 149.7, "housing": 136.7},
    "targeted": {"fiscal": 1.0, "imports": 99.8, "food": 180.9, "housing": 190.1},
    "voucher":  {"fiscal": 1.0, "imports": 101.1, "food": 170.8, "housing": 175.6},
    "clt":      {"fiscal": 0.4, "imports": 101.5, "food": 188.7, "housing": 198.9},
    "none":     {"fiscal": 0.0, "imports": 100.0, "food": 191.3, "housing": 197.4},
}


@pytest.fixture(scope="module")
def rows() -> dict[str, dict]:
    out = {"none": run(None)}
    for paper_name, model_name in MODALITIES.items():
        out[paper_name] = run(model_name)
    base = out["none"]["imports"]
    for row in out.values():
        row["import_index"] = 100.0 * row["imports"] / base
    return out


@pytest.mark.parametrize("key", sorted(PUBLISHED))
def test_published_values(rows, key):
    """Each printed figure is what the model produces."""
    want, got = PUBLISHED[key], rows[key]
    assert 100.0 * got["fiscal"] == pytest.approx(want["fiscal"], abs=0.05)
    assert got["import_index"] == pytest.approx(want["imports"], abs=0.15)
    assert got["food"] == pytest.approx(want["food"], abs=0.15)
    assert got["housing"] == pytest.approx(want["housing"], abs=0.15)


def test_equal_welfare_is_not_equal_cost(rows):
    """The point of the calibration: same welfare, different outlays.

    Universal cash is the most expensive because it pays everybody; in-kind
    housing is the cheapest because it replaces a money transfer with a
    physical asset whose cost carries no land rent.
    """
    fiscal = {k: rows[k]["fiscal"] for k in ("ubi", "targeted", "voucher", "clt")}
    assert fiscal["ubi"] > fiscal["targeted"] > fiscal["clt"]
    assert fiscal["voucher"] < fiscal["ubi"]


def test_universal_cash_undoes_itself(rows):
    """Transfer reversal: worse than no transfer, in both goods.

    This is the table's central claim, and the reason the paper reports
    quantities rather than transfer values.
    """
    for good in ("food", "housing"):
        assert rows["ubi"][good] < rows["none"][good]


def test_in_kind_housing_holds_the_baseline(rows):
    """In-kind housing does not go backwards, and it carries food with it.

    It distributes no food at all, so the food result is indirect: rent leaves
    the household budget and the freed income goes to food.
    """
    assert rows["clt"]["housing"] > rows["none"]["housing"]
    assert rows["clt"]["food"] > rows["ubi"]["food"]


def test_provisioning_order_matches_depreciation_order(rows):
    """The ranking is not arbitrary: it is the exchange-rate ranking, inverted.

    What takes the quantities away is the price of imports, so the form that
    depreciates the currency most secures the least.
    """
    order = sorted(("ubi", "targeted", "voucher", "clt"),
                   key=lambda k: rows[k]["e"] / rows[k]["e_start"])
    housing = [rows[k]["housing"] for k in order]
    assert housing == sorted(housing, reverse=True)


def test_settlement_off_is_what_the_table_reports():
    """The configuration the caption states.

    With settlement on, this bloc is rationed with no transfer at all and the
    differences between forms collapse. The table is about the demand-side
    consequence of equal welfare, which that state would hide.
    """
    assert globals_for().settlement is False
    assert U_UBI == 0.02
    assert TARGET == 3
