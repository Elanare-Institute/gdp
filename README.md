# Fugitive and non-fugitive provisioning: a closed-world simulation

Simulation code for the paper *From Fugitive Cash to Non-Fugitive
Provisioning*. It asks a narrow question: under the constraints a peripheral
economy actually faces — it must settle its imports in someone else's money,
and its lenders reassess it whenever a country like it fails — what does a
large cash transfer do, and what does supplying the same thing in kind do
instead?

The answer is reported as pressure and as orderings, never as a verdict. The
model covers the run-up to a bloc becoming unable to sustain itself and stops
there.

## What the model does

Four blocs by default, or a 25-point grid over the two axes that matter, run
quarterly for thirty years.

- **The world is closed.** One bloc's imports are another's exports; one
  bloc's capital outflow is another's inflow. World demand for tradables
  against world capacity sets a world price that every bloc pays through its
  exchange rate. The adding-up identities hold to machine precision and are
  asserted in the tests.
- **Transfers are nominal.** How the amount is updated — pinned to output,
  frozen, or indexed to last period's inflation — is a policy choice, so a
  transfer can lose its value and the cost of protecting it can feed the
  inflation it is protecting against.
- **Imports must be settled.** Non-reserve blocs are capped at exports plus
  reserves plus new borrowing, and the shortfall is rationed. The reserve
  issuer settles in money it prints, which is the one structural privilege
  the model grants it.
- **Defaults propagate.** When a bloc fails, lenders withdraw from everything
  that resembles it on the two axes. Adjustment stops being continuous.
- **Labour demand can fall worldwide.** One progress variable drives potential
  capacity up, unit costs down, constrained-household labour income down, and
  capital income up. Production is not potential: capacity nobody can buy from
  stands idle.

## The two axes

The four archetypes bundle everything together — the bloc that cannot borrow
is also the one that must import its food — so nothing computed on them can
say which does the work. `model/twoaxis.py` samples the two independently:

- **creditworthiness** — what lenders will advance, what they charge, what
  reserves the bloc holds
- **self-sufficiency** — how much of its food, goods and construction it must
  buy abroad

Reserve-currency status is a third, binary attribute, independent of both.

## Layout

```
model/          the simulation. Standard library only
  core.py         bloc dynamics and the two-pass world clearing
  trade.py        trade shares, adding-up, capital-account balance
  world.py        world price and capacity
  settlement.py   the settlement constraint and contagion
  indexation.py   how a transfer's nominal amount moves
  subsistence.py  whether the budget reaches the subsistence bundle
  automation.py   falling labour demand
  twoaxis.py      bloc generation on two independent axes
  policy.py       what authorities do when they fall behind

analysis/       experiments and figures (numpy, matplotlib)
specs/          what each phase was supposed to do, and why
reports/        what it actually did, including what was retracted
tests/          602 tests; legacy/ holds v6 and v7's regression suite
legacy/         v6 and v7, frozen. Do not modify
```

## Running it

```sh
uv sync --extra analysis --extra dev
uv run python -m pytest tests/ -q        # 602 passed, 3 skipped
uv run python analysis/v8a_run.py        # closing the world
uv run python analysis/v8c_run.py        # indexation
uv run python analysis/v8d_run.py        # settlement, on the two-axis plane
uv run python analysis/v8u_run.py        # the transfer needed to live on
uv run python analysis/v8pe_run.py       # falling labour demand
uv run python analysis/v8_phase_diagram.py
```

Each `*_run.py` writes JSON to `results/`; the matching `*_figures.py` reads it
and writes to `figures/`. Raw `.jsonl` output is not tracked — it is large and
regenerable — but the summaries the reports cite are.

Everything is seeded. The same inputs give the same outputs.

## Reading the results

`reports/` is the record, in order: `PHASE0`–`PHASE3` for v6 and v7, then
`PHASE_V8_A` (closing the world), `V8_RELEAK` (re-reading leakage in levels),
`PHASE_V8_C` (indexation), `PHASE_V8_D` (settlement and contagion),
`PHASE_V8_U` (sizing a transfer by what people need), `PHASE_V8_POSTEMP`
(falling labour demand).

The reports include the results that were withdrawn, and why. A leakage
ordering inverted once rates were read as levels. An fx gradient turned out to
be group structure. A self-defeating loop turned out to be a quarterly rate
multiplied as an annual one. A comparison of indexation rules was not on the
same footing until the transfer's own erosion entered the requirement. Each
correction is recorded where the original claim was made, rather than quietly
replaced.

## What it does not do

- **It stops at the point of collapse.** What follows economic breakdown —
  the loss of a government's authority, civil conflict, its externalisation as
  war, displacement — is social and political. Putting it in the model would
  mean coefficients deciding the conclusion.
- **It solves only constrained households.** Unconstrained households enter
  through a marginal propensity to consume and nothing more. Aggregate demand
  and aggregate consumption cannot be computed; every claim about quantities,
  welfare and budgets is about the constrained block.
- **It cannot show an allocation below subsistence.** The demand system clips
  there. The budget indicator says when the money stops reaching the bundle;
  what happens to a household in that position is outside the model.
- **It contains no trade-bloc formation, migration, conflict or policy
  reversal.** It therefore does not predict whether any real economy fails.

Thresholds in the code are reporting cut-offs, not structural boundaries, and
are swept rather than chosen. "CLT" stands for housing supplied as a
non-fugitive real asset, of which a community land trust is the example the
paper draws on — not a reconstruction of that institution. "Voucher" is a
control for establishing where an in-kind grant is equivalent to cash, not a
proposal.

## Licence

Not yet decided. Ask before redistributing.
