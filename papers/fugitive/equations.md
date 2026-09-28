# Equations extracted from the model code

Every equation below is transcribed from the v8 core in `model/`. Nothing is
added, simplified, or completed: where the code has no equation for something
Section 3 describes in words, that is recorded as **[no equation in code]**
rather than filled in.

**Source of the extraction.** The instruction named `simulation/src/model.py`
and `simulation/src/config.py`. Those two files are a two-sector CES general
equilibrium model of the aggregate labour share, with fugacity `phi` as an
*input* parameter acting through four channels; they contain no Stone–Geary
demand, no transfer modality, no world market, no settlement constraint, no
automation, and no `step()` method. None of the equations Section 3 requires
exist there. The model Section 3 describes is `model/` (v8 core, 3512 lines),
and that is what was read:

| Module | Lines | Supplies |
|---|---|---|
| `model/household.py` | 289 | §3.3 demand, §3.4 modalities and calibration |
| `model/subsistence.py` | 203 | §3.8 budget ratio, required transfer |
| `model/world.py` | 155 | §3.5 demand-pressure index |
| `model/settlement.py` | 176 | §3.6 settlement, borrowing, contagion |
| `model/automation.py` | 108 | §3.7 automation |
| `model/indexation.py` | 146 | §3.4 cost-of-living adjustment |
| `model/core.py` | 873 | §3.1 period order, exchange rate, inflation |
| `model/trade.py` | 127 | §3.2 closed-world accounting |
| `model/policy.py` | 79 | §3.6 policy response |
| `model/twoaxis.py` | 153 | §3.2 the two-axis plane |
| `model/params.py` | 521 | parameter table |
| `model/heterogeneous.py` | 537 | quantile detail, CLT amortisation |
| `model/collapse.py` | 145 | world pressure measures |
| `analysis/v8pe_run.py` | — | §3.8 feasibility classification |

Notation is converted to paper form throughout. `P^W` denotes the
demand-pressure index the code calls `world.price`; see the note at §3.5 on why
it is not a world price level.

---

## Notation

| Paper | Code | Meaning |
|---|---|---|
| $P^W_t$ | `world.price` | demand-pressure index on traded goods |
| $Z_t$ | `world.capacity` | world traded-goods supply capacity, real |
| $e_{it}$ | `s["e"]` | bloc $i$'s nominal exchange rate |
| $m^F_i, m^G_i, m^H_i$ | `b.m_F, b.m_G, b.m_H` | import content of food, general goods, construction |
| $p^F_{it}, p^G_{it}, p^H_{it}$ | `pF, pG, pH` | bloc $i$'s prices of food, general goods, housing |
| $\gamma^k$ | `g.sub_F, sub_G, sub_H` | subsistence coefficient for good $k$ |
| $\beta^k$ | `g.beta_F, beta_G, beta_H` | marginal budget share for good $k$ |
| $I^c_{it}$ | `Ic` | constrained block's income |
| $\sigma_i$ | `b.c_income_share` | constrained block's income share of GDP |
| $n_i$ | `b.c_pop_share` | constrained share of population |
| $Y_{it}$ | `s["Y"]` | bloc output |
| $\Pi_{it}$ | `s["P"]` | bloc price level |
| $\pi_{it}$ | `s["pi"]` | bloc inflation, annual |
| $T_{it}$ | `nominal` | transfer, nominal |
| $u_i$ | `b.size` | programme size parameter |
| $s_{it}$ | `s["scarcity"]` | share of last period's imports left unsettled |
| $a_t$ | `progress` | automation progress |
| $\Delta$ | `DT` $= 0.25$ | quarter, in years |

---

## §3.3 Household demand

### Subsistence requirement

Two floors coexist in the code, deliberately and with different definitions.
`model/subsistence.py` states the inconsistency in its module docstring and
records it as belonging in the paper's limitations.

**The behavioural floor**, which enters the household's own problem
(`household.py:122`), is proportional to current income:

$$\bar{x}^k_{it} = \gamma^k \, I^c_{it}, \qquad k \in \{F, G, H\}$$

$$I^c_{it} = \sigma_i \, Y_{it} \, \phi^{\text{inc}}_{it} \, \lambda_t$$

where $\gamma^F = 0.35$, $\gamma^G = 0.05$, $\gamma^H = 0.25$;
$\phi^{\text{inc}}_{it}$ is the income-linkage factor (below) and $\lambda_t$
the automation labour multiplier (§3.7). Note `g.subsistence_scale` does
**not** enter here — `household.py:117` states this explicitly, because letting
it in would change the allocation and the welfare calibration.

**The measured floor**, used only by the budget indicator
(`subsistence.py:77`), is a *quantity* fixed once at a reference income
$I^{c,\text{ref}}_i$ and then held constant:

$$\hat{x}^k_{iq} = \gamma^k \, I^{c,\text{ref}}_i \, \theta \, \zeta_q$$

where $\theta$ = `g.subsistence_scale` and $\zeta_q$ = `t.sub_scale` is the
quantile's subsistence tilt. The two floors disagree, and
`subsistence.py:33-39` states why: a floor proportional to income cannot fall
short, because if income halves so does the floor.

### Stone–Geary / LES demand

From `les_demand` (`household.py:29-71`). Let $M$ be cash income available for
market purchases and $f^k$ the in-kind quantity received of good $k$
(zero unless a voucher or CLT grant is running). Full income is

$$\tilde{M} = M + \sum_k p^k f^k$$

and supernumerary income is

$$\Omega = \tilde{M} - \sum_k p^k \bar{x}^k$$

The interior allocation is

$$x^k = \bar{x}^k + \frac{\beta^k \max(\Omega, 0)}{p^k}, \qquad \sum_k \beta^k = 1$$

with $\beta^F = 0.25$, $\beta^G = 0.45$, $\beta^H = 0.30$. The clip at zero is
an implementation measure, not a claim: `subsistence.py:7-14` records that the
model does not represent an allocation below subsistence, because Stone–Geary
utility is undefined there.

**The extramarginal corner.** When an in-kind grant exceeds what the household
would freely choose, that good's quantity is pinned and the remaining budget is
reallocated. Let $B = \{k : x^k < f^k\}$ be the binding set and
$\mathcal{F}$ its complement (`household.py:59-68`):

$$x^k = f^k \qquad \text{for } k \in B$$

$$x^k = \bar{x}^k + \frac{\beta^k}{\sum_{j \in \mathcal{F}} \beta^j} \cdot \frac{\max(\Omega', 0)}{p^k} \qquad \text{for } k \in \mathcal{F}$$

$$\Omega' = M + \sum_{k \notin B} p^k f^k - \sum_{k \in \mathcal{F}} p^k \bar{x}^k$$

This is what makes an *infra-marginal* voucher equivalent to cash and an
extramarginal one not; it is solved explicitly rather than assumed
(`household.py:44-48`).

Market expenditure is what the household buys over and above the grant:

$$b^k = p^k \max(0, \; x^k - f^k)$$

### Utility

From `utility` (`household.py:74-82`):

$$U = \sum_k \beta^k \ln(x^k - \bar{x}^k), \qquad U = -10^9 \text{ if any } x^k \le \bar{x}^k$$

### Prices

From `household.py:126-135`. The domestic component costs 1 by normalisation,
scaled by the automation cost multiplier $c_t$ (§3.7); the traded component is
the demand-pressure index converted at the bloc's exchange rate and marked up
by unsettled scarcity:

$$\tau_{it} = e_{it} \, P^W_t \cdot \frac{1}{\max(10^{-6},\; 1 - \min(0.95, s_{it}))}$$

$$p^F_{it} = \left[ (1 - m^F_i) \, c_t + m^F_i \, \tau_{it} \right] \cdot \varphi_{it}$$

$$p^G_{it} = (1 - m^G_i) \, c_t + m^G_i \, \tau_{it}$$

where $\varphi_{it}$ = `food_factor` is a food-price shock multiplier (1.0 in
the main series). Rationing enters as a *price*, not as an absent quantity:
`household.py:128-132` states that a bloc rationed to half its import demand
faces a traded-goods price that has risen enough to clear the quantity it can
actually pay for, which is how the settlement constraint reaches what a
household eats.

Housing rent $p^H_{it}$ is not a formula but the solution of a market-clearing
condition. Supply is anchored so the no-transfer economy clears at
$p^H = 1$ (`household.py:137`):

$$S^0_i = \bar{x}^H + \beta^H \left( I^c_i - \bar{x}^F - \bar{x}^G - \bar{x}^H \right)$$

and $p^H_{it}$ solves, by bisection on $[0.05, 20]$ over 60 iterations
(`household.py:197-210`),

$$\frac{b^H(p^H)}{p^H} = S^0_i \, (p^H)^{\varepsilon_i \, \mu}$$

where $\varepsilon_i$ = `b.eps_supply` is the short-run rental supply elasticity
and $\mu$ = `g.eps_mult`.

### Constrained / unconstrained

There is **no endogenous branching condition in the code.** The constrained
block is a fixed share of population $n_i$ = `b.c_pop_share` earning a fixed
share of income $\sigma_i$ = `b.c_income_share`, both structural parameters of
the bloc (`params.py:61-62`). Only the constrained block's household problem is
solved; the unconstrained block enters solely through a marginal propensity to
consume (`core.py:104-107`):

$$\Delta I^u_{it} = \text{cash}^u_{it} + \Delta \text{rent}_{it} + D_{it}$$

$$D_{it} = \sigma_i \, Y_{it} \left( 1 - \lambda_t \right)$$

where $D_{it}$ is labour income displaced by automation, accruing to the owners
of the machines, and $\text{cash}^u$ includes the land payment under CLT. Its
contribution to import demand is $\text{mpc}^u \, m^G_i \, \Delta I^u_{it}$
with $\text{mpc}^u = 0.3$.

This is the limitation CLAUDE.md requires be stated in one place: the
unconstrained household's problem is never solved, so aggregate provisioning
and aggregate demand cannot be summed. Within the constrained block, quantile
heterogeneity is available (`params.py:123-143`), with $k$ equal-population
quantiles whose income weights are

$$w_q = 1 + \rho \left( \frac{2q}{k-1} - 1 \right), \qquad q = 0, \ldots, k-1$$

$$\zeta_q = \min\left( 1.35, \; \left( \frac{\bar{w}}{w_q} \right)^{0.5} \right)$$

where $\rho$ = `g.income_spread` $= 0.6$ and $\bar{w}$ the mean weight. The
exponent 0.5 damps the tilt: `params.py:33-37` records that a full inverse tilt
drives the poorest quantile below subsistence before any transfer is made,
which would make its utility undefined rather than merely low.

### Import equation

From `household.py:214-241`. The code carries imports in **two measures that
are not interchangeable** — spending, for the balance of payments, and
quantity, for the world market:

$$M^{\text{sp}}_{it} = m^F_i \, b^F_{it} + m^G_i \, b^G_{it} \qquad \text{(spending)}$$

$$M^{\text{q}}_{it} = m^F_i \, x^F_{it} + m^G_i \, x^G_{it} \qquad \text{(quantity)}$$

$$\text{Dom}_{it} = (1 - m^F_i) \, b^F_{it} + (1 - m^G_i) \, b^G_{it}$$

Government procurement adds a second component. For CLT, construction is sized
in real units so its import content is already a quantity; the effective import
share is reduced by domestic-sourcing policy but the structural parameter
$m^H_i$ is not changed (`household.py:228-232`):

$$\tilde{m}^H_i = m^H_i \left( 1 - d_i \right), \qquad d_i = \texttt{clt\_domestic\_sourcing}$$

$$M^{\text{gov}}_{it} = \tilde{m}^H_i \, C^{\text{build}}_{it}$$

For a voucher, `fiscal` is a spending figure, so the real quantity procured is
that divided by the food price (`household.py:238-241`):

$$M^{\text{gov}}_{it} \mathrel{+}= m^F_i \, F_{it}, \qquad M^{\text{gov,q}}_{it} \mathrel{+}= m^F_i \, \frac{F_{it}}{p^F_{it}}$$

Total import demand facing the world, from `core.py:109-123`, is the
no-programme level plus the programme increment:

$$M_{it} = M^{\text{sp},0}_{it} + \underbrace{\left( M^{\text{sp},1}_{it} - M^{\text{sp},0}_{it} \right) + M^{\text{gov}}_{it} + \text{mpc}^u m^G_i \Delta I^u_{it}}_{\Delta M_{it}}$$

`core.py:118-121` records why the level and not the increment must clear: a
world-wide programme would otherwise show up as demand against zero capacity.

Note the deflator used for the unconstrained block's quantity contribution is
the *general good's* price, not $P^W$ (`core.py:110-113`):

$$\Delta M^{\text{q}}_{it} = \left( M^{\text{q},1}_{it} - M^{\text{q},0}_{it} \right) + M^{\text{gov,q}}_{it} + \frac{\text{mpc}^u \, m^G_i \, \Delta I^u_{it}}{p^G_{it}}$$

### Income linkage

From `_income_factor` (`core.py:147-170`). Three rules, with
$Y_{i0}$ the opening output:

$$\phi^{\text{inc}}_{it} = \begin{cases}
1 & \texttt{growth\_linked} \\
Y_{i0} / Y_{it} & \texttt{cpi\_linked} \\
Y_{i0} / \left( Y_{it} \max(10^{-6}, \Pi_{it}) \right) & \texttt{fixed\_nominal}
\end{cases}$$

Under `cpi_linked` income holds its opening real value, which against a growing
$Y$ means a falling share — `params.py:418-424` cites Van Mechelen & Marchal
(2012) for this being what minimum income protection actually does in most of
the OECD, and makes it the main series.

### Saving and foreign-asset allocation

From `_foreign_asset_purchase` (`core.py:132-145`). Unconstrained households
save $1 - \text{mpc}^u$ of what reaches them and place a share $\omega$ abroad:

$$\omega_{it} = \min\left( 0.9, \; \max\left( 0, \; \omega^0_i + \nu_\omega \left( rp_{it} + \max(0, \dot{e}_{it}/\Delta) \right) \right) \right)$$

$$FA_{it} = \left( 1 - \text{mpc}^u \right) \Delta I^u_{it} \, \omega_{it} \, o_i$$

where $\omega^0_i$ = `b.omega0` is the baseline foreign-asset share,
$\nu_\omega$ = `g.omega_sens` $= 2.0$, $rp_{it}$ the risk premium,
$\dot{e}_{it}$ = `s["de"]` last period's exchange-rate change, and $o_i$ =
`b.openness`. This is a **capital outflow, not a purchase of goods**
(`core.py:138-144`): it buys a claim on another bloc, so it enters the capital
account and not world demand for tradables. This is the assumption CLAUDE.md
flags as testable via `G.capital_inflow_to_demand` (default 0.0).

The risk premium it responds to, from `_risk_premium` (`core.py:183-189`):

$$rp_{it} = \kappa_r \max(0, d_{it} - \bar{d}) + \tfrac{\kappa_r}{2} \max(0, \pi_{it} - 0.06) + \tfrac{\kappa_r}{2} f_i \max(0, d_{it} - 0.6)$$

where $\kappa_r$ = `g.risk_coeff` $= 0.15$, $d_{it}$ the debt ratio, $f_i$ =
`b.fx_share` the foreign-currency share of debt, and $\bar{d} = 0.9$ unless the
reserve privilege sensitivity is on. A contagion term is added in `core.py:396`:

$$rp_{it} \mathrel{+}= \kappa_c \, \chi_{it}, \qquad \kappa_c = \texttt{contagion\_risk\_premium} = 0.05$$

---

## §3.4 Transfer modalities and welfare equivalence

### The programme budget

From `household.py:145-148`. Under an indexation rule the transfer is a nominal
amount carried across periods; without one it is a fixed share of real output:

$$\mathcal{B}_{it} = \begin{cases}
u_i \, Y_{it} & \text{no indexation rule (\texttt{gdp\_linked})} \\
T_{it} & \text{under an indexation rule}
\end{cases}$$

The `gdp_linked` case makes the transfer a *real* quantity immune to inflation,
which `indexation.py:1-7` records as the reason the paper's first claim could
not be tested before Phase C.

### The four modalities

From `household.py:150-195`.

**Universal cash (UBI).** The whole budget is paid out, split between the two
blocks by population share (`household.py:151-155`):

$$T^{\text{cash},c}_{it} = \mathcal{B}_{it} \, n_i, \qquad T^{\text{cash},u}_{it} = \mathcal{B}_{it} \left( 1 - n_i \right), \qquad F_{it} = \mathcal{B}_{it}$$

where $n_i$ = `b.c_pop_share` unless `g.universal_split` overrides it, and
$F_{it}$ is the fiscal cost.

**Targeted cash.** The whole budget goes to the constrained block
(`household.py:156-161`):

$$T^{\text{cash},c}_{it} = \mathcal{B}_{it}, \qquad T^{\text{cash},u}_{it} = 0, \qquad F_{it} = \mathcal{B}_{it}$$

**Food voucher.** The grant is a *quantity* of food, entering the demand system
as $f^F$ (`household.py:162-170`):

$$f^F_{it} = \begin{cases} u_i \, Y_{it} & \text{no indexation rule} \\ \mathcal{B}_{it} / p^F_{it} & \text{under an indexation rule} \end{cases}$$

$$F_{it} = p^F_{it} \, f^F_{it}$$

The second branch is how inflation reaches an in-kind transfer: what the budget
buys falls as food gets dearer (`household.py:163-167`).

**In-kind housing (CLT).** The grant is a real quantity of housing, entering as
$f^H$ (`household.py:171-195`):

$$f^H_{it} = \begin{cases} u_i \, Y_{it} & \text{no indexation rule} \\ \mathcal{B}_{it} & \text{under an indexation rule} \end{cases}$$

Note the asymmetry with the voucher: the CLT quantity is **not** divided by a
price. `household.py:180-184` states the reason — once built the units stay
built, which is the sense in which an in-kind programme holds its real value.

Fiscal cost splits into structure and land:

$$C^{\text{build}}_{it} = \left( 1 - \ell_i \right) f^H_{it} \left( 1 + \kappa_s \, d_i \right)$$

$$C^{\text{land}}_{it} = \ell_i \left( 1 - \delta_L \right) \frac{r_{it} + 0.02}{r_{i0} + 0.02} \, f^H_{it}$$

$$F_{it} = C^{\text{build}}_{it} + C^{\text{land}}_{it}$$

where $\ell_i$ = `b.land_share` is land rent's share of gross rent,
$\delta_L$ = `g.land_discount` $= 0.5$ the acquisition discount,
$\kappa_s$ = `g.sourcing_cost_kappa` the markup for domestic sourcing, $r_{i0}$
the pre-transfer interest level. `household.py:187-189` records why the markup
exists: local supply is thinner than the import market, so substituting away
from imports is paid for in price. The land payment accrues to landowners and
enters $\text{cash}^u$ (`household.py:227, 246`).

For `clt_cash`, a per-capita cash component is added alongside
(`household.py:176-178`):

$$T^{\text{cash},c}_{it} \mathrel{+}= u^{\text{cash}}_i Y_{it}, \qquad F_{it} \mathrel{+}= u^{\text{cash}}_i Y_{it}$$

### CLT durable-asset cost amortisation

From `annuity_factor` and the rent credit in `heterogeneous.py:30-41, 329-341`.
Rent is a flow and construction is a stock, so rent credited against the
programme's annual cost is annualised at the bloc's own cost of capital over
the building's life:

$$A(r_i, n) = \frac{r_i}{1 - (1 + r_i)^{-n}}, \qquad A = 1/n \text{ if } |r_i| < 10^{-12}$$

with $n$ = `g.housing_life_years` $= 40$. Under the main series
(`clt_rent_to_landlords = False`) the trust collects the rent on its own units
and credits it against cost:

$$R^{\text{CLT}}_{it} = R_{it} \cdot \frac{f^H_{it}}{S^0_i (p^H_{it})^{\varepsilon_i \mu} + f^H_{it}} \cdot A(r_i, n)$$

and $R_{it}$ is reduced by the uncredited amount. Deriving the factor from the
bloc's own interest rate means a high-rate bloc amortises faster, instead of
every bloc sharing one invented constant (`heterogeneous.py:34-36`).

A finite build rate phases the programme in (`heterogeneous.py:116-144`),
expressed as a share of the existing **housing stock** per year, not of GDP:

$$f^{H,\text{avail}}_{it} = \min\left( u_i Y_{it}, \; \beta_{\text{build}} \, H^{\text{stock}}_i \max(0, t - t^{\text{start}}_i) \right)$$

### Welfare-equivalent calibration

From `calibrate` (`household.py:251-274`), mirrored for quantiles in
`heterogeneous.py:394-427`. The calibration is a **bisection at introduction
prices**, not a closed form. With $Y_i = $ `b.gdp`, $e_i = 1$, $r_i = $
`b.rate`, the target is UBI's constrained-household utility at size $u$:

$$U^\star_i = U_i\!\left( \text{ubi}, \; u \right)$$

and for each $\text{mod} \in \{\text{cash\_t}, \text{voucher}, \text{clt}\}$
the calibrated size solves

$$u^{\text{mod}}_i = \{\, u' : U_i(\text{mod}, u') = U^\star_i \,\}$$

found by 60 bisection steps on $u' \in [0, 0.5]$. The search returns a bound
rather than a solution in two flagged cases: `_infeasible_` when
$U_i(\text{mod}, u') < U^\star_i - 10^{-6}$ at convergence, and `_at_ceiling_`
when $u'$ reaches the ceiling $0.5$ (`heterogeneous.py:421-427`).

An equal-*cost* alternative is also available
(`equal_cost_size`, `household.py:277-289`), solving
$F_i(\text{mod}, u') / Y_i = \bar{F}$ by the same bisection on $[0, 1]$.

### Cost-of-living adjustment

From `advance` (`indexation.py:104-146`). Three rules:

$$T_{it} = \begin{cases}
u_i \, Y_{it} & \texttt{gdp\_linked} \\
T_{i,t^{\text{start}}} & \texttt{fixed\_nominal} \\
T_{i,t-1} \left( 1 + \pi_{i,t-1} \Delta \right) & \texttt{cpi\_indexed}
\end{cases}$$

with the opening amount $T_{i,t^{\text{start}}} = u_i Y_{i,t^{\text{start}}}$ in
all three cases.

Two details the code is explicit about. First, indexation uses the **previous**
period's inflation (`indexation.py:19-21`): indexing to the current period would
make the transfer and the inflation it causes simultaneous equations, and real
indexation arrangements lag. Second, the rate is scaled by $\Delta$
(`core.py:75-78`): `pi` is annual and the model runs quarterly, so passing the
annual rate would index four times too fast and the transfer would outrun the
price level it is supposed to track.

Indexation also stops when a bloc's state stops being updated
(`core.py:66-71`), so a defaulted bloc's transfer cannot gain real value against
a frozen price level.

### Real value of the transfer

From `TransferState` (`indexation.py:65-101`). Two numeraires, because they
answer different questions:

$$\text{RV}_{it} = \frac{T_{it} / \Pi_{it}}{T_{i,t^{\text{start}}} / \Pi_{i,t^{\text{start}}}} \qquad \text{(CPI reading)}$$

$$\text{HV}_{it} = \frac{T_{it} / p^H_{it}}{T_{i,t^{\text{start}}} / p^H_{i,t^{\text{start}}}} \qquad \text{(housing reading)}$$

Both are reported as 1.0 before the programme starts. `indexation.py:87-94`
records the asymmetry the second measure exists to show: a transfer that holds
its value in consumption goods can still lose it in housing, because rent is a
price the transfer itself pushes up, and an in-kind housing programme has no
such problem.

---

## §3.5 World market and price formation

**$P^W$ is a demand-pressure index, not a world price level.** The code calls it
`world.price` and normalises it to 1 at the start; its role is mechanical, to
transmit aggregate demand pressure into each bloc's import price through the
exchange rate. It does not correspond to any observable aggregate. The
extraction preserves this reading throughout.

### The demand-pressure index

From `output_gap` and `step_world` (`world.py:96-145`). The gap is excess world
demand over capacity, as a fraction of capacity:

$$g_t = \frac{\sum_i M^{\text{q}}_{it}}{Z_t} - 1$$

The index evolves in logs, with a gap term and a slow reversion to a **fixed**
anchor $P^W_0 = 1$:

$$\hat{\pi}^W_t = \kappa_w \, g_t - \lambda_w \ln\!\left( \frac{P^W_t}{P^W_0} \right)$$

$$P^W_{t+1} = \min\left( \bar{P}, \; \max\left( 10^{-6}, \; P^W_t \exp\left[ \min(\hat{\pi}^W_t \Delta, \; 20) \right] \right) \right)$$

where $\kappa_w = 0.35$, $\lambda_w = 0.05$, and $\bar{P} = 10^{12}$ is a
**reporting limit, not a claim about where the world stops**
(`world.py:135-139`): a configuration sitting at the cap is one whose
indexation loop ran away.

Two implementation choices the code justifies at length. The reversion is taken
in logs because written on the level — $\lambda(P/P_0 - 1)$ — the restoring term
grows without bound in the price and drives the level negative
(`world.py:125-129`). And the anchor does **not** drift with the price, because
an anchor that did would not stop a transitory gap from shifting the level
permanently (`world.py:44-49`); a *sustained* gap still moves the level without
limit.

### Import price and the bloc's food price

From `import_price` (`world.py:148-155`) and `household.py:134`:

$$p^{\text{imp}}_{it} = P^W_t \, e_{it}$$

$$\boxed{\; p^F_{it} = (1 - m^F_i) \, c_t + m^F_i \, e_{it} \, P^W_t \;}$$

(shown here at $\varphi = 1$ and $s_{it} = 0$; the full form with the scarcity
markup and food shock is at §3.3.) A bloc with high self-sufficiency (low
$m^F_i$) is insulated from movements in $P^W$; a bloc with low
self-sufficiency is exposed. `household.py:7-14` records that at $P^W = 1$ this
is algebraically identical to the v7 form $p^F = 1 + m^F(e - 1)$.

### World capacity

From `world_growth` and `baseline_capacity_path` (`world.py:60-93`,
`core.py:191-241`). Capacity growth is demand-weighted, not supply-weighted
(`world.py:66-69`):

$$\hat{g}^Z_t = \frac{\sum_i w_i \max\left( 0.005, \; g^{\text{base}}_i - \delta_i t \right)}{\sum_i w_i}, \qquad w_i = M^{\text{q}}_{it}$$

The capacity **path is fixed once** from a no-transfer run and shared by every
configuration, so that a programme's demand cannot call forth the supply to
meet it (`core.py:191-205`). Opening capacity is normalised on the first
period's no-transfer demand, so the baseline opens at a zero gap
(`world.py:85-93`):

$$Z_0 = \sum_i M^{\text{q}}_{i0}$$

The path is a fixed point, iterated up to 12 rounds to tolerance $10^{-9}$
(`core.py:233-241`), each pass adopting the previous pass's realised demand.
`core.py:212-232` records why: adopting the assumed growth rates left capacity
outgrowing real demand by 3% over thirty years and the no-transfer world
drifting into a 0.45%/year deflation with no cause in the model.

### Bloc inflation

From `core.py:643-656`. Inflation is the sum of three pushes on the **change**
in the rate, not a level the rate is set to:

$$\text{world}_{it} = \frac{1}{\Delta} \left( \frac{P^W_{t+1} \max(0.05, e_{it} + \dot{e}_{it})}{P^W_t \, e_{it}} - 1 \right)$$

$$\theta_i = \min\left( 0.9, \; o_i \, \frac{m^F_i + m^G_i}{2} \right)$$

$$\text{pull}_{it} = 0.15 \, \eta_f \max\left( 0, \; \frac{\Delta \text{Dom}_{it}}{Y_{it}} \right)$$

$$\Delta \pi_{it} = \left[ 0.3 \, \theta_i \, \text{world}_{it} + \text{pull}_{it} - 0.1 \max(0, \pi_{it} - 0.03) \right] \Delta$$

$$\pi_{i,t+1} = \max(-0.02, \; \pi_{it} + \Delta \pi_{it}), \qquad \Pi_{i,t+1} = \max(10^{-6}, \; \Pi_{it}(1 + \pi_{i,t+1} \Delta))$$

where $\eta_f$ = `g.fiscal_mult` $= 1.2$. `core.py:637-641` records why the
world term is a push on the *change* rather than a level: written as a level it
makes an appreciating bloc's inflation track the exchange rate down to the
floor, which is not what an appreciation does to a price index in which most
items are domestic.

---

## §3.6 International settlement and monetary hierarchy

### Settlement constraint

From `settle` (`settlement.py:74-87`). For a non-exempt bloc:

$$\boxed{\; \mathcal{K}_{it} = \max\left( 0, \; X_{it} + R_{it} + B_{it} \right), \qquad M^{\text{real}}_{it} = \min\left( M_{it}, \; \mathcal{K}_{it} \right) \;}$$

where $X_{it}$ are export earnings, $R_{it}$ foreign-exchange reserves,
$B_{it}$ new borrowing. For the exempt bloc — the reserve issuer, which settles
in money it issues — $M^{\text{real}}_{it} = M_{it}$ and $\mathcal{K}_{it}$ is
reported as $M_{it}$.

### Borrowing capacity

From `borrowing_capacity` (`settlement.py:61-71`), with the contagion-adjusted
standing from `core.py:463-469`:

$$\varsigma_{it} = \varsigma_i \max\left( 0, \; 1 - \chi_{it} \right)$$

$$B_{it} = \max\left( 0, \; Y_{it} \, \bar{b} \, \Delta \, \varsigma_{it} \max\left( 0, \; 1 - \nu_b \, rp_{it} \right) \right)$$

where $\varsigma_i$ = `b.credit_standing` is what lenders will advance as a
multiple of the global limit, $\bar{b}$ = `g.borrow_limit` $= 0.05$/year,
$\nu_b$ = `g.borrow_sensitivity` $= 4.0$, and $\chi_{it}$ the contagion state.
At a premium of $1/\nu_b$ the window is shut. `settlement.py:66-69` notes this
is part of the constraint, not a privilege: the same formula applies to the
reserve bloc on the occasions when it faces the constraint at all.

`params.py:88-94` records why credit standing is separate from opening debt:
giving a low-credit bloc a *low* opening debt makes its risk premium small and
its borrowing window wide, which inverts the axis.

### Import rationing

From `SettlementOutcome` (`settlement.py:45-58`) and `core.py:480-493`:

$$\text{Rationed}_{it} = \max\left( 0, \; M_{it} - M^{\text{real}}_{it} \right), \qquad s_{i,t+1} = \frac{\text{Rationed}_{it}}{M_{it}}$$

with $s_{i,t+1} = 0$ when $M_{it} = 0$, and "bound" recorded when
$\text{Rationed}_{it} > 10^{-12}$. The real quantity is cut in the same
proportion (`core.py:475-479`):

$$M^{\text{q,real}}_{it} = M^{\text{q}}_{it} \cdot \frac{M^{\text{real}}_{it}}{M_{it}}$$

Rationing is applied **before** the world re-clears, because one bloc's cut
imports remove another bloc's exports (`core.py:472-474`). The scarcity share
is carried into the *next* period's prices, not this one's: a bloc discovers it
cannot pay after it has tried to (`core.py:480-483`).

Reserves are carried forward by (`settlement.py:120-129`):

$$R_{i,t+1} = \max\left( 0, \; R_{it} + X_{it} - M^{\text{real}}_{it} + K_{it} \Delta \right)$$

### The reserve issuer's exemption and its limit

From `reserve_currency_demand` and `reserve_is_constrained`
(`settlement.py:90-117`). Reserves are held to settle future imports, so what
makes them worth holding is what they will buy:

$$\mathcal{D}_t = \mathcal{D}_0 \left( \frac{1}{P^W_t} \right)^{\epsilon_R}$$

$$\mathcal{D}_0 = \sum_{i \, : \, \neg\text{reserve}} Y_{i0} \, o_i$$

The exemption holds while other blocs want the currency, and is lost below a
share of opening demand:

$$\text{exempt}_{it} = \text{reserve}_i \wedge \neg\left[ \mathcal{D}_t < \bar{\theta}_R \, \mathcal{D}_0 \right]$$

with $\epsilon_R$ = `reserve_demand_elasticity` $= 0.5$ and
$\bar{\theta}_R$ = `reserve_demand_threshold` $= 0.7$. Both are swept rather
than chosen: `settlement.py:96-98` states the elasticity has no empirical basis
here, and `params.py:259-262` states the threshold is a reporting cut-off — the
model is asked where the exemption goes, not told.

### Exchange rate adjustment

From `core.py:608-624`. What presses on the rate is the demand for foreign
exchange, not the payments actually made, so the unsettled shortfall is added
back (`core.py:604-611`):

$$\text{pressure}_{it} = CA_{it} - \text{Rationed}_{it}$$

$$\text{flow}_{it} = \left( K_{it} + \eta_b \frac{\text{pressure}_{it}}{Y_{it}} \right) \Delta$$

$$\dot{e}_{it} = -2 \, \text{flow}_{it} \, e_{it} + \eta_x \left( 1 - e_{it} \right) \Delta$$

$$e_{i,t+1} = \max\left( 0.05, \; e_{it} + \dot{e}_{it} \right)$$

where $\eta_b$ = `g.bop_coeff` $= 0.5$ and $\eta_x$ = `g.export_switch` $=
0.05$ (expenditure switching pulls the rate back towards parity). The response
is **proportional, not additive**: `core.py:613-619` records that v7's additive
rule $\dot{e} = -2\,\text{flow}$ moves the rate by the same number of points
whether it stands at 2.0 or 0.2, so a bloc under sustained surplus walks the
rate into the floor and stops responding — and in a closed world that is not an
edge case, since somebody must run the surplus matching everyone else's deficit.

`core.py:604-611` gives the reason for the shortfall term: a bloc rationed
because its reserves ran out shows a *better* current account afterwards, and
reading that as strength would have its currency appreciate precisely when it
can no longer pay its bills.

Under the reserve-privilege sensitivity only (off by default):

$$\dot{e}_{it} \leftarrow \rho_d \, \dot{e}_{it}, \qquad \dot{e}_{it} \leftarrow \max\left( \dot{e}_{it}, \; \underline{e}_R - e_{it} \right)$$

### Contagion

From `axis_distance`, `contagion_weight`, `decay_contagion`
(`settlement.py:142-176`) and `core.py:748-779`. Distance is Euclidean in the
two axes that decide whether a bloc can pay for its imports:

$$\text{dist}(i, j) = \sqrt{ \left( \frac{|\varsigma_i - \varsigma_j|}{1.6} \right)^2 + \left( \frac{|m^F_i - m^F_j|}{0.25} \right)^2 }$$

The withdrawal falls off exponentially with distance:

$$w(\text{dist}) = \exp\left( -\frac{\text{dist}}{\varrho} \right), \qquad \varrho = \texttt{contagion\_reach} = 0.5$$

On a default by bloc $j$, every other bloc is marked down:

$$\chi_{it} \leftarrow \min\left( 1, \; \chi_{it} + \varkappa \, w\!\left( \text{dist}(i, j) \right) \right), \qquad \varkappa = \texttt{contagion\_strength}$$

and the shock decays geometrically over `contagion_years` $= 5$:

$$\chi_{i,t+1} = \chi_{it} \left( 1 - \frac{\Delta}{Y_c} \right) \text{ if } \Delta < Y_c, \text{ else } 0$$

`params.py:284-294` records that the reserve issuer is **not** exempted on the
main path: granting the exemption directly would reinstate v7's privileges in
another form. Instead, a flight-to-reserve channel gives it relief as a
*consequence* (`core.py:765-779`):

$$W_t = \sum_{i \neq j} \varkappa \, w\!\left( \text{dist}(i, j) \right) Y_{it}$$

$$\chi_{it} \leftarrow \max\left( 0, \; \chi_{it} - \frac{\phi_R W_t}{\max(10^{-9}, Y_{it})} \right), \qquad R_{it} \mathrel{+}= \phi_R W_t \qquad \text{(reserve bloc)}$$

with $\phi_R$ = `g.flight_to_reserve`. The money pulled out of a region is not
held as cash; it buys the safest claim on offer, so the issuer receives an
inflow *because* others are being withdrawn from (`core.py:771-776`).

### Policy response

From `respond` (`policy.py:46-79`). Both triggers are **gaps, not levels**, and
nothing names a target (`policy.py:60-62`): an authority told to hold inflation
at two per cent would be told the answer.

$$\text{gap}_{it} = \max\left( 0, \; \hat{g}^W_t - \hat{g}_{it} \right) + \max\left( 0, \; \bar{r}^W_t - (r_{it} - \pi_{it}) \right)$$

$$\text{impulse}_{it} = \varsigma_P \, \text{gap}_{it}$$

$$\text{rate cut}_{it} = \text{impulse}_{it} (1 - \varphi_F), \qquad \text{fiscal boost}_{it} = \text{impulse}_{it} \, \varphi_F$$

where $\varsigma_P$ = `g.policy_response`, $\varphi_F$ =
`g.policy_fiscal_share` $= 0.5$, and the world yardsticks are size-weighted
(`core.py:379-390`). The rule is identical in every bloc (`policy.py:15-21`):
the periphery's predicament comes from the settlement constraint and from debt
in someone else's money, never from a weaker reaction function.

The policy rate follows a target (`core.py:691-693`):

$$r^{\text{tgt}}_{it} = \max\left( 0, \; 0.02 + 0.5 (\pi_{it} - 0.02) + 0.3 \, rp_{it} - \text{rate cut}_{it} \right)$$

$$r_{i,t+1} = \max\left( 0, \min\left( 0.30, \; r_{it} + 0.2 \left( r^{\text{tgt}}_{it} - r_{it} \right) \Delta \right) \right)$$

### Closed-world accounting

From `model/trade.py`. These are the identities CLAUDE.md requires be fixed by
test:

$$S_{ij} = \frac{w_j \, e_j^{\,\epsilon_T}}{\sum_{k \neq i} w_k \, e_k^{\,\epsilon_T}} \quad (j \neq i), \qquad S_{ii} = 0, \qquad w_i = Y_{i0} \, o_i$$

$$X_j = \sum_i S_{ij} M_i \quad \Longrightarrow \quad \sum_i X_i = \sum_i M_i$$

$$CA_i = X_i - M_i \quad \Longrightarrow \quad \sum_i CA_i = 0$$

$$K_i = \tilde{K}_i - \left( \sum_j \tilde{K}_j \right) \frac{w_i}{\sum_j w_j} \quad \Longrightarrow \quad \sum_i K_i = 0$$

$$\dot{\text{NFA}}_i = CA_i + K_i \quad \Longrightarrow \quad \sum_i \dot{\text{NFA}}_i = 0$$

where the desired flow $\tilde{K}_i$ is (`core.py:391-418`):

$$\tilde{K}_{it} = o_i \nu_K \left( r_{it} - rp_{it} - \bar{r}_t \right) + o_i \nu_g \left( \hat{g}_{it} - \hat{g}^W_t \right) - FA_{it}$$

with $\nu_K$ = `g.cap_sens` $= 0.8$ and $\nu_g$ =
`g.growth_differential_sensitivity` $= 0.4$. Adding-up tolerance is $10^{-10}$
(`trade.py:23`).

### The two-axis plane

From `model/twoaxis.py`. Each bloc is a point $(c, \psi) \in [0,1]^2$, with
every structural parameter linearly interpolated between the archetypes'
endpoints:

$$\text{param} = \text{lo} + (\text{hi} - \text{lo}) \cdot \text{position}$$

| Axis | Parameter | worst → best |
|---|---|---|
| credit $c$ | `rate` | $0.070 \to 0.020$ |
| | `omega0` | $0.35 \to 0.05$ |
| | `credit_standing` | $0.15 \to 1.6$ |
| | `reserve_standing` | $0.25 \to 1.75$ |
| self-sufficiency $\psi$ | `m_F` | $0.30 \to 0.10$ |
| | `m_G` | $0.35 \to 0.15$ |
| | `m_H` | $0.25 \to 0.05$ |

$$f_i = (1 - c_i) \cdot 0.85 \qquad \text{(foreign-currency debt share)}$$

The grid is a $5 \times 5$ lattice, $c_i = i/4$, $\psi_j = j/4$, giving 25
blocs, with the reserve issuer placed at $(1.0, 1.0)$.

---

## §3.7 Post-employment transition

### Automation progress

From `progress` (`automation.py:59-69`). Exponential approach, not a straight
line: the early years move fastest and the last of the work is hardest to
automate.

$$a_t = a_{\max} \left( 1 - e^{-\varsigma_a t} \right)$$

with $a_{\max}$ = `automation_max` (0 is the baseline series) and $\varsigma_a$
= `automation_speed` $= 0.15$/year. Neither is an estimate, so both are swept.

### The four derived consequences

From `state` (`automation.py:72-85`). All four move together because they are
one phenomenon seen from four sides; giving each its own schedule would let the
model be tuned until it said what was wanted (`automation.py:74-78`).

$$\text{capacity:} \qquad \Lambda_t = 1 + \alpha_K \, a_t$$

$$\text{unit cost:} \qquad c_t = \max\left( 0.05, \; 1 - \alpha_C \, a_t \right)$$

$$\text{labour income:} \qquad \lambda_t = \max\left( 0, \; 1 - \alpha_L \, a_t \right)$$

$$\text{capital shift:} \qquad D_{it} = \sigma_i \, Y_{it} \left( 1 - \lambda_t \right)$$

where $\alpha_K$ = `automation_capacity` $= 1.0$, $\alpha_C$ =
`automation_cost` $= 0.3$, $\alpha_L$ = `automation_labour` $= 1.0$. The fourth
is not a separate parameter: it is the identity
$\text{displaced} = 1 - \lambda_t$ (`automation.py:48-56`), and what labour
stops receiving accrues to the owners of the machines — the unconstrained block
(`core.py:100-103`).

$\Lambda_t$ enters potential output, $c_t$ enters the domestic component of
prices (§3.3), $\lambda_t$ multiplies the constrained block's income (§3.3),
and $D_{it}$ enters $\Delta I^u$ (§3.3).

### Demand-constrained output

**The code does not write $Y = \min(\text{demand}, \text{capacity})$.**
Potential works as a drag on the *growth rate*, not as a hard ceiling on the
level. From `core.py:666-668`:

$$\text{slack}_{it} = \max\left( 0, \; 1 - \frac{Y_{it}}{Y^{\text{pot}}_{it}} \right)$$

$$\hat{g}_{it} = g^{\text{eff}}_{it} + 0.08 \, \eta_f \frac{\Delta \text{Dom}_{it}}{Y_{it}} - \alpha_D \, \text{slack}_{it} + \sum \text{penalties}$$

with $\alpha_D$ = `demand_constrains_output` (0 is the pre-post_employment
behaviour; swept). `params.py:385-401` records the reason it exists: automation
raises what could be made and removes the income that would buy it, so demand
and capacity stop moving together and output has to be able to fall short.

The reduced-form growth penalties, switchable via `g.growth_penalties`
(`params.py:346-364`, applied `core.py:669-676`):

$$\text{capital:} \; +0.15 \, \text{flow}_{it} \qquad \text{debt:} \; -0.02 \max(0, d_{it} - 1.2)$$

$$\text{inflation:} \; -0.03 \max(0, \pi_{it} - 0.05) \qquad \text{fx:} \; -0.01 \left| \dot{e}_{it} \right|$$

$$g^{\text{eff}}_{it} = \max\left( 0.005, \; g^{\text{base}}_i - \delta_i t \right)$$

$$Y_{i,t+1} = \max\left( 0.01, \; Y_{it} \left( 1 + \hat{g}_{it} \Delta \right) \right)$$

Potential output carries the bloc's own realised trend **times** the automation
multiplier (`core.py:709-716`):

$$Y^{\text{trend}}_{i,t+1} = \max\left( 0.01, \; Y^{\text{trend}}_{it} \left( 1 + g^{\text{eff}}_{it} \Delta \right) \right), \qquad Y^{\text{pot}}_{i,t+1} = Y^{\text{trend}}_{i,t+1} \Lambda_t$$

Trend growth belongs to both numerator and denominator — it is not what
automation did — so it is the *automation* multiplier alone that opens the gap.
`core.py:704-708` records that deriving potential from $Y$ directly made
utilisation identically $1/\Lambda_t$: arithmetic, not a finding.

### Capacity overhang, not utilisation

From `capacity_overhang` (`automation.py:88-108`):

$$\text{CO}_{it} = \min\left( 1, \; \max\left( 0, \; \frac{Y_{it}}{Y^{\text{pot}}_{it}} \right) \right)$$

The docstring is emphatic that this is **not a utilisation rate for the
economy**: the unconstrained block's consumption bundle is never solved, so its
demand is not in the numerator in any full sense. Reading it as "this share of
the world's plant stands idle" would claim an aggregate the model cannot
compute. Named for what it is, so the reading cannot drift.

---

## §3.8 Outcome measures and feasibility

### Budget ratio

From `BudgetPosition` and `bundle_cost` (`subsistence.py:55-98`):

$$\boxed{\; \text{BR}_{iqt} = \frac{I_{iqt} + T_{iqt}}{\sum_k p^k_{it} \, \hat{x}^k_{iq}} \;}$$

$$I_{iqt} = \sigma_i \, Y_{it} \, \phi^{\text{inc}}_{it} \, \lambda_t \, \varpi_q, \qquad T_{iqt} = T_{it} \, n_q$$

where $\hat{x}^k_{iq}$ are the reference quantities fixed once (§3.3),
$\varpi_q$ = `t.income_share`, $n_q$ = `t.pop_share`. Shortfall is
$\max(0, 1 - \text{BR})$, and $\text{BR} = \infty$ when the bundle costs
nothing. The quantities come from the reference point and do not move; only the
prices do (`subsistence.py:90-98`).

Reported across quantiles as $\min$, median, and share below one
(`subsistence.py:154-163`).

`subsistence.py:41-42` records the wording discipline: the ratio says whether
the money reaches the bundle, not whether the household is fed. The word
"meets" is deliberately avoided.

### Required transfer

From `required_transfer` (`subsistence.py:166-203`). Solved **directly, not
searched**, because the ratio is linear in the transfer:

$$T^\star_{it} = \frac{1}{\max(0.05, \, \text{RV}_{it})} \max_q \left[ \frac{\max\left( 0, \; \sum_k p^k_{it} \hat{x}^k_{iq} - I_{iqt} \right)}{n_q} \right]$$

and $u^\star_{it} = T^\star_{it} / Y_{it}$.

The erosion divisor $\text{RV}_{it}$ is the transfer's real value ratio under
the indexation rule in force (§3.4): a transfer that has lost half its value
must have been twice as large to begin with. `subsistence.py:186-190` records
that without it, an unindexed transfer looked cheaper than it is and the
comparison of rules was not on the same footing.

This is a **time series, not a constant** (`subsistence.py:175-180`): as the
currency falls and the demand-pressure index rises, the bundle costs more and
the transfer needed to pay for it grows — and the blocs where it grows fastest
are the ones that can least afford to pay it.

### The fixed point

From `analysis/v8pe_run.py:85-98`. Because the transfer changes prices, $u^\star$
is a fixed point, found by iterating over a discrete candidate grid
$\mathcal{U} = \{0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12\}$ for at most 6
rounds:

$$u^{(0)} = u^\star_i \big|_{u = 0}, \qquad u^{(n+1)} = \min\{ u \in \mathcal{U} : u \ge u^{(n)} \}$$

$$u^{\text{FP}}_i = u^{(n+1)} \quad \text{if} \quad u^\star_i \big|_{u = u^{(n+1)}} \le u^{(n+1)}$$

If no candidate satisfies this within 6 rounds, $u^{\text{FP}}_i$ is undefined —
each additional unit of transfer raises the requirement by more than one unit.

Note the need is taken as the **maximum over the run's sampled years**
(`v8pe_run.py:78-79`), not the terminal value:
$u^\star_i \big|_u = \max_t \text{need}_{it}$.

### Deliverable transfer

From `v8pe_run.py:100-108`. The largest candidate whose average rationing stays
within a tolerance:

$$\bar{r}_i \big|_u = \frac{1}{N_s} \sum_{t} s_{it} \big|_u$$

$$u^{\text{del}}_i(\varepsilon) = \max\left\{ u \in \mathcal{U} : \bar{r}_i \big|_u \le \varepsilon \right\}$$

undefined when the set is empty. Tolerances swept:
$\varepsilon \in \{0, 0.05, 0.10\}$.

### Feasibility classification

From `v8pe_run.py:110-121`. Four categories, evaluated per bloc:

$$\textbf{already constrained:} \qquad \bar{r}_i \big|_{u=0} > 10^{-12}$$

$$\textbf{no fixed point:} \qquad u^{\text{FP}}_i \text{ undefined} \;\wedge\; u^\star_i \big|_{u=0} > 0$$

$$\textbf{trapped}(\varepsilon): \qquad u^\star_i \big|_{u=0} > 0 \;\wedge\; \bar{r}_i \big|_{u=0} \le 10^{-12} \;\wedge\; \left( u^{\text{FP}}_i \text{ undefined} \;\vee\; u^{\text{del}}_i(\varepsilon) \text{ undefined} \;\vee\; u^{\text{FP}}_i > u^{\text{del}}_i(\varepsilon) \right)$$

$$\textbf{feasible:} \qquad \text{none of the above}$$

Note the categories are **not mutually exclusive by construction** — the
figure code (`analysis/v8pe_figures.py:55-60`) imposes an order of precedence:
already constrained, then no fixed point, then trapped, then clear.

### Crisis decomposition

From `core.py:721-731`. Three continuous excess measures against reporting
cut-offs $\text{THR} = \{d: 1.5, \pi: 0.15, e: 2.0\}$:

$$\text{exc}_{it} = \left( \frac{\max(0, d_{it} - 1.5)}{1.5}, \; \frac{\max(0, \pi_{it} - 0.15)}{0.15}, \; \frac{\max(0, e_{it} - 2.0)}{2.0} \right)$$

$$\text{in crisis}_{it} \iff \exists \, x \in \text{exc}_{it} : x > 0$$

$$\text{severity}_i = \sum_t \left( \sum \text{exc}_{it} \right) \Delta, \qquad \text{duration}_i = \sum_t \mathbb{1}[\text{in crisis}_{it}]$$

with onsets counted on transitions into crisis, and insolvency recorded at
$d_{it} > $ `default_thr` $= 3.0$, after which the bloc's state stops being
updated. `params.py:27-29` states these are reporting cut-offs, not structural
boundaries.

### World pressure measures

From `model/collapse.py`. Levels and changes only; nothing is compared against
a cut-off (`collapse.py:21`).

$$\pi^W_t = \frac{P^W_t}{P^W_{t-1}} - 1 \qquad \text{(year-on-year)}$$

$$\text{gap}_t = \frac{\sum_i M^{\text{q}}_{it}}{Z_t} - 1$$

$$\text{solvent demand share} = \frac{\sum_{i \, : \, \text{alive}} M^{\text{q}}_{iT}}{\sum_i M^{\text{q}}_{iT}}$$

`collapse.py:40-44` records why the last is reported alongside inflation:
$\pi^W$ can be low for opposite reasons — a world that stayed calm, and a world
that went quiet after its blocs defaulted — so the default count and the peak
are carried with it. The `WorldPressure` record has **no verdict field, by
design** (`collapse.py:35-38`).

### Leakage measures

From `core.py:791-816`. Imports and foreign-asset purchases are reported
**apart**, because only the first buys goods on the world market:

$$\text{import}^{\text{abs}}_i = 100 \, \frac{\sum_t \Delta M_{it}}{\sum_t Y_{it}}, \qquad \text{FA}^{\text{abs}}_i = 100 \, \frac{\sum_t FA_{it}}{\sum_t Y_{it}}$$

$$\text{import per fiscal}_i = \frac{\sum_t \Delta M_{it}}{\sum_t F_{it}}, \qquad \text{FA per fiscal}_i = \frac{\sum_t FA_{it}}{\sum_t F_{it}}$$

The combined rate measures are retained for continuity with the open-world
phases but flagged in the code as **not the right measure of pressure on the
world goods market** (`core.py:809-811`, `core.py:809-811`):

$$\text{leak}^{\text{abs}}_i = 100 \frac{\sum_t (\Delta M_{it} + FA_{it})}{\sum_t Y_{it}}, \qquad \text{leak per welfare}_i = \frac{\sum_t (\Delta M_{it} + FA_{it})}{\sum_t \Delta U_{it}}$$

with the last reported as null when $\sum_t \Delta U_{it} \le 10^{-12}$.

---

## Algorithm 1: Simulation order per period

There is **no `step()` method** in the code. The main loop is `_run`
(`model/core.py:274-786`), a single `for step in range(STEPS + 1)` over
$\text{STEPS} = 120$ quarters ($\Delta = 0.25$, 30 years). The order below is
transcribed from that loop.

**Initialisation** (once, `core.py:294-345`)

1. Solve the shared world capacity path from a no-transfer run
   (`baseline_capacity_path`), iterating capacity and realised demand to a
   fixed point (≤ 12 rounds, tol $10^{-9}$).
2. Initialise each bloc's state: $Y_{i0} = $ `b.gdp`, $d_{i0} = $ `b.debt`,
   $e_{i0} = 1$, $\pi_{i0} = 0.02$, $\Pi_{i0} = 1$, $p^H_{i0} = 1$,
   $R_{i0} = $ `initial_reserves` $\cdot Y_{i0} \cdot$ `reserve_standing`,
   $s_{i0} = 0$, $\chi_{i0} = 0$, alive.
3. Fix the trade share matrix $S$ and export weights $w$.
4. Fix each quantile's subsistence bundle as quantities (`reference_bundles`),
   at the opening or the no-transfer baseline output per
   `subsistence_reference`.
5. Record opening reserve-currency demand $\mathcal{D}_0$.

**Per quarter** $t = \text{step} \cdot \Delta$

6. **Pass 1 — demands at current prices** (`_bloc_demand`, no state written).
   For each bloc: advance the transfer's nominal amount under the indexation
   rule using *last* period's inflation; compute the income factor and the
   automation multipliers; solve household demand twice — with the programme
   and without it — at prices carrying last period's scarcity; form import
   demand in both spending and quantity terms, domestic demand, and the fiscal
   cost.
7. **Set world capacity** from the shared path (at step 0, normalise on the
   first period's no-transfer demand so the baseline opens at a zero gap).
8. **Pass 2 — the world clears.** Recompute trade shares if the trade
   elasticity is on; distribute imports across suppliers to get exports;
   form current accounts.
9. **Capital account.** Compute each bloc's risk premium (including the
   contagion term), its desired net flow from the rate and growth differentials
   less household foreign-asset purchases, then balance the desires to sum to
   zero by size. Optionally charge part of an inflow to world goods demand
   (sensitivity, default off) and re-clear.
10. **Settlement** (when `g.settlement`). Compute reserve-currency demand and
    whether the issuer retains its exemption; for each bloc compute borrowing
    capacity from contagion-adjusted standing and the risk premium; ration
    imports to exports + reserves + borrowing. Re-clear exports and current
    accounts on the settled quantities, update reserves, and store the
    rationing share for **next** period's prices.
11. **Record** (every 4th step). Bloc series; real quantities secured; budget
    positions and the required transfer; capacity overhang; world price,
    capacity, demand, and demand from solvent blocs only.
12. **Accrue net foreign assets** for every bloc, insolvent ones included, so
    the adding-up identity holds.
13. **Per-bloc state update** (skipped for insolvent blocs). In order: debt
    accumulation; exchange-rate change from the external position plus the
    unsettled shortfall, with expenditure switching; inflation from the world
    push, domestic pull, and central-bank drag; growth from the trend, the
    fiscal impulse, the demand-slack drag, and the switchable penalties; the
    policy impulse; then write $d, K, e, p^H, \pi, \Pi, Y^{\text{trend}},
    Y^{\text{pot}}, Y, r$.
14. **Crisis flags and insolvency.** Accumulate duration, severity, and onsets
    against the reporting cut-offs; mark insolvent above `default_thr`.
15. **Contagion.** Decay existing shocks; for each default this step, mark down
    every other bloc by distance on the two axes; route the withdrawn amount to
    the reserve issuer if the flight channel is on.
16. **Advance the demand-pressure index** from the world output gap and the
    reversion term.

Note the ordering constraint the closure depends on (`core.py:5-16`): all blocs
compute demands *before* any state is written, so one bloc's imports become
another's export revenue within the same quarter rather than leaving the model.

---

## Table: Parameters, state variables, and initial conditions

### Global parameters (`model/params.py`, class `G`)

| Symbol | Code | Description | Baseline |
|---|---|---|---|
| $\eta_f$ | `fiscal_mult` | fiscal multiplier | 1.2 |
| $\nu_K$ | `cap_sens` | capital-flow sensitivity to the rate differential | 0.8 |
| $\kappa_r$ | `risk_coeff` | risk-premium coefficient | 0.15 |
| — | `compress` | inflation's compression of the real debt ratio | 0.3 |
| — | `deficit_share` | share of the transfer financed by deficit | 0.6 |
| $\eta_b$ | `bop_coeff` | external position → FX pressure | 0.5 |
| $\eta_x$ | `export_switch` | expenditure switching on depreciation | 0.05 |
| $\epsilon_T$ | `trade_elasticity` | how strongly depreciation wins market share | 0.0 |
| $\nu_\omega$ | `omega_sens` | foreign-asset share's sensitivity to risk | 2.0 |
| $\text{mpc}^u$ | `mpc_u` | unconstrained households' MPC | 0.3 |
| $\delta_L$ | `land_discount` | CLT land acquisition discount | 0.5 |
| — | `default_thr` | debt ratio at which the bloc stops being updated | 3.0 |
| $\mu$ | `eps_mult` | uniform multiplier on rental supply elasticity | 1.0 |
| $\kappa_s$ | `sourcing_cost_kappa` | cost markup for domestic CLT sourcing | 0.0 |
| $\gamma^F$ | `sub_F` | food subsistence coefficient | 0.35 |
| $\gamma^G$ | `sub_G` | general-goods subsistence coefficient | 0.05 |
| $\gamma^H$ | `sub_H` | housing subsistence coefficient | 0.25 |
| $\beta^F$ | `beta_F` | marginal budget share, food | 0.25 |
| $\beta^G$ | `beta_G` | marginal budget share, general goods | 0.45 |
| $\beta^H$ | `beta_H` | marginal budget share, housing | 0.30 |
| $k$ | `n_types` | income quantiles in the constrained block | 1 |
| $\rho$ | `income_spread` | income inequality within the block | 0.6 |
| — | `clt_capacity` | CLT tenancies as a share of the block | None |
| — | `allocation_rule` | tenancy allocation | `priority_low_income` |
| $\beta_{\text{build}}$ | `clt_build_rate` | construction per year, share of housing stock | None |
| $n$ | `housing_life_years` | building life, for amortisation | 40 |
| — | `clt_rent_to_landlords` | whether landlords collect CLT rent | False |
| — | `universal_split` | override for the UBI population split | None |
| $\kappa_w$ | `kappa_w` | pass-through from world output gap to $\hat\pi^W$ | 0.35 |
| $\lambda_w$ | `lambda_w` | reversion of $P^W$ toward its anchor | 0.05 |
| — | `world_supply_growth` | override for world capacity growth | None |
| — | `settlement` | ration imports to payable foreign exchange | False |
| — | `initial_reserves` | opening reserves, share of GDP | 0.15 |
| $\bar{b}$ | `borrow_limit` | new foreign borrowing per year, share of GDP | 0.05 |
| $\nu_b$ | `borrow_sensitivity` | how sharply the window closes with the premium | 4.0 |
| $\epsilon_R$ | `reserve_demand_elasticity` | reserve demand's elasticity to purchasing power | 0.5 |
| $\bar{\theta}_R$ | `reserve_demand_threshold` | share of opening demand below which the exemption lapses | 0.7 |
| $\varkappa$ | `contagion_strength` | lender withdrawal from blocs resembling a defaulter | 0.0 |
| $\varrho$ | `contagion_reach` | how far the reassessment travels | 0.5 |
| $Y_c$ | `contagion_years` | how long a withdrawal takes to unwind | 5.0 |
| — | `contagion_exempts_reserve` | exempt the issuer by assertion (sensitivity) | False |
| $\phi_R$ | `flight_to_reserve` | capital running to the reserve currency on default | 0.0 |
| $\kappa_c$ | `contagion_risk_premium` | extra premium for a fully hit bloc | 0.05 |
| $\varsigma_P$ | `policy_response` | how hard authorities lean against falling behind | 0.0 |
| $\nu_g$ | `growth_differential_sensitivity` | capital chasing the growth gap | 0.4 |
| $\varphi_F$ | `policy_fiscal_share` | share of the impulse carried by spending | 0.5 |
| — | `growth_penalties` | which reduced-form growth terms are active | all four |
| $a_{\max}$ | `automation_max` | how far labour demand ultimately falls | 0.0 |
| $\varsigma_a$ | `automation_speed` | approach to the ceiling, per year | 0.15 |
| $\alpha_K$ | `automation_capacity` | potential output gained per unit of automation | 1.0 |
| $\alpha_C$ | `automation_cost` | unit cost saved per unit of automation | 0.3 |
| $\alpha_D$ | `demand_constrains_output` | how strongly demand shortfall holds output down | 0.0 |
| $\alpha_L$ | `automation_labour` | labour income lost per unit of automation | 1.0 |
| — | `income_linkage` | how constrained income keeps up | `growth_linked` |
| — | `subsistence_reference` | where the measured bundle is fixed | `start` |
| $\theta$ | `subsistence_scale` | multiplier on the Stone–Geary floor | 1.0 |
| — | `indexation` | how the transfer's nominal amount moves | `gdp_linked` |
| — | `capital_inflow_to_demand` | share of an inflow spent on tradables on arrival | 0.0 |
| — | `reserve_privilege` | re-enable v7's exogenous reserve advantages | False |
| — | `reserve_flight` | safe-haven inflow (privilege only) | 0.3 |
| — | `reserve_thr` | higher debt threshold (privilege only) | 1.5 |
| $\rho_d$ | `reserve_damp` | damped exchange rate (privilege only) | 0.5 |
| $\underline{e}_R$ | `reserve_flight_floor` | floor on the reserve rate (privilege only) | 0.3 |

Horizon and cut-offs (`params.py:23-40`): `YEARS` $= 30$, $\Delta = 0.25$,
`STEPS` $= 120$; `THR` $= \{d: 1.5, \pi: 0.15, e: 2.0, \text{app}: 0.3\}$;
`SUB_SCALE_EXPONENT` $= 0.5$, `SUB_SCALE_CAP` $= 1.35$.

### Per-bloc parameters (`model/params.py`, class `Bloc`)

| Symbol | Code | Description | Default |
|---|---|---|---|
| $Y_{i0}$ | `gdp` | initial GDP | — |
| $d_{i0}$ | `debt` | initial debt / GDP | — |
| $r_{i0}$ | `rate` | initial policy rate | — |
| $g^{\text{base}}_i$ | `base_growth` | trend growth | — |
| $\delta_i$ | `decay` | decline in trend growth per year | — |
| $o_i$ | `openness` | trade openness | — |
| $f_i$ | `fx_share` | foreign-currency share of debt | — |
| — | `reserve` | issues the reserve currency | False |
| $K_{i0}$ | `capital` | initial capital stock | 1.0 |
| $\sigma_i$ | `c_income_share` | constrained households' income / GDP | 0.2 |
| $n_i$ | `c_pop_share` | constrained share of population | 0.4 |
| $m^F_i$ | `m_F` | import content of food | 0.2 |
| $m^G_i$ | `m_G` | import content of other goods | 0.25 |
| $m^H_i$ | `m_H` | import content of construction | 0.1 |
| $\varepsilon_i$ | `eps_supply` | short-run rental supply elasticity | 0.4 |
| $\ell_i$ | `land_share` | land rent share of gross rent | 0.35 |
| $\omega^0_i$ | `omega0` | baseline foreign-asset share of saving | 0.2 |
| $t^{\text{start}}_i$ | `start` | year the programme begins | 999.0 |
| — | `modality` | none / ubi / cash_t / voucher / clt / clt_cash / cash_cheap | none |
| $u_i$ | `size` | programme size | 0.0 |
| $\varsigma_i$ | `credit_standing` | what lenders will advance, multiple of the limit | 1.0 |
| — | `reserve_standing` | opening reserves, multiple of the global share | 1.0 |
| $d_i$ | `clt_domestic_sourcing` | policy effort to source construction at home | 0.0 |

### The four archetypes (`params.py:498-517`)

| | A (reserve) | B (non-reserve adv.) | C (emerging) | D (developing) |
|---|---|---|---|---|
| `gdp` | 1.2 | 0.9 | 0.5 | 0.2 |
| `debt` | 0.85 | 0.70 | 0.40 | 0.35 |
| `rate` | 0.020 | 0.025 | 0.050 | 0.070 |
| `base_growth` | 0.018 | 0.020 | 0.035 | 0.038 |
| `decay` | 0.0003 | 0.0003 | 0.0005 | 0.0008 |
| `openness` | 1.0 | 1.0 | 0.75 | 0.50 |
| `fx_share` | 0.0 | 0.15 | 0.45 | 0.65 |
| `reserve` | True | False | False | False |
| `capital` | 1.3 | 1.0 | 0.45 | 0.20 |
| `c_income_share` | 0.15 | 0.15 | 0.20 | 0.25 |
| `c_pop_share` | 0.30 | 0.30 | 0.40 | 0.50 |
| `m_F` | 0.10 | 0.15 | 0.20 | 0.30 |
| `m_G` | 0.15 | 0.25 | 0.30 | 0.35 |
| `m_H` | 0.05 | 0.08 | 0.15 | 0.25 |
| `eps_supply` | 0.3 | 0.3 | 0.4 | 0.5 |
| `land_share` | 0.45 | 0.40 | 0.35 | 0.30 |
| `omega0` | 0.05 | 0.15 | 0.25 | 0.35 |

### State variables carried per bloc (`core.py:294-303`)

$Y$ output; $d$ debt ratio; $r$ policy rate; $\pi$ inflation;
$\pi^{\text{lag}}$ last period's inflation; $e$ exchange rate;
$\dot{e}$ its change; $K$ capital stock; NFA net foreign assets;
$\Pi$ price level; $p^H$ clearing rent; $R$ reserves; $s$ scarcity;
unsettled; $\chi$ contagion; $\hat{g}$ growth rate; $Y_0$ opening output;
$Y^{\text{pot}}$ potential; $Y^{\text{trend}}$ trend; alive;
$t^{\text{insolv}}$.

Plus, per bloc, a `TransferState` (`indexation.py:36-50`): nominal;
$\Pi$ at start; $p^H$ at start; nominal at start; started.

### Post-employment sweep values (`analysis/v8pe_run.py:36-49`)

$a_{\max} \in \{0, 0.2, 0.4, 0.6, 0.8\}$; floor share
$\in \{0.65, 0.75, 0.85, 0.95\}$; indexation
$\in \{\texttt{cpi\_indexed}, \texttt{fixed\_nominal}\}$; rationing tolerance
$\in \{0, 0.05, 0.10\}$; $\mathcal{U} = \{0, 0.01, 0.02, 0.03, 0.04, 0.06,
0.08, 0.12\}$. Run configuration: `n_types=5`, `clt_capacity=1.0`,
`clt_build_rate=0.02`, `settlement=True`, `trade_elasticity=1.0`,
`policy_response=1.0`, `contagion_strength=1.0`, `flight_to_reserve=0.5`,
`income_linkage="cpi_linked"`, grid of $5 \times 5 = 25$ blocs.

---

## Discrepancies between the code and the draft's prose

Checked against `~/Downloads/05_draft (4).md` §3. Recorded, not fixed.

**1. §3.7: "Actual output in each period is the minimum of demand and
capacity."** The draft states this twice (§3.5 final paragraph and §3.7 second
paragraph) and the instruction asked for $Y = \min(\text{demand},
\text{capacity})$. **The code contains no such expression.** Potential output
enters as a drag on the growth *rate*, scaled by `demand_constrains_output`
(`core.py:666-668`), and that parameter's default is **0.0** — so in the default
configuration output does not respond to the demand shortfall at all.
`params.py:385-401` is explicit that zero "is the v6 behaviour, carried through
every phase so far: output follows a trend growth rate plus a fiscal impulse,
and pays no attention to whether anybody can buy what is made." Output can also
exceed potential, which $\min(\cdot)$ would forbid. The draft sentence
overstates what is implemented.

**2. §3.7: "potential productive capacity rises by a factor of $(1 + a \cdot
\text{capacity\_gain})$, unit costs fall by a factor of $(1 - a \cdot
\text{cost\_reduction})$, labour's share of income falls by a factor of $(1 - a
\cdot \text{labour\_loss})$."** The functional forms match
`automation.py:82-84`, but the draft omits the floors the code imposes:
$\max(0.05, \cdot)$ on the cost multiplier and $\max(0, \cdot)$ on the labour
multiplier. Also, the draft says "labour's **share of income**" falls; the code
multiplies the constrained block's **income level** (`core.py:90-91`), and the
share falls only as a consequence.

**3. §3.8: "The required transfer level, $u^\star$, is the transfer as a share
of GDP that keeps the poorest quintile's budget ratio at one."** The code takes
the **maximum over quantiles** of the per-quantile gap
(`subsistence.py:192-200`), which is the quantile with the largest shortfall
relative to its population share — not necessarily the poorest by income. With
the default `n_types = 1` there is only one type and the distinction is empty;
under the post-employment configuration (`n_types = 5`) it is not. Separately,
the draft omits the erosion divisor (`subsistence.py:203`), which is what puts
indexation rules on a comparable footing.

**4. §3.8: "The model solves for this fixed point iteratively."** True, but the
draft implies a continuous solve. The code searches a **discrete 8-point grid**
$\mathcal{U}$ for at most 6 rounds (`v8pe_run.py:85-98`), so $u^{\text{FP}}$ is
grid-valued and "no fixed point" means "not found on this grid within 6
rounds."

**5. §3.8: the three-then-four state classification.** The draft presents
"already constrained / trapped / feasible" with "no fixed point" emerging as a
fourth state at high automation. In the code all four flags are computed
independently for every cell and are **not mutually exclusive**; the precedence
that makes them a partition lives in the figure code
(`v8pe_figures.py:55-60`), not in the model.

**6. §3.3: "The share of constrained households is endogenous and responds to
changes in labour income, transfers, and prices."** It is **not endogenous.**
`b.c_pop_share` and `b.c_income_share` are structural constants of the bloc
(`params.py:61-62`), fixed for the whole run. Nothing in `core.py` or
`household.py` updates either. This is a direct contradiction.

**7. §3.3: "Household saving is a fixed share of income above subsistence."**
The code has no saving rate on the constrained block's income at all. Saving
appears only for the **unconstrained** block, as $1 - \text{mpc}^u$ of the
income increment that reaches it (`core.py:145`), and only that increment —
not its whole income — is modelled.

**8. §3.2: "a reserve-currency bloc with high creditworthiness and moderate
self-sufficiency."** In the grid the reserve issuer is placed at $(1.0, 1.0)$,
the corner **both** axes favour, i.e. high self-sufficiency, not moderate
(`twoaxis.py:101-112`). The archetype A has `m_F = 0.10`, the most
self-sufficient of the four.

**9. §3.6: "Borrowing capacity is a function of the bloc's creditworthiness,
which deteriorates with its net foreign-asset position."** NFA does **not**
enter `borrowing_capacity` (`settlement.py:61-71`) nor the risk premium
(`core.py:183-189`). The premium responds to the **debt ratio** $d$, inflation,
and the foreign-currency debt share. NFA is accumulated and reported but never
feeds back into borrowing. `twoaxis.py:44-49` states the opening debt level is
deliberately kept off the credit axis.

**10. §3.6: "Financial contagion operates through a similarity channel …
distance … on the creditworthiness–self-sufficiency plane."** The code's
`axis_distance` uses `credit_standing` and **`m_F` alone**
(`settlement.py:151-152`), not the full self-sufficiency axis (which also moves
`m_G` and `m_H`). Since the grid moves all three together this is monotone in
$\psi$, but the stated distance is not the plane distance.

**11. §3.5: "The supply capacity of the world grows at a rate calibrated to the
no-transfer baseline's realised output."** Capacity is calibrated to the
baseline's realised **traded-goods demand**, not output
(`core.py:233-241`, `world.py:85-93`).

**12. §3.4: "Under cost-of-living adjustment, the nominal transfer is increased
each period by the previous period's rate of consumer-price inflation."** The
code multiplies by the **quarterly** fraction $\pi_{t-1}\Delta$, not the annual
rate (`core.py:75-78`). The draft's wording, read with the stated quarterly
step, would index four times too fast — which is the error the code comment
says it is avoiding.

**13. §3.1: "the following sequence operates"** and the draft's §3.1 ordering
place import rationing before the price effect within the same period. The code
carries the rationing share into the **next** period's prices
(`core.py:480-483`). The draft's §3.6 wording ("domestic prices rise to reflect
the scarcity") does not state the lag.

**14. §3.4: "All four modalities are calibrated to deliver the same level of
household welfare at the moment of introduction."** True in intent, but the
code flags two failure modes the draft does not mention: `_infeasible_` when
the bisection converges below the target utility, and `_at_ceiling_` when the
required size hits the search bound of 0.5 of GDP
(`heterogeneous.py:421-427`). Whether either fires in the reported runs is a
result, not a given.

### Vocabulary check

CLAUDE.md forbids certain terms. Counts from `~/Downloads/05_draft (4).md`
(case-insensitive):

| Term | Count | Status |
|---|---|---|
| "world-price" (hyphenated) | 7 | lines 15, 17, 99, 189, 195, 205, 281 |
| "world price" (spaced) | 2 | lines 195, 233 |
| "dominant strategy" | **2** | lines 195, 233 — **violation** |
| "Nash" | **1** | line 195 — **violation** |
| "Pareto" | **1** | line 195 — **violation** |
| "optimal" | 1 | line 247, but see below |
| "indexation" | 0 | clear |
| "periphery" / "peripheral" | 0 | clear |

**Three forbidden terms are present, all in §4.5 and §6.2.** CLAUDE.md's
absolute rule: *"「optimal / Nash / Pareto / dominant strategy」という語は、
形式的に計算しない限り使わない"*. The model computes none of these — there is no
best-response calculation, no strategy space, and no welfare ordering across
blocs anywhere in `model/`.

Line 195 (§4.5) contains all three at once:

> "each bloc's **dominant strategy** (adjust) leads to a **Nash** outcome that
> is **Pareto**-inferior to collective non-adjustment, but non-adjustment is not
> individually rational…"

Line 233 (§6.2) repeats one:

> "When each country individually adjusts its transfers for inflation — the
> **dominant strategy** — …"

The underlying result (each bloc adjusting raises traded-goods demand for all,
including itself) is what the model does show; the game-theoretic vocabulary
attributes to it a formal structure that was not computed. Recorded, not fixed.

The one "optimal" (line 247) is in a literature discussion of the farm-size /
yield relationship, not a claim about the model's configurations — outside the
rule's scope.

**On "world price".** §3.5 itself is careful: it states $P_w$ "is a
model-internal intermediate variable, not a quantity that corresponds to any
observable real-world aggregate." But **"world-price externality"** (§1, §4.5,
§6.2) and the bare "world price" at lines 195 and 233 read as though $P^W$ were
a price level, which the instruction says to avoid. Flagged for the author's
decision; not changed.

The draft's §3.5 also writes the food price as
$p_F = (1 - m_F) + m_F \cdot e \cdot P_w$, omitting the automation cost
multiplier $c_t$ on the domestic term. That is correct for the baseline series
($a = 0 \Rightarrow c_t = 1$) but not under post-employment.
