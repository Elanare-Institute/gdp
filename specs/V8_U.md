# v8 — 基礎需要基準による u の導出

**対象**：CLAUDE.md「u の導出」。給付水準を政策変数ではなく
**「制約世帯が生活を維持できる水準」**から導く。

---

## 0. モデルが表現しないこと（最重要）

**このモデルは基礎需要を下回る配分を表現しない。**

`model/household.py` の需要式は

```python
x = {k: sub[k] + beta[k] * max(supern, 0) / p[k] for k in goods}
```

であり、超過所得 `supern` が負でもゼロにクリップされる。
したがって**確保量は常に基礎需要以上**になる。
Stone–Geary の効用が基礎需要以下で定義されないための実装上の措置である。

**`budget_ratio` は予算の不足を測る指標であって、確保量の不足ではない。**
予算が基礎需要の費用に届かなくなったことは分かるが、
**その先で何が起きるかを本モデルは扱わない。**

> **「充足」「充足率」という語を使わない。** 予算が届いているかであって、
> 実際に足りているかではない。

---

## 1. 指標

```
budget_ratio = (制約世帯の所得 + 給付) / 基礎需要バンドルの費用
```

- 分子：`c_income_share · Y + 給付額`（現物給付は評価額で計上）
- 分母：`Σ_k p_k · sub_k`（`sub_k` は所得比で定義、`p_k` は当期の価格）

**1.0 を下回った分が不足である。** 連続量として報告し、閾値で判定しない。

報告単位：**ブロック別・分位別**。
最貧分位（`q1`）は `sub_scale` が大きいので先に 1.0 を割る。

---

## 2. u の導出

```
必要な u(t) = 最貧分位の budget_ratio を 1.0 に保つ最小の給付水準
```

**価格が動けば必要な u も動くので、これは時系列である。**
自給率の低いブロックほど輸入食料が高いので、必要な u は大きくなる。

### 2.1 出力

- ブロック別の必要な u の推移（二軸平面上）
- その u を実際に配ったときに決済制約に当たるか

---

## 3. 中心的な検証

**同じ平面上で二つを比較する。**

| 量 | 定義 |
|---|---|
| `u_needed` | budget_ratio を 1.0 に保つ水準 |
| `u_affordable` | 決済制約に当たらずに配れる上限 |

**`u_needed > u_affordable` の領域が、逃げ場のない位置である。**

領域の**有無と広さ**を報告する。最適配置の特定には踏み込まない。

### 3.1 領域がどう動くか

次を変えて領域の広さを出す：

- 伝播の有無（`contagion_strength`）
- 政策反応の強さ（`policy_response`）
- 指数化規則（`fixed_nominal` / `cpi_indexed`）

---

## 4. 制約世帯の所得の連動規則

`budget_ratio` が 1.0 を大きく上回るのは、30年で実質 GDP が約2倍になり、
制約世帯の所得もそれに**比例して伸びている**ためである（`Ic = c_income_share · Y`）。

これを給付の指数化と同じ形で政策変数にする。

| 規則 | 更新 | 備考 |
|---|---|---|
| `growth_linked` | `Ic(t) = c_income_share · Y(t)` | 現行。**上限側の参考** |
| **`cpi_linked` ★** | `Ic(t) = Ic(0) · P_i(t)` | **主系列**。物価にのみ連動 |
| `fixed_nominal` | `Ic(t) = Ic(0)` | 連動なし |

### 4.1 実証的な根拠

Van Mechelen & Marchal (2012), *Struggle for Life: Social Assistance Benefits,
1992–2009*, GINI Discussion Paper 55, AIAS（EU25か国＋米3州＋ノルウェー）:

- 物価連動が多数派の制度である。物価に連動する国では
  「social assistance benefits may be expected to remain fairly stable
  at least in real terms」。
- 賃金が物価より速く伸びる局面では、物価連動の給付は**賃金対比で侵食される**。
  「when wages grow faster than prices … benefit levels subjected to such
  price-linking mechanisms may be more prone to welfare erosion than benefit
  amounts that are regularly adjusted to wage developments」。
- 1990年代の西欧では賃金対比の侵食が広範に起きた。
  「Compared with average wages, they lagged most significantly behind in the
  Netherlands, Norway, Sweden, Ireland and the United Kingdom」。

> **文献確認で判明した訂正**（レビュー指示の内容から3点）：
> 1. **年は 2012**（GINI DP 55, July 2012）。書籍版は 2013年（Palgrave）。
> 2. **オランダは純粋な賃金連動国**である
>    （"Pure wage indexation is only applied in the Netherlands and in Denmark"）。
>    1980–90年代に連動が政治判断で切られたという記述は本文にない。
>    賃金対比で最も遅れたという記述はあるが、機構は別である。
> 3. **カナダは本論文の対象国に含まれない**（EU＋米3州＋ノルウェーのみ）。
>    カナダ1990年代の例は本論文では裏付けられない。
>
> Nelson (2011) は本文で引用されているが、Nelson/Wang 2016 は未確認である。

したがって主系列を `cpi_linked` とするのは、
**「物価には追随するが成長には追随しない」という OECD 諸国の制度の実態**に基づく。

### 4.2 post_employment との整合

post_employment では労働所得が自動化によって減り、給付に置き換わる。
**世帯所得の連動は自動的に `cpi_linked` 側に寄る**ので、
主系列の選択は post_employment とも一貫する。

### 4.3 基礎需要の水準

`sub_F + sub_G + sub_H = 0.65`（所得の65%）は較正されていない。
**流動性制約世帯（hand-to-mouth）としては余裕がありすぎる**可能性がある。
この層は所得の大半が食料と住居に消える。

0.65 から 0.95 まで掃引し、**どの水準から逃げ場のない領域が現れるか**を
反応面として報告する。理論境界 1.0 を超えない範囲で刻む。

**「結果が出るまで上げる」のではなく、当初の設定が不自然だった可能性の検証である。**

---

## 5. 既存結果への影響

**確保量に関する既存の結果は変わらない。**
`budget_ratio` は観測量であって、家計の配分にも動学にも入らない。
Phase C の表1（確保量の経路）が一致することを**回帰テストで固定する**。

---

## 6. テストで固定すること

- `budget_ratio` が確保量にも動学にも影響しないこと（Phase C の出力と一致）。
- 価格が上がると `budget_ratio` が下がること。
- 給付が増えると `budget_ratio` が上がること。
- 最貧分位が最初に 1.0 を割ること。
- 自給率の低いブロックほど `u_needed` が大きいこと。
- `u_needed > u_affordable` の領域が存在すること（あるいは存在しないこと）。
