# 実データ指示書: Asset-Relative Wage Index の構築

## 目的

論文 "The Wage That Vanished" の Section 2 で提示する Figure 4 を、シミュレーション出力ではなく実際の経済データから構築する。1971–2024の米国の賃金を複数の資産価格で割った時系列を作成し、「名目では右肩上がり、CPI実質ではほぼ横ばい、資産建てでは崩落」という対比を実データで可視化する。

## データソース（全てFRED or 公開データ）

### 賃金（分子）
- **Average Hourly Earnings of Production and Nonsupervisory Employees, Total Private**
  - FRED series: `AHETPI`（旧コード `CES0500000008`）
  - 1964年〜現在、月次
  - 名目ドル
  - これが最も長い連続時給系列。median hourly earningsは1979年開始なので使えない

### 資産価格（分母）
1. **金価格**: London Bullion Market Association (LBMA) PM Fix
   - 利用可能なFRED代替: ICE Benchmark Administration の Gold Fixing Price
   - FRED series: `GOLDPMGBD228NLBM`（日次）→ 月次平均に変換
   - 1968年〜現在
   - 1968–1971/8は公式ペグ$35に近いが市場価格あり

2. **住宅価格**: S&P/Case-Shiller U.S. National Home Price Index
   - FRED series: `CSUSHPINSA`
   - 1987年1月〜現在、月次
   - **1987年以前は使えない**。代替として全米中位住宅販売価格:
   - FRED series: `MSPUS`（Median Sales Price of Houses Sold）
   - 1963年Q1〜現在、四半期
   - 四半期データなので月次に補間するか四半期のまま使う

3. **株式**: S&P 500
   - FRED series: `SP500`（日次終値）→ 月次平均に変換
   - 1928年〜現在
   - **注意**: これは価格指数のみで配当を含まない。Total Return Index は Yahoo Finance (`^SP500TR`) から取れるが1988年〜のみ
   - 論文では price index を使い、配当再投資込みだとさらに崩落が激しい旨を注記する

4. **CPI**: Consumer Price Index for All Urban Consumers
   - FRED series: `CPIAUCSL`
   - 1947年〜現在、月次

5. **CRB商品バスケット**（参考・コントロール用）
   - Refinitiv/CoreCommodity CRB Index は FRED にない
   - 代替: Producer Price Index - All Commodities (`PPIACO`)、1913年〜現在
   - または WTI原油 (`DCOILWTICO`) を個別商品の代表として

## データ取得方法

```python
# FRED API（fredapiパッケージ）または pandas-datareader
# API key不要の方法: FRED CSVダウンロード
import pandas as pd

# 各系列のFRED URL pattern:
# https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES_ID&cosd=1964-01-01&coed=2024-12-31

series = {
    'wage': 'AHETPI',           # Average hourly earnings, production workers
    'gold': 'GOLDPMGBD228NLBM', # Gold fixing price, daily → resample monthly
    'housing': 'MSPUS',         # Median sales price of houses sold, quarterly
    'sp500': 'SP500',           # S&P 500, daily → resample monthly
    'cpi': 'CPIAUCSL',          # CPI-U, monthly
    'ppi_commodities': 'PPIACO' # PPI all commodities, monthly
}
```

## 構築する系列（全て 1971 = 100 にインデックス化）

### 系列1: 名目賃金指数
```
nominal_index(t) = wage(t) / wage(1971) * 100
```

### 系列2: CPI実質賃金指数
```
real_index(t) = [wage(t) / cpi(t)] / [wage(1971) / cpi(1971)] * 100
```

### 系列3: 金建て賃金指数
```
gold_index(t) = [wage(t) / gold(t)] / [wage(1971) / gold(1971)] * 100
```

### 系列4: 住宅建て賃金指数
```
housing_index(t) = [wage(t) / housing(t)] / [wage(1971) / housing(1971)] * 100
```
住宅価格は四半期なので、賃金も四半期平均にダウンサンプルするか、住宅価格を線形補間で月次にアップサンプル。

### 系列5: 株式建て賃金指数
```
sp500_index(t) = [wage(t) / sp500(t)] / [wage(1971) / sp500(1971)] * 100
```

### 系列6: 商品建て賃金指数（コントロール）
```
commodity_index(t) = [wage(t) / ppi_commodities(t)] / [wage(1971) / ppi_commodities(1971)] * 100
```

## 出力要件

### Figure 4a: メインの6系列比較（論文用）
- 全6系列を1つのグラフに
- Y軸: 対数スケール（系列間のオーダー差が大きいため）
- X軸: 1971–2024
- 線の色:
  - 名目: 青
  - CPI実質: 緑
  - 金建て: オレンジ/赤
  - 住宅建て: 紫
  - 株式建て: ピンク
  - 商品建て: 灰色（破線）
- 基準線: y=100（1971年水準）を薄い水平線で
- 各線の終端値を右端にアノテーション
- 論文掲載品質: 300 dpi、フォント12pt以上、軸ラベル明確

### Figure 4b: ロバストネス（起点変更）
- 金建て賃金を4つの異なる起点で比較:
  - 1971 = 100
  - 1975 = 100（ポストペグ調整後）
  - 1980 = 100（金価格スパイク後）
  - 1985 = 100（安定期）
- 定性的パターン（持続的低下）が起点に依存しないことを確認

### Figure 4c: 住宅建ての詳細（Case-Shiller版）
- 1987年以降のみ
- Case-Shiller全米指数 (`CSUSHPINSA`) と中位住宅価格 (`MSPUS`) の両方で計算
- 両者が整合することを確認

### Figure 4d: 新興国の金建て賃金（国際比較）

米国以外の国について金建て賃金を構築し、φの構造が先進国固有なのかグローバルなのかを検証する。

**対象国と期間:**
- 中国: 2000–2024（データ可用性に依存）
- ブラジル: 1995–2024
- インド: 2000–2024
- 日本: 1971–2024（先進国の対照群として）
- 米国: 1971–2024（ベースライン比較用）

**データソース:**

| 国 | 賃金データ | ソース | 通貨 |
|---|---|---|---|
| 中国 | 都市部平均賃金（城镇单位就业人员平均工资） | 国家統計局 (NBS) / ILO ILOSTAT | CNY |
| ブラジル | 平均月間賃金 (Rendimento médio) | IBGE PNAD | BRL |
| インド | 製造業平均日給 or PLFS賃金データ | Annual Survey of Industries / RBI | INR |
| 日本 | 毎月勤労統計・現金給与総額 | 厚労省 / FRED `JPNWAG` or ILO | JPY |
| 米国 | AHETPI | BLS / FRED | USD |

**金価格の通貨換算:**
- 金はロンドンドル建て（LBMA fix）で取引される
- 各国の金建て賃金 = 現地通貨賃金 / (LBMA金ドル価格 × 現地通貨/ドル為替レート)
- 為替レートソース: FRED（`DEXCHUS`, `DEXBZUS`, `DEXINUS`, `DEXJPUS`）or BIS

**構築する系列:**
```
gold_wage_country(t) = wage_local(t) / [gold_usd(t) × fx_rate(t)]
```
各国を独自の起点年 = 100 にインデックス化。

**期待される結果と解釈:**

1. **中国**: 2000年代は金建てでも上昇している可能性が高い（実質賃金が年10%超で成長、金は年15–20%上昇の期間もあるが賃金が追いつくか）。ただし2010年代以降は金価格高騰で横ばい→低下に転じる可能性。もし上昇期があれば「great doublingの供給側にいる国では、資本流入がlabor's outside optionを一時的に改善した」と読める——φの枠組みと整合
2. **ブラジル・インド**: commodity boom期（2003–2011）に賃金が上がったが、金建てでの持続性は疑わしい。boom後に反転していれば「一次産品依存の成長はフロー的で、asset-relative賃金の持続的改善にはならない」
3. **日本**: 1971–1995は上昇（高度成長→バブル期）、1995–2024は崩落。特に2012年以降の円安×金高騰でドル建て・金建ての二重崩落が予想される。日本のケースは「先進国でもφの拡大＋通貨安で崩落が加速しうる」ことを示す

**プロットの仕様:**
- 5カ国を1つのパネルに。各国の起点年が異なるため、共通起点（2000年 = 100）で重ねるのが比較に最適
- 2000年以前のデータがある国（米国・日本）は薄い線で延長表示
- Y軸: 対数スケール
- 色: 米国=青、中国=赤、日本=緑、ブラジル=黄、インド=紫

**データ取得の注意:**
- 中国のNBS賃金データは年次のみ。月次・四半期は不可能な場合あり→年次で構築
- インドの賃金データは断絶が多い（ASI→PLFS移行）。ILOSTATの統合系列を優先
- ブラジルはPNAD→PNAD Contínua移行（2012年）で接合が必要
- **データが取れない国は無理に含めない。取れた国だけで図を作り、取れなかった国は注記する**
- ILOSTATのバルクダウンロード（CSV）: https://ilostat.ilo.org/data/ の "Earnings and labour cost" → "Mean nominal monthly earnings of employees" (EAR_4MTH_SEX_ECO_CUR_NB) を使うと複数国を一括取得できる可能性あり

### 数値レポート（results_realdata.json）
```json
{
  "base_year": 1971,
  "end_year": 2024,
  "wage_1971_nominal": 3.45,
  "wage_2024_nominal": 30.XX,
  "index_values_2024": {
    "nominal": XXX,
    "cpi_real": XXX,
    "gold": XX,
    "housing": XX,
    "sp500": X,
    "commodity_ppi": XXX
  },
  "decline_pct": {
    "gold": -XX,
    "housing": -XX,
    "sp500": -XX,
    "commodity_ppi": "+XX (wages outpaced)"
  },
  "robustness": {
    "gold_1975_base_decline_pct": -XX,
    "gold_1980_base_decline_pct": -XX,
    "gold_1985_base_decline_pct": -XX
  },
  "international_gold_wages": {
    "base_year": 2000,
    "countries": {
      "USA": {"index_2024": XX, "change_pct": -XX},
      "Japan": {"index_2024": XX, "change_pct": -XX},
      "China": {"index_2024": XX, "change_pct": "XX (may be positive)"},
      "Brazil": {"index_2024": XX, "change_pct": -XX},
      "India": {"index_2024": XX, "change_pct": "XX"}
    },
    "notes": "Countries with unavailable data omitted. See README for data gaps."
  }
}
```

## ファイル構成

```
realdata/
├── README.md
├── requirements.txt          # pandas, matplotlib, seaborn, requests
├── src/
│   ├── fetch_data.py         # FREDからCSVダウンロード
│   ├── fetch_intl_data.py    # ILO/NBS/IBGE等から新興国データ取得
│   ├── construct_indices.py  # 米国6系列の構築
│   ├── construct_intl.py     # 新興国金建て賃金の構築
│   └── plot_figures.py       # Figure 4a, 4b, 4c, 4d の生成
├── data/
│   └── (ダウンロードしたCSVが入る)
├── output/
│   ├── fig4a_asset_relative_wages.png
│   ├── fig4b_gold_robustness.png
│   ├── fig4c_housing_detail.png
│   ├── fig4d_international_gold_wages.png
│   └── results_realdata.json
└── tests/
    └── test_indices.py       # 基本的な整合性テスト
```

## コード品質
- 型ヒント使用
- docstring（英語）
- データ取得失敗時のエラーハンドリング
- FRED API key不要（CSV直接ダウンロード方式）
- 全データをローカルにキャッシュ（再実行時にダウンロードしない）

## 注意事項

- **1971年の金価格**: 1971年8月にニクソンショック。年平均は約$41/oz（年初$37、年末$44付近）。年平均を使う
- **住宅価格の接合**: MSPUS（1963–現在、四半期）とCSUSHPINSA（1987–現在、月次）は水準が異なる（MSPUSはドル建て実額、Case-Shillerは指数）。メインのFigure 4aではMSPUSを使用（実額で割り算が直感的）。Case-Shillerは4cのロバストネス用
- **S&P 500**: 1971年1月の月次平均は約92ポイント。2024年12月は約5,900。price only index で約64倍。Total return だと数百倍になるが、price only を基本とし注記で言及
- **商品バスケット**: PPI All Commoditiesが「賃金に負ける」ことを確認すること。これは論文の主張（wealth-storing assetsのみで崩落、消費的商品では崩落しない）の実データ検証になる
- 全ての図は論文掲載品質。スタイルは simulation の fig1–fig5 と統一（seaborn-v0_8-whitegrid相当）

## 実行環境
- Python 3.10+
- 依存: pandas, matplotlib, seaborn, requests
- ネットワーク: fred.stlouisfed.org へのアクセスが必要
