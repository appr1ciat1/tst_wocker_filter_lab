# TW Stocker — v8.5 約束優化四策略（v8.5 / GUARD / SURGE / SURGE PRO）

固定純 **v8.5 橫向動量**為唯一 baseline，在「**相對 v8.5**」的硬約束下
（MDD 不深於 baseline + 3pp、逐年 OOS Sharpe 不低於 baseline、PBO < 0.5、交易數不因過度濾網崩掉）
做 **constrained parameter search + regime-aware sizing**，產出三個正式優化策略 + 純 v8.5 基準。
每日收盤後自動全期回測、產出四份報表並部署到 GitHub Pages。

> **更新 2026-06-27**：v9 Hybrid Tiered（Core-Satellite + Vol Target）經多次驗證**報酬與回撤皆遜於 v8.5**，
> 已淘汰為背景參考（見文末）。現以 **v8.5 家族四策略**為主線。舊版 README 的 v9 headline 不再沿用。

---

## 📊 線上報表（GitHub Pages，每工作日收盤後自動更新）

**https://appr1ciat1.github.io/tw_stocker_filter_lab/** — 獨立 Filter Lab 策略選單

| 卡片 | 內容 |
|---|---|
| [v8.5 CONFIRMED FILTER](https://appr1ciat1.github.io/tw_stocker_filter_lab/paper_trading_v85_confirmed.html) | v8.5 第一件事獨立濾網版 |
| [SURGE PRO CONFIRMED FILTER](https://appr1ciat1.github.io/tw_stocker_filter_lab/paper_trading_surge_pro_confirmed.html) | SURGE PRO 第一件事獨立濾網版 |
| [v8.5 300K](https://appr1ciat1.github.io/tw_stocker_filter_lab/paper_trading_v85_300k.html) | v8.5 第二件事獨立 30 萬版 |
| [SURGE PRO 300K](https://appr1ciat1.github.io/tw_stocker_filter_lab/paper_trading_surge_pro_300k.html) | SURGE PRO 第二件事獨立 30 萬版 |
| [Capital Rotation Alert](https://appr1ciat1.github.io/tw_stocker_filter_lab/capital_rotation_alert.html) | SURGE PRO 第三件事，三日確認警示 |

> 報表數字為**全期 2019-01-01 → 回測當日**（`--eval-start 2019-01-01`），與下表一致；
> 點時間值，會隨資料更新而漂移，非恆定保證。

## 五個隔離新版本（2026-07-16）

依三項需求拆成五版，避免把進場濾網、30 萬配置與輪動警示混在一起造成歸因錯誤；原四個正式策略完全不變。

| 母策略 | 件別 | 註冊名 | 2019–2026 同資料結果 | 判定 |
|---|---|---|---|---|
| v8.5 | 隔夜／龍頭／籌碼確認 | `momentum_v85_confirmed` | 年化 42.5%、Sharpe 1.73、MDD -20.9%、822 筆 | **PASS** |
| v8.5 | 30 萬配置 | `momentum_v85_300k` | 年化 39.5%、MDD -24.2% | **PASS** |
| SURGE PRO | 隔夜／龍頭／籌碼確認 | `mom_surge_pro_confirmed` | 年化 50.1%、MDD -23.6% | 研究版，未勝母策略 |
| SURGE PRO | 30 萬配置 | `mom_surge_pro_300k` | 年化 60.5%、MDD -23.6% | **30萬執行版** |
| SURGE PRO | 資金輪動警示 | `mom_surge_pro_rotation_alert` | 交易績效與母策略相同 | **警示專用，不自動交易** |

第一件事資料嚴格以台股開盤前的完整美股交易日為準，例如 7/16 台股開盤只使用 7/15 的 S&P 500、SOX、TSM ADR 開盤／日內區間／收盤；個股另映射全球龍頭。融資、融券、借券保留原始數量後，以前一日變化做橫斷面標準化，避免前視與規模偏誤。

第二件事的 paper 頁：[`paper_trading_v85_300k.html`](paper_trading_v85_300k.html) 與 [`paper_trading_surge_pro_300k.html`](paper_trading_surge_pro_300k.html)。第三件事的每日監控頁是 [`capital_rotation_alert.html`](capital_rotation_alert.html)，頁面完整列出 192 次三日確認事件，且每一筆可展開查看全部 5,596 筆逐股結果；可重現原始檔為 [`capital_rotation_history_10y.json`](capital_rotation_history_10y.json) 與根目錄下的 `capital_rotation_*_10y.csv/json`。20% 回撤命中率未優於配對基準，因此此版本只在收盤後產生下一交易日可見警報，不改變持倉；回撤日期與前一交易日只記錄歷史結果，不得解讀為預測機率或賣出期限。

---

## ⚠️ 2026-07-25 稽核後的定位（先讀這一段）

本系統**尚未證明具備可宣稱的統計優勢**。以下是實測結論，不是保守措辭：

> 🚨 **先讀這個：下表的數字不穩定，而且不穩定的幅度大於宣稱的優勢。**
>
> 2026-07-26 實測（同一份程式、同一組參數、**同一個回測視窗 2019-01-01~2026-07-16**，
> 只有「下載資料的日期」相差一天）：
>
> | v8.5 | 年化 | MDD | 交易 |
> |---|---:|---:|---:|
> | 07-25 下載（＝下表與四份報表的來源） | 27.5% | −34.7% | 845 |
> | 07-26 下載 | **18.8%** | −36.7% | 827 |
>
> 年化差 **8.7pp**，比任何一個策略對池的 α 都大。成因是資料源會**回溯調整歷史價格**：
> 七月是台股除權息旺季（抽樣 40 檔就有 14 檔近一個月內除息，含華碩 42 元、聯發科 24.5 元、
> 聯詠 23 元、廣達 15.6 元），調整後整段歷史被重新縮放 → 動量排名改變 → 選股改變 → 路徑改變。
>
> 已排除的可能：**不是**管線問題。CLI 路徑（ai_report）與插件路徑（build_paper_page）
> 用同一天的資料跑出 17.17% vs 17.0%、MDD 皆 −36.7%、交易皆 830 筆——兩層等價成立。
> 也**不是**池縮水（115 檔全數下載成功）。
>
> ⇒ **任何單一數字都只在「那次下載」成立。** 要可重現就必須保留凍結快照
> （`twstk.data.contract.freeze_snapshot` 已實作，但快照目前被 `.gitignore` 排除，
> 只留 manifest，所以既有的發布數字事後無法驗證）。這是目前最該優先處理的缺口。

以下每個數字都直接取自四份報表（點進去可核對），窗口為 n=1766 個交易日：

| 對照（全期 2019-01~2026-07，20bps 滑價，含息） | 年化 | Sharpe | MDD |
|---|---:|---:|---:|
| **116 池等權買進持有**（＝什麼都不做） | 30.1% | **1.60 全場最高** | -29.4% |
| 0050 買進持有 | 31.6% | 1.39 | -33.8% |
| GUARD | 30.9% | 1.31 | -21.0% |
| SURGE | 34.0% | 1.27 | -29.5% |
| SURGE PRO | 32.9% | 1.25 | -29.6% |
| v8.5 baseline | 27.5% | 1.15 | -34.7% |

**三個必須知道的事實：**

1. **沒有任何策略的 Sharpe 贏過「等權抱著自己挑的 116 檔」。**
   對池的 α 經 Newey-West 修正後 t = 0.93~1.24，**全部不顯著**（門檻 1.96）。
   Monte Carlo（bootstrap 配對差額）四個策略的 95% CI **全部含 0**。

2. **2022 是樣本內唯一的空頭年，所有策略都比「什麼都不做」慘 4.8~14.4pp。**
   池 -8.4%；GUARD -13.2%（-4.8pp，最抗跌）、v8.5 -17.9%（-9.4pp）、
   SURGE PRO -18.9%（-10.4pp）、**SURGE -22.8%（-14.4pp，最慘）**。
   ⇒ 全期 MDD 較淺是**路徑效應，不是空頭保護**。
   逐年勝池（排除不完整的 2019/2026）：SURGE 4/6、SURGE PRO 3/6、GUARD 3/6、**v8.5 2/6**。

3. **唯一顯著的正面證據在「訊號層」，不在完整策略。**
   真實前瞻紀錄（2026-04~07，非重跑模擬）：固定 20 日持有下，
   訊號相對池 **+5.31pp/窗、配對 t = 3.44（顯著）**；
   但加上 TP/SL 出場後降到 +1.04pp（t=1.06），再加成本剩 +0.44pp（t=0.45）。
   ⇒ **出場規則吃掉約 80% 的訊號價值**（該期為極端多頭，停利上限特別傷；
   空頭時停損應有價值，需更長樣本才能分辨）。

**這代表什麼**：這是一套**研究系統**，其可驗證的價值目前在「每日訊號提示看哪幾檔」，
而不是「這個包裝好的策略年化 X%」。任何新想法必須先贏過池等權，才談得上有價值。

詳見 `audit_reports/`：稽核報告、Codex ablation 稽核、定位重寫、策略決策文件。

---

## 四策略（全期 2019-01~2026-07，動態 Top-60，20bps 滑價，`consec_loss_limit=3`）

★數字為 **2026-07-25 修正後**（同日時序穿越 + 零滑價 + `.TWO` 代碼三個 P0 已修）。
修正前發布的 39~58% 已作廢，兩者差異見下表。

| 策略 | 註冊名 | 年化 | Sharpe | MDD | Calmar | 交易 | 對池 α (NW t) |
|---|---|---:|---:|---:|---:|---:|---|
| **v8.5 baseline** | `momentum_v85` | 27.5% | 1.15 | -34.7% | 0.79 | 845 | +0.9% (0.10) ❌ |
| **GUARD** | `mom_guard` | 30.9% | **1.31** | **-21.0%** | **1.47** | 920 | +8.7% (1.06) ❌ |
| **SURGE** | `mom_surge` | **34.0%** | 1.27 | -29.5% | 1.15 | 770 | +15.0% (1.51) ❌ |
| **SURGE PRO** | `mom_surge_pro` | 32.9% | 1.25 | -29.6% | 1.11 | 684 | +7.5% (0.81) ❌ |

<details><summary>修正前 vs 修正後（點開）</summary>

| 策略 | 舊發布年化 | 新 | 舊 MDD | 新 |
|---|---:|---:|---:|---:|
| v8.5 | 39.3% | 27.5% | -39.8% | -34.7% |
| GUARD | 46.7% | 30.9% | -24.4% | -21.0% |
| SURGE | 49.5% | 34.0% | -25.7% | -29.5% |
| SURGE PRO | **58.0%** | **32.9%** | **-19.1%** | **-29.6%** |

三個 P0 修正：(1) 同日時序穿越——開盤下單用到當日收盤價與當日盤中回款；
(2) 滑價預設為 0，所有發布績效都是零滑價名目回測（現為 20bps）；
(3) `.TWO` 代碼被 `replace('.TW','')` 剝成 `5274O`。
</details>

- **⚠️ 已作廢的宣稱**：「同時改善報酬與回撤」不成立（見 2022）；
  「PBO 0.09–0.34、DSR ≈ 1.0、nested walk-forward」**找不到任何產出檔佐證**
  （`walk_forward_results.csv` / `monte_carlo_results.json` 皆不存在，
  git 從未追蹤過任何 WF/MC 結果）。

## 各策略定義（單一真實來源：`strategies/optimized_v85.py`）

四策略共用同一組 v8.5 評分（Mom×3 + Trend×1，`MomentumV85.prepare()`），差別只在事件引擎的風控/sizing 參數：

- **GUARD**：`sl_atr=3.5` + graduated regime（`regime_floor=0` 最弱全出）+ breadth-aware + `dynamic_topk`；**不加碼**。
  2026-07 起加入 **corr_select 相關性分散選股**（效率前緣/共變異數觀點）：greedy 依評分選股，候選與
  (持倉∪已選) 的 60 日相關 >0.7 之檔數達 2 即跳過（同聚落最多 2 檔）。驗證：ann 43.2→51.6%、
  Sharpe 1.54→1.78、MDD -34.2→-26.8%、2022 -14.9→-10.3%，鄰域 (0.65–0.75 / 40–80日) 全穩健。
  **同機制在 v8.5 / SURGE / SURGE PRO 測試皆變差**（加碼型策略的 alpha 靠集中聚落）故僅 GUARD 採用；
  cap=1（嚴禁同群）亦全軍覆沒——台股動量聚落太強，完全禁掉會錯過主升段。
- **SURGE**：GUARD 去風險不變 + **分段強勢加碼**（只在 0050>MA60/MA20 且 breadth 高、VIX 低時放大單筆）。
  四段式單筆部位：弱 0% / 強 12.5% / 更強(breadth≥.65,VIX≤20) 14.5% / 最強(breadth≥.75,VIX≤15) 17%。
- **SURGE PRO**：SURGE 去風險不變 + **更激進分段加碼**（VIX 門檻放寬 28、tier 倍數更高、`max_regime_scale=1.9`、`hold_days=25`）。
  四段式：弱 0% / 強 12.5% / 更強 17% / 最強 18.5%。
- **v8.5 baseline**：`momentum_v85`，binary regime、無去風險、無加碼。

引擎（向後相容）新增：`EventDrivenBacktester(strong_tiers=[(breadth,vix,mult),...])` 分段加碼、`run(..., vix_series=)` 可注入 VIX、
`corr_select_max/window/cap` 相關性分散選股（greedy，跳過與持倉∪已選高相關者，不回補——寧缺勿濫）。

---

## 四策略實際差異

四策略「**選股訊號相同、弱勢去風險邏輯相同**」，差別只在「強勢時加碼的積極度」與「進場挑剔度」。

以下是 2019-01 ~ 2026-07、20bps 滑價、含息的**實測值**（點進各報表可見逐筆交易）：

| 策略 | 機制 | 年化 | MDD | Calmar | Sharpe | 2022（唯一空頭年） | 對池 α | NW t | 逐年勝池 |
|---|---|---|---|---|---|---|---|---|---|
| **SURGE** | 去風險 + 分段加碼 | **+34.0%** | −29.5% | 1.15 | 1.27 | −22.8%（輸池 **14.4pp**，四者最差） | +11.3% | 1.24 | 5/8 |
| **SURGE PRO** | 去風險 + 最激進分段加碼 | +32.9% | −29.6% | 1.11 | 1.25 | −18.9%（輸池 10.4pp） | +11.3% | 1.18 | 4/8 |
| **GUARD** | 純去風險、不加碼、相關性分散（同群 ≤2 檔） | +30.9% | **−21.0%** | **1.47** | **1.31** | **−13.2%**（輸池 4.8pp，四者最好） | +10.5% | 1.24 | 4/8 |
| **v8.5** | 純動量、永遠滿倉、不去風險不加碼 | +27.5% | −34.7% | 0.79 | 1.15 | −17.9%（輸池 9.4pp） | +8.2% | 0.93 | 3/8 |
| *116 池等權買進持有* | *不擇時、不停損、不加碼* | *+30.1%* | *−29.4%* | — | ***1.60*** | ***−8.4%*** | — | — | — |

**只有兩套在效率前緣上**：SURGE（年化最高）與 GUARD（MDD／Calmar／Sharpe／2022 四項最好）。
**SURGE PRO 與 v8.5 沒有任何一項第一**——SURGE PRO 的分段加碼層在 ablation 中也未通過統計驗證。

> ⚠️ **沒有單一「最好」，而且目前沒有依據宣稱任何一個優於「什麼都不做」**：
> 四者 Sharpe 1.15~1.31 **全數低於池等權買進持有的 1.60**；對池 α 經 Newey-West 修正後
> t = 0.93~1.24，**全部不顯著**（門檻 1.96）；而在唯一的空頭年 **四者全部輸給池 4.8~14.4pp**。
>
> 先前 README 這裡有一張「什麼行情該用哪一套」的對照表，**已整段移除**，兩個原因：
> 1. 修正同日時序穿越與零滑價後排名整個改變，表內指派全部過期；
> 2. 更根本地，那張表宣稱情境法則，但全部證據只有一條 8 年路徑、**其中只有一個空頭年**。n=1 推不出法則。
>    當時表內還寫著「💥 崩盤 → **SURGE**（最防守）、崩盤年 OOS 最佳」——SURGE 恰恰是 2022 **最差**的那一個。
>    錯在最危險的方向：叫人在崩盤時選最不抗跌的。
>
> 同樣作廢的還有「SURGE PRO 報酬／Sharpe／Calmar 全期最高、PBO 居首」——PBO/DSR 找不到產出檔佐證，
> 且該 ablation 的判定經稽核證實方法無效（hurdle 以觀測均值重新置中，導致 0/22 試驗可能通過）。
>
> Paper 頁仍追蹤 **SURGE PRO**，理由是**前瞻紀錄連續性**（`forward_log.csv` 已累積至 2026-07-24），
> 不是因為它最好。換追蹤標的會斷掉這條唯一的樣本外證據鏈。

---

## 策略方法論：每一層的邏輯、方法與理論依據

策略由五層組成，每層各自解決一個問題、各有明確的理論依據。**訊號決定「買什麼」，其餘四層決定「買多少、何時買、何時出、怎麼不騙自己」。**

### 第 1 層｜訊號：橫斷面動量（買什麼）
- **做法**：v8.5 評分 = 20 日動量 ×3 + 60MA 趨勢 ×1，在流動性 Top-60 池內做**橫斷面排名**；每日取「評分 ≥2.0 且站上 60MA」的 Top-7。
- **依據**：橫斷面動量效應（Jegadeesh & Titman, 1993 — 過去 3–12 個月贏家續強）；趨勢/MA 濾網（時間序列動量，Moskowitz–Ooi–Pedersen, 2012 的概念）。
- **關鍵定位**：動量評分本身就是「預期報酬 μ 的自有觀點」。Markowitz 框架的死穴是 μ 極難估（用歷史平均會整條前緣都錯）——本系統用動量觀點取代歷史均值來回答 μ，即 **Black–Litterman「以主觀觀點修正 μ」的精神**；因此 **Σ（共變異數）只用於風險端，絕不讓它決定買什麼**（見第 4 層）。

### 第 2 層｜出場：波動標準化風控（何時出）
- **做法**：ATR 停利 4×／停損 3–3.5×（Wilder ATR，引擎內部計算）＋最長持有 20–25 日強制出場＋開盤跳空 >1.5×ATR 放棄進場＋**連續虧損 3 筆熔斷**暫停新倉（`consec_loss_limit=3`）＋權益回撤暫停。
- **依據**：以 ATR 把停損距離「波動標準化」，使高低波動股承擔一致的風險預算（固定百分比停損做不到）；熔斷是對「訊號失效期」的簡單貝氏防禦——連錯代表環境變了，先停手。

### 第 3 層｜Regime：環境決定倉位（何時買、買多大）
- **做法**：graduated regime（0050 相對 60MA/20MA 漸進調整倉位，`regime_floor=0` 最弱時全出）＋市場廣度 breadth＋VIX。SURGE/SURGE PRO 再加**分段強勢加碼** `strong_tiers`：僅在「廣度高＋VIX 低」的強勢環境把單筆從 10% 分段放大到 12.5–18.5%。
- **依據**：動量策略最大的尾部風險是「動量崩潰」（momentum crash，Daniel & Moskowitz, 2016——崩盤反彈期動量重挫），regime 濾網+去風險就是對此的直接防禦；2022 升息年的實測（v8.5 -41% MDD vs GUARD/SURGE -21~-27%）是本系統的活教材。
- **界線（測試得出）**：加碼只做「環境門檻的橫斷面放大」，**不做時間序列 Vol Target 槓桿調節**——v9（Portfolio Vol Target + Core-Satellite）實測報酬回撤皆遜於 v8.5 且 PBO 0.94，已淘汰。

### 第 4 層｜組合：相關性分散（效率前緣／共變異數觀點）
- **理論**：Markowitz (1952) 效率前緣——組合風險 σₚ=√(wᵀΣw) 取決於**資產間相關性**而非個股波動加總；同漲同跌的持股沒有分散效果。入門實作參考：[FinvestNote《用 Python 繪製效率前緣》](https://finvestnote.com/posts/efficient-frontier-and-python/)。
- **本系統的實證問題**：動量訊號天生擠同族群。實測 Top-7 訊號股平均 pairwise 相關 0.37、聚落內 0.6–0.8（如華邦電×南亞科 0.82、群創×彩晶 0.68）→ 有效獨立注數 N/(1+(N−1)ρ̄) **≈ 2.2 注**——名義上分散 7 檔、實際像押 2 檔，是 MDD 的隱形來源。
- **落地（GUARD 的 `corr_select`，2026-07）**：greedy 依評分選股，候選與（持倉∪已選）60 日相關 >0.7 的檔數達 2 即跳過（同聚落 ≤2 檔）、不回補。驗證：ann 43→52%、Sharpe 1.54→1.78、MDD -34→-27%、2022 年 -15→-10%，鄰域參數全穩健。
- **兩個由驗證劃出的界線**：(a) **不做完整 Markowitz/MVP**——風險極小化會系統性剔除高動量高波動股（該文 MVP 示範：JNJ 佔 53.77%、NVDA/META 壓到 0%），等於殺死動量 alpha；Σ 只管分散，μ 由動量觀點決定。(b) **加碼型策略（SURGE/SURGE PRO）不用 corr_select**——它們的 alpha 正是「強勢時集中壓動量聚落」，實測加上後年化 -4~-12pp；嚴禁同群（cap=1）在台股也全面變差，同聚落 ≤2 檔才是甜蜜點。

### 第 5 層｜驗證：如何避免騙自己（所有參數的守門員）
- **相對基準約束搜尋**：任何新配置必須「MDD 不深於 v8.5+3pp、逐年 OOS Sharpe 不低於基準、交易數不因過度濾網崩掉」才收。
- **過擬合檢驗**：PBO/CSCV（Bailey et al.）<0.5、Deflated Sharpe（Bailey & López de Prado）、nested walk-forward（train→select→test）。
- **穩健性紀律**：參數採用前做鄰域掃描（要的是 plateau 不是刀鋒值），且**只驗證不挑事後最佳**；新舊配置比較必須**同一次 run 內 off vs on**（yfinance 資料修訂＋路徑依賴會讓跨日數字漂移，拿舊數字對照會得出假結論）。
- **負結果也保留**：測過但不採用——完整 Markowitz/MVP、v9 Vol Target（PBO 0.94）、法人 5/10/20 日籌碼出場（`inst_hold_exit`）、低勝率風報比 gate（`low_wr_rr_gate`）、corr_select 用於加碼型策略、cap=1 嚴禁同群。待評估：Top-K 內逆波動（1/ATR）加權、SURGE PRO/SURGE 70/30 資金混搭。

---

## 跑法（CLI）

```bash
# 任一策略全期回測（twstk runner；不加 --eval-start 即全期）
python -m twstk.backtest.runner --strategy mom_surge_pro --start-date 2019-01-01

# 列出所有策略
python -m twstk.backtest.runner --list

# 產生 HTML 報表（ai_report.py；務必加 --eval-start 看全期、--consec-loss-limit 3）
python ai_report.py --no-hybrid-tiered --consec-loss-limit 3 \
  --sl-atr 3.5 --hold-days 25 --position-size 0.10 \
  --regime-floor 0.0 --dynamic-topk --dynamic-gap-filter \
  --regime-sizing --strong-regime-mult 1.25 --strong-breadth-min 0.55 --strong-vix-max 28.0 \
  --max-regime-scale 1.9 --strong-tiers "0.62,18,1.7;0.72,15,1.85" \
  --start-date 2019-01-01 --eval-start 2019-01-01      # ← SURGE PRO
```

完整指令對照與驗證細節見 [`artifacts/v85_optimization_result.md`](artifacts/v85_optimization_result.md)。

### ⚠️ 兩個會誤導結果的關鍵 gotcha
1. **ai_report 必加 `--consec-loss-limit 3`**：此旗標預設 99（停用連損熔斷），少了它 MDD 會從 -23% 惡化到 -40%。
   `momentum_v85` / twstk runner 走引擎預設 3，故自帶此保護。
2. **ai_report 看全期數字必加 `--eval-start 2019-01-01`**：不加時用預設視窗，會嚴重低估激進策略
   （例：SURGE PRO 會顯示偏低、MDD 偏深、順序甚至反轉）。twstk runner 不加 `--eval-start` 即全期。

---

## 每日 pipeline（GitHub Actions）

`.github/workflows/update_ai_report.yml`（每工作日 UTC 09:17 + 可手動 `workflow_dispatch`）：
依序全期重跑 **v8.5 → GUARD → SURGE → SURGE PRO**，每份報表的標題改為各自策略名，
產出 `report_v85 / report_guard / report_surge / report_surge_pro.html` + 各自圖表；
**SURGE PRO 最後跑**，使 `stock_report.html` 與 `artifacts/orders_*.json` = SURGE PRO，故
`paper_tracker.py` 追蹤的是 **SURGE PRO** 的每日訊號。`deploy_pages.yml` 接著部署到 GitHub Pages。

> workflow 需 `permissions: contents: write` 才能 commit 報表回 repo（已設）。

---

## 架構（三套件 + 策略插件）

```
twstk/            三層基礎設施
  data/           純資料層（yfinance 股價/0050、appr1ciat1 三大法人、SBL、美股訊號）
  backtest/       歷史回測 CLI（runner）+ 指標
  paper/          每日 paper 模擬
strategies/       策略插件（@register）：momentum_v85 / optimized_v85(mom_guard/surge/surge_pro)
                  / sector_rotation_v2 / hybrid_tiered_v9 / reversal / ew_momentum
strategy/         共用引擎與因子（event_backtest 事件引擎、ai_strategy 評分、risk_metrics…）
validation/       PBO(CSCV) / Deflated Sharpe
research/          constrained_search / validate_* / tune_surge_broad …（搜尋與驗證工具）
ai_report.py      事件驅動回測 + HTML 報表/交易計畫產生器
paper_tracker.py  讀當日訂單追蹤 TP/SL/時間出場，累積 paper 權益曲線
```

新策略：在 `strategies/` 下用 `@register("name")` 註冊，並於 `strategies/registry.py` 匯入即可。

---

## 背景 / 歷史（為何回到 v8.5 家族）

- **v8.5 Momentum**：個股 cross-sectional ranking（Mom 20d×3 + Trend 60MA×1），台股 0050/MA60 regime。優化前的底層 alpha。
- **Sector Rotation v2**：美股 SPY/VIX/SOX regime + 板塊資金流 + 板塊內動量。MDD 較深（~-38%），僅適合當小比例 sleeve（10–25%），不宜當主體。
- **v9 Hybrid Tiered**（已淘汰）：在 v8.5 上疊 Portfolio Vol Target + Core-Satellite。多次驗證
  （含 2026-06 行情、nested walk-forward PBO=0.94）顯示**報酬與回撤皆遜於 v8.5**，故不再作為主線。
- 測過但**未採用**：法人 5/10/20 日籌碼退場（`inst_hold_exit`）與低勝率風報比 gate（`low_wr_rr_gate`）——
  在本約束下都降低目標、無正貢獻；引擎仍保留旗標可選用。

> 績效皆為點時間回測，**非未來保證**。請以最新一次回測 / walk-forward OOS 為準。
