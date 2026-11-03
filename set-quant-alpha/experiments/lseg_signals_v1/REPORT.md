# lseg_signals_v1 - do the other LSEG datasets carry alpha beyond S5's score? (2026-11-03)

Code: `lseg_signals_v1.py` (stages `features`, `discover`, `holdout`). Tables: `output/tables.md`.

* **Protocol.** Frozen in `output/protocol_frozen.txt` at 2026-11-03T01:58:42Z, before any signal value was computed.
* **Decision.** Written to `output/discovery_decision.txt` (02:09:02Z) before any 2022-26 number existed; the holdout
  was computed once.
* **Universe.** Survivorship-corrected L, tradable top-100 (as fundamentals_v1).
* **Data.** 15 new LSEG groups for the 263 SET100-ever stocks (`scripts/fetch_lseg.py extra`): price targets, DPS,
  revenue and FY2 EPS forecasts, EPS dispersion, SmartEstimate, recommendation counts, StarMine Combined Alpha,
  shares outstanding, free float, ESG, quarterly EPS with report timestamps, I/B/E/S surprises, quarterly financials
  with original announcement times, dividend events.

**Verdict: two signals pass every gate; the strategy does not.**

* **Earnings surprise (`surp`, post-earnings drift).** Passes in both periods with the same sign, and adds information
  beyond S5's score:
  * 2013-21: 5-day IC t 3.7, Fama-MacBeth (FM) t 3.1;
  * 2022-26: IC t 3.9, FM t 3.9.
* **StarMine Combined Alpha (`ca`).** Passes at both horizons in both periods (2022-26: 20-day IC t 3.5, FM t 3.9).
  * Its pre-launch history may be backfilled, so the 2013-21 part is not trusted.
  * The 2022-26 values were published live by LSEG, so that half is a genuine out-of-sample record of a vendor model.
  * As pre-registered, it is not in any composite.
* **The strategy fails the holdout.**
  * The pick F = (ML + V8 + blend of 11 new signals) / 3 beat S5 in 2013-21: 17.3% vs 16.0%, Sharpe 1.00 vs 0.91.
  * In 2022-26 it lost: 4.5% vs 7.7%, below TDEX's 6.7%.
  * Sharpe difference vs S5 −0.15 (95% CI −0.79 to +0.45).
  * The blend's marginal information fell from FM t 3.6 to 1.1.
* **The same rotation as fundamentals_v1.**
  * Analyst-revision signals (dispersion, DPS and price-target revisions, SUE) worked in 2013-21 and faded after.
    Dispersion even flipped sign.
  * Forward dividend yield, dead in 2013-21, was the strongest 2022-26 signal (IC t 4.6).
  * The earnings surprise is the one analyst-based signal that held in both periods.

## Point-in-time handling

| Data | Use |
| --- | --- |
| Monthly estimate snapshots | next trading day after the snapshot, carried ≤ 70 days |
| Quarterly EPS, surprise | trading day after the report timestamp (100% of rows have one) |
| Quarterly financials | trading day after the original announcement time (100% present); discrete quarters, TTM over 4 |
| Shares outstanding | shifted one more month |
| ESG | fiscal year end + 180 days |
| StarMine CA | as published; flagged for possible backfill before launch |

## Stage 1 - signals (sign-adjusted so the prior is positive; IC vs forward alpha, NW t; FM controls C2, size, beta, momentum)

| Signal | Prior | Discovery 2013-21: IC t h5 / h20 | FM t h5 / h20 | Holdout 2022-26: IC t h5 / h20 | FM t h5 / h20 | Pass |
| --- | --- | --- | --- | --- | --- | --- |
| ptup: price-target upside | + | −1.3 / −1.3 | 1.3 / 0.3 | **−2.5 / −2.4** | −1.8 / −2.2 | no (wrong sign) |
| ptrev3: price-target revision | + | 3.3 / 2.0 | 0.1 / −0.1 | 1.5 / 0.8 | −0.6 / −1.5 | no |
| sue: earnings vs a year ago | + | 2.9 / 1.7 | **2.2** / 1.4 | 1.8 / 0.9 | 0.4 / 0.1 | no |
| **surp: I/B/E/S surprise** | + | **3.7** / 2.4 | **3.1** / 2.0 | **3.9 / 3.0** | **3.9 / 3.1** | **yes (h5)** |
| smartdiv: SmartEstimate − consensus | + | 3.9 / 3.3 | 1.0 / 1.5 | 1.4 / 1.8 | 1.1 / 1.3 | no |
| disp: forecast dispersion | − | **5.7 / 3.9** | **3.2 / 2.5** | −1.0 / −0.6 | −2.4 / −1.3 | no (flipped) |
| fdy: forward dividend yield | + | 1.4 / 0.6 | 1.5 / 0.6 | **4.6 / 4.2** | 4.4 / 4.0 | no (2022-26 only) |
| dpsrev3: DPS revision | + | 3.8 / 3.3 | 1.7 / **2.5** | 1.3 / 1.8 | −0.2 / 0.5 | no |
| gpa: gross profit / assets | + | 0.6 / 0.6 | −0.2 / −0.1 | −0.2 / 0.1 | 0.2 / 0.3 | no |
| accr: accruals | − | 0.5 / 0.4 | 1.2 / 1.3 | 1.3 / 1.8 | 2.3 / 2.2 | no |
| iss: share issuance | − | 2.4 / 1.5 | **3.0** / 2.3 | 1.5 / 1.4 | 0.9 / 1.2 | no |
| esg | two-sided | −0.3 / −0.4 | 1.0 / −0.4 | 3.5 / 3.2 | 2.5 / 2.6 | no (2022-26 only) |
| **ca: StarMine Combined Alpha** | + | **5.4 / 4.0** | **3.4 / 2.9** | **3.3 / 3.5** | **2.9 / 3.9** | **yes (flagged)** |
| LC_all: blend of the 11 above (excl. esg, ca) | | 6.5 / 3.9 | 3.6 / 2.5 | 2.6 / 2.7 | 1.1 / 1.7 | |
| LC_sel: disp, dpsrev3, iss, sue, surp | | 6.4 / 4.3 | 3.6 / 3.1 | 2.3 / 2.0 | 0.0 / 0.6 | |

## Stage 3 - strategies (L, top-10, buffer 30, weekly, EW, net of 0.168% per side)

| Strategy | 2013-21 CAGR / Sharpe / α t | 2022-26 CAGR / Sharpe / max DD | 2013-26 CAGR / Sharpe |
| --- | --- | --- | --- |
| A: S5 (C2) | 16.0% / 0.91 / 3.06 | 7.7% / 0.54 / −27.6% | 13.0% / 0.79 |
| E: ML + V8 + LC_sel | 16.9% / 0.98 / 3.43 | 3.2% / 0.30 / −28.4% | 11.9% / 0.77 |
| **F: ML + V8 + LC_all (pick)** | 17.3% / 1.00 / **3.63** | 4.5% / 0.39 / −27.8% | 12.7% / 0.81 |
| G: ML + V8 + FC + LC_all | 11.6% / 0.72 / 2.48 | 7.7% / 0.61 / −20.9% | 10.2% / 0.69 |
| TDEX | 3.9% / 0.31 | 6.7% / 0.54 | 4.8% / 0.38 |

Gates: S1 information pass (LC_all FM t 3.6), S2 pass, S3 **fail** (Sharpe 0.39 < 0.54, CAGR 4.5% < TDEX 6.7%).

## What it means

* **No S5 change, no S7 from the blend.** A blend that is 80% analyst-revision signals inherits their post-2021 fade.
* **Earnings surprise is the best new candidate in the whole project.**
  * It passed pre-registered gates in both periods.
  * Its information is marginal to S5 (FM t 3.1 and 3.9).
  * It has a well-known economic story (post-earnings announcement drift).
  * It is clean point-in-time: it uses the report timestamp.
* **StarMine Combined Alpha confirms that analyst-plus-value composites predict Thai returns live.**
  * As a licensed vendor score it needs an LSEG licence to use.
* **Forward-test candidates** (registered separately, forward data only):
  * `surp` added to C2;
  * a value tilt (forward dividend yield; fundamentals_v1's bp/ep).
  * The dividend-yield evidence is 2022-26 only.

## Deviations

* **D1.** The first fetch crashed on mixed-type columns (ESG returned '' for missing values). Columns are now cast
  before saving, and nothing in the method changed.
* **D2.** `LC_sel` was formed from signals passing G1+G2 at either horizon, as the protocol states. dpsrev3 qualified
  at h20 only.

## Not usable

* LSEG's investor-type ownership ("Holdings by Foreign Investors") covers only filed holders. KBANK shows 0-1%
  foreign holding, against about 48% in reality, so it was not used.
* Per-stock foreign ownership needs SET or NVDR data.

## Trials

31 (26 signal tests + 2 composites + 3 strategies). `validity.TRIALS["lseg_signals_v1"] = 31`, so `N_TRIALS` = 788.

## Reproduce

    cd ~/stock-markets-analytics-zoomcamp-pathorn
    uv run --group lseg python set-quant-alpha/scripts/fetch_lseg.py extra      # LSEG Workspace + key; ~30 min
    for s in features discover holdout; do
      uv run python set-quant-alpha/experiments/lseg_signals_v1/lseg_signals_v1.py $s; done   # holdout runs once
