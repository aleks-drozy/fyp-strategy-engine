# Phase 6 — Cross-Instrument Confirmation of the Exit Edge — Design Spec

**Date:** 2026-07-14
**Owner:** Aleksandrs Drozdovs
**Status:** approved (brainstorming), pending spec review
**Program:** `fyp-strategy-engine` — Phase 6 (the proof/disproof capstone)

## Purpose

Phase 5 found the program's only net-positive result: smarter exits (trailing / partial) + a volatility
filter flipped the 2-year OOS point estimate to **+$9,795 (net PF 1.07)** — but it **failed the
pre-registered confidence gate** (bootstrap CI lower 0.81 < 1.0) and the edge was concentrated in one
25-trade fold. The verdict was *promising, not proven*.

Phase 6 performs the only legitimate proof: **confirm the frozen Phase-5 hypothesis on independent data
it has never touched** — new instruments (ES, YM), an independent-vendor NQ, and a much longer history
(more folds, more trades — directly attacking the small-sample reason Phase 5 failed).

Two honest outcomes, both valuable:
- **PROVEN:** the exit edge replicates on independent instruments with statistical confidence.
- **DISPROVEN:** it does not replicate → Phase 5's near-miss was instrument-/period-specific luck.
Either is the program's definitive answer. The design must make both outcomes credible.

## The cardinal rule: confirmation, not exploration

**Nothing is re-searched, re-tuned, or re-selected in response to the new data.** Every degree of
freedom was frozen in Phase 5 (its committed grid, objective, floors, cost logic, fold shape). Phase 6
runs the frozen machinery unchanged on new data and reads out the answer. Any modification prompted by
seeing Phase-6 results = a new experiment, disclosed as such.

## Data (acquired + validated before anything else)

Source: Kaggle `bpwqsdd/us-futures-1-minute-candlesticks` — real, **unadjusted** 1-minute futures bars:
- **ES_2** (S&P 500 e-mini) — truly independent of everything studied so far.
- **YM_2** (Dow e-mini) — truly independent.
- **NQ_2** (Nasdaq e-mini) — same instrument as Phases 1–5 but an **independent vendor + unadjusted
  prices + longer history**; treated as a *replication* check, not an independent confirmation.
Spike-verified on the small NQ_1 file: genuine price levels (21,474 Dec-2024 → 17,419 in the Apr-2025
selloff, matching reality), 0.25-grid OHLC, 1-min bars.

**Format & conversion:** files are Python **pickles of `Candle` objects** (`o,h,l,c,t`; **no volume**).
A one-time converter (restricted unpickler that only admits `Candle` + numpy scalars — never blind
`pickle.load`) → clean **parquet** per instrument. `volume` is synthesized as 0 and **documented unused**
(signals/exits use OHLC only; verified: ifvg/cisd/ema/atr touch only o/h/l/c).

**Validation (adapt the Phase-1 validator, per instrument; hardened after adversarial review):**
- OHLC invariants, ordering, 1-min spacing dominance, date range, per-instrument tick grid
  (ES 0.25, NQ 0.25, **YM integer**).
- **Divergent-duplicate handling (Blocker B1):** the ~16k duplicate timestamps are not assumed identical
  re-prints. The validator **compares dup rows** and reports `n_dups_identical` vs `n_dups_divergent` +
  divergence percentiles. Dedup rule (frozen): within each dup group keep the row minimizing
  `|open − previously_kept_close|` (price-continuity; tie → keep-first). A pre-registered **anomaly-window
  detector** (in-session bad prints: bar range > K× a rolling robust range measure, K frozen) flags
  windows; trades overlapping a flagged window are **excluded identically in every arm** and counted.
- **Contract-roll splices (Blocker B2):** the data is unadjusted spliced front-month; quarterly splices
  inject phantom jumps. The validator **detects roll boundaries**: per quarterly month, the largest
  |session-break gap| inside a frozen day-window (days 5–18), flagged when > K× that year's median
  session-break |gap| (K frozen); hard-fails if the count deviates materially from ~4/yr; asserts the
  detected gap signs are mixed (a splice artifact, not one-directional drift). `roll_boundaries` are
  committed in the report. **Mitigation (frozen): trades whose holding window spans a detected roll
  boundary are EXCLUDED identically in every arm** (never force-flattened — that would add a new exit
  path to frozen machinery), with counts disclosed. Exposure is small by measurement (median hold 17 min;
  0.5% of trades > 24h) but the sign is systematic, so exclusion is the only verdict-safe choice.
  Detection doubles as committed **proof of unadjusted-ness**.
- **Timezone inference + proof (hardened):** timestamps are naive. Evidence must hold **per DST regime
  and per year**: quiet-hour==17 and reopen==18 under both EDT and EST and in every calendar year (a
  fixed-offset/UTC-stamped series would drift an hour seasonally and FAIL). Localize to `US/Eastern`.
  **Hard gate** — ambiguity stops the phase.
- **Per-fold session integrity:** per-half-year session-day counts (accept ∈ [115,135]) and per-session
  09:30–10:30 bar-completeness (a counted session has ≥ 50 of 60 in-session minutes); missing/incomplete
  sessions listed and disclosed; folds with materially degraded coverage flagged in the results.
- **Cross-vendor sanity (hard gate):** on the 2022→2025-01 overlap, per-day correlation of 1-min
  log-returns between indep-NQ and Phase-1 NQ on common bars — fail any day below a frozen floor (~0.9);
  assert lag-0 correlation exceeds ±1-min lags (bar-stamp-convention proof); the back-adjustment offset
  (≈ +2,655 → ≈ 0) reported. Proves both vendors describe the same market, timestamped the same way.
- Commit `validation_report_p6.json` — all of the above evidence, per instrument.

## Instrument spec (new, engine-level)

`InstrumentSpec(tick_size, tick_value, pt_value)` threaded through the engine + cost model (Phase 5
hardcoded NQ's $20/pt):
- **ES:** tick 0.25 = $12.50 → $50/pt. **NQ:** tick 0.25 = $5 → $20/pt. **YM:** tick 1.0 = $5 → $5/pt.
- Costs (pre-registered, same structure as Phase 5): `COMMISSION_RT=$5`; slippage 1 tick entry + 1 tick
  on every market exit (per Phase 5's reason-aware table), valued at each instrument's tick_value;
  sensitivity 0×/1×/2× reported.
- **Regression lock:** with the NQ spec + Phase-1 data, the engine reproduces Phase-5 results exactly.

## Pre-registered hypotheses (fixed BEFORE seeing any Phase-6 backtest; revised after adversarial review)

**H-A (SOLE PRIMARY — the only hypothesis that can yield the program-level PROVEN):** the **frozen
Phase-5 walk-forward procedure** — identical 20-combo grid, net-PF selection, MIN_IS_TRADES=50
(pre-filter basis), rolling 12mo-train/6mo-test folds tiled over each instrument's full history — beats
the base config (`fixed_1_5R`, `off`) net-of-costs, out-of-sample, on the independent instruments.
**Composition condition (closes the semantic false-PROVEN):** on each independent instrument, a
**majority of per-fold selections must fall in the smart-exit family** ({trail_swing, partial_1R,
breakeven_1R} × any vol filter, or any exit × {p25,p50,p75}) — i.e. H-A cannot be declared PROVEN while
mostly selecting base-like configs; the full per-fold pick table is published.

**H-B (secondary, config-level REPLICATION — cannot yield the program-level PROVEN):** the two fixed
configs, no selection anywhere: **B1 = `partial_1R`+`p50`** (Phase 5's modal pick, 2/4 folds) and
**B2 = `trail_swing`+`p50`** (the pick of the profitable fold). Verdicts are labeled *"config-level
replication: supported / not supported"*, gated at a **Bonferroni-adjusted 2.5th-percentile** CI lower
bound (two configs share the secondary family). **Disclosure:** Phase 5's third distinct pick,
`trail_swing`+`off` (F3), is excluded because it lacks the vol-filter component that is part of the
hypothesis being confirmed; this exclusion is declared here, in advance, not after seeing results.
`p50` thresholds are computed per fold from that instrument's train window exactly as Phase 5.

**Honesty note on "independence":** ES and YM are ~0.9-correlated US equity-index futures. They are
independent of *Phase-5's data*, not of each other; cross-instrument agreement is one confirmation on
correlated markets, never counted as two independent confirmations. The statistics below (day-cluster
bootstrap) are constructed to respect this.

## Pre-registered success rule (ordered, exhaustive, mutually exclusive — revised after adversarial review)

**Gating statistic:** all gates use **net R-multiples**, `net_R = net_pnl / (risk × pt_value)` — unit-risk
normalized so ES's $50/pt cannot outvote YM's $5/pt; dollar PF is reported as descriptive only.
**CI method:** **calendar-day cluster bootstrap** — resample trade-DAYS with replacement; a sampled day
carries ALL its pooled ES+YM trades together (kills the same-event-booked-twice duplication); instrument
composition stratified (n_ES from ES-days, n_YM from YM-days); `n_boot = 10,000`, fixed seed, basic
(pivotal) intervals; constants frozen. **Floors:** pooled **n ≥ 150 trades AND ≥ 100 distinct trade-days**;
per-instrument **n ≥ 150 tuned OOS trades** required for conditions 1–2 to be evaluable (else the verdict
caps at INCONCLUSIVE).

**Conditions (evaluated for H-A on ES and YM):**
1. Tuned net stitched-OOS **R-based PF > 1.0** on **both** ES and YM;
2. Tuned beats base net R-PF by **≥ 0.10** on both (margin also reported as net expectancy per R);
3. **Pooled ES+YM day-cluster CI lower bound (5th pct) > 1.0** on the R-based PF, floors met;
4. indep-NQ directional agreement (tuned ≥ base net R-PF), with the pre/post-2023 split published
   (segment definition pinned: pre = entry-date < 2023-01-01).
**Robustness guards (part of condition 3):** tuned beats base per-fold (net) in a **majority of eligible
folds on EACH independent instrument**, AND the pooled R-PF stays > 1.0 with per-instrument margins
positive after **removing the single most profitable fold** (leave-one-out).

**Verdict decision table (applied in order; first match wins; encoded verbatim in `run_phase6.py`):**
1. **PROVEN** iff conditions 1∧2∧3(+guards)∧4 all pass.
2. else **DISPROVEN** iff (a) tuned **strictly <** base net R-PF on both ES and YM **AND** the pooled
   tuned-minus-base margin's day-cluster CI **upper** bound < 0.10; **OR** (b) the pooled tuned R-PF
   day-cluster CI **upper** bound (95th pct, same method/seed) **< 1.0**. (Symmetric with PROVEN: it
   takes statistical confidence to disprove, exactly as to prove — ties and low power cannot force it.)
3. else **PARTIAL** iff (a) conditions 1–2 pass on exactly one independent instrument; or (b) conditions
   1–3 pass but condition 4 fails ("proven on independents, unreplicated on NQ" — stated in those words).
4. else **INCONCLUSIVE** (the explicit catch-all: between the PROVEN and DISPROVEN CI bars, floor
   failures, composition-condition failures, and every other outcome). No undefined branch exists.

**Freeze mechanics:** a `docs/phase6_freeze.json` containing the config hash, the serialized decision
table, every rule constant (n_boot, seed, percentiles, cluster unit, floors, margins, roll/anomaly
detector constants, tiling rule, hypothesis hierarchy + composition condition) is **committed to git
BEFORE `run_phase6.py` exists** — the freeze is git-timestamped, not self-attested. The runner verifies
its own config hashes to the frozen value and refuses to run otherwise (mechanical single-shot).
All combos' OOS R-PF per instrument recorded (selection-luck context). **Sensitivity readouts (context,
never gates):** (i) recompute excluding ES/YM test windows overlapping 2023-01→2025-12 (the
hypothesis-formation era — does the headline survive?); (ii) pre/post-2020 regime split; (iii) cost
multipliers 0×/1×/2×; (iv) fold-cluster CI as a diagnostic alongside the day-cluster gate; (v) the
design's minimum detectable effect published (~pooled R-PF ≈ 1.10 at these ns).

## Components

```
data/convert_p6.py         # restricted unpickler -> parquet per instrument (one-time; documented)
nqdata/load_p6.py          # load_instrument("ES"|"NQ"|"YM") -> ET-indexed OHLC(V=0) frame
validate_p6.py             # per-instrument validation + tz-inference proof -> validation_report_p6.json
strategy/instrument.py     # InstrumentSpec + the three frozen specs
backtest/{engine,costs}.py # thread InstrumentSpec (pt_value/tick_value); NQ default preserves Phase 1-5
tuning/walkforward_p6.py   # thin wrapper: tile Phase-5 folds over each instrument's history;
                           #   H-A (frozen procedure) + H-B (fixed configs) runners
run_phase6.py              # single-shot -> phase6_results.json + charts
notebooks/06_cross_instrument_proof.ipynb ; WRITEUP_PHASE6.md
tests/ (converter safety, tz inference, instrument specs, engine regression at NQ spec,
        fold tiling, H-B fixed-config no-selection, leak guards reused)
```

## Testing (TDD)

- **Converter:** restricted unpickler rejects non-Candle globals (security); round-trips a synthetic
  pickle; output parquet matches input values exactly.
- **TZ inference:** synthetic frames with known maintenance-gap placement → correct zone; ambiguous →
  raises (hard gate).
- **InstrumentSpec:** cost math per instrument (a YM stop-exit loser pays $5+$5+$5; an ES one
  $5+$12.50+$12.50); engine regression at the NQ spec reproduces the Phase-5 golden path.
- **Fold tiling:** folds tile each instrument's actual history (no fold past the data edge; half-open;
  disjoint — reuse Phase-4/5 guards); works for histories of differing lengths.
- **H-B runner:** fixed config, provably NO selection anywhere (a test asserts the selector is never
  invoked); vol p50 threshold train-only per fold (reuse the Phase-5 leak test pattern).
- **Smoke:** short real slice per instrument end-to-end.

## Non-Goals

- **No** new exit modes, grids, filters, or parameters. **No** re-tuning on Phase-6 data. **No** blending
  instruments into one pot for selection. **No** attempt to fix Phase-5's NQ result — this phase asks
  only "does it replicate?".

## Risks

- **Data quality unknowns** (naive timestamps, vendor gaps, no volume) → hard validation gates before
  any backtest; the tz-inference proof is committed evidence, not an assumption.
- **Memory** (~400 MB pickles → millions of Python objects) → convert once, streaming-write parquet,
  then only parquet is touched.
- **History may be shorter than hoped** → fold count reported; if an instrument yields < 4 OOS folds it
  cannot support a PROVEN verdict on its own (stated in advance).
- **The likely outcome is DISPROVEN or PARTIAL** → that is the honest, publishable answer; the writeup
  leads with it either way. The prior (from 5 phases) leans negative; the design cannot be accused of
  favoring it — the bar is symmetric and pre-registered.
