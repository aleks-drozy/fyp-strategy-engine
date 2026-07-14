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

**Validation (adapt the Phase-1 validator, per instrument):**
- OHLC invariants, duplicate/ordering, 1-min spacing dominance, date range, per-instrument tick grid
  (ES 0.25, NQ 0.25, YM 1.0).
- **Timezone inference + proof:** timestamps are naive. Infer the zone by locating the CME maintenance
  hour (17:00–18:00 ET daily gap) and weekend gaps; assert the inferred offset is stable across DST
  changes (i.e. the gap tracks ET wall-clock, else convert accordingly). Localize to `US/Eastern`
  exactly like Phase 1. This inference is a **hard gate**: if the session structure is ambiguous, stop.
- **Cross-vendor sanity:** on the overlap window, indep-NQ (unadjusted) vs the Phase-1 NQ (back-adjusted)
  must show the expected time-varying offset (≈ +2,655 → ≈ 0) and near-identical intraday *shapes*
  (returns correlation ≈ 1 on common bars). Proves both vendors describe the same market.
- Commit a `validation_report_p6.json` (like Phase 1) — rows, range, violations, inferred tz evidence.

## Instrument spec (new, engine-level)

`InstrumentSpec(tick_size, tick_value, pt_value)` threaded through the engine + cost model (Phase 5
hardcoded NQ's $20/pt):
- **ES:** tick 0.25 = $12.50 → $50/pt. **NQ:** tick 0.25 = $5 → $20/pt. **YM:** tick 1.0 = $5 → $5/pt.
- Costs (pre-registered, same structure as Phase 5): `COMMISSION_RT=$5`; slippage 1 tick entry + 1 tick
  on every market exit (per Phase 5's reason-aware table), valued at each instrument's tick_value;
  sensitivity 0×/1×/2× reported.
- **Regression lock:** with the NQ spec + Phase-1 data, the engine reproduces Phase-5 results exactly.

## Pre-registered hypotheses (fixed BEFORE seeing any Phase-6 backtest)

**H-A (primary, procedure-level):** the **frozen Phase-5 walk-forward procedure** — identical 20-combo
grid, net-PF selection, MIN_IS_TRADES=50 (pre-filter basis), rolling 12mo-train/6mo-test folds tiled
over each instrument's full history — beats the base config (`fixed_1_5R`, `off`) net-of-costs,
out-of-sample, on the independent instruments.

**H-B (secondary, config-level):** the two exit configs Phase 5's selection actually favored, tested as
**fixed configs with no selection at all**:
- **B1 = `partial_1R` + `p50`** (selected in 2 of 4 Phase-5 folds — the modal pick), and
- **B2 = `trail_swing` + `p50`** (the pick in the fold that produced the profit).
Declaring both (rather than cherry-picking the profitable one) is deliberate; the multiple-comparison
cost (2 configs × 3 instruments) is disclosed and the success rule demands cross-instrument consistency,
which a single lucky cell cannot satisfy. For H-B's vol threshold, `p50` is computed per fold from that
instrument's train window exactly as in Phase 5 (scale-free ATR%, train-only — no new freedom).

## Pre-registered success rule (the proof bar)

Evaluated per instrument on the stitched net OOS record (full history, all folds). Let ES and YM be the
**independent set**; indep-NQ is a **replication** readout reported alongside.

**PROVEN** requires, for H-A (or for a single named H-B config across the board — no mixing):
1. Net stitched-OOS PF **> 1.0** on **both** ES and YM;
2. Beats base net PF by **≥ 0.10** on both;
3. **Pooled ES+YM** bootstrap CI lower bound (5th pct, fixed seed) **> 1.0** with pooled **n ≥ 150**;
4. Directional agreement on indep-NQ (tuned ≥ base net PF).

**PARTIALLY SUPPORTED:** conditions 1–2 on exactly one independent instrument, or pooled CI lower ≤ 1.0
with point estimates positive. **DISPROVEN:** tuned ≤ base on both independent instruments (net), or
pooled point estimate ≤ 1.0. Every sub-condition's value is reported regardless.

Config-freeze: grids, folds, floors, costs, instrument specs, hypotheses, and this rule are hashed
(SHA-256) + git-SHA'd into `phase6_results.json` before the single-shot run. All combos' OOS PF per
instrument recorded (selection-luck context, as in Phase 5).

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
