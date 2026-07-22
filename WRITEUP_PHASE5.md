# Phase 5 — Costs + Exits + Volatility Filter

## Honest headline

**Pre-registered verdict: `robust_improvement = FALSE` — a null result.**
And also: the **closest this program has ever come**. For the first time
across five phases, the tuned configuration's stitched out-of-sample
profit factor crosses breakeven **net of realistic trading costs**
(commission + slippage) — PF **1.0707** on **228** OOS trades, **+$9,795**
total net P&L, **+$42.96** expectancy per trade. The untuned base
(`fixed_1_5R`, no volatility filter, same costs) stays net-unprofitable
over the same 401 trades: PF **0.9006**, **−$25,587.50**, expectancy
**−$63.81** per trade.

It still fails the pre-registered success rule — but it fails on exactly
**one** of five conditions, and that one condition is the statistical-
confidence gate the design was built to enforce. This is reported as a
null, not softened, because that is what the frozen rule says. It is also
reported as the program's best result, because that is also true, and
burying it would be its own kind of dishonesty.

### The five pre-registered conditions

Frozen in `docs/specs/2026-07-13-phase5-exits-costs-volfilter-design.md`
*before* the run. A positive verdict requires **all five**:

| # | Condition | Required | Actual | Passes? |
|---|---|---|---|---|
| a | Tuned net stitched-OOS PF > 1.0 | > 1.0 | **1.0707** | **PASS** |
| b | Tuned − base net stitched-OOS PF margin ≥ 0.10 | ≥ 0.10 | **+0.1701** (1.0707 − 0.9006) | **PASS** |
| c | Tuned beats base net PF in ≥ 3 of 4 folds | ≥ 3/4 | **3/4** (loses only Fold 1) | **PASS** |
| d | Tuned net stitched-OOS PF ≥ 75th percentile of the 20 combos' stitched net-OOS PF | ≥ p75 | **1.0707 ≥ 1.0466** (median 0.9862) | **PASS** |
| e | OOS-sample gate: n ≥ 60 **AND** bootstrap CI lower bound (5th pct) > 1.0 | n≥60 & CI-lo>1.0 | n=**228** (pass) but CI-lo = **0.807** (fail) | **FAIL** |

`robust_improvement = false` because condition (e) — the only gate — fails.
228 trades comfortably clears the `MIN_OOS_TRADES = 60` floor, but a fixed-
seed bootstrap (1,000 resamples, seed 42) over the stitched net-OOS PF puts
the 5th-percentile lower bound at **0.807**, well under the 1.0 threshold
the rule demands (95th-percentile upper bound: 1.394). In plain terms: the
point estimate says "profitable," but the resampling distribution says
"we cannot rule out that this is a below-1.0 strategy that got lucky in
this particular 228-trade sample." Conditions (a)–(d) all pass — this is
not a marginal, everything-barely-fails null like earlier phases; it is a
null that clears four of five bars and is stopped by the one bar designed
specifically to catch exactly this failure mode (see "The Fold-4
concentration problem" below for *why* the CI is so wide).

## The result that almost got there

| Stitched OOS (2024–25, 4 rolling folds, never seen during selection) | Tuned | Base |
|---|---|---|
| Net profit factor | **1.0707** | 0.9006 |
| Net win rate | 50.9% | 39.4% |
| Net total P&L | **+$9,795** | **−$25,587.50** |
| Net max drawdown | $21,030 | $47,092.50 |
| Net expectancy / trade | **+$42.96** | **−$63.81** |
| Gross total P&L (pre-cost) | +$13,405 | −$20,362.50 |
| Trade count | 228 | 401 |

Two things co-reported here on purpose, per the pre-registered plan:
**profit factor alone is a confounded metric across heterogeneous exit
shapes** — `trail_swing` has unbounded upside while `fixed_1_5R` caps at
1.5R, so a high PF built on a few large trail winners is not the same
statistical animal as a high PF built on many small, capped wins. That is
exactly why expectancy-per-trade is reported alongside PF on every table
in this document: **+$42.96/trade net, on 228 trades, is a genuinely
different (and much more legible) fact than "PF 1.07."**

## Per-fold detail

Objective: maximize **net** in-sample PF among eligible combos (eligibility
= fold's pre-filter in-session signal count ≥ `MIN_IS_TRADES=50`, identical
across all 20 combos so the filter under test can't disqualify itself);
tie-break higher net total P&L, then lower net max drawdown. **0 of 4
folds fell back** to the default combo.

| Fold | Test window | Selected exit / vol filter | IS net PF (n) | OOS tuned PF (n) | OOS base PF (n) | Tuned wins? | Pick's OOS percentile |
|---|---|---|---|---|---|---|---|
| F1 | 2024-01 .. 2024-07 | `partial_1R` / `p50` | 0.919 (113) | **0.723** (30) | 0.880 (102) | **no** | 35th |
| F2 | 2024-07 .. 2025-01 | `partial_1R` / `p50` | 0.967 (113) | **0.751** (80) | 0.626 (110) | yes | 75th |
| F3 | 2025-01 .. 2025-07 | `trail_swing` / `off` | 0.882 (213) | **1.168** (93) | 1.106 (93) | yes | 40th |
| F4 | 2025-07 .. 2025-12 | `trail_swing` / `p50` | 1.022 (113) | **2.248** (25) | 1.069 (96) | yes | 85th |

Fold 1 is the one loss: `partial_1R`/`p50` is selected there too (the
same combo as F2), but the OOS window is unkind to it — PF 0.723 on 30
trades, worse than the 102-trade base that fold. The in-sample PF that
picked it (0.919) was itself under 1.0, so this fold's selection was
"least-bad of a weak in-sample field," not a genuine edge — and it shows.
Folds 2–4 all beat base, with Fold 4 doing so by a wide and, as the next
section explains, structurally important margin.

## The Fold-4 concentration problem — why condition (e) is right to say no

This is the central honesty finding of Phase 5, and it is the reason the
pre-registered CI gate exists at all.

**Fold 4 alone contributes +$18,845 in net P&L on just 25 trades** — more
than the entire stitched net result (+$9,795). **The other three folds
combined net −$9,050** (F1 −$3,860 + F2 −$14,085 + F3 +$8,895). Strip out
Fold 4 and the "net-profitable strategy" headline disappears; the entire
program-first result rests on one 25-trade, 6-month window
(`trail_swing`/`p50`, OOS PF 2.248, expectancy **+$753.80/trade** — nearly
20x the stitched expectancy).

This is not a data error or a bug — `trail_swing`'s unbounded upside is
exactly what it is designed to do when a trend cooperates, and Fold 4
(2025-07 to 2025-12) evidently had trending conditions this combination
caught well. But a result that lives almost entirely in one fold, out of
four, is precisely the situation a naive point-estimate ("PF > 1, ship it")
would over-claim on. **This is exactly what the bootstrap CI (condition e)
is built to catch**: resampling the 228 stitched trades with replacement
routinely draws samples that under-represent Fold 4's concentrated run,
and the 5th-percentile outcome across 1,000 such resamples is a PF of
0.807 — below breakeven. The pre-registered rule doing its job here is the
point: a fragile, one-fold, statistically-uncertified result was
prevented from being reported as "Phase 5 solved profitability."

## What helped, and what didn't

**The winning levers were exit management, not entry selectivity.**
Entry parameters were frozen at the same Pine defaults across all 20
grid combos and all 4 folds — Phase 5 asks *only* "given the Phase-4-null
default entry, do smarter exits + a volatility filter help?", not "should
entry be re-tuned?" (Phase 4 already answered that one; see disclosure ii
below.)

- **`trail_swing`** (let winners run — trailing stop only, no fixed
  target once +1R activates) was selected in 2 of 4 folds (F3, F4) and
  produced the two largest OOS wins.
- **`partial_1R`** (bank 0.5 unit at +1R, run the remainder to breakeven-
  stop / 3R) was selected in 2 of 4 folds (F1, F2) — a lower-variance
  compromise that still under-performed base in F1.
- **`fixed_1_5R`** (the Phase-2 baseline, capped 1.5R target) and
  **`breakeven_1R`** and **`time_stop`** were never selected in any fold —
  though 3 of these 3 modes' `p50`/`p75` variants *would* have stitched
  competitively (see the selection-luck null below), so "never selected"
  should not be read as "never competitive."
- **The `p50` volatility filter** (ATR% ≥ the train-window's 50th
  percentile of in-session signal-bar ATR%) was selected in 3 of 4 folds
  (F1, F2, F4); `off` (no filter) was selected once (F3, where the wider
  `trail_swing` exit apparently didn't need the extra selectivity).

**Selection-luck null (condition d's distribution):** all 20 combos' own
stitched net-OOS PF were computed independently (each fold picks its own
best in-sample combo, but *every* combo's OOS trades are also tracked
across all 4 folds, giving each of the 20 a full stitched-OOS PF). The
tuned/selected pick (1.0707) sits **above the p75 of that distribution
(1.0466)** — condition (d) passes — but only **5 of the 20 fixed combos**
(`time_stop`/p75 1.100, `trail_swing`/p50 1.097, `time_stop`/p50 1.091,
`trail_swing`/p75 1.087, `fixed_1_5R`/p75 1.073) would have stitched
*higher* than the adaptive, per-fold-selected pick. In other words: a
handful of the *simplest* fixed combos — including the plain `fixed_1_5R`
baseline exit paired with the `p75` filter — would have done as well or
better than adaptively re-selecting exit/filter every 6 months. The
per-fold adaptive selection process clears the null only marginally, the
same caution Phase 4's selection-luck analysis raised about its own
144-combo grid.

## Eligibility — how thin the filtered arms got

Selection floor (`MIN_IS_TRADES=50`) is evaluated on each fold's
**pre-filter** in-session signal count (identical across all 20 combos —
a property of the entry base, not the exit/vol arm under test), so `p50`/
`p75` arms are never auto-disqualified by the very filter being tested.
But realized trade counts still shrink hard as the filter tightens
(averaged across all 5 exit modes × 4 folds, 20 combos per row):

| Vol filter | Avg IS realized trades | Avg OOS realized trades |
|---|---|---|
| `off` | 206.3 | 100.4 |
| `p25` | 166.8 | 76.5 |
| `p50` | 115.1 | 52.2 |
| `p75` | 58.2 | 35.8 |

`p75` cuts realized OOS trades to roughly a third of `off` — e.g. Fold 1's
`fixed_1_5R`/`p75` cell realizes only 9 OOS trades from 48 in-sample. The
selected combos in this study land on `p50` (3 of 4 folds) or `off` (1 of
4), never on the thinnest `p75` tier, which is at least consistent with
the selection process not chasing an over-filtered, too-thin sample —
but the table above is exactly the kind of "how thin did it get"
transparency the pre-registered plan required regardless of which tier
won.

## Cost sensitivity — does the verdict survive?

Stitched net PF / total P&L at commission+slippage multipliers 0× (no
costs), 1× (pre-registered base case), 2× (pessimistic/realistic — see
disclosure iii):

| Multiplier | Tuned PF | Tuned total P&L | Base PF | Base total P&L |
|---|---|---|---|---|
| 0× (gross) | 1.0979 | +$13,405 | 0.9198 | −$20,362.50 |
| **1× (base case)** | **1.0707** | **+$9,795** | **0.9006** | **−$25,587.50** |
| 2× (pessimistic) | 1.0441 | +$6,185 | 0.8820 | −$30,812.50 |

The **qualitative** verdict is stable: tuned stays PF > 1 and base stays
PF < 1 at every multiplier tested, and both `tuned_pf_gt_1` and
`tuned_beats_base_margin_ge_0_10` hold at 0×, 1×, and 2×. But the margin
compresses steadily as costs rise (tuned P&L: +$13,405 → +$9,795 →
+$6,185), which is the expected direction — real trading costs eat real
edge — and reinforces that this is a **thin**, not commanding, margin over
breakeven even before the statistical-confidence question is asked.

## Mandatory disclosures

1. **Cumulative multiple testing.** Phase 5 is the **third** pre-registered
   pass over the same ~3-year back-adjusted NQ series — after Phase 2's
   build/validate pass and Phase 4's 144-combo walk-forward sweep. No
   formal multiple-comparisons correction is applied across phases; each
   phase's own within-phase null control (Phase 4's selection-luck null,
   Phase 5's 20-combo null + bootstrap CI) only guards *that phase's*
   internal search, not the fact that this is the third bite at the same
   apple. Read the cross-phase pattern (three passes, three nulls, one of
   which reaches a positive point estimate) as weak evidence at best.
2. **Conditional on the Phase-4-null default entry.** Phase 4 tuned entry
   parameters (`fvg_threshold`, `rr`, `ema_length`, `swing_lookback`) via
   walk-forward search and returned a null (`robust_improvement=false`,
   tuned OOS PF topped out at 0.9945, still under 1.0). Phase 5 does
   **not** re-tune entry — it fixes entry at the plain Pine defaults and
   asks only whether exits + costs + a volatility filter can rescue a
   config that Phase 4 already showed sits near breakeven. This result is
   therefore a statement about exit/filter management layered on a known-
   marginal entry, not a fresh, independent test of the whole strategy.
3. **Arbitrary pre-registered choices.** The `partial_1R` remainder target
   (3R), the cost constants ($5 commission round-trip, 1 tick entry
   slippage, 1 tick exit slippage on market fills, $5/tick), and
   `TIME_STOP_ET = "11:00"` were fixed by design judgment before the run,
   not fit to the data. In particular, **1-tick stop slippage is
   optimistic** for NQ in a fast-moving market — the 2× cost band above is
   the more realistic pessimistic case, and the verdict's qualitative
   direction (tuned > 1, base < 1) does survive it, but the margin is
   visibly thinner there than at 1×.
4. **n=4 folds is descriptive, not inferential**, and the fold windows
   overlap (each fold's 12-month train window shares 6 months with its
   neighbors), which mechanically inflates any apparent cross-fold
   agreement — the same caveat Phase 4 raised about its own parameter-
   stability table applies here to the exit/vol-filter selection pattern
   (`p50` in 3 of 4 folds).
5. **The Fold-4 concentration** (+$18,845 on 25 trades vs. −$9,050 net
   across the other three folds combined) is the single most important
   caveat in this document — covered in its own section above, repeated
   here because it is the direct, mechanical reason condition (e)'s
   bootstrap CI lower bound falls under 1.0.
6. **Provenance note on `git_sha`.** `phase5_results.json` records
   `config_hash: 71ced75e…` and `git_sha: 37fa2dd…`. The recorded SHA is
   the **design-freeze parent commit** (the commit that landed the frozen
   grid, folds, cost model, and success rule), not the commit
   that produced the runner/results file itself — this project's
   single-shot workflow runs the sweep against the frozen design and
   commits the results *afterward*. The config hash is the load-bearing
   audit trail here: it was independently recomputed against the frozen
   design at write time and reproduces byte-identically, which is the
   actual proof this run executed against the pre-registered design
   unmodified — the `git_sha` field is a secondary, informational pointer
   to *which* frozen design, not a claim that the working tree was clean
   at that exact SHA.
7. **`trail_swing` can exit slightly negative even after +1R activates —
   this is spec-correct, not a bug.** `trail_swing`'s stop only trails the
   rolling swing level once price reaches +1R; it does not lock in
   breakeven or any positive P&L on activation. Real OOS trades confirm
   this: e.g. Fold 3 trades on 2025-03-14 (r ≈ **−0.098R**) and 2025-03-19
   (r ≈ **−0.276R**) both activated the trail past +1R and still closed
   net negative when the swing level gave back more than the initial move
   before the trail caught up. This is the intended behavior of a pure
   trailing exit (no breakeven floor), not a defect in the same-bar
   stop-first sequencing.

## Charts

Generated by `run_phase5.py`, committed under `charts/`:

- `charts/phase5_equity_curve.png` — stitched OOS cumulative P&L: tuned
  (net) vs. base (net) vs. tuned (gross).
- `charts/phase5_oos_pf_per_fold.png` — per-fold net OOS profit factor,
  tuned vs. base.
- `charts/phase5_selected_exit_vol_stability.png` — selected exit mode +
  volatility filter per fold (stability table).
- `charts/phase5_cost_sensitivity.png` — stitched net total P&L at
  0×/1×/2× cost multipliers.
- `charts/phase5_combo_null.png` — all 20 combos' own stitched net-OOS PF
  (the selection-luck null), with the adaptive pick and the median marked.

See `notebooks/05_costs_exits_volfilter.ipynb` for the same result
narrated end-to-end with the fold table, eligibility table, cost
sensitivity, and all five charts inline — loaded from the committed
`phase5_results.json` (the sweep itself is not re-run by the notebook).

---

## Program epilogue (Phases 1–5)

1. **Phase 1 — data foundation.** ~3 years of 1-minute NQ OHLCV, loaded,
   parsed, and coverage-checked before any strategy logic touched it.
2. **Phase 2 — faithful rebuild, edge located in selectivity.** A
   bar-by-bar Python port of the FYP IFVG+CISD Pine strategy, validated
   against two real trade logs with no fitting. Good directional/coverage
   fidelity, but the raw default parameters over-trade ~4x and
   under-perform the real "optimised" logs — a gap that looked like
   selectivity/tuning, not a broken port.
3. **Phase 3 — correctly not run**, its own pre-registered precondition
   (a robust Phase-4 improvement) never having been met.
4. **Phase 4 — tuning entry params, honestly, is a null.** A pre-
   registered walk-forward sweep of four entry parameters nearly erases
   the default's OOS loss (−$907.50 vs. −$20,362.50) but doesn't cross
   breakeven (tuned OOS PF 0.9945) and misses the pre-registered margin —
   directional confirmation of the Phase-2 hypothesis, not a tradeable
   edge.
5. **Phase 5 — exits + costs + a volatility filter: the best result yet,
   and still not a proven edge.** With entry parameters frozen at their
   Phase-4-null defaults, smarter trade management (`trail_swing` letting
   winners run, `partial_1R` banking gains early, a `p50` ATR% filter for
   selectivity) pushes the stitched, cost-adjusted OOS point estimate
   **above breakeven for the first time in this program** (PF 1.0707,
   +$9,795, four of five pre-registered conditions passed). But the
   pre-registered statistical-confidence gate — built specifically to
   catch a result this concentrated — correctly refuses to certify it: the
   bootstrap CI lower bound sits at 0.807, and the entire net edge traces
   to one 25-trade fold.

The through-line across all five phases is not "we found an edge" and it
is not "we didn't" — it is that **every headline number in this program
was defined by a rule fixed before the result was known**, and every
result, favorable or not, was reported against that rule rather than
around it. Phase 5 is the sharpest expression of that discipline: it is
the phase that came closest to a positive result, and the same honest
machinery that let it get that close is exactly what stopped it from
being oversold as more than it is. A promising signal, clearly
delineated from a proven one, is the deliverable — arguably more so than
any single profit-factor number in this document.
