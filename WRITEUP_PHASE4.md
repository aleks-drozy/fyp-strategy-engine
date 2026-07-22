# Phase 4 — Walk-Forward Parameter Tuning

## Honest headline

**Pre-registered verdict: `robust_improvement = FALSE` — a null result.**
Walk-forward tuning of `fvg_threshold`, `rr`, `ema_length`, and
`swing_lookback` — selected in-sample, applied unchanged out-of-sample,
across 4 rolling 12-month-train / 6-month-test folds spanning 2024-01
through 2025-12 — does **not** clear the falsifiable, pre-registered bar
for "tuning robustly beats the Pine defaults." This is reported as-is, not
softened. A rigorously measured null is a legitimate, portfolio-grade
result: it means the question was asked honestly and the answer, on this
data, is no.

The **pre-registered success rule** (fixed in `docs/specs/2026-07-13-phase4-parameter-tuning-design.md`
*before* the run) required **all four** of the following. Two failed:

| # | Condition | Required | Actual | Passes? |
|---|---|---|---|---|
| a | Tuned stitched-OOS PF > 1.0 (absolute profitability floor) | > 1.0 | **0.9945** | **FAIL** |
| b | Tuned OOS PF − default OOS PF ≥ 0.10 | ≥ 0.10 | **+0.0747** | **FAIL** |
| c | Tuned beats default OOS PF in ≥ 3 of 4 folds | ≥ 3/4 | **3/4** | PASS |
| d | Tuned stitched-OOS PF > median-combo selection-luck null | > null | **0.9945 > 0.9396** | PASS |

`robust_improvement = false` because (a) and (b) fail: the tuned strategy's
out-of-sample profit factor is still marginally under 1.0 (breakeven), and
the margin over the untuned default (+0.07) falls well short of the
pre-registered +0.10 threshold that was set to rule out noise-level
"improvement." Conditions (c) and (d) do pass — tuning is not *nothing* —
which is exactly why this is reported as a **null**, not a **failure**: the
directional evidence for the Phase-2 selectivity hypothesis (see below) is
real, it just doesn't clear the bar for a *robust, tradeable* edge.

## The directional finding — selectivity nearly erases the default's loss

Even though tuning doesn't cross into profitability, it moves the needle
substantially in the direction Phase 2 predicted. Phase 2's honest
conclusion was that the real "optimised" track record's edge over the raw
default-parameter signal looked like **selectivity** — fewer, better
trades — rather than a different core mechanic. Phase 4's OOS result
corroborates that directly:

| Stitched OOS (2024-01 .. 2025-12, 2 years never seen during tuning) | Tuned | Default |
|---|---|---|
| Profit factor | **0.9945** | 0.9198 |
| Win rate | 40.3% | 39.4% |
| Total P&L | **−$907.50** | **−$20,362.50** |
| Max drawdown | $34,370 | $43,552.50 |
| Trade count | **233** | 401 |

Walk-forward tuning nearly **erases** the default's two-year loss
(−$907.50 vs. −$20,362.50 — a >20x reduction in dollar loss) while trading
**42% fewer** times (233 vs. 401) and lifting profit factor from 0.92 to
0.99. It gets there almost entirely by being **more selective**, not by
finding a fundamentally different edge — exactly the Phase-2 hypothesis.
The honest reading: **selectivity helps, directionally and substantially,
but even parameters chosen to be optimal in-sample do not reach
profitability out-of-sample on this data.** The real track record's full
edge is not reproduced by tuning these four levers alone.

## Per-fold detail

Objective: maximize in-sample profit factor subject to ≥50 in-sample
trades (tie-break: higher trade count, then lower max drawdown); default
params used as a fallback if no combo meets the trade floor (**0 folds
fell back** — every fold found a qualifying combo on its own terms).

| Fold | Test window | Selected (fvg / rr / ema / swing) | IS PF (n) | OOS PF tuned | OOS PF default | Tuned wins? |
|---|---|---|---|---|---|---|
| F1 | 2024-01 .. 2024-06 | 0.05 / 1.5 / 50 / 5 | 1.011 (n=69) | 0.921 | 0.907 | yes |
| F2 | 2024-07 .. 2024-12 | 0.05 / 1.5 / 50 / 5 | 1.019 (n=57) | 1.087 | 0.639 | yes |
| F3 | 2025-01 .. 2025-06 | 0.02 / 3.0 / 20 / 5 | 1.194 (n=161) | 0.886 | 1.126 | **no** |
| F4 | 2025-07 .. 2025-12 | 0.02 / 1.0 / 10 / 5 | 1.209 (n=179) | 1.153 | 1.090 | yes |

Fold 3 is the one loss: the strongest in-sample profit factor of any fold
(1.194) collapsed to the weakest out-of-sample result (0.886, *worse* than
the default's 1.126 on the same window) — the single largest
in-sample-to-OOS reversal in the study, and it happens to be the fold the
default would have won on its own. This is a useful reminder that
"in-sample winner" and "OOS winner" are not the same thing, fold by fold,
even when the aggregate stitched result favors tuning.

## The overfitting gap (in-sample PF vs. OOS PF)

| Fold | IS PF | OOS PF | Decay (IS − OOS) |
|---|---|---|---|
| F1 | 1.011 | 0.921 | +0.090 |
| F2 | 1.019 | 1.087 | **−0.068** (OOS beat IS) |
| F3 | 1.194 | 0.886 | **+0.308** (largest decay) |
| F4 | 1.209 | 1.153 | +0.056 |

In-sample PF ranges 1.01–1.21; OOS PF ranges 0.89–1.15. Three of four folds
show the expected in-sample-to-OOS decay (the selected combo looked better
on the data it was chosen from than on unseen data); one fold (F2) went the
other way, which is itself a reminder that with n=4 the "decay" pattern is
not a reliable per-fold signal — only the aggregate matters, and the
aggregate is a decay from marginal in-sample edge to marginal-negative OOS
edge. Chart: `charts/phase4_is_vs_oos_pf.png`.

## Is the pick special, or just lucky? — the selection-luck null

For every fold, all 144 grid combinations' OOS profit factors were
recorded (not just the selected one), giving a **selection-luck null**: if
a randomly chosen combo does about as well OOS as the one selection
picked, the in-sample selection process bought nothing.

| Fold | Selected pick's OOS percentile* | Median-combo OOS PF | Selected OOS PF | Combos within winner's IS-PF bootstrap CI (of 144) |
|---|---|---|---|---|
| F1 | 30.6% | 1.051 | 0.921 | 123 |
| F2 | 97.2% | 0.771 | 1.087 | 122 |
| F3 | 55.6% | 0.828 | 0.886 | 88 |
| F4 | 44.4% | 1.183 | 1.153 | 72 |

\* Fraction of the 144 combos' OOS PF the selected pick exceeds.

The picture is mixed, not clean: in F2 the selection was genuinely special
(97th percentile — almost the best of all 144 combos would have done OOS),
but in F1 the selected pick actually did *worse* than the median combo
(30.6th percentile) — in-sample selection pointed the wrong way that fold.
Averaged across folds the picks sit close to the middle of the pack. Just
as telling: **72–123 of the 144 combos' in-sample profit factors fall
inside a bootstrap confidence interval around the winner's** — meaning in
every fold, roughly half to over 85% of the grid was statistically
indistinguishable from the "winner" using only in-sample data. Selection
is closer to a coin-flip among a large cluster of similar combos than to
picking one standout. This is precisely why the pre-registered success
rule requires beating the **median**-combo null (condition d), not just
beating the default — and precisely why, even though condition (d) passes
in aggregate (0.9945 > 0.9396), it should not be read as "the tuning
process reliably found a special combo." Chart:
`charts/phase4_selection_luck_null.png`.

## Parameter stability — read with the n=4 caveat front and center

| Parameter | F1 | F2 | F3 | F4 |
|---|---|---|---|---|
| `fvg_threshold` | 0.05 | 0.05 | 0.02 | 0.02 |
| `rr` | 1.5 | 1.5 | 3.0 | 1.0 |
| `ema_length` | 50 | 50 | 20 | 10 |
| `swing_lookback` | **5** | **5** | **5** | **5** |

`swing_lookback = 5` (the tightest value in its grid, `{5, 8, 12}`, and
tighter than the Pine default of 8) is the **only** parameter selected
consistently across all four folds. `fvg_threshold`, `rr`, and `ema_length`
scatter across their grids fold to fold with no visible pattern.

**This is descriptive, not evidence of a real optimum.** Two structural
reasons, stated plainly and not glossed over:

1. **n = 4 folds has no statistical power.** Four paired observations
   cannot support a significance claim about "the" optimal parameter for
   any of the four levers — this table is a description of what happened
   in this one run, not an inference about the population of possible
   markets/regimes.
2. **The train windows overlap.** F1's train window (2023-01..2023-12) and
   F2's (2023-07..2024-06) share six months of data; F2 and F3 share six
   months; F3 and F4 share six months. The four selected parameter-sets
   are therefore **not independent draws** — consecutive folds are
   partially re-selecting on the same underlying data, which mechanically
   *inflates* any apparent agreement (including `swing_lookback`'s
   4-for-4 record). A truly independent 4-fold study would be expected to
   show less agreement than this one does, all else equal.

Chart: `charts/phase4_param_stability.png`.

## Multiple-testing and pre-registration disclosure

- **144 grid combinations** (`fvg_threshold ∈ {0, 0.02, 0.05, 0.10}` ×
  `rr ∈ {1.0, 1.5, 2.0, 3.0}` × `ema_length ∈ {10, 20, 50}` ×
  `swing_lookback ∈ {5, 8, 12}`) were evaluated **per fold**, purely on
  in-sample data, to select each fold's params — a real multiple-testing
  surface, which is exactly why the selection-luck null above (comparing
  the pick against the full 144-combo OOS distribution, not just against
  the default) is load-bearing rather than decorative.
- **Pre-registration freeze.** The grid, the four fold date windows, the
  50-trade in-sample selection floor, and the objective
  (`max_pf_min50trades`) were committed as constants in `tuning/grid.py`
  and `tuning/walkforward.py` *before* `run_phase4.py` was executed. The
  run is single-shot: the stitched-OOS number was observed exactly once.
  `phase4_results.json` records a SHA-256 hash of the frozen config
  (`config_hash: ebdcc2e9…dd4c4d`) and the git commit SHA the run executed
  against (`git_sha: 45603bb…093eb`) as an audit trail — any future change
  to the grid, folds, floor, or objective after seeing this OOS result
  would require a new dated spec and an explicitly-labelled new
  experiment, never a silent re-run.
- Run cost: 635.8 seconds for the full grid × 4-fold walk-forward
  (including the null-control pass of all 144 combos' OOS on every fold).

## Other caveats

- **`fvg_threshold`'s effective strictness drifts across the sample.** It
  is a *percent* gap filter (`gap / price × 100`), but the underlying
  series is back-adjusted and its price level drifts materially over the
  window (roughly +2,655 points in early 2023 down to ~0 by 2025, per the
  Phase-2 writeup). A fixed percentage therefore maps to a different
  absolute point-gap filter at the start of the sample than at the end,
  which confounds any attempt to read `fvg_threshold`'s selected values as
  a stable preference — its selection in F1/F2 (0.05) vs. F3/F4 (0.02)
  could reflect this drift as easily as a genuine regime difference.
- **Back-adjusted, continuous price series.** As in Phase 2, `load_nq()`
  is a back-adjusted continuous NQ series, not the real, unadjusted
  front-month prices the strategy would trade live. Every comparison in
  this study is strategy-vs-strategy (tuned vs. default) on the *same*
  series, which is a valid comparison for this question — but no PF, P&L,
  or drawdown figure above should be read as a live-tradeable estimate.
- **Boundary handling.** A trade still open at a fold's `test_end` is
  dropped from the stitched OOS (the next fold starts flat); no trade
  spans a window boundary or a contract roll (one trade/day, intraday), so
  per-trade dollar figures and PF are back-adjustment-safe for this
  comparison. Fold boundaries are half-open (`[start, end)`), verified
  contiguous and non-overlapping, covering 2024-01 through 2025-12-11
  exactly.
- **No leakage, verified structurally.** Indicators are computed once over
  the full series (causal — each bar's value depends only on bars ≤ it)
  and then sliced per fold via positional `searchsorted`; a dedicated test
  asserts `train_slice.index.max() < test_slice.index.min()` for every
  fold and that sliced arrays exactly match direct `.loc[window]` slices.
  Parameter selection reads only each fold's train-window in-sample
  trades — it never touches that fold's test window.

## Tie-back to Phase 2

Phase 2's honest finding was that the raw default-parameter engine
recovers most of the real track record's trade-days and directions but
over-trades roughly 4x relative to the real, "optimised" logs, and
underperforms them on PF and win rate — with the gap looking like
selectivity/tuning rather than a different core mechanic. Phase 4 tested
that hypothesis directly by tuning the mechanic's own knobs
(`fvg_threshold`, `rr`, `ema_length`, `swing_lookback`) via honest
walk-forward optimization, and the result **supports the hypothesis
directionally while falling short of it in magnitude**: selectivity
(233 vs. 401 trades) very nearly closes the gap to breakeven (PF 0.99 vs.
0.92, loss cut from −$20,362.50 to −$907.50) but does not cross it. The
remaining gap to the real track record's reported performance likely
requires either additional discretionary filtering beyond these four
parameters, a different execution/session model, or genuinely
non-stationary edge that this back-adjusted, 3-year sample cannot recover
via parameter search alone.

## Charts

All charts generated by `run_phase4.py`, committed under `charts/`:

- `charts/phase4_equity_curve.png` — stitched OOS cumulative P&L, tuned vs.
  default.
- `charts/phase4_oos_pf_per_fold.png` — per-fold OOS profit factor, tuned
  vs. default (bars).
- `charts/phase4_param_stability.png` — selected parameter value per fold
  (stability table).
- `charts/phase4_is_vs_oos_pf.png` — in-sample vs. OOS profit factor per
  fold (the overfitting-gap chart).
- `charts/phase4_selection_luck_null.png` — all 144 combos' OOS PF
  distribution per fold, with the selected pick and the median-combo null
  marked.

See `notebooks/04_parameter_tuning.ipynb` for the same result narrated
end-to-end with the fold table and all five charts inline, loaded from the
committed `phase4_results.json` (the walk-forward sweep itself takes ~10.6
minutes and is not re-run by the notebook).

---

## The 4-phase program: what we learned

This project was built as four phases, each answering a question the
previous phase raised — and the arc across all four is itself the
deliverable, as much as any single number:

1. **Phase 1 — data foundation.** Built and validated the raw data
   pipeline: ~3 years of 1-minute NQ OHLCV, load/parsing/coverage checks.
   No strategy logic yet — just making sure the ground truth was solid
   before building anything on top of it.
2. **Phase 2 — faithfully rebuild the strategy, and find where the edge
   likely lives.** Ported the FYP IFVG+CISD strategy from Pine to Python
   bar-by-bar, with no fitting to any real trade log, and validated it
   against two real TradingView track records. Found and fixed a genuine
   off-by-one bug in the ported CISD logic along the way. The honest
   result: good directional/coverage fidelity (76–80% recall), but the raw
   default parameters underperform the real "optimised" track record and
   over-trade ~4x — a gap that looked like it came from
   selectivity/tuning rather than a broken port. This is the hypothesis
   Phase 4 exists to test.
3. **Phase 3 — correctly not run.** The original program design
   pre-registered a Phase 3 (a large-sample Monte Carlo / ML re-run of the
   strategy) explicitly **conditional on Phase 4 finding a robust
   improvement** — it was designed to run "AFTER, on the tuned trades, if
   tuning helps" (`docs/specs/2026-07-13-phase4-parameter-tuning-design.md`).
   Phase 4's pre-registered verdict is `robust_improvement = false`, so
   Phase 3's gating condition was never met, and it was skipped by design
   rather than run anyway to manufacture another number. Skipping a
   planned phase because its own precondition failed is a small thing, but
   it is the same discipline as the rest of this program applied to the
   program's own structure, not just to individual results.
4. **Phase 4 — test the tuning hypothesis, rigorously, out-of-sample.**
   Parameterized the engine, built a pre-registered walk-forward
   optimizer with a falsifiable success rule and a selection-luck null
   control, and ran it single-shot. The result: tuning **directionally
   confirms** the Phase-2 selectivity hypothesis (fewer, more selective
   trades very nearly erase the default's OOS loss) but **does not
   robustly cross into profitability** by the standard set before the run
   was executed. Reported as the honest null it is.

The throughline is not "we found an edge" — it is that **every claim in
this program was tested against a bar set before the result was known**,
and reported whether or not it flattered the strategy: a rebuild that
under-recovers the real track record's edge (Phase 2), a downstream phase
correctly skipped when its precondition failed (Phase 3), and a tuning
study that gets close to breakeven but not across it (Phase 4). That
discipline — pre-registration, out-of-sample-only headlines, null
controls, and reporting negative results (and skipped phases) as plainly
as positive ones — is the actual portfolio artifact here, arguably more
than any single profit-factor number.
