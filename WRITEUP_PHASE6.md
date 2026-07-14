# Phase 6 — Cross-Instrument Confirmation: the Exit Edge Is DISPROVEN

## Pre-registered verdict

**DISPROVEN — with statistical confidence.** The Phase-5 exit edge does not replicate on
independent instruments. Across ~10 years and two independent equity-index futures (ES, YM;
17 walk-forward folds each; 1,402 tuned out-of-sample trades over 1,005 trade-days), the frozen
Phase-5 procedure's pooled net R-multiple profit factor is **0.905**, with a day-cluster bootstrap
90% CI of **[0.813, 0.990]** — the **upper bound is below breakeven**, firing the pre-registered
DISPROVEN clause (b): with high confidence, the tuned strategy loses money net of costs on data it
has never seen. Both fixed-config replications also fail (B1 `partial_1R+p50`: pooled R-PF 0.904,
Bonferroni CI-low 0.794; B2 `trail_swing+p50`: 0.836, CI-low 0.726 — "not supported").

| Instrument (~10y, 17 folds) | Tuned R-PF (n) | Base R-PF (n) | Tuned net $ | Base net $ |
|---|---|---|---|---|
| **ES** (independent) | 0.847 (574) | 0.829 (1,527) | −$21,348 | −$39,735 |
| **YM** (independent) | 0.948 (828) | 0.876 (1,624) | −$9,435 | −$19,675 |
| **NQ** (indep vendor, replication) | 0.851 (914) | **0.899** (1,726) | −$82,438 | −$96,798 |

Everything was frozen before the run: hypotheses, grid, folds, floors, cost model, the gating
statistic (net R-multiples), the CI method (calendar-day cluster bootstrap, stratified by
instrument, n=10,000, seed 42, basic intervals), and the ordered 4-verdict decision table —
committed as `docs/phase6_freeze.json` (hash `699fd485…`) **before `run_phase6.py` existed**
(git history proves the ordering); the runner recomputes the hash and refuses to run on mismatch.

## What the verdict means

1. **Phase 5's near-miss was period-specific luck, not a real edge.** On 2024–25 NQ, smarter exits
   flipped the point estimate net-positive (+$9,795, PF 1.07) but failed the significance gate.
   Phase 6 answers the question that gate raised: run the same frozen procedure on 10 years of two
   independent instruments (and a cleaner NQ), and it loses everywhere. On clean NQ the tuned
   procedure is actually **worse than base** (0.851 vs 0.899) — `nq_agreement: False`.
2. **The exit story survives in miniature, but it cannot rescue the strategy.** Consistent with
   Phases 4–5, trade management *narrows* losses on the independent instruments (ES −$40k→−$21k;
   YM −$20k→−$9k; margins +0.018 / +0.071 R-PF) — but both arms stay under water, and the margins
   are below the pre-registered 0.10 bar. Better exits on a losing entry signal produce a smaller
   loss, not a profit.
3. **No hiding regime.** Sensitivity cuts (context, never gates): excluding the 2023+
   hypothesis-formation era → 0.898; pre-2020 → 0.871; post-2020 → 0.922; NQ pre-2023 → 0.887;
   NQ post-2023 → 0.735. Unprofitable in every slice. Leave-one-out and per-fold-majority guards
   also fail (no instrument has a majority of folds with tuned > base).

**The program's final answer:** the FYP IFVG+CISD strategy — entries and exits, default or tuned —
has **no robust net edge**. That answer is now statistically certified, cross-instrument, on the
cleanest data in the program, under a pre-registered symmetric rule that could equally well have
said PROVEN.

## A finding of independent value: the data forensics

Phase 6's hard validation gates caught, root-caused, and *proved* four defect classes in the data
this program had been using since Phase 1 (the Kaggle back-adjusted NQ set):

- **A ±60-minute DST-handling bug on 104 scattered days** — Phase-1 bars are stamped exactly one
  hour off (proof: ~0 correlation aligned, 0.88–0.99 at a ±60-min lag; lag histogram cleanly at
  ±60). On those days, Phases 2–5 were literally trading the wrong hour.
- **Intra-day splices** (part of a day shifted), **missing sessions** (weekday holes), and
- **Fast-market infidelity** — e.g. the 2023-05-03 FOMC hour, where the Phase-1 path diverges 80+
  points from the CME-grid data and re-converges after (hourly correlation 1.00 everywhere except
  that hour).

Every one of the 188 disputed overlap days was attributed to a Phase-1 defect via a three-rung
evidence ladder (whole-day lag match → intra-day splice halves → an indep-ES *referee*: real NQ
co-moves with real ES at 0.93–0.98; garbage cannot). Zero days impeach the new data. The new
vendor's ES/NQ/YM set also had defects — a contaminated maintenance hour (dropped: those bars are
untradable) and a one-minute bar-labeling offset (normalized: empirically decisive, 0.905 vs ~0) —
all detected by gates, fixed by frozen market-structure-grounded rules, and disclosed in
`data/validation_report_p6.json`.

**Retrospective honesty note:** Phases 2–5's NQ results were measured on partly time-shifted data.
Their *internal* comparisons (tuned vs base on the same data) remain valid, and Phase 6 — on clean
data — now supersedes them as the definitive measurement. The defects likely explain part of
Phase 2's imperfect trade-matching against the real TradingView logs.

## Method integrity (what makes this verdict trustworthy)

- **Confirmation, not exploration:** zero re-tuning — the frozen Phase-5 machinery ran verbatim
  (`walk_forward_p5`, byte-reused; H-B provably never invokes the selector — a test monkeypatches
  it to raise). The engine at the NQ spec reproduces the exact Phase-5 result to 12 decimal places.
- **Statistics built for the data's dependence:** unit-risk R-multiples (so ES's $50/pt cannot
  outvote YM's $5/pt), day-cluster bootstrap (a resampled day carries both instruments' trades —
  ES/YM are ~0.9-correlated and same-day trades are near-duplicates), basic/pivotal intervals,
  floors on trades AND days, symmetric CI-gated DISPROVEN (ties and low power cannot force it).
- **Roll safety:** unadjusted futures splice ~4×/year; trades spanning a detected roll boundary
  are excluded identically in every arm (0 such trades occurred — median hold is 17 minutes).
- **One reporting-layer defect found and fixed after the first run** (an instrument-order mislabel
  in the conditions assembly — `es, ym, nq` unpacked in the wrong order). Disclosed here; the
  verdict was unaffected (the pooled CI uses explicit per-instrument keys, and the deterministic
  re-run reproduced identical numbers with corrected labels).

## Disclosures

- ES and YM are independent of Phase-5's data, **not of each other** (~0.9-correlated US equity
  indices); the day-cluster CI is constructed to respect exactly that.
- The H-B family declared 2 of Phase 5's 3 distinct fold-picks; `trail_swing+off` was excluded in
  advance (lacks the vol-filter component of the hypothesis). H-B used a Bonferroni 2.5th-pct gate.
- Cost model: $5 RT commission + 1 tick entry + 1 tick per market exit, per-instrument tick values;
  1-tick stop slippage is optimistic — which makes an *unprofitable* verdict conservative (real
  costs would only worsen it). Cost sensitivity: at **0× costs** the tuned arm still loses on ES
  (R-PF 0.983) and is only marginally gross-profitable on YM (1.058) — realistic costs erase the
  YM sliver (→ 0.948) and 2× costs push both deep under (0.733 / 0.851). The edge is too thin to
  survive any plausible cost assumption.
- Session-integrity gaps (vendor missing sessions) and all exclusion counts are in the committed
  validation report. Fold windows within an instrument overlap in training data (rolling), and
  n = 17 folds/instrument is descriptive at fold level — the gate operates at trade-day level.

## Program epilogue (Phases 1–6)

1. **P1** built a reproducible data foundation → 2. **P2** faithfully rebuilt the strategy and
found the edge was selectivity, not the signal → 4. **P4** pre-registered tuning: null →
5. **P5** costs + exits: first net-positive point estimate, failed the significance gate ("promising,
not proven") → 6. **P6** acquired 10 years × 3 instruments of cleaner data, froze the hypothesis,
and got the definitive answer: **DISPROVEN** — and, along the way, forensically caught defects in
the original dataset that no one knew were there.

Every phase was pre-registered, adversarially reviewed before build, regression-locked, and
reported as-is: one near-miss and otherwise honest nulls, ending in a certified disproof. The
deliverable of this program is not a trading edge — it is the demonstrated ability to find out,
rigorously, whether one exists.
