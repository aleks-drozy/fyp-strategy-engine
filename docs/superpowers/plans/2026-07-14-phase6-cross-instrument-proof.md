# Phase 6 — Cross-Instrument Confirmation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Prove or disprove the Phase-5 exit edge by running the FROZEN Phase-5 hypothesis on ~10 years of independent instruments (ES, YM) plus an independent-vendor NQ — pre-registered, single-shot, net of per-instrument costs.

**Architecture:** One-time safe pickle→parquet conversion + hard validation gates (incl. a timezone-inference proof); an `InstrumentSpec` threaded through the engine/costs (NQ-spec regression-locked to Phase 5); fold tiling over each instrument's full history; two runners — H-A (the frozen Phase-5 walk-forward procedure, unchanged) and H-B (two fixed configs, provably no selection); a single-shot runner evaluating the pre-registered per-instrument + pooled success rule.

**Tech Stack:** Python, pandas 2.2.3, numpy 2.1.3, pyarrow (add), pytest, matplotlib. Reuses `tuning/walkforward{,_p5}.py`, `backtest/*`, `strategy/*`.

## Global Constraints

- **CONFIRMATION, NOT EXPLORATION.** Nothing is re-tuned in response to Phase-6 data. The grid (20 combos), objective (net PF, MIN_IS_TRADES=50 pre-filter basis), fold shape (12mo/6mo rolling), exit sequencing, and cost structure are the FROZEN Phase-5 artifacts, reused verbatim. Any change after seeing a Phase-6 backtest = new experiment, disclosed.
- **Spike-established facts (from `p6_spike`, committed evidence):** each file ≈ 3.4M bars, 2015-03-26 → 2025-01-29 (~9.8y); 0 OHLC violations; 0 NaN; ~16k duplicate timestamps (dedup keep-first + sort, like Phase 1); naive timestamps ARE **ET wall-clock** (quietest weekday hour = 17:00 = CME maintenance; reopens at 18:00). The production validator must re-prove tz on the cleaned data and hard-fail if ambiguous.
- **Instruments (frozen):** `ES: tick 0.25, tick_value $12.50, pt_value $50` · `NQ: tick 0.25, $5, $20` · `YM: tick 1.0, $5, $5`. Costs = Phase-5 structure (commission $5 RT; 1 tick entry + 1 tick per market exit, valued per instrument; multiplier 0×/1×/2× reported). YM prices must sit on an integer grid (validator asserts per-instrument grid).
- **Volume is absent** in the source (`Candle{o,h,l,c,t}`); synthesized as 0 and unused (signals/exits read OHLC only — verified in Phases 2–5).
- **Fold tiling:** rolling 12mo-train/6mo-test half-open folds tiled from `first_bar + 12mo` boundary-aligned to calendar halves (Jan-01/Jul-01), last test window clipped at the data edge; expected ≈ 17 folds/instrument. Reuse Phase-4/5 half-open searchsorted slicing + disjointness raises verbatim.
- **Hypotheses (frozen; revised per adversarial review — Blocker B7):** **H-A is the SOLE primary** (only route to the program-level PROVEN), with a **composition condition**: on each independent instrument a majority of per-fold selections must be smart-exit-family configs (not base-like); pick tables published. H-B1 = fixed `partial_1R`+`p50`, H-B2 = fixed `trail_swing`+`p50` are **secondary "config-level replication"** readouts gated at a Bonferroni **2.5th-pct** CI bound; they cannot produce PROVEN. `trail_swing`+`off`'s exclusion (lacks the vol-filter component of the hypothesis) is declared in advance.
- **Gating statistic (Blocker B6):** all gates use **net R-multiples** (`net_R = net_pnl/(risk × pt_value)`); dollar PF descriptive only. **CI (Blocker B5): calendar-day cluster bootstrap** — resample trade-DAYS with replacement, a sampled day carries ALL its pooled ES+YM trades; stratified instrument composition; `n_boot=10,000`, fixed seed, basic (pivotal) intervals. Floors: pooled **n ≥ 150 trades AND ≥ 100 days**; **per-instrument n ≥ 150** tuned OOS trades for conditions 1–2 to be evaluable.
- **Verdict decision table (Blocker B3/B4 — ordered, exhaustive, encoded verbatim in the runner):**
  1. **PROVEN** iff (1) tuned R-PF > 1.0 on both ES+YM; (2) margin ≥ 0.10 vs base on both; (3) pooled day-cluster CI-low (5th pct) > 1.0 **+ robustness guards** (per-fold majority on EACH instrument; leave-one-out most-profitable-fold survives); (4) indep-NQ agreement (pre/post-2023 split published, pre = entry < 2023-01-01).
  2. else **DISPROVEN** iff (a) tuned **strictly <** base on both ES+YM AND pooled margin CI-**upper** < 0.10; OR (b) pooled tuned R-PF CI-**upper** (95th pct) < 1.0. (CI-gated — ties/noise/low power cannot force it.)
  3. else **PARTIAL** iff conds 1–2 on exactly one independent instrument, or conds 1–3 pass + cond 4 fails.
  4. else **INCONCLUSIVE** (explicit catch-all; no undefined branch).
- **Pre-registration freeze (Important I5 — mechanical, git-timestamped):** commit `docs/phase6_freeze.json` (config hash, serialized decision table, every rule constant — n_boot/seed/percentiles/cluster unit/floors/margins/roll+anomaly detector constants/tiling rule/hypothesis hierarchy/composition condition) in Task 3, **BEFORE `run_phase6.py` exists**. The runner recomputes its config hash and **refuses to run** on mismatch. Single-shot.
- **Sensitivity readouts (context, never gates):** exclude-2023+ ES/YM windows (formation-era overlap, Important I8); pre/post-2020 regime split; cost 0×/1×/2×; fold-cluster CI diagnostic; published MDE (~pooled R-PF 1.10).
- **Safe unpickling (converter):** restricted `Unpickler.find_class` allowlisting ONLY `Candle`, numpy scalar/dtype/ndarray/`_reconstruct`, `datetime.datetime`; everything else raises. Documented security note. Parquet is the only artifact the pipeline reads afterward.
- **Base config** = `fixed_1_5R` + `off`, same engine, same instrument spec, net — the control on every instrument.

---

## Task 1: Converter + loader + validation gates

**Files:** Create `data/convert_p6.py`, `nqdata/load_p6.py`, `validate_p6.py`; Test `tests/test_convert_p6.py`, `tests/test_load_p6.py`, `tests/test_validate_p6.py`. Modify `requirements.txt` (+`pyarrow==17.0.0`).

**Interfaces:** `convert(src_path, out_parquet) -> dict` (n_in, n_out, dups_dropped); `load_instrument(sym: str) -> pd.DataFrame` (lowercase o/h/l/c + volume=0, tz-aware **US/Eastern** DatetimeIndex, sorted, deduped — same contract as `load_nq`); `validate_instrument(df, sym) -> dict` (Phase-1-style report + `tz_evidence`); `main()` writes `data/validation_report_p6.json` (committed).

- [ ] **Step 1 (tests first):** converter — a synthetic pickle of 5 Candles (one dup timestamp, one out-of-order) round-trips to parquet sorted+deduped; a pickle containing a forbidden global (e.g. `os.system`) raises `UnpicklingError`. **Divergent-dup rule (B1):** a synthetic dup group where keep-first breaks price continuity but the continuity rule (`min |open − prev_kept_close|`, tie→first) picks the consistent row — assert the continuity pick. Loader — parquet → contract frame (tz-aware ET, columns, volume=0). Validator — flags a synthetic OHLC violation; **tz-inference hardened (I2)**: evidence per DST regime AND per year (quiet==17, reopen==18 under EDT and EST every year); a synthetic fixed-offset (UTC-5 year-round) series **fails**; a true-ET series passes. **Roll detection (B2):** synthetic series with 4 injected quarterly session-break jumps in day-window 5–18 → all 4 detected, `roll_boundaries` listed; a no-roll series → hard-fail on count. **Anomaly windows (B1):** an injected in-session bad print (range > K× rolling robust range) is flagged. Per-instrument grid: YM close 0.25 off integer → violation.
- [ ] **Step 2:** implement converter (SafeUnpickler, allowlist pinned to the observed `(module,name)` pairs; stream: load→arrays→drop objects→parquet), loader, validator. Report fields: `n_rows, date_min, date_max, pct_1min_spacing_after_sort, n_ohlc_violations, n_nan, n_dups_identical, n_dups_divergent, dup_divergence_pctiles, grid_ok, session_days, per_halfyear_session_days (accept [115,135]), per_session_completeness (≥50/60 min), tz_evidence{quiet_hour_by_dst_regime, reopen_hour_by_dst_regime, quiet_hour_by_year}, roll_boundaries[], n_anomaly_windows`. **Cross-vendor gate (I1):** on the 2022→2025-01 overlap, per-day 1-min log-return correlation indep-NQ vs Phase-1 NQ ≥ 0.9 every day (hard gate), lag-0 > ±1-min lags, offset curve reported.
- [ ] **Step 3:** run the real conversion + validation for ES/NQ/YM (foreground; minutes). Commit `data/validation_report_p6.json` (parquets gitignored under `data/parquet_p6/`). Sanity: spans ≈ 2015-03→2025-01, 0 violations, tz evidence green in every year/regime, rolls ≈ 4/yr each.
- [ ] **Step 4:** `pytest tests/ -q` all green. **Commit:** `feat: phase 6 data pipeline (safe converter + hardened validation gates)`.

## Task 2: InstrumentSpec + engine threading (regression-locked)

**Files:** Create `strategy/instrument.py`; Modify `backtest/engine.py`, `backtest/costs.py`; Test `tests/test_instrument.py`, `tests/test_engine_p6_regression.py`.

**Interfaces:** `InstrumentSpec(sym, tick_size, tick_value, pt_value)`; `SPECS = {"ES":…, "NQ":…, "YM":…}`; `run_execution(..., spec: InstrumentSpec = SPECS["NQ"])`; `CostModel.net_pnl(..., spec)` uses `spec.tick_value`; engine P&L uses `spec.pt_value`.

- [ ] **Step 1 (tests first):** cost math per instrument (YM stop-loser pays $5+$5+$5; ES stop-loser $5+$12.50+$12.50; NQ unchanged). **Regression lock:** with `SPECS["NQ"]` defaults, (a) the Phase-2 golden fixture reproduces trade-for-trade; (b) a stored Phase-5 spot value (e.g. F4 tuned net OOS PF 2.2480132450 from `phase5_results.json`) reproduces via `walk_forward_p5` on the Phase-1 data — proving the threading changed nothing.
- [ ] **Step 2:** implement; default parameter values keep every existing call site working unchanged.
- [ ] **Step 3:** `pytest tests/ -q` all green. **Commit:** `feat: InstrumentSpec threaded through engine+costs (NQ regression-locked)`.

## Task 3: Fold tiling + H-A/H-B runners

**Files:** Create `tuning/walkforward_p6.py`; Test `tests/test_walkforward_p6.py`.

**Interfaces:** `make_folds_tiled(index) -> list[Fold]`; `run_HA(df, spec, roll_boundaries) -> dict` (frozen `walk_forward_p5` over the tiled folds — reuse it, do not reimplement); `run_HB(df, spec, config, roll_boundaries) -> dict` (fixed config per fold — **no selector call anywhere**); both return per-fold + stitched net tuned/base trades (with `net_R`) + all-combo OOS R-PFs (H-A only) + roll/anomaly exclusion counts.

- [ ] **Step 1 (tests first):** **Tiling pinned (I4):** "test starts at every Jan-01/Jul-01 ET boundary `t` with `t ≥ first_bar + 12mo` and `t < last_bar`; train = `[t−12mo, t)` exactly (never extended); test = `[t, min(t+6mo, data_end))`; final fold formed only if its test span ≥ 3mo" — synthetic 3-year index → exact expected boundaries; <18mo index → raises; stub-fold rule tested both sides of 3mo. **No-selection proof:** monkeypatch `select_params_p5` to raise → `run_HB` completes. **Roll exclusion (B2):** a synthetic trade whose hold spans an injected roll boundary is excluded from BOTH tuned and base arms and counted; a same-day trade isn't. Vol-threshold leak test reused. Stitched-base == stitched H-B when config==base.
- [ ] **Step 2:** implement as thin wrappers over `walkforward_p5` machinery (`_precompute_p5`, `_slice_layer_p5`, `_vol_threshold`, `_net_metrics`); folds passed in, spec threaded; roll/anomaly windows applied as post-run trade filters (identical across arms).
- [ ] **Step 3 (I5 — the freeze, BEFORE any runner exists):** write + commit `docs/phase6_freeze.json`: config hash (SHA-256 over instruments/specs/costs/grid/folds-rule-text/floors/margins/n_boot/seed/percentiles/cluster-unit/detector constants/hypothesis hierarchy/composition condition) + the serialized verdict decision table. Separate commit, so the freeze is **git-timestamped before `run_phase6.py` exists**.
- [ ] **Step 4:** `pytest tests/ -q` green. **Commit:** `feat: phase 6 fold tiling + H-A/H-B runners + pre-registration freeze`.

## Task 4: Single-shot run + results + charts

**Files:** Create `run_phase6.py`; outputs `phase6_results.json` + `charts/phase6_*.png` (committed). Test `tests/test_smoke_phase6.py`.

- [ ] **Step 1:** runner — **first verifies its recomputed config hash equals `docs/phase6_freeze.json` and refuses to run otherwise (mechanical single-shot).** For each instrument: `load_instrument` → tiled folds → `run_HA` + `run_HB(B1)` + `run_HB(B2)` + base, roll/anomaly exclusions applied identically. Gating statistic = **net R-multiples**; pooled ES+YM CI = **calendar-day cluster bootstrap** (resample days; a day carries all its ES+YM trades; stratified composition; n_boot=10,000, fixed seed, basic/pivotal intervals); floors: pooled n≥150 ∧ days≥100; per-instrument n≥150 for conds 1–2 evaluability. Assemble `phase6_results.json`: frozen hash + git_sha, per-instrument per-hypothesis stitched net R-metrics + per-fold tables + pick tables (H-A composition condition) + fold/trade/exclusion counts, indep-NQ pre/post-2023 split, pooled records, the **verdict decision table evaluated verbatim in order** (H-A → PROVEN/DISPROVEN/PARTIAL/INCONCLUSIVE with every sub-condition value incl. the robustness guards: per-fold majority per instrument + leave-one-out; H-B1/H-B2 → "config-level replication supported/not" at the 2.5th-pct bound), sensitivity readouts (exclude-2023+ windows; pre/post-2020 split; cost 0×/1×/2×; fold-cluster CI diagnostic; MDE statement), run_seconds. Cap PFs at PF_CAP before distribution comparisons; dollar PF reported as descriptive only.
- [ ] **Step 2:** charts — (1) per-instrument stitched net OOS equity (tuned H-A vs base); (2) verdict matrix (hypothesis × instrument net PF vs base); (3) pooled ES+YM equity + CI annotation; (4) per-fold PF scatter per instrument; (5) cost-sensitivity bars.
- [ ] **Step 3:** smoke test — **must not peek at Phase-6 data**: run on synthetic bars or Phase-1 NQ data only, asserting mechanics (fold counts, schema, exclusion filters, no exceptions). Then RUN `run_phase6.py` foreground — the heaviest run of the program (3 instruments × ~17 folds × 20 combos for H-A; per-instrument precompute reuse makes it tractable; report runtime). Commit results + charts.
- [ ] **Step 4:** `pytest tests/ -q` green. **Commit:** `feat: phase 6 single-shot cross-instrument confirmation run`.

## Task 5: Writeup + notebook + final review/merge

- [ ] **Step 1:** `WRITEUP_PHASE6.md` — lead with the verdict per hypothesis (PROVEN/PARTIAL/DISPROVEN) and the rule's sub-conditions; per-instrument tables; the indep-NQ overlap split; pooled CI; selection-luck context; disclosures (confirmation-not-exploration statement, cost optimism, fold overlap within an instrument, the 2015–2025 regime span, volume-absent note); program epilogue updated (Phases 1–6). Honest either way — a DISPROVEN closes the arc as "the near-miss did not replicate"; a PROVEN must survive every stated gate.
- [ ] **Step 2:** `notebooks/06_cross_instrument_proof.ipynb` (loads the JSON; no re-run) + README Phase-6 section.
- [ ] **Step 3:** final whole-branch review (strongest model) → fix Critical/Important → `pytest` green → merge to master → vault update. **Hold the GitHub push for Alex.**

## Self-review notes
- **Revised after a 3-lens adversarial review that returned DO-NOT-RUN with 7 blockers — all closed in-plan:** (B1) continuity-based divergent-dup handling + anomaly windows; (B2) roll-boundary detection + identical-across-arms trade exclusion; (B3) ordered exhaustive 4-verdict decision table (adds INCONCLUSIVE), encoded verbatim; (B4) CI-gated symmetric DISPROVEN (strict inequality + upper-bound conditions); (B5) day-cluster bootstrap replacing the invalid i.i.d. one; (B6) net-R gating statistic replacing dollar PF; (B7) H-A sole primary + composition condition, H-B demoted to Bonferroni-gated replication.
- **Importants folded in:** cross-vendor correlation gate (I1); per-DST-regime/per-year tz evidence (I2); per-fold session integrity (I3); tiling rule pinned (I4); git-timestamped freeze file committed before the runner exists + mechanical single-shot (I5); fold-majority + leave-one-out robustness guards (I6); per-instrument n≥150 floors + published MDE (I7); exclude-formation-era sensitivity readout (I8).
- **Correctness locks:** NQ-spec regression to Phase-2 golden + a pinned Phase-5 value; converter security allowlist (pinned pairs); tz hard gate; Phase-4/5 leak guards reused verbatim.
- **Scrutinize in final review:** the freeze-before-runner git ordering, the day-cluster bootstrap implementation, the roll-exclusion symmetry across arms, the decision-table encoding vs the frozen text, and that H-A truly reuses `walk_forward_p5` (not a re-implementation).
