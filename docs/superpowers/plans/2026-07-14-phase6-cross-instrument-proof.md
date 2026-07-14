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
- **Hypotheses (frozen):** H-A = frozen Phase-5 walk-forward procedure per instrument. H-B1 = fixed `partial_1R`+`p50`; H-B2 = fixed `trail_swing`+`p50` (both declared; no mixing; p50 = per-fold train-window ATR% percentile exactly as Phase 5).
- **Pre-registered success rule (per spec):** PROVEN for H-A (or one named H-B config consistently) iff: (1) net stitched-OOS PF > 1.0 on BOTH ES and YM; (2) margin vs base ≥ 0.10 on both; (3) pooled ES+YM bootstrap CI lower (5th pct, fixed seed) > 1.0 with pooled n ≥ 150; (4) indep-NQ directional agreement (tuned ≥ base). PARTIAL/DISPROVEN per spec. Every value reported.
- **Indep-NQ overlap disclosure:** its 2023-01→2025-01 segment overlaps the period the hypothesis was formed on. Report indep-NQ SPLIT: pre-2023 (fresh-in-time) vs 2023+ (overlap) segments, plus combined; it counts only as replication either way.
- **Pre-registration freeze:** SHA-256 config hash covering instruments/specs/costs/grid/folds/floors/hypotheses/success-rule constants + git SHA into `phase6_results.json`; single-shot; per-instrument all-combo OOS PF recorded (H-A) for selection-luck context.
- **Safe unpickling (converter):** restricted `Unpickler.find_class` allowlisting ONLY `Candle`, numpy scalar/dtype/ndarray/`_reconstruct`, `datetime.datetime`; everything else raises. Documented security note. Parquet is the only artifact the pipeline reads afterward.
- **Base config** = `fixed_1_5R` + `off`, same engine, same instrument spec, net — the control on every instrument.

---

## Task 1: Converter + loader + validation gates

**Files:** Create `data/convert_p6.py`, `nqdata/load_p6.py`, `validate_p6.py`; Test `tests/test_convert_p6.py`, `tests/test_load_p6.py`, `tests/test_validate_p6.py`. Modify `requirements.txt` (+`pyarrow==17.0.0`).

**Interfaces:** `convert(src_path, out_parquet) -> dict` (n_in, n_out, dups_dropped); `load_instrument(sym: str) -> pd.DataFrame` (lowercase o/h/l/c + volume=0, tz-aware **US/Eastern** DatetimeIndex, sorted, deduped — same contract as `load_nq`); `validate_instrument(df, sym) -> dict` (Phase-1-style report + `tz_evidence`); `main()` writes `data/validation_report_p6.json` (committed).

- [ ] **Step 1 (tests first):** converter — a synthetic pickle of 5 Candles (one dup timestamp, one out-of-order) round-trips to parquet sorted+deduped; a pickle containing a forbidden global (e.g. `os.system`) raises `UnpicklingError`. Loader — parquet → contract frame (tz-aware ET, columns, volume=0). Validator — flags a synthetic OHLC violation; **tz-inference**: a synthetic week with the quiet hour at 17:00 & reopens at 18:00 passes with `tz_evidence.inferred="ET"`; shifting the pattern 5 hours → raises (hard gate). Per-instrument grid: YM close 0.25 off integer → violation counted.
- [ ] **Step 2:** implement converter (SafeUnpickler per Global Constraints; stream: load→arrays→drop objects→parquet via pyarrow), loader, validator (`n_rows, date_min, date_max, pct_1min_spacing_after_sort, n_ohlc_violations, n_nan, n_dups_dropped, grid_ok, session_days, tz_evidence{quiet_hour, reopen_hour, n_reopens}`).
- [ ] **Step 3:** run the real conversion + validation for ES/NQ/YM (foreground; minutes). Commit `data/validation_report_p6.json` (NOT the parquets — gitignore `data/parquet_p6/`). Sanity: spans ≈ 2015-03→2025-01, 0 violations, quiet_hour==17, reopen==18 for all three.
- [ ] **Step 4:** `pytest tests/ -q` all green. **Commit:** `feat: phase 6 data pipeline (safe converter + ET loader + validation gates)`.

## Task 2: InstrumentSpec + engine threading (regression-locked)

**Files:** Create `strategy/instrument.py`; Modify `backtest/engine.py`, `backtest/costs.py`; Test `tests/test_instrument.py`, `tests/test_engine_p6_regression.py`.

**Interfaces:** `InstrumentSpec(sym, tick_size, tick_value, pt_value)`; `SPECS = {"ES":…, "NQ":…, "YM":…}`; `run_execution(..., spec: InstrumentSpec = SPECS["NQ"])`; `CostModel.net_pnl(..., spec)` uses `spec.tick_value`; engine P&L uses `spec.pt_value`.

- [ ] **Step 1 (tests first):** cost math per instrument (YM stop-loser pays $5+$5+$5; ES stop-loser $5+$12.50+$12.50; NQ unchanged). **Regression lock:** with `SPECS["NQ"]` defaults, (a) the Phase-2 golden fixture reproduces trade-for-trade; (b) a stored Phase-5 spot value (e.g. F4 tuned net OOS PF 2.2480132450 from `phase5_results.json`) reproduces via `walk_forward_p5` on the Phase-1 data — proving the threading changed nothing.
- [ ] **Step 2:** implement; default parameter values keep every existing call site working unchanged.
- [ ] **Step 3:** `pytest tests/ -q` all green. **Commit:** `feat: InstrumentSpec threaded through engine+costs (NQ regression-locked)`.

## Task 3: Fold tiling + H-A/H-B runners

**Files:** Create `tuning/walkforward_p6.py`; Test `tests/test_walkforward_p6.py`.

**Interfaces:** `make_folds_tiled(index) -> list[Fold]` (calendar-half aligned, ≥12mo train, half-open, clipped at edge); `run_HA(df, spec) -> dict` (frozen `walk_forward_p5` over the tiled folds — reuse it, do not reimplement); `run_HB(df, spec, config: StrategyParams) -> dict` (fixed config per fold: compute the p50 train threshold, run test window — **no selector call anywhere**); both return per-fold + stitched net tuned/base + all-combo OOS PFs (H-A only).

- [ ] **Step 1 (tests first):** tiling — synthetic 3-year index → expected fold boundaries (halves), all half-open/disjoint/clipped; a <18mo index → raises (cannot form a fold). **No-selection proof:** monkeypatch `select_params_p5` to raise → `run_HB` still completes (never calls it). Vol-threshold leak test reused (train-only). Stitched-base equals stitched H-B when config==base.
- [ ] **Step 2:** implement as thin wrappers over `walkforward_p5` machinery (`_precompute_p5`, `_slice_layer_p5`, `_vol_threshold`, `_net_metrics`); folds passed in, spec threaded.
- [ ] **Step 3:** `pytest tests/ -q` green. **Commit:** `feat: phase 6 fold tiling + H-A/H-B runners (no-selection proven)`.

## Task 4: Single-shot run + results + charts

**Files:** Create `run_phase6.py`; outputs `phase6_results.json` + `charts/phase6_*.png` (committed). Test `tests/test_smoke_phase6.py`.

- [ ] **Step 1:** runner — for each instrument: `load_instrument` → tiled folds → `run_HA` + `run_HB(B1)` + `run_HB(B2)` + base. Assemble: config_hash (whole frozen design incl. instrument specs + hypotheses + rule constants), git_sha, per-instrument per-hypothesis stitched net metrics + per-fold tables + fold counts + trade counts, indep-NQ pre/post-2023 split, pooled ES+YM records per hypothesis, the **success rule** evaluated for H-A, H-B1, H-B2 (each sub-condition + verdict ∈ {PROVEN, PARTIAL, DISPROVEN} per the spec's definitions), cost sensitivity 0×/1×/2×, run_seconds. Cap PFs at PF_CAP before distribution comparisons.
- [ ] **Step 2:** charts — (1) per-instrument stitched net OOS equity (tuned H-A vs base); (2) verdict matrix (hypothesis × instrument net PF vs base); (3) pooled ES+YM equity + CI annotation; (4) per-fold PF scatter per instrument; (5) cost-sensitivity bars.
- [ ] **Step 3:** smoke test (small grid/short slice/one instrument). Then RUN `run_phase6.py` foreground — expect the heaviest run of the program (3 instruments × ~17 folds × 20 combos for H-A; precompute reuse per instrument makes it tractable; report runtime). Commit results + charts.
- [ ] **Step 4:** `pytest tests/ -q` green. **Commit:** `feat: phase 6 single-shot cross-instrument confirmation run`.

## Task 5: Writeup + notebook + final review/merge

- [ ] **Step 1:** `WRITEUP_PHASE6.md` — lead with the verdict per hypothesis (PROVEN/PARTIAL/DISPROVEN) and the rule's sub-conditions; per-instrument tables; the indep-NQ overlap split; pooled CI; selection-luck context; disclosures (confirmation-not-exploration statement, cost optimism, fold overlap within an instrument, the 2015–2025 regime span, volume-absent note); program epilogue updated (Phases 1–6). Honest either way — a DISPROVEN closes the arc as "the near-miss did not replicate"; a PROVEN must survive every stated gate.
- [ ] **Step 2:** `notebooks/06_cross_instrument_proof.ipynb` (loads the JSON; no re-run) + README Phase-6 section.
- [ ] **Step 3:** final whole-branch review (strongest model) → fix Critical/Important → `pytest` green → merge to master → vault update. **Hold the GitHub push for Alex.**

## Self-review notes
- **Proof integrity:** frozen-procedure reuse (no new degrees of freedom); H-B provably selection-free; success rule symmetric (can prove OR disprove); pre-registered + hashed; indep-NQ overlap disclosed and split.
- **Correctness locks:** NQ-spec regression to Phase-2 golden + a pinned Phase-5 value; converter security allowlist; tz hard gate re-proven on cleaned data; per-instrument grid checks; Phase-4/5 leak guards reused verbatim.
- **Scrutinize in final review:** fold tiling boundaries (calendar halves vs data edge), pooled-CI construction (pool trades, then bootstrap), the H-B no-selection proof, per-instrument cost math, and that H-A truly reuses the frozen `walk_forward_p5` (not a re-implementation).
