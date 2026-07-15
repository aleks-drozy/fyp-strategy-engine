"""Phase-6 pre-registration freeze (Important I5 — git-timestamped, mechanical).

This module is the SINGLE SOURCE OF TRUTH for the frozen Phase-6 design:
`build_frozen_config()` serializes every decision that could flatter the
verdict — grids, fold-tiling rule, hypotheses + hierarchy + composition
condition, the full ordered verdict decision table, every statistical
constant (cluster unit, n_boot, seed, percentiles, floors, margins), the
cost constants per instrument, and the roll/anomaly-exclusion rules —
and `config_hash()` is its SHA-256.

`main()` writes docs/phase6_freeze.json. That file is committed BEFORE
run_phase6.py exists (git history proves the ordering). run_phase6.py
imports THIS builder, recomputes the hash, and REFUSES TO RUN if it does
not equal the committed frozen value — any post-hoc change to any frozen
constant is mechanically detectable and blocks the single shot.
"""

from __future__ import annotations

import hashlib
import json

FREEZE_PATH = "docs/phase6_freeze.json"


def build_frozen_config() -> dict:
    from strategy.instrument import SPECS
    from tuning.grid_p5 import build_grid_p5
    from tuning.walkforward import MIN_IS_TRADES, PF_CAP
    from tuning.walkforward_p6 import (
        BASE_CONFIG, HB_CONFIGS, FOLD_TZ, TRAIN_MONTHS, TEST_MONTHS, MIN_LAST_TEST_MONTHS,
    )

    grid = [
        {"exit_mode": p.exit_mode, "vol_filter": p.vol_filter}
        for p in build_grid_p5()
    ]
    return {
        "phase": 6,
        "purpose": "cross-instrument confirmation of the Phase-5 exit edge; confirmation, not exploration",
        "instruments": {
            s.sym: {"tick_size": s.tick_size, "tick_value": s.tick_value, "pt_value": s.pt_value}
            for s in SPECS.values()
        },
        "independent_set": ["ES", "YM"],
        "replication_readout": "NQ (independent vendor; pre/post-2023 split published, pre = entry < 2023-01-01)",
        "cost_model": {
            "commission_rt": 5.0, "slippage_ticks_entry": 1, "slippage_ticks_exit_market": 1,
            "limit_exits_pay_zero": True, "sensitivity_multipliers": [0.0, 1.0, 2.0],
        },
        "grid_20": grid,
        "selection": {
            "objective": "max net PF among eligible combos; tie -> higher net total_pnl -> lower net max_drawdown",
            "min_is_trades_pre_filter_basis": MIN_IS_TRADES,
            "pf_cap": PF_CAP,
        },
        "fold_tiling_rule": {
            "tz": FOLD_TZ,
            "text": (
                "test starts at every Jan-01/Jul-01 ET boundary t with t >= first_bar + 12mo and "
                "t < last_bar; train = [t - 12mo, t) exactly (never extended); "
                "test = [t, min(t + 6mo, data_end)); final fold formed only if its test span >= 3 months"
            ),
            "train_months": TRAIN_MONTHS, "test_months": TEST_MONTHS,
            "min_last_test_months": MIN_LAST_TEST_MONTHS,
        },
        "hypotheses": {
            "H_A": {
                "role": "SOLE PRIMARY (only route to the program-level PROVEN)",
                "definition": "the frozen Phase-5 walk-forward procedure (walk_forward_p5, verbatim) beats the base config net-of-costs OOS on the independent instruments",
                "composition_condition": (
                    "on each independent instrument, a majority of per-fold selections must be "
                    "smart-exit-family configs (exit_mode != fixed_1_5R OR vol_filter != off); "
                    "pick tables published"
                ),
            },
            "H_B1": {"role": "secondary config-level replication", "exit_mode": "partial_1R", "vol_filter": "p50",
                     "note": str(HB_CONFIGS["B1_partial_1R_p50"])},
            "H_B2": {"role": "secondary config-level replication", "exit_mode": "trail_swing", "vol_filter": "p50",
                     "note": str(HB_CONFIGS["B2_trail_swing_p50"])},
            "H_B_gate": "Bonferroni-adjusted 2.5th-percentile CI lower bound (two configs share the secondary family)",
            "excluded_and_why": "trail_swing+off (Phase-5 F3 pick) lacks the vol-filter component of the hypothesis being confirmed; declared in advance",
            "base_config": str(BASE_CONFIG),
        },
        "gating_statistic": "net R-multiples: net_R = net_pnl / (risk * pt_value); dollar PF descriptive only",
        "ci_method": {
            "kind": "calendar-day cluster bootstrap; a sampled day carries ALL its pooled ES+YM trades; stratified instrument composition",
            "n_boot": 10000, "seed": 42, "interval": "basic (pivotal)",
            "primary_lower_pct": 5.0, "secondary_bonferroni_lower_pct": 2.5, "disproven_upper_pct": 95.0,
        },
        "floors": {
            "pooled_min_trades": 150, "pooled_min_days": 100,
            "per_instrument_min_trades_for_conditions_1_2": 150,
        },
        "margin_min_r_pf": 0.10,
        "verdict_decision_table_ordered": [
            "1. PROVEN iff ALL of: (1) tuned net R-PF > 1.0 on both ES and YM; (2) tuned - base net R-PF >= 0.10 on both; (3) pooled ES+YM day-cluster CI lower (5th pct) > 1.0 AND per-fold majority tuned>base on EACH independent instrument AND leave-one-out (drop single most profitable fold) keeps pooled R-PF > 1.0 with per-instrument margins positive; (4) indep-NQ agreement (tuned >= base net R-PF).",
            "2. else DISPROVEN iff (a) tuned STRICTLY < base net R-PF on both ES and YM AND the pooled tuned-minus-base margin day-cluster CI UPPER bound < 0.10; OR (b) pooled tuned R-PF day-cluster CI UPPER bound (95th pct) < 1.0.",
            "3. else PARTIAL iff (a) conditions 1-2 pass on exactly one independent instrument; or (b) conditions 1-3 pass and condition 4 fails ('proven on independents, unreplicated on NQ').",
            "4. else INCONCLUSIVE (explicit catch-all; no undefined branch).",
        ],
        "exclusion_rules": {
            "roll_spanning_trades": "excluded identically in every arm (entry trading-day < roll date <= exit trading-day); counts disclosed",
            "maintenance_hour": "[17:00, 18:00) ET bars dropped for all instruments (untradable; vendor-contamination window)",
            "bar_label_shift": "+1 minute applied to all three instruments (empirically decisive: 0.905 vs ~0)",
        },
        "sensitivity_readouts_never_gates": [
            "exclude ES/YM test windows overlapping 2023-01..2025-12 (hypothesis-formation era)",
            "pre/post-2020 regime split", "cost multipliers 0x/1x/2x",
            "fold-cluster CI diagnostic alongside the day-cluster gate",
            "published MDE (~pooled R-PF 1.10 at these ns)",
        ],
        "single_shot": "run_phase6.py recomputes this config hash and refuses to run unless it equals the committed frozen value",
    }


def config_hash(cfg: dict | None = None) -> str:
    cfg = cfg if cfg is not None else build_frozen_config()
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> dict:
    cfg = build_frozen_config()
    out = {"config_hash": config_hash(cfg), "frozen_config": cfg}
    with open(FREEZE_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    r = main()
    print(json.dumps({"config_hash": r["config_hash"]}, indent=2))
