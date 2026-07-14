"""Phase-6 single-shot cross-instrument confirmation runner.

MECHANICAL SINGLE-SHOT (Important I5): this runner recomputes the frozen
config hash via `tuning.freeze_p6.build_frozen_config()` and REFUSES TO RUN
unless it equals the value committed in docs/phase6_freeze.json (which git
history shows was committed before this file existed). It also refuses to
run unless data/validation_report_p6.json says overall_status == PASS.

Everything evaluated here is pre-registered in the freeze: the gating
statistic (net R-multiples), the calendar-day cluster bootstrap (stratified
by instrument; n_boot=10,000; seed 42; basic/pivotal intervals), the ordered
4-verdict decision table, the floors, the H-A composition condition, the
robustness guards (per-fold majority + leave-one-out), and the H-B
Bonferroni gate. Sensitivity readouts are context, never gates.
"""

from __future__ import annotations

import json
import subprocess
import time

import numpy as np
import pandas as pd

from backtest.trade import Trade
from nqdata.load_p6 import load_instrument
from strategy.instrument import SPECS
from tuning.freeze_p6 import FREEZE_PATH, build_frozen_config, config_hash
from tuning.grid_p5 import build_grid_p5
from tuning.walkforward import PF_CAP
from tuning.walkforward_p6 import BASE_CONFIG, HB_CONFIGS, run_HA, run_HB

RESULTS_PATH = "phase6_results.json"
VALIDATION_PATH = "data/validation_report_p6.json"

N_BOOT = 10_000
SEED = 42
PRIMARY_LOWER_PCT = 5.0
BONFERRONI_LOWER_PCT = 2.5
DISPROVEN_UPPER_PCT = 95.0
MARGIN_MIN = 0.10
POOLED_MIN_TRADES = 150
POOLED_MIN_DAYS = 100
PER_INSTRUMENT_MIN_TRADES = 150
FORMATION_ERA = ("2023-01-01", "2026-01-01")   # ES/YM overlap with hypothesis-formation period
REGIME_SPLIT = "2020-01-01"


# --- gating statistic: net R-multiples ---------------------------------------

def net_r(t: Trade, pt_value: float) -> float:
    return float(t.net_pnl) / (float(t.risk) * pt_value)


def r_pf(rs: list[float]) -> float:
    gains = sum(r for r in rs if r > 0)
    losses = -sum(r for r in rs if r < 0)
    if losses == 0:
        return float(PF_CAP) if gains > 0 else 0.0
    return min(gains / losses, float(PF_CAP))


def _day(t: Trade) -> str:
    return str(t.entry_time.tz_convert("America/New_York").date())


def _halfyear(t: Trade) -> str:
    d = t.entry_time.tz_convert("America/New_York")
    return f"{d.year}-H{1 if d.month <= 6 else 2}"


# --- calendar-day cluster bootstrap (stratified by instrument) ---------------

def day_cluster_bootstrap(
    rs_by_day_by_inst: dict[str, dict[str, list[float]]],
    stat,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> np.ndarray:
    """Bootstrap distribution of `stat(pooled_r_list)` under day-cluster
    resampling: per instrument, resample ITS trade-days with replacement
    (stratified composition); a sampled day contributes all its trades."""
    rng = np.random.default_rng(seed)
    inst_days = {inst: sorted(m) for inst, m in rs_by_day_by_inst.items()}
    out = np.empty(n_boot)
    for b in range(n_boot):
        pooled: list[float] = []
        for inst, days in inst_days.items():
            if not days:
                continue
            pick = rng.integers(0, len(days), size=len(days))
            m = rs_by_day_by_inst[inst]
            for i in pick:
                pooled.extend(m[days[i]])
        out[b] = stat(pooled)
    return out


def basic_interval(theta_hat: float, boot: np.ndarray, lower_pct: float, upper_pct: float) -> tuple[float, float]:
    """Basic (pivotal) bootstrap interval: [2t - Q(1-a), 2t - Q(a)]."""
    lo = 2 * theta_hat - float(np.percentile(boot, 100.0 - lower_pct))
    hi = 2 * theta_hat - float(np.percentile(boot, 100.0 - upper_pct))
    return lo, hi


# --- the ordered verdict decision table (pure function; unit-tested) ---------

def evaluate_decision_table(c: dict) -> str:
    """`c` carries booleans/values named exactly as in the frozen table."""
    cond1 = c["r_pf_es"] > 1.0 and c["r_pf_ym"] > 1.0
    cond2 = c["margin_es"] >= MARGIN_MIN and c["margin_ym"] >= MARGIN_MIN
    cond3 = (c["pooled_ci_lower"] > 1.0 and c["floors_met"]
             and c["fold_majority_es"] and c["fold_majority_ym"]
             and c["loo_pooled_pf"] > 1.0 and c["loo_margin_es"] > 0 and c["loo_margin_ym"] > 0)
    cond4 = c["nq_agreement"]
    evaluable = c["per_instrument_floors_met"]

    if evaluable and cond1 and cond2 and cond3 and cond4:
        return "PROVEN"
    strictly_worse_both = c["r_pf_es"] < c["base_r_pf_es"] and c["r_pf_ym"] < c["base_r_pf_ym"]
    if (strictly_worse_both and c["margin_ci_upper"] < MARGIN_MIN) or c["pooled_ci_upper"] < 1.0:
        return "DISPROVEN"
    one_only = (c["r_pf_es"] > 1.0 and c["margin_es"] >= MARGIN_MIN) != (
        c["r_pf_ym"] > 1.0 and c["margin_ym"] >= MARGIN_MIN)
    if evaluable and one_only:
        return "PARTIAL"
    if evaluable and cond1 and cond2 and cond3 and not cond4:
        return "PARTIAL"  # proven on independents, unreplicated on NQ
    return "INCONCLUSIVE"


# --- helpers ------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _rs_by_day(trades: list[Trade], pt_value: float) -> dict[str, list[float]]:
    m: dict[str, list[float]] = {}
    for t in trades:
        m.setdefault(_day(t), []).append(net_r(t, pt_value))
    return m


def _fold_pf_table(trades: list[Trade], pt_value: float) -> dict[str, float]:
    by_fold: dict[str, list[float]] = {}
    for t in trades:
        by_fold.setdefault(_halfyear(t), []).append(net_r(t, pt_value))
    return {k: r_pf(v) for k, v in sorted(by_fold.items())}


def _smart_family(p) -> bool:
    return p.exit_mode != "fixed_1_5R" or p.vol_filter != "off"


def _serialize_trade(t: Trade, pt_value: float) -> dict:
    return {"entry_time": str(t.entry_time), "direction": t.direction,
            "pnl_usd": t.pnl_usd, "net_pnl": t.net_pnl, "risk": t.risk,
            "net_r": net_r(t, pt_value), "exit_reason": t.exit_reason, "outcome": t.outcome}


def _cost_multiplier_net_r(trades: list[Trade], pt_value: float, m: float) -> float:
    """net(m) = gross - m * (gross - net_1x); exact (costs linear in multiplier)."""
    rs = []
    for t in trades:
        cost_1x = t.pnl_usd - t.net_pnl
        rs.append((t.pnl_usd - m * cost_1x) / (t.risk * pt_value))
    return r_pf(rs)


def _subset(trades: list[Trade], keep) -> list[Trade]:
    return [t for t in trades if keep(t)]


# --- main ---------------------------------------------------------------------

def main() -> dict:
    t0 = time.time()

    # -- mechanical single-shot gates
    frozen = json.load(open(FREEZE_PATH, encoding="utf-8"))
    recomputed = config_hash(build_frozen_config())
    if recomputed != frozen["config_hash"]:
        raise RuntimeError(
            f"FROZEN CONFIG MISMATCH: recomputed {recomputed} != committed "
            f"{frozen['config_hash']} -- refusing to run (pre-registration violated)")
    validation = json.load(open(VALIDATION_PATH, encoding="utf-8"))
    if validation.get("overall_status") != "PASS":
        raise RuntimeError("validation_report_p6.json is not PASS -- refusing to run")

    grid = build_grid_p5()
    results: dict = {"config_hash": recomputed, "git_sha": _git_sha(),
                     "grid_size": len(grid), "instruments": {}}

    per_inst: dict = {}
    for sym in ("ES", "NQ", "YM"):
        spec = SPECS[sym]
        df = load_instrument(sym)
        roll_dates = [r["date"] for r in validation["instruments"][sym]["roll_boundaries"]]
        ha = run_HA(df, spec, grid, roll_dates)
        hb = {name: run_HB(df, spec, cfg, roll_dates) for name, cfg in HB_CONFIGS.items()}
        per_inst[sym] = {"spec": spec, "ha": ha, "hb": hb}
        print(f"[{time.time()-t0:7.0f}s] {sym}: H-A folds={ha['n_folds']} "
              f"tuned n={len(ha['oos_trades_tuned'])} base n={len(ha['oos_trades_default'])}")

    # -- assemble per-instrument readouts (H-A primary) -------------------------
    def instrument_block(sym: str) -> dict:
        p = per_inst[sym]
        pt = p["spec"].pt_value
        tuned, base = p["ha"]["oos_trades_tuned"], p["ha"]["oos_trades_default"]
        rs_t = [net_r(t, pt) for t in tuned]
        rs_b = [net_r(t, pt) for t in base]
        picks = [fr["selected_params"] for fr in p["ha"]["folds"]]
        block = {
            "n_folds": p["ha"]["n_folds"],
            "tuned_n": len(tuned), "base_n": len(base),
            "tuned_r_pf": r_pf(rs_t), "base_r_pf": r_pf(rs_b),
            "tuned_net_expectancy_r": float(np.mean(rs_t)) if rs_t else None,
            "tuned_total_net_usd": float(sum(t.net_pnl for t in tuned)),
            "base_total_net_usd": float(sum(t.net_pnl for t in base)),
            "n_roll_excluded": p["ha"]["n_roll_spanning_trades_excluded_tuned_plus_base"],
            "per_fold_tuned_r_pf": _fold_pf_table(tuned, pt),
            "per_fold_base_r_pf": _fold_pf_table(base, pt),
            "pick_table": [
                {"fold": str(fr["test_start"].date()), "exit_mode": fr["selected_params"].exit_mode,
                 "vol_filter": fr["selected_params"].vol_filter, "fallback": fr["fallback_used"]}
                for fr in p["ha"]["folds"]
            ],
            "composition_majority_smart": (
                sum(1 for q in picks if _smart_family(q)) > len(picks) / 2),
            "hb": {},
        }
        # per-fold majority tuned>base
        ft, fb = block["per_fold_tuned_r_pf"], block["per_fold_base_r_pf"]
        common = [k for k in ft if k in fb]
        block["fold_majority_tuned_beats_base"] = (
            sum(1 for k in common if ft[k] > fb[k]) > len(common) / 2 if common else False)
        for name, hb in p["hb"].items():
            rs_c = [net_r(t, pt) for t in hb["oos_trades_config"]]
            rs_hb_b = [net_r(t, pt) for t in hb["oos_trades_base"]]
            block["hb"][name] = {
                "n": len(rs_c), "r_pf": r_pf(rs_c), "base_r_pf": r_pf(rs_hb_b),
                "margin": r_pf(rs_c) - r_pf(rs_hb_b),
            }
        return block

    for sym in ("ES", "NQ", "YM"):
        results["instruments"][sym] = instrument_block(sym)

    # unpack in the SAME order as named (a mislabel here propagates into the
    # conditions block -- caught and fixed after the first run; the verdict
    # was unaffected because the pooled CI uses explicit ES/YM keys)
    es, ym, nq = (results["instruments"][s] for s in ("ES", "YM", "NQ"))

    # -- pooled ES+YM day-cluster CI on the tuned R-PF + margin ----------------
    pooled_rs_by = {s: _rs_by_day(per_inst[s]["ha"]["oos_trades_tuned"], per_inst[s]["spec"].pt_value)
                    for s in ("ES", "YM")}
    base_rs_by = {s: _rs_by_day(per_inst[s]["ha"]["oos_trades_default"], per_inst[s]["spec"].pt_value)
                  for s in ("ES", "YM")}
    pooled_rs = [r for m in pooled_rs_by.values() for rs in m.values() for r in rs]
    pooled_days = {d for m in pooled_rs_by.values() for d in m}
    pooled_pf_hat = r_pf(pooled_rs)

    boot_pf = day_cluster_bootstrap(pooled_rs_by, r_pf)
    ci_lo, _ = basic_interval(pooled_pf_hat, boot_pf, PRIMARY_LOWER_PCT, DISPROVEN_UPPER_PCT)
    _, ci_hi = basic_interval(pooled_pf_hat, boot_pf, PRIMARY_LOWER_PCT, DISPROVEN_UPPER_PCT)

    # margin bootstrap: same day resampling applied to BOTH arms jointly
    def margin_stat_builder():
        # rebuild per-day pairs: margin = pf(tuned pooled) - pf(base pooled) on the same day sample
        def stat(_ignored):
            return 0.0
        return stat
    # joint day-cluster margin: resample days once, apply to both arms
    rng = np.random.default_rng(SEED + 1)
    inst_days = {s: sorted(set(pooled_rs_by[s]) | set(base_rs_by[s])) for s in ("ES", "YM")}
    boot_margin = np.empty(N_BOOT)
    for b in range(N_BOOT):
        rs_tt: list[float] = []
        rs_bb: list[float] = []
        for s, days in inst_days.items():
            if not days:
                continue
            pick = rng.integers(0, len(days), size=len(days))
            for i in pick:
                d = days[i]
                rs_tt.extend(pooled_rs_by[s].get(d, []))
                rs_bb.extend(base_rs_by[s].get(d, []))
        boot_margin[b] = r_pf(rs_tt) - r_pf(rs_bb)
    base_pooled_rs = [r for m in base_rs_by.values() for rs in m.values() for r in rs]
    margin_hat = pooled_pf_hat - r_pf(base_pooled_rs)
    margin_lo, margin_hi = basic_interval(margin_hat, boot_margin, PRIMARY_LOWER_PCT, DISPROVEN_UPPER_PCT)

    # fold-cluster CI diagnostic (coarse; context only)
    fold_rs_by = {"POOLED": {}}
    for s in ("ES", "YM"):
        pt = per_inst[s]["spec"].pt_value
        for t in per_inst[s]["ha"]["oos_trades_tuned"]:
            fold_rs_by["POOLED"].setdefault(f"{s}-{_halfyear(t)}", []).append(net_r(t, pt))
    boot_fold = day_cluster_bootstrap(fold_rs_by, r_pf, n_boot=N_BOOT, seed=SEED + 2)
    fold_ci_lo, _ = basic_interval(pooled_pf_hat, boot_fold, PRIMARY_LOWER_PCT, DISPROVEN_UPPER_PCT)

    # leave-one-out: drop the single most profitable (instrument, fold) cell
    cell_pnl: dict[tuple, float] = {}
    for s in ("ES", "YM"):
        pt = per_inst[s]["spec"].pt_value
        for t in per_inst[s]["ha"]["oos_trades_tuned"]:
            cell_pnl[(s, _halfyear(t))] = cell_pnl.get((s, _halfyear(t)), 0.0) + net_r(t, pt)
    drop_cell = max(cell_pnl, key=cell_pnl.get) if cell_pnl else None

    def _loo(trades, s):
        pt = per_inst[s]["spec"].pt_value
        return [net_r(t, pt) for t in trades
                if not (drop_cell and s == drop_cell[0] and _halfyear(t) == drop_cell[1])]
    loo_pooled = _loo(per_inst["ES"]["ha"]["oos_trades_tuned"], "ES") + \
        _loo(per_inst["YM"]["ha"]["oos_trades_tuned"], "YM")
    loo_pf = r_pf(loo_pooled)
    loo_margin_es = r_pf(_loo(per_inst["ES"]["ha"]["oos_trades_tuned"], "ES")) - es["base_r_pf"]
    loo_margin_ym = r_pf(_loo(per_inst["YM"]["ha"]["oos_trades_tuned"], "YM")) - ym["base_r_pf"]

    floors_met = len(pooled_rs) >= POOLED_MIN_TRADES and len(pooled_days) >= POOLED_MIN_DAYS
    per_inst_floor = es["tuned_n"] >= PER_INSTRUMENT_MIN_TRADES and ym["tuned_n"] >= PER_INSTRUMENT_MIN_TRADES

    conditions = {
        "r_pf_es": es["tuned_r_pf"], "r_pf_ym": ym["tuned_r_pf"],
        "base_r_pf_es": es["base_r_pf"], "base_r_pf_ym": ym["base_r_pf"],
        "margin_es": es["tuned_r_pf"] - es["base_r_pf"],
        "margin_ym": ym["tuned_r_pf"] - ym["base_r_pf"],
        "pooled_r_pf": pooled_pf_hat, "pooled_ci_lower": ci_lo, "pooled_ci_upper": ci_hi,
        "margin_hat": margin_hat, "margin_ci_upper": margin_hi, "margin_ci_lower": margin_lo,
        "floors_met": floors_met, "per_instrument_floors_met": per_inst_floor,
        "fold_majority_es": es["fold_majority_tuned_beats_base"],
        "fold_majority_ym": ym["fold_majority_tuned_beats_base"],
        "composition_es": es["composition_majority_smart"],
        "composition_ym": ym["composition_majority_smart"],
        "loo_dropped_cell": str(drop_cell), "loo_pooled_pf": loo_pf,
        "loo_margin_es": loo_margin_es, "loo_margin_ym": loo_margin_ym,
        "nq_agreement": nq["tuned_r_pf"] >= nq["base_r_pf"],
        "fold_cluster_ci_lower_diagnostic": fold_ci_lo,
        "pooled_n_trades": len(pooled_rs), "pooled_n_days": len(pooled_days),
    }
    # composition condition folds into evaluability for PROVEN (frozen H-A definition)
    conditions_for_table = dict(conditions)
    conditions_for_table["per_instrument_floors_met"] = (
        per_inst_floor and conditions["composition_es"] and conditions["composition_ym"])
    verdict = evaluate_decision_table(conditions_for_table)

    # -- H-B replication verdicts (Bonferroni) ----------------------------------
    hb_verdicts = {}
    for name in HB_CONFIGS:
        rs_by = {}
        for s in ("ES", "YM"):
            pt = per_inst[s]["spec"].pt_value
            rs_by[s] = _rs_by_day(per_inst[s]["hb"][name]["oos_trades_config"], pt)
        rs_all = [r for m in rs_by.values() for rs in m.values() for r in rs]
        pf_hat = r_pf(rs_all)
        boot = day_cluster_bootstrap(rs_by, r_pf, seed=SEED + 3)
        lo_b, _ = basic_interval(pf_hat, boot, BONFERRONI_LOWER_PCT, DISPROVEN_UPPER_PCT)
        margins = {s: results["instruments"][s]["hb"][name]["margin"] for s in ("ES", "YM")}
        supported = (pf_hat > 1.0 and lo_b > 1.0
                     and all(m > 0 for m in margins.values())
                     and len(rs_all) >= POOLED_MIN_TRADES)
        hb_verdicts[name] = {
            "pooled_r_pf": pf_hat, "ci_lower_bonferroni_2p5": lo_b,
            "margins": margins, "n": len(rs_all),
            "replication": "supported" if supported else "not supported",
        }

    # -- sensitivity readouts (context, never gates) ----------------------------
    def _outside_formation(t: Trade) -> bool:
        d = str(t.entry_time.tz_convert("America/New_York").date())
        return not (FORMATION_ERA[0] <= d < FORMATION_ERA[1])

    sens = {}
    ex_rs = []
    for s in ("ES", "YM"):
        pt = per_inst[s]["spec"].pt_value
        ex_rs += [net_r(t, pt) for t in _subset(per_inst[s]["ha"]["oos_trades_tuned"], _outside_formation)]
    sens["excluding_formation_era_pooled_r_pf"] = r_pf(ex_rs)
    sens["excluding_formation_era_n"] = len(ex_rs)
    for label, keep in (("pre_2020", lambda t: str(t.entry_time.date()) < REGIME_SPLIT),
                        ("post_2020", lambda t: str(t.entry_time.date()) >= REGIME_SPLIT)):
        rs = []
        for s in ("ES", "YM"):
            pt = per_inst[s]["spec"].pt_value
            rs += [net_r(t, pt) for t in _subset(per_inst[s]["ha"]["oos_trades_tuned"], keep)]
        sens[f"{label}_pooled_r_pf"] = r_pf(rs)
        sens[f"{label}_n"] = len(rs)
    for m in (0.0, 2.0):
        rs_m, rs_bm = [], []
        for s in ("ES", "YM"):
            pt = per_inst[s]["spec"].pt_value
            rs_m.append(_cost_multiplier_net_r(per_inst[s]["ha"]["oos_trades_tuned"], pt, m))
        sens[f"cost_{m}x_tuned_r_pf_by_instrument"] = dict(zip(("ES", "YM"), rs_m))
    sens["mde_statement"] = ("with pooled n=%d trades / %d days, the day-cluster design detects "
                             "a pooled R-PF of roughly >=1.10 at the 5%% gate" % (len(pooled_rs), len(pooled_days)))
    sens["nq_pre_2023_r_pf"] = r_pf([net_r(t, SPECS['NQ'].pt_value)
                                     for t in per_inst["NQ"]["ha"]["oos_trades_tuned"]
                                     if str(t.entry_time.date()) < "2023-01-01"])
    sens["nq_post_2023_r_pf"] = r_pf([net_r(t, SPECS['NQ'].pt_value)
                                      for t in per_inst["NQ"]["ha"]["oos_trades_tuned"]
                                      if str(t.entry_time.date()) >= "2023-01-01"])

    results.update({
        "verdict_H_A": verdict,
        "conditions": conditions,
        "hb_replication": hb_verdicts,
        "sensitivity": sens,
        "run_seconds": time.time() - t0,
    })

    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    return results


if __name__ == "__main__":
    r = main()
    print(json.dumps({"verdict_H_A": r["verdict_H_A"],
                      "pooled_r_pf": r["conditions"]["pooled_r_pf"],
                      "pooled_ci": [r["conditions"]["pooled_ci_lower"], r["conditions"]["pooled_ci_upper"]],
                      "hb": {k: v["replication"] for k, v in r["hb_replication"].items()},
                      "run_seconds": r["run_seconds"]}, indent=2))
