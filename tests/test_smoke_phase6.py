"""Phase-6 runner mechanics — unit tests only (NEVER touches Phase-6 data:
the single-shot discipline means no OOS number is observed before the run)."""
import numpy as np
import pytest

from run_phase6 import (
    MARGIN_MIN,
    basic_interval,
    day_cluster_bootstrap,
    evaluate_decision_table,
    r_pf,
)


def _base_conditions(**over) -> dict:
    c = {
        "r_pf_es": 1.2, "r_pf_ym": 1.15, "base_r_pf_es": 0.95, "base_r_pf_ym": 0.9,
        "margin_es": 0.25, "margin_ym": 0.25,
        "pooled_ci_lower": 1.05, "pooled_ci_upper": 1.4,
        "margin_ci_upper": 0.5, "floors_met": True, "per_instrument_floors_met": True,
        "fold_majority_es": True, "fold_majority_ym": True,
        "loo_pooled_pf": 1.1, "loo_margin_es": 0.1, "loo_margin_ym": 0.1,
        "nq_agreement": True,
    }
    c.update(over)
    return c


def test_verdict_proven_when_all_conditions_pass():
    assert evaluate_decision_table(_base_conditions()) == "PROVEN"


def test_verdict_not_proven_when_ci_lower_below_1():
    c = _base_conditions(pooled_ci_lower=0.98)
    assert evaluate_decision_table(c) != "PROVEN"


def test_verdict_disproven_needs_ci_confidence_not_just_a_tie():
    # strictly worse on both + margin CI upper < 0.10 -> DISPROVEN
    c = _base_conditions(r_pf_es=0.8, r_pf_ym=0.7, margin_es=-0.15, margin_ym=-0.2,
                         pooled_ci_lower=0.6, margin_ci_upper=0.05)
    assert evaluate_decision_table(c) == "DISPROVEN"
    # merely tied/slightly worse WITHOUT CI confidence -> INCONCLUSIVE, not DISPROVEN
    c2 = _base_conditions(r_pf_es=0.94, r_pf_ym=0.89, margin_es=-0.01, margin_ym=-0.01,
                          pooled_ci_lower=0.7, pooled_ci_upper=1.3, margin_ci_upper=0.4)
    assert evaluate_decision_table(c2) == "INCONCLUSIVE"


def test_verdict_disproven_by_pooled_upper_below_1():
    # a real CI with upper < 1 necessarily has lower < 1 too (PROVEN can't fire)
    c = _base_conditions(pooled_ci_lower=0.7, pooled_ci_upper=0.97)
    assert evaluate_decision_table(c) == "DISPROVEN"


def test_verdict_partial_one_instrument_only():
    c = _base_conditions(r_pf_ym=0.9, margin_ym=0.0, pooled_ci_lower=0.9)
    assert evaluate_decision_table(c) == "PARTIAL"


def test_verdict_partial_when_nq_disagrees():
    c = _base_conditions(nq_agreement=False)
    assert evaluate_decision_table(c) == "PARTIAL"


def test_verdict_inconclusive_on_floor_failure():
    c = _base_conditions(per_instrument_floors_met=False)
    assert evaluate_decision_table(c) == "INCONCLUSIVE"


def test_no_undefined_branch_random_sweep():
    rng = np.random.default_rng(0)
    for _ in range(500):
        c = _base_conditions(
            r_pf_es=float(rng.uniform(0.3, 2.0)), r_pf_ym=float(rng.uniform(0.3, 2.0)),
            base_r_pf_es=float(rng.uniform(0.3, 2.0)), base_r_pf_ym=float(rng.uniform(0.3, 2.0)),
            margin_es=float(rng.uniform(-1, 1)), margin_ym=float(rng.uniform(-1, 1)),
            pooled_ci_lower=float(rng.uniform(0.3, 1.5)), pooled_ci_upper=float(rng.uniform(0.5, 2.0)),
            margin_ci_upper=float(rng.uniform(-0.5, 1.0)),
            floors_met=bool(rng.random() < 0.8), per_instrument_floors_met=bool(rng.random() < 0.8),
            fold_majority_es=bool(rng.random() < 0.5), fold_majority_ym=bool(rng.random() < 0.5),
            loo_pooled_pf=float(rng.uniform(0.5, 1.5)),
            loo_margin_es=float(rng.uniform(-0.5, 0.5)), loo_margin_ym=float(rng.uniform(-0.5, 0.5)),
            nq_agreement=bool(rng.random() < 0.5),
        )
        assert evaluate_decision_table(c) in {"PROVEN", "DISPROVEN", "PARTIAL", "INCONCLUSIVE"}


def test_day_cluster_bootstrap_deterministic_and_clustered():
    rs_by = {"ES": {"2024-01-02": [1.5, -1.0], "2024-01-03": [0.8]},
             "YM": {"2024-01-02": [1.5], "2024-01-04": [-1.0, 2.0]}}
    b1 = day_cluster_bootstrap(rs_by, r_pf, n_boot=200, seed=1)
    b2 = day_cluster_bootstrap(rs_by, r_pf, n_boot=200, seed=1)
    assert np.array_equal(b1, b2)  # deterministic under fixed seed
    assert len(b1) == 200 and np.isfinite(b1).all()


def test_basic_interval_pivots():
    boot = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    lo, hi = basic_interval(1.0, boot, 5.0, 95.0)
    assert lo < 1.0 < hi
    assert lo == pytest.approx(2 * 1.0 - np.percentile(boot, 95.0))
