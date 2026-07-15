"""Phase-6 charts from the committed phase6_results.json (no re-run)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

r = json.load(open("phase6_results.json", encoding="utf-8"))
c = r["conditions"]
INST = ("ES", "YM", "NQ")

# 1) verdict matrix: tuned vs base R-PF per instrument
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(INST))
tuned = [r["instruments"][s]["tuned_r_pf"] for s in INST]
base = [r["instruments"][s]["base_r_pf"] for s in INST]
ax.bar(x - 0.18, tuned, 0.34, label="Tuned (H-A, frozen Phase-5 procedure)", color="#2F6F6A")
ax.bar(x + 0.18, base, 0.34, label="Base (fixed 1.5R, no filter)", color="#B0473C", alpha=0.75)
ax.axhline(1.0, color="k", lw=1, ls="--")
ax.text(2.45, 1.005, "breakeven", fontsize=8)
ax.set_xticks(x, [f"{s}\n(~10y, 17 folds)" for s in INST])
ax.set_ylabel("net R-multiple profit factor (OOS)")
ax.set_title(f"Phase 6 — cross-instrument confirmation: verdict {r['verdict_H_A']}")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("charts/phase6_verdict_matrix.png", dpi=150); plt.close(fig)

# 2) pooled CI vs breakeven
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.errorbar([c["pooled_r_pf"]], [0],
            xerr=[[c["pooled_r_pf"] - c["pooled_ci_lower"]], [c["pooled_ci_upper"] - c["pooled_r_pf"]]],
            fmt="o", color="#2F6F6A", capsize=6, lw=2, markersize=9)
ax.axvline(1.0, color="#B0473C", lw=1.5, ls="--")
ax.text(1.003, 0.28, "breakeven (PF = 1.0)", color="#B0473C", fontsize=9)
ax.set_yticks([])
ax.set_xlabel("pooled ES+YM net R-PF, day-cluster bootstrap 90% basic CI")
ax.set_title(f"Pooled {c['pooled_r_pf']:.3f}  CI [{c['pooled_ci_lower']:.3f}, {c['pooled_ci_upper']:.3f}] — upper bound < 1.0 ⇒ DISPROVEN")
fig.tight_layout(); fig.savefig("charts/phase6_pooled_ci.png", dpi=150); plt.close(fig)

# 3) per-fold tuned R-PF scatter
fig, ax = plt.subplots(figsize=(10, 4.5))
colors = {"ES": "#2F6F6A", "YM": "#A9791F", "NQ": "#58626F"}
for s in INST:
    table = r["instruments"][s]["per_fold_tuned_r_pf"]
    ks = sorted(table)
    ax.plot(range(len(ks)), [min(table[k], 3.0) for k in ks], "o-", ms=4, lw=0.8,
            label=f"{s} (median {np.median(list(table.values())):.2f})", color=colors[s], alpha=0.8)
ax.axhline(1.0, color="k", lw=1, ls="--")
ax.set_xlabel("fold (half-year, 2016H2 → 2024H2)")
ax.set_ylabel("tuned net R-PF per fold (capped at 3)")
ax.set_title("Per-fold out-of-sample performance — no instrument sustains PF > 1")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("charts/phase6_per_fold.png", dpi=150); plt.close(fig)

# 4) sensitivity cuts
s = r["sensitivity"]
cuts = [("all pooled ES+YM", c["pooled_r_pf"]),
        ("excl. 2023+ (formation era)", s["excluding_formation_era_pooled_r_pf"]),
        ("pre-2020", s["pre_2020_pooled_r_pf"]),
        ("post-2020", s["post_2020_pooled_r_pf"]),
        ("NQ pre-2023", s["nq_pre_2023_r_pf"]),
        ("NQ post-2023", s["nq_post_2023_r_pf"])]
fig, ax = plt.subplots(figsize=(8, 4))
y = np.arange(len(cuts))
vals = [v for _, v in cuts]
ax.barh(y, vals, color=["#2F6F6A" if v >= 1 else "#B0473C" for v in vals], alpha=0.8)
ax.axvline(1.0, color="k", lw=1, ls="--")
ax.set_yticks(y, [k for k, _ in cuts], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("tuned net R-PF")
ax.set_title("Sensitivity cuts — unprofitable in every slice")
fig.tight_layout(); fig.savefig("charts/phase6_sensitivity.png", dpi=150); plt.close(fig)

print("4 charts written to charts/phase6_*.png")
