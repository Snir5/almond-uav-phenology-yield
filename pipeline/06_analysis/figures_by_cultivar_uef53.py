#!/usr/bin/env python3
"""figures_by_cultivar_uef53.py - cultivar-stratified figures (UEF, 53), same-year
bloom-yield only, all measured yield years + predicted. -> Results_Analysis/08_UEF53_Rerun/."""
import os, sys, csv, math, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
BASE = find_thesis_root()
FIG = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun"); os.makedirs(FIG, exist_ok=True)
SC = os.environ.get("SCRATCH_DIR", FIG); os.makedirs(SC, exist_ok=True)
R = json.load(open(os.path.join(BASE, "Results_Analysis", "hypothesis_by_cultivar_uef53.json")))
YEARS = [2021, 2022, 2023, 2024]; MEAS = [2022, 2023, 2024]; CCOL = {"53": "#2E75B6", "UEF": "#C0392B"}
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3})
allrows = [r for r in load_master() if r.get("cultivar") in {"UEF", "53"}]
def col(rs, c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in rs], float)
CP = {y: float(np.nanmean(col(allrows, f"Chill_Portions_{y}"))) for y in YEARS}


def save(fig, name):
    for d in (FIG, SC): fig.savefig(os.path.join(d, name), dpi=140, bbox_inches="tight")
    plt.close(fig)


def fci(r, n):
    if n < 4 or abs(r) >= 1: return (np.nan, np.nan)
    z = 0.5 * math.log((1 + r) / (1 - r)); se = 1 / math.sqrt(n - 3)
    return (math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se))


# join measured yield
yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
ymeas = defaultdict(dict); coord = {}
for r in yc: ymeas[r["tree_id"]][int(r["year"])] = float(r["net_kernel_yield_per_tree_kg"]); coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))
mlat, mlon = col(allrows, "Latitude"), col(allrows, "Longitude")
def hav(a, b, d, e):
    p = math.pi / 180; x = math.sin((d - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(d * p) * math.sin((e - b) * p / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(max(0, x)))
cand = []
for t, (la, lo) in coord.items():
    ds = np.array([hav(la, lo, a, b) for a, b in zip(mlat, mlon)])
    for j in np.where(ds <= 8)[0]: cand.append((float(ds[j]), t, int(j)))
cand.sort(); uy = set(); um = set(); mp = {}
for d, t, j in cand:
    if t in uy or j in um: continue
    mp[t] = j; uy.add(t); um.add(j)


def ebi_yield(C, y):
    E = []; Yv = []
    for t, j in mp.items():
        if allrows[j].get("cultivar") != C: continue
        v = ymeas[t].get(y); e = allrows[j].get(f"EBI_Norm_{y}")
        if v is not None and e is not None: E.append(float(e)); Yv.append(float(v))
    return np.array(E), np.array(Yv)


# ============ FIG A: EBI -> yield same-year, BY CULTIVAR (headline) ============
fig = plt.figure(figsize=(16.5, 5.4)); gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1])
# A1 coefficient plot: r by year for each cultivar
ax = fig.add_subplot(gs[0, 0])
off = {"UEF": 0.12, "53": -0.12}
for C in ["UEF", "53"]:
    for y in MEAS:
        d = R[C]["H3_EBI_yield_sameyear"]["measured"][f"EBI{y}_vs_measY{y}"]
        r = d["r"]; n = d["n"]; lo, hi = fci(r, n)
        yy = y + off[C]
        ax.errorbar(r, yy, xerr=[[max(0, r - lo)], [max(0, hi - r)]] if lo == lo else None,
                    fmt="o", color=CCOL[C], capsize=4, ms=9)
        ax.text(r, yy + 0.06, f"{r:+.2f}{'*' if d['sig'] not in ('ns','') else ''} (n={n})", ha="center", fontsize=8)
ax.axvline(0, color="k", lw=.9); ax.set_yticks(MEAS); ax.set_ylim(2021.4, 2024.6)
ax.set_xlim(-1, 1.05); ax.set_xlabel("Same-year EBI→yield  Pearson r")
ax.set_title("A. Bloom→yield slope within each cultivar\nUEF always +, cv-53 flips - (95% CI)", fontweight="bold", fontsize=10)
from matplotlib.lines import Line2D
ax.legend([Line2D([0], [0], color=CCOL["UEF"], marker="o", ls=""), Line2D([0], [0], color=CCOL["53"], marker="o", ls="")],
          ["UEF", "cv 53"], loc="lower right", fontsize=9)
# A2 UEF 2023 scatter, A3 53 2023 scatter
for k, C in enumerate(["UEF", "53"]):
    ax = fig.add_subplot(gs[0, 1 + k])
    E, Yv = ebi_yield(C, 2023)
    ax.scatter(E, Yv, s=42, alpha=.75, color=CCOL[C], edgecolors="white", lw=.5)
    b1, b0 = np.polyfit(E, Yv, 1); xl = np.linspace(E.min(), E.max(), 20); ax.plot(xl, b1 * xl + b0, color=CCOL[C], lw=2)
    d = R[C]["H3_EBI_yield_sameyear"]["measured"]["EBI2023_vs_measY2023"]
    ax.set_title(f"{'B' if k==0 else 'C'}. {C} 2023 (n={d['n']})\nr={d['r']:+.2f} {d['sig']} slope={d['linreg']['slope']:+.1f}", fontweight="bold", fontsize=10)
    ax.set_xlabel("EBI_Norm 2023"); ax.set_ylabel("Measured yield (kg/tree)")
fig.suptitle("EBI → measured yield, same year, computed WITHIN each cultivar (UEF vs 53)", fontweight="bold", fontsize=13)
fig.tight_layout(); save(fig, "C1_EBI_Yield_byCultivar.png")

# ============ FIG B: yield by year + predicted validation, BY CULTIVAR ============
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
a = ax[0]; w = .38
for C, o in [("53", -w / 2), ("UEF", w / 2)]:
    ms = [R[C]["H2_climate_yield"]["measured_by_year"][str(y)]["mean"] for y in MEAS]
    a.bar([y + o for y in MEAS], ms, width=w, color=CCOL[C], label=f"{C} measured")
ax2 = a.twinx(); a.set_zorder(ax2.get_zorder() + 1); a.patch.set_visible(False)
ax2.plot(MEAS, [CP[y] for y in MEAS], "k--o", lw=1.5, label="Chill Portions")
a.set_title("A. Measured yield by year within cultivar\n2022 'on' year; CP dashed (n=3, descriptive)", fontweight="bold", fontsize=10)
a.set_xlabel("Year"); a.set_ylabel("Mean measured yield (kg/tree)"); ax2.set_ylabel("Chill Portions"); a.set_xticks(MEAS); a.legend(loc="upper right", fontsize=9)
a = ax[1]
labels = []; vals = []; cols = []
for C in ["UEF", "53"]:
    for y in [2022, 2023]:
        v = R[C]["V_predicted_vs_measured"][f"validation_{y}"]
        labels.append(f"{C}\n{y}"); vals.append(v.get("pearson_r", np.nan)); cols.append(CCOL[C])
xb = np.arange(len(labels))
a.bar(xb, vals, color=cols, alpha=.85)
for x, v in zip(xb, vals): a.text(x, v + (0.02 if v >= 0 else -0.05), f"{v:+.2f}", ha="center", fontsize=9)
a.axhline(0, color="k", lw=.9); a.set_xticks(xb); a.set_xticklabels(labels, fontsize=9)
a.set_ylim(-0.4, 0.4); a.set_ylabel("predicted vs measured  Pearson r")
a.set_title("B. Model 'predicted' vs measured, within cultivar\nno tree-level skill (r≈0, all ns; no pred 2024)", fontweight="bold", fontsize=10)
fig.suptitle("Yield by year and model validation - within each cultivar", fontweight="bold", fontsize=13)
fig.tight_layout(); save(fig, "C2_Yield_Validation_byCultivar.png")

# ============ FIG C: phenology BY CULTIVAR ============
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
a = ax[0]
for C in ["UEF", "53"]:
    cv = [R[C]["P_phenology"]["EBI_cv_by_year"][str(y)] for y in YEARS]
    a.plot(YEARS, cv, "o-", color=CCOL[C], lw=2, ms=8, label=f"{C} (Levene F={R[C]['P_phenology']['levene_EBI_var_across_years']['F']}***)")
a.set_title("A. Bloom heterogeneity (EBI CV) falls within each cultivar\nboth become more synchronous; cv-53 most", fontweight="bold", fontsize=10)
a.set_xlabel("Year"); a.set_ylabel("EBI CV"); a.set_xticks(YEARS); a.legend(fontsize=9)
a = ax[1]
for C in ["UEF", "53"]:
    e = col([r for r in allrows if r.get("cultivar") == C], "EBI_Norm_2023")
    ng = col([r for r in allrows if r.get("cultivar") == C], "NGRDI_Norm_2023")
    m = ~np.isnan(e) & ~np.isnan(ng)
    a.scatter(e[m], ng[m], s=12, alpha=.4, color=CCOL[C], label=f"{C} r={R[C]['P_phenology']['EBI_vs_NGRDI_2023']['r']}***")
a.set_title("B. Bloom (EBI) vs greenness (NGRDI) 2023, within cultivar\nnegative in both: flower vs leaf trade-off", fontweight="bold", fontsize=10)
a.set_xlabel("EBI_Norm 2023"); a.set_ylabel("NGRDI_Norm 2023"); a.legend(fontsize=9)
fig.suptitle("Phenological support - within each cultivar (UEF, 53)", fontweight="bold", fontsize=13)
fig.tight_layout(); save(fig, "C3_Phenology_byCultivar.png")
print("saved C1/C2/C3")
