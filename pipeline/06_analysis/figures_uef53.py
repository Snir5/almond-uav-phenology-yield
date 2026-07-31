#!/usr/bin/env python3
"""
figures_uef53.py - regenerate the key thesis figures restricted to UEF + 53,
Chill Portions only. Reads thesis_results_uef53.json for the headline stats and
the master for the raw distributions. Path-robust via geo_guardrails.
Outputs into Results_Analysis/08_UEF53_Rerun/ (and the scratch dir).
"""
import os, sys, json, math, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master, field_boundary
from stats_utils import r_pvalue

BASE = find_thesis_root()
FIGDIR = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
os.makedirs(FIGDIR, exist_ok=True)
SC = os.environ.get("SCRATCH_DIR", FIGDIR); os.makedirs(SC, exist_ok=True)
R = json.load(open(os.path.join(BASE, "Results_Analysis", "thesis_results_uef53.json")))

KEEP = {"UEF", "53"}; YEARS = [2021, 2022, 2023, 2024]
CCOL = {"53": "#2E75B6", "UEF": "#C0392B"}
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3})

rows = [r for r in load_master() if r.get("cultivar") in KEEP]
cult = np.array([r.get("cultivar") for r in rows])
def col(c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in rows], float)
ebi = {y: col(f"EBI_Norm_{y}") for y in YEARS}
X = col("X_UTM"); Y = col("Y_UTM")


def sv(*keys):
    d = R
    for k in keys: d = d[k]
    return d


# ============ FIG 1: EBI temporal dynamics (UEF+53) ============
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
ax = axes[0]
means = [sv("S2_ebi_dist", str(y), "mean") if str(y) in R["S2_ebi_dist"] else sv("S2_ebi_dist", y, "mean") for y in YEARS]
sds = [sv("S2_ebi_dist", str(y), "sd") if str(y) in R["S2_ebi_dist"] else sv("S2_ebi_dist", y, "sd") for y in YEARS]
ax.errorbar(YEARS, means, yerr=sds, fmt="o-", color="#C0392B", lw=2, ms=9, capsize=5, label="Mean ± SD")
for x, m in zip(YEARS, means): ax.annotate(f"{m:.3f}", (x, m), xytext=(6, 8), textcoords="offset points", fontsize=9)
ax.set_title("A. EBI mean rises modestly (UEF + 53)\nall inter-year shifts p<0.001", fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("EBI_Norm"); ax.set_xticks(YEARS); ax.legend()
ax2 = axes[1]
parts = ax2.violinplot([ebi[y][~np.isnan(ebi[y])] for y in YEARS], positions=YEARS, widths=0.7, showmeans=True)
for pc in parts["bodies"]: pc.set_facecolor("#2E75B6"); pc.set_alpha(.55)
sdtxt = "  ".join(f"{y}:SD={s:.3f}" for y, s in zip(YEARS, sds))
ax2.set_title("B. Distribution tightens over time\n" + sdtxt, fontweight="bold", fontsize=9.5)
ax2.set_xlabel("Year"); ax2.set_ylabel("EBI_Norm"); ax2.set_xticks(YEARS)
fig.suptitle("EBI temporal dynamics - UEF + 53 only (cultivar 54 removed)", fontweight="bold", fontsize=13)
fig.tight_layout()
for d in (FIGDIR, SC): fig.savefig(os.path.join(d, "F1_EBI_Temporal_UEF53.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

# ============ FIG 2: cultivar effects over years ============
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
ax = axes[0]
Fs = [sv("S3_cultivar_anova_ebi", str(y), "F") for y in YEARS]
etas = [sv("S3_cultivar_anova_ebi", str(y), "eta2") for y in YEARS]
b = ax.bar([str(y) for y in YEARS], Fs, color="#8E44AD", alpha=.85)
for rect, f, e in zip(b, Fs, etas):
    ax.text(rect.get_x() + rect.get_width() / 2, f, f"F={f:.0f}\nη²={e:.2f}", ha="center", va="bottom", fontsize=8.5)
ax.set_title("A. Cultivar effect on EBI collapses by 2024\n(UEF vs 53 one-way ANOVA)", fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("F statistic"); ax.set_ylim(0, max(Fs) * 1.25)
ax = axes[1]
for u in ["53", "UEF"]:
    ms = [sv("S3_cultivar_anova_ebi", str(y), "means", u) for y in YEARS]
    ax.plot(YEARS, ms, "o-", color=CCOL[u], lw=2, ms=8, label=u)
ax.set_title("B. Cultivar means converge (rank flips ~2023)", fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("Mean EBI_Norm"); ax.set_xticks(YEARS); ax.legend(title="Cultivar")
fig.suptitle("Cultivar effects on bloom (EBI) - UEF + 53", fontweight="bold", fontsize=13)
fig.tight_layout()
for d in (FIGDIR, SC): fig.savefig(os.path.join(d, "F2_Cultivar_Effects_UEF53.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

# ============ FIG 3: Moran's I + 2023 spatial map ============
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
ax = axes[0]
Is = [sv("S4_moran", str(y), "I") for y in YEARS]
ax.plot(YEARS, Is, "o-", color="#16A085", lw=2, ms=9)
for x, i in zip(YEARS, Is): ax.annotate(f"{i:.3f}**", (x, i), xytext=(6, 8), textcoords="offset points", fontsize=9)
ax.axhline(0, color="k", lw=.7)
ax.set_title("A. Moran's I of EBI (10 m band) - clustering every year, peak 2023", fontweight="bold", fontsize=10)
ax.set_xlabel("Year"); ax.set_ylabel("Moran's I"); ax.set_xticks(YEARS)
ax = axes[1]
v = ebi[2023]; m = ~np.isnan(v) & ~np.isnan(X) & ~np.isnan(Y)
sc = ax.scatter(X[m], Y[m], c=v[m], cmap="RdYlGn", s=14, vmin=np.nanpercentile(v[m], 5), vmax=np.nanpercentile(v[m], 95))
hx, hy = field_boundary(X[m], Y[m]); ax.plot(hx, hy, color="#555", lw=1.1, ls="--", alpha=.6)
plt.colorbar(sc, ax=ax, label="EBI_Norm 2023"); ax.set_aspect("equal")
ax.set_title("B. 2023 EBI spatial pattern (I=0.35)", fontweight="bold"); ax.set_xlabel("X_UTM (m)"); ax.set_ylabel("Y_UTM (m)")
fig.suptitle("Spatial autocorrelation of bloom - UEF + 53", fontweight="bold", fontsize=13)
fig.tight_layout()
for d in (FIGDIR, SC): fig.savefig(os.path.join(d, "F3_Moran_Spatial_UEF53.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

# ============ FIG 4: Chill Portions x EBI x exceedance ============
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
cp = [sv("S5_exceedance", str(y), "chill_portions") for y in YEARS]
ax = axes[0]; ax2 = ax.twinx()
ax.bar([y - .15 for y in YEARS], cp, width=.3, color="#4aa3df", label="Chill Portions")
ax2.plot(YEARS, sds, "o-", color="#e07b4a", lw=2, label="EBI SD")
ax.set_title("A. Chilling (CP) vs EBI heterogeneity", fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("Chill Portions", color="#4aa3df"); ax2.set_ylabel("EBI_Norm SD", color="#e07b4a"); ax.set_xticks(YEARS)
ax = axes[1]
g60 = [sv("S5_exceedance", str(y), "pct_gt_0.60") for y in YEARS]
g65 = [sv("S5_exceedance", str(y), "pct_gt_0.65") for y in YEARS]
ax.scatter(cp, g60, s=80, color="#2E75B6", label="EBI>0.60")
ax.scatter(cp, g65, s=80, color="#e07b4a", label="EBI>0.65")
for x, yy, yr in zip(cp, g60, YEARS): ax.annotate(str(yr), (x, yy), xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.set_title("B. Bloom-bright fraction vs chilling\n(computed from data, UEF+53)", fontweight="bold", fontsize=9.5)
ax.set_xlabel("Chill Portions"); ax.set_ylabel("% trees exceeding"); ax.legend(fontsize=9)
ax = axes[2]
cm = sv("S5_climate_ebi_year", "sd")
items = sorted(cm.items(), key=lambda kv: kv[1]["r"])
names = [k for k, _ in items]; rs = [v["r"] for _, v in items]
ax.barh(names, rs, color=["#C0392B" if x < 0 else "#27AE60" for x in rs])
ax.axvline(0, color="k", lw=.8)
ax.set_title("C. Climate vs EBI SD (r, n=4, directional)", fontweight="bold", fontsize=9.5)
ax.set_xlabel("Pearson r"); ax.tick_params(labelsize=8)
fig.suptitle("Climate (Chill Portions) × EBI heterogeneity - UEF + 53", fontweight="bold", fontsize=13)
fig.tight_layout()
for d in (FIGDIR, SC): fig.savefig(os.path.join(d, "F4_Climate_CP_EBI_UEF53.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

# ============ FIG 5: measured yield x EBI x cultivar ============
yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
ymeas = defaultdict(dict); coord = {}
for r in yc:
    ymeas[r["tree_id"]][int(r["year"])] = float(r["net_kernel_yield_per_tree_kg"]); coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))
mlat = col("Latitude"); mlon = col("Longitude")
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
E = []; Yv = []; CU = []
for t, j in mp.items():
    if ymeas[t].get(2023) is not None and rows[j].get("EBI_Norm_2023") is not None:
        E.append(float(rows[j]["EBI_Norm_2023"])); Yv.append(ymeas[t][2023]); CU.append(rows[j].get("cultivar"))
E = np.array(E); Yv = np.array(Yv); CU = np.array(CU)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
ax = axes[0]
for u in ["53", "UEF"]:
    s = CU == u
    ax.scatter(E[s], Yv[s], s=45, alpha=.75, c=CCOL[u], edgecolors="white", lw=.5, label=u)
    b1, b0 = np.polyfit(E[s], Yv[s], 1); xl = np.linspace(E[s].min(), E[s].max(), 30)
    ax.plot(xl, b1 * xl + b0, color=CCOL[u], lw=1.6)
rr = sv("S6_within_cultivar_slope")
ax.set_title(f"A. Yield×EBI reverses by cultivar\nUEF r={rr['UEF']['r']:+.2f}** · 53 r={rr['53']['r']:+.2f}* · interaction F={sv('S6_ANCOVA_yield2023','EBIxcultivar_interaction','F')}***", fontweight="bold", fontsize=9.5)
ax.set_xlabel("EBI_Norm 2023"); ax.set_ylabel("Measured yield (kg/tree)"); ax.legend(title="Cultivar")
ax = axes[1]
gb = [[Yv[i] for i in range(len(Yv)) if CU[i] == u] for u in ["53", "UEF"]]
bp = ax.boxplot(gb, patch_artist=True, tick_labels=["53", "UEF"], medianprops=dict(color="black", lw=2))
for patch, u in zip(bp["boxes"], ["53", "UEF"]): patch.set_facecolor(CCOL[u]); patch.set_alpha(.7)
ya = sv("S6_yield_cultivar_anova"); yc2 = sv("S6_yield_by_cultivar")
ax.set_title(f"B. Yield by cultivar  F={ya['F']}*** η²={ya['eta2']}\n53={yc2['53']['mean']}kg · UEF={yc2['UEF']['mean']}kg", fontweight="bold", fontsize=9.5)
ax.set_xlabel("Cultivar"); ax.set_ylabel("Measured yield (kg/tree)")
fig.suptitle("Measured (ground-truth) 2023 yield - UEF + 53 (n=151 matched)", fontweight="bold", fontsize=13)
fig.tight_layout()
for d in (FIGDIR, SC): fig.savefig(os.path.join(d, "F5_Measured_Yield_UEF53.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

print("Saved 5 figures to", FIGDIR)
print(os.listdir(FIGDIR))
