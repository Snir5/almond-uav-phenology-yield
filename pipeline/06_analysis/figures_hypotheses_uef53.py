#!/usr/bin/env python3
"""figures_hypotheses_uef53.py - figures for the H1/H2/H3 + validation + phenology
battery (UEF+53, Chill Portions). Path-robust. -> Results_Analysis/08_UEF53_Rerun/."""
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
R = json.load(open(os.path.join(BASE, "Results_Analysis", "hypothesis_tests_uef53.json")))
YEARS = [2021, 2022, 2023, 2024]; CCOL = {"53": "#2E75B6", "UEF": "#C0392B"}
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3})
rows = [r for r in load_master() if r.get("cultivar") in {"UEF", "53"}]
cult = np.array([r.get("cultivar") for r in rows])
def col(c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in rows], float)
CP = {y: float(np.nanmean(col(f"Chill_Portions_{y}"))) for y in YEARS}


def save(fig, name):
    for d in (FIG, SC): fig.savefig(os.path.join(d, name), dpi=140, bbox_inches="tight")
    plt.close(fig)


def fisher_ci(r, n):
    if n < 4 or abs(r) >= 1: return (np.nan, np.nan)
    z = 0.5 * math.log((1 + r) / (1 - r)); se = 1 / math.sqrt(n - 3)
    lo, hi = z - 1.96 * se, z + 1.96 * se
    return (math.tanh(lo), math.tanh(hi))


# join measured yield
yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
ymeas = defaultdict(dict); coord = {}
for r in yc: ymeas[r["tree_id"]][int(r["year"])] = float(r["net_kernel_yield_per_tree_kg"]); coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))
mlat, mlon = col("Latitude"), col("Longitude")
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

# ============ HFIG1: H1 climate (CP) -> EBI ============
ebi_mean = {y: float(np.nanmean(col(f"EBI_Norm_{y}"))) for y in YEARS}
ebi_sd = {y: float(np.nanstd(col(f"EBI_Norm_{y}"), ddof=1)) for y in YEARS}
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
cpx = [CP[y] for y in YEARS]
a = ax[0]
a.scatter(cpx, [ebi_mean[y] for y in YEARS], s=90, color="#C0392B")
for x, y, yr in zip(cpx, [ebi_mean[y] for y in YEARS], YEARS): a.annotate(str(yr), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
h = R["H1_climate_EBI"]["CP_vs_meanEBI"]
a.set_title(f"A. Chill Portions vs mean EBI\nr={h['r']} (n=4, ns; perm p={h['perm_p']})", fontweight="bold", fontsize=10)
a.set_xlabel("Chill Portions"); a.set_ylabel("Mean EBI_Norm")
a = ax[1]
a.scatter(cpx, [ebi_sd[y] for y in YEARS], s=90, color="#e07b4a")
for x, y, yr in zip(cpx, [ebi_sd[y] for y in YEARS], YEARS): a.annotate(str(yr), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
h2 = R["H1_climate_EBI"]["CP_vs_sdEBI"]
a.set_title(f"B. Chill Portions vs EBI spread (SD)\nr={h2['r']} (n=4, ns)", fontweight="bold", fontsize=10)
a.set_xlabel("Chill Portions"); a.set_ylabel("EBI_Norm SD")
a = ax[2]
ceil = R["H1_climate_EBI"]["between_year_variance_ceiling_eta2"]
a.bar(["Between-year\n(climate ceiling)", "Within-year\n(tree-to-tree)"], [ceil * 100, (1 - ceil) * 100], color=["#4aa3df", "#bbbbbb"])
a.text(0, ceil * 100 + 2, f"{ceil*100:.1f}%", ha="center", fontweight="bold")
a.text(1, (1 - ceil) * 100 - 8, f"{(1-ceil)*100:.1f}%", ha="center", fontweight="bold", color="white")
a.set_title("C. Variance partition of EBI\nClimate can explain at most the between-year slice", fontweight="bold", fontsize=10)
a.set_ylabel("% of total EBI variance"); a.set_ylim(0, 100)
fig.suptitle("H1: Climate (Chill Portions) vs EBI - between-year, n=4, directional only", fontweight="bold", fontsize=13)
fig.tight_layout(); save(fig, "H1_Climate_EBI_UEF53.png")

# ============ HFIG2: predicted vs measured validation ============
fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
for k, yr in enumerate([2023, 2022]):
    a = ax[k]; mv = []; pv = []
    for t, j in mp.items():
        m = ymeas[t].get(yr); p = rows[j].get(f"predicted_Yield_{yr}")
        if m is not None and p is not None: mv.append(m); pv.append(float(p))
    mv = np.array(mv); pv = np.array(pv)
    a.scatter(mv, pv, s=42, alpha=.75, color="#8E44AD", edgecolors="white", lw=.5)
    lim = [0, max(mv.max(), pv.max()) * 1.05]
    a.plot(lim, lim, "k--", lw=1.3, label="1:1 (perfect)")
    v = R["V_predicted_vs_measured"][f"validation_{yr}"]
    a.set_title(f"{yr}: predicted vs measured (n={v['n']})\nr={v['pearson_r']} ns · CCC={v['Lin_CCC']} · RMSE={v['RMSE']} · bias={v['bias_pred_minus_meas']:+.2f}", fontweight="bold", fontsize=9.5)
    a.set_xlabel("Measured yield (kg/tree)"); a.set_ylabel("Predicted (modelled) yield (kg/tree)")
    a.set_xlim(lim); a.set_ylim(lim); a.legend(fontsize=9); a.set_aspect("equal")
fig.suptitle("Validation: modelled 'predicted' yield does NOT track measured yield at tree level", fontweight="bold", fontsize=12.5)
fig.tight_layout(); save(fig, "H_Validation_PredVsMeas_UEF53.png")

# ============ HFIG3: H3 EBI -> yield ============
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
a = ax[0]
labels = [("EBI23→Y23 (same)", "EBI2023_vs_measY2023"), ("EBI22→Y23 (lag)", "EBI2022_lag_measY2023"),
          ("EBI22→Y22 (same)", "EBI2022_vs_measY2022"), ("EBI24→Y24 (same)", "EBI2024_vs_measY2024"),
          ("EBI23→Y24 (lag)", "EBI2023_lag_measY2024"), ("NGRDI23→Y23", "NGRDI2023_vs_measY2023")]
ys = list(range(len(labels)))[::-1]
for y, (lab, key) in zip(ys, labels):
    d = R["H3_EBI_yield"]["measured"][key]; r = d["r"]; n = d["n"]; lo, hi = fisher_ci(r, n)
    a.errorbar(r, y, xerr=[[r - lo], [hi - r]], fmt="o", color="#E67E22", capsize=4, ms=8)
    a.text(hi + .02, y, f"r={r:+.2f} (n={n})", va="center", fontsize=8.5)
a.axvline(0, color="k", lw=.8); a.set_yticks(ys); a.set_yticklabels([l for l, _ in labels], fontsize=9)
a.set_xlim(-.7, .9); a.set_title("A. Pooled bloom→yield: all ns (95% CI)\nno single index-year predicts measured yield", fontweight="bold", fontsize=10)
a.set_xlabel("Pearson r (measured yield)")
a = ax[1]
E = []; Yv = []; CU = []
for t, j in mp.items():
    if ymeas[t].get(2023) is not None and rows[j].get("EBI_Norm_2023") is not None:
        E.append(float(rows[j]["EBI_Norm_2023"])); Yv.append(ymeas[t][2023]); CU.append(rows[j].get("cultivar"))
E = np.array(E); Yv = np.array(Yv); CU = np.array(CU)
for u in ["53", "UEF"]:
    s = CU == u; a.scatter(E[s], Yv[s], s=40, alpha=.75, color=CCOL[u], edgecolors="white", lw=.4, label=u)
    b1, b0 = np.polyfit(E[s], Yv[s], 1); xl = np.linspace(E[s].min(), E[s].max(), 20); a.plot(xl, b1 * xl + b0, color=CCOL[u], lw=1.8)
anc = R["H3_EBI_yield"]["ANCOVA_measured2023"]; wc = R["H3_EBI_yield"]["within_cultivar_measured2023"]
a.set_title(f"B. But the slope REVERSES by cultivar (ANCOVA)\ninteraction F={anc['EBIxcultivar_interaction']['F']}*** · UEF r={wc['UEF']['r']:+.2f}** · 53 r={wc['53']['r']:+.2f}*", fontweight="bold", fontsize=9.5)
a.set_xlabel("EBI_Norm 2023"); a.set_ylabel("Measured yield (kg/tree)"); a.legend(title="Cultivar")
fig.suptitle("H3: EBI ↔ measured yield - pooled null, cultivar-conditional reversal", fontweight="bold", fontsize=12.5)
fig.tight_layout(); save(fig, "H3_EBI_Yield_UEF53.png")

# ============ HFIG4: phenology support ============
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
a = ax[0]; ax2 = a.twinx()
sd = [ebi_sd[y] for y in YEARS]; cv = [R["P_phenology"]["P1_bloom_synchrony"]["EBI_cv_by_year"][str(y)] for y in YEARS]
a.plot(YEARS, sd, "o-", color="#e07b4a", lw=2, label="EBI SD")
ax2.plot(YEARS, cv, "s--", color="#16A085", lw=2, label="EBI CV")
lev = R["P_phenology"]["P1_bloom_synchrony"]["levene_EBI_variance_across_years"]
a.set_title(f"P1. Bloom synchrony rises (spread falls)\nLevene F={lev['F']}*** variances differ across years", fontweight="bold", fontsize=9.5)
a.set_xlabel("Year"); a.set_ylabel("EBI SD", color="#e07b4a"); ax2.set_ylabel("EBI CV", color="#16A085"); a.set_xticks(YEARS)
a = ax[1]
e = col("EBI_Norm_2023"); ng = col("NGRDI_Norm_2023"); m = ~np.isnan(e) & ~np.isnan(ng)
a.scatter(e[m], ng[m], s=10, alpha=.4, color="#5b6cff")
b1, b0 = np.polyfit(e[m], ng[m], 1); xl = np.linspace(e[m].min(), e[m].max(), 20); a.plot(xl, b1 * xl + b0, "k-", lw=1.6)
en = R["P_phenology"]["P2_NGRDI_convergent"]["EBI_vs_NGRDI_2023_alltrees"]
a.set_title(f"P2. Bloom (EBI) vs greenness (NGRDI) 2023\nr={en['r']}*** - indices trade off (flowers vs leaves)", fontweight="bold", fontsize=9.5)
a.set_xlabel("EBI_Norm 2023"); a.set_ylabel("NGRDI_Norm 2023")
a = ax[2]
same = R["P_phenology"]["P3_bloom_yield_timing"]["same_year_2023"]["r"]
lag = R["P_phenology"]["P3_bloom_yield_timing"]["lag_prevbloom_2022to2023"]["r"]
a.bar(["Same-season\nEBI23→Y23", "Prior-season\nEBI22→Y23"], [same, lag], color=["#E67E22", "#95a5a6"])
a.axhline(0, color="k", lw=.8)
a.set_title("P3. Timing: same-season bloom carries\nthe (weak, cultivar-split) yield signal", fontweight="bold", fontsize=9.5)
a.set_ylabel("Pearson r (measured Y2023)")
fig.suptitle("Phenological support - bloom synchrony, index specificity, timing (UEF+53)", fontweight="bold", fontsize=12.5)
fig.tight_layout(); save(fig, "H_Phenology_Support_UEF53.png")

# ============ HFIG5: 20-tree 3-year panel (2022-2024) ============
PAN = R["PANEL_3yr_2022_2024"]
panel = [t for t in ymeas if all(y in ymeas[t] for y in [2022, 2023, 2024]) and t in mp]
Yp = {y: np.array([ymeas[t][y] for t in panel]) for y in [2022, 2023, 2024]}
cup = np.array([rows[mp[t]].get("cultivar") for t in panel])
yrs3 = [2022, 2023, 2024]
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
a = ax[0]
for i, t in enumerate(panel):
    a.plot(yrs3, [Yp[y][i] for y in yrs3], "-", color=CCOL[cup[i]], alpha=.35, lw=1)
for u in ["53", "UEF"]:
    a.plot(yrs3, [Yp[y][cup == u].mean() for y in yrs3], "o-", color=CCOL[u], lw=3, ms=10, label=f"{u} mean")
rm = PAN["RM_ANOVA_year"]
a.set_title(f"A. Same 20 trees, 3 years (spaghetti = tree)\nyear effect F={rm['F']}*** eta^2={rm['eta2_year']}; 2022 was an 'on' year", fontweight="bold", fontsize=9.5)
a.set_xlabel("Year"); a.set_ylabel("Measured yield (kg/tree)"); a.set_xticks(yrs3); a.legend(title="Cultivar")
a = ax[1]
w = .38
a.bar([y - w / 2 for y in yrs3], [PAN["yield_by_year"][str(y)]["mean_53"] for y in yrs3], width=w, color=CCOL["53"], label="53")
a.bar([y + w / 2 for y in yrs3], [PAN["yield_by_year"][str(y)]["mean_UEF"] for y in yrs3], width=w, color=CCOL["UEF"], label="UEF")
for y in yrs3:
    s = PAN["cultivar_by_year"][str(y)]["sig"]
    a.text(y, max(PAN["yield_by_year"][str(y)]["mean_53"], PAN["yield_by_year"][str(y)]["mean_UEF"]) + .15, s, ha="center", fontweight="bold")
a.set_title("B. Cultivar gap concentrates in 2023 (off-year)\n53 nearly fails (1.1 kg) while UEF holds (4.1 kg)", fontweight="bold", fontsize=9.5)
a.set_xlabel("Year"); a.set_ylabel("Mean yield (kg/tree)"); a.set_xticks(yrs3); a.legend(title="Cultivar")
a = ax[2]
rmat = []
for u in ["53", "UEF"]:
    rmat.append([PAN["EBI_yield_by_year_cultivar"][str(y)].get(u, {}).get("r", np.nan) for y in yrs3])
im = a.imshow(rmat, cmap="RdBu_r", vmin=-.6, vmax=.6, aspect="auto")
a.set_xticks(range(3)); a.set_xticklabels(yrs3); a.set_yticks([0, 1]); a.set_yticklabels(["cv 53", "UEF"])
for i in range(2):
    for j in range(3):
        a.text(j, i, f"{rmat[i][j]:+.2f}", ha="center", va="center", fontweight="bold", color="black")
plt.colorbar(im, ax=a, label="EBI→yield r")
a.set_title("C. Within-tree EBI→yield sign recurs\n53 negative (2023-24), UEF positive; n=9/cell", fontweight="bold", fontsize=9.5)
fig.suptitle("3-year yield panel (2022-2024): year effect, cultivar×year gap, recurring EBI-yield reversal", fontweight="bold", fontsize=12)
fig.tight_layout(); save(fig, "H_Panel_3yr_UEF53.png")

print("saved:", [f for f in os.listdir(FIG) if f.startswith(("H1_", "H3_", "H_"))])
