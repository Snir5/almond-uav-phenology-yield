#!/usr/bin/env python3
"""compare_ebi_v2.py — old mean-of-pixel EBI vs new EBI-of-means (v2).
Shows within-year robustness (Spearman, cultivar F, EBI->yield) and the residual
cross-year GSD dependence. -> Results_Analysis/08_UEF53_Rerun/EBI_v2_Comparison.png"""
import os, sys, csv, math, json
import numpy as np, openpyxl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
BASE = find_thesis_root()
FIG = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun"); os.makedirs(FIG, exist_ok=True)
SC = os.environ.get("SCRATCH_DIR", FIG)
YEARS = [2021, 2022, 2023, 2024]; GSD = {2021: 1.2244, 2022: 1.4721, 2023: 1.3307, 2024: 2.2424}
CCOL = {"53": "#2E75B6", "UEF": "#C0392B"}
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3})


def load(p):
    wb = openpyxl.load_workbook(p, read_only=True, data_only=True); ws = wb.active
    raw = list(ws.iter_rows(values_only=True)); H = raw[0]; M = [dict(zip(H, r)) for r in raw[1:]]; wb.close(); return M


v2 = load(os.path.join(BASE, "4band_mosaic", "Master_Trees_EBIofMeans_v2.xlsx"))
v2by = {r["Zone_Value"]: r for r in v2}
can = load_master()
cult = {r["Zone_Value"]: r.get("cultivar") for r in can}
rows = [r for r in can if r.get("cultivar") in {"UEF", "53"}]
carr = np.array([r.get("cultivar") for r in rows])


def rank(x):
    o = x.argsort(); rk = np.empty(len(x)); rk[o] = np.arange(len(x)); return rk


# within-year Spearman old vs new, and cultivar F old/new
spear = {}; F_old = {}; F_new = {}
for y in YEARS:
    a = np.array([v2by.get(r["Zone_Value"], {}).get(f"EBI_Norm_ofMeans_{y}") for r in rows], float)
    b = np.array([r.get(f"EBI_Norm_{y}") for r in rows], float)
    m = ~np.isnan(a) & ~np.isnan(b)
    spear[y] = float(np.corrcoef(rank(a[m]), rank(b[m]))[0, 1])

    def F(getter):
        e = np.array([getter(r) for r in rows], float)
        A = e[carr == "53"]; B = e[carr == "UEF"]; A = A[~np.isnan(A)]; B = B[~np.isnan(B)]
        na, nb = len(A), len(B); sp2 = ((na - 1) * A.var(ddof=1) + (nb - 1) * B.var(ddof=1)) / (na + nb - 2)
        t = (B.mean() - A.mean()) / math.sqrt(sp2 * (1 / na + 1 / nb)); return t * t
    F_old[y] = F(lambda r: r.get(f"EBI_Norm_{y}"))
    F_new[y] = F(lambda r: v2by.get(r["Zone_Value"], {}).get(f"EBI_ofMeans_{y}"))

# cross-year raw SD and mean (new EBI_ofMeans) + GSD corr
sd_new = []; mean_new = []
for y in YEARS:
    e = np.array([v2by.get(r["Zone_Value"], {}).get(f"EBI_ofMeans_{y}") for r in rows], float); e = e[~np.isnan(e)]
    sd_new.append(e.std(ddof=1)); mean_new.append(e.mean())
g = [GSD[y] for y in YEARS]
gsd_sd_r = float(np.corrcoef(g, sd_new)[0, 1])

# EBI->yield 2023 by cultivar, old vs new
mlat = np.array([r.get("Latitude") for r in can], float); mlon = np.array([r.get("Longitude") for r in can], float)
yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
ymeas = defaultdict(dict); coord = {}
for r in yc: ymeas[r["tree_id"]][int(r["year"])] = float(r["net_kernel_yield_per_tree_kg"]); coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))
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
ry = {"old": {}, "new": {}}
for C in ["UEF", "53"]:
    Eo = []; En = []; Y = []
    for t, j in mp.items():
        cr = can[j]
        if cr.get("cultivar") != C: continue
        vr = v2by.get(cr.get("Zone_Value")); yv = ymeas[t].get(2023)
        eo = cr.get("EBI_Norm_2023"); en = vr.get("EBI_ofMeans_2023") if vr else None
        if yv is not None and eo is not None and en is not None: Eo.append(eo); En.append(en); Y.append(yv)
    ry["old"][C] = float(np.corrcoef(Eo, Y)[0, 1]); ry["new"][C] = float(np.corrcoef(En, Y)[0, 1])

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
a = ax[0]
a.bar([str(y) for y in YEARS], [spear[y] for y in YEARS], color="#16A085")
for i, y in enumerate(YEARS): a.text(i, spear[y] - 0.06, f"{spear[y]:.3f}", ha="center", color="white", fontweight="bold", fontsize=9)
a.set_ylim(0.9, 1.0); a.set_title("A. Within-year tree ranking is unchanged\nSpearman(new EBI-of-means, old EBI) per year", fontweight="bold", fontsize=9.5)
a.set_ylabel("Spearman rho")
a = ax[1]; w = .38; xb = np.arange(4)
a.bar(xb - w / 2, [F_old[y] for y in YEARS], width=w, color="#95a5a6", label="old (mean-of-pixel)")
a.bar(xb + w / 2, [F_new[y] for y in YEARS], width=w, color="#8E44AD", label="new (EBI-of-means)")
a.set_xticks(xb); a.set_xticklabels(YEARS); a.set_title("B. Cultivar effect on EBI: same story\n(UEF vs 53 ANOVA F per year)", fontweight="bold", fontsize=9.5)
a.set_ylabel("F statistic"); a.legend(fontsize=8)
a = ax[2]; xb = np.arange(2)
a.bar(xb - w / 2, [ry["old"]["UEF"], ry["old"]["53"]], width=w, color="#95a5a6", label="old")
a.bar(xb + w / 2, [ry["new"]["UEF"], ry["new"]["53"]], width=w, color="#8E44AD", label="new")
a.axhline(0, color="k", lw=.8); a.set_xticks(xb); a.set_xticklabels(["UEF", "cv 53"])
a.set_title(f"C. EBI->yield 2023 by cultivar: unchanged\n(cross-year SD still GSD-linked r={gsd_sd_r:+.2f})", fontweight="bold", fontsize=9.5)
a.set_ylabel("Pearson r (measured yield)"); a.legend(fontsize=8)
fig.suptitle("EBI-of-means (v2) vs old mean-of-pixel EBI: within-year inference is robust; cross-year heterogeneity still GSD-sensitive", fontweight="bold", fontsize=11.5)
fig.tight_layout()
for d in (FIG, SC): fig.savefig(os.path.join(d, "EBI_v2_Comparison.png"), dpi=140, bbox_inches="tight")
out = {"within_year_spearman": {y: round(spear[y], 4) for y in YEARS},
       "cultivar_F_old": {y: round(F_old[y], 1) for y in YEARS}, "cultivar_F_new": {y: round(F_new[y], 1) for y in YEARS},
       "EBI_yield_2023_old": {k: round(v, 3) for k, v in ry["old"].items()},
       "EBI_yield_2023_new": {k: round(v, 3) for k, v in ry["new"].items()},
       "new_raw_mean_by_year": {y: round(mean_new[i], 4) for i, y in enumerate(YEARS)},
       "new_raw_sd_by_year": {y: round(sd_new[i], 4) for i, y in enumerate(YEARS)},
       "GSD_vs_newSD_r": round(gsd_sd_r, 3)}
json.dump(out, open(os.path.join(FIG, "EBI_v2_Comparison.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
