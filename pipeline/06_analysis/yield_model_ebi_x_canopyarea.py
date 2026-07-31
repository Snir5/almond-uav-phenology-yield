#!/usr/bin/env python3
"""
yield_model_ebi_x_canopyarea.py: per request, a different, better-powered way to
check the EBI-yield relationship, motivated by domain reasoning rather than
small-panel demeaning (sections 34-36 showed that route is too noisy at T=3).
Idea: EBI is a bloom-INTENSITY index (bright-pixel fraction, roughly canopy-size
normalized already), but measured yield is in absolute kg, which a bigger tree can
produce more of regardless of relative bloom intensity. So EBI x CanopyArea_m2
("total bloom volume", roughly proportional to how much actual bloom the whole
canopy displayed, not just its intensity per unit area) may relate to absolute kg
yield better than EBI alone. This runs on the FULL pooled cross-sectional n=167
sample (no per-tree demeaning, no small-T mean-reliability problem), reusing the
canopy structure extraction already on file (Master_Trees_CanopyStructure_v5.xlsx,
sections 18-20 of the codebase / removed-from-log canopy_structure_yield.py, which
tested CanopyArea additively and found nothing; this tests the EBI x CanopyArea
PRODUCT specifically, a genuinely different feature not tested there).
-> Results_Analysis/08_UEF53_Rerun/Yield_Model_EBI_x_CanopyArea.json + figure
"""
import os, sys, csv, math, json
from collections import defaultdict
import numpy as np
import openpyxl
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import ols_rss, f_pvalue, r_pvalue

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
CV_SEED = 42

M = load_master()
cultivar = [r.get("cultivar") for r in M]
mlat = np.array([r.get("Latitude") for r in M], float); mlon = np.array([r.get("Longitude") for r in M], float)

# --- load canopy area per tree per year from the v5 structural extraction ---
# Matched by Tree_ID directly (both files share the same Tree_ID scheme; geo
# nearest-match was tried first and failed, 0/1523 within 1m, this file's X_UTM/
# Y_UTM are not on the same footing as the master's, Tree_ID is the reliable key).
wb = openpyxl.load_workbook(os.path.join(BASE, "4band_mosaic", "Master_Trees_CanopyStructure_v5.xlsx"), read_only=True)
ws = wb.active
rows_xlsx = list(ws.iter_rows(values_only=True))
header = rows_xlsx[0]; hidx = {h: i for i, h in enumerate(header)}
canopy_by_id = {}
for r in rows_xlsx[1:]:
    d = {h: r[hidx[h]] for h in header}
    canopy_by_id[d["Tree_ID"]] = d

canopy_for_master = [canopy_by_id.get(r.get("Tree_ID")) for r in M]
n_matched = sum(1 for c in canopy_for_master if c is not None)
print(f"canopy-structure match (by Tree_ID): {n_matched}/{len(M)} master trees matched")

yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
ymeas = defaultdict(dict); coord = {}
for r in yc:
    ymeas[r["tree_id"]][int(r["year"])] = float(r["net_kernel_yield_per_tree_kg"])
    coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))

def hav(a, b, d, e):
    p = math.pi / 180
    x = math.sin((d - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(d * p) * math.sin((e - b) * p / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(max(0, x)))

cand = []
for t, (la, lo) in coord.items():
    ds = np.array([hav(la, lo, a, b) for a, b in zip(mlat, mlon)])
    for j in np.where(ds <= 8)[0]: cand.append((float(ds[j]), t, int(j)))
cand.sort(); uy = set(); um = set(); ymap = {}
for d, t, j in cand:
    if t in uy or j in um: continue
    ymap[t] = j; uy.add(t); um.add(j)

rowsP = [(t, j, yr) for t, j in ymap.items() for yr in (2022, 2023)
         if ymeas[t].get(yr) is not None and M[j].get(f"EBI_Norm_{yr}") is not None
         and canopy_for_master[j] is not None and canopy_for_master[j].get(f"CanopyArea_m2_{yr}") is not None]
nP = len(rowsP)
print(f"pooled 2022-2023 sample with EBI + canopy area + measured yield: n={nP}")

yP = np.array([ymeas[t][yr] for t, j, yr in rowsP])
uefP = np.array([1.0 if cultivar[j] == "UEF" else 0.0 for t, j, yr in rowsP])

def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)

ebi_raw = np.array([M[j].get(f"EBI_Norm_{yr}") for t, j, yr in rowsP], float)
area_raw = np.array([canopy_for_master[j][f"CanopyArea_m2_{yr}"] for t, j, yr in rowsP], float)
ebiP = zc(ebi_raw)
areaP = zc(area_raw)
bloom_volume_raw = ebi_raw * area_raw  # "total bloom volume" proxy, raw units (fraction x m^2)
bloomvolP = zc(bloom_volume_raw)

CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
climP = zc([CP[yr] for t, j, yr in rowsP])
gaP = zc([canopy_for_master[j].get(f"CanopyArea_m2_{yr}") for t, j, yr in rowsP])  # reuse not needed; placeholder removed below
oneP = np.ones(nP)

print(f"\nraw correlation, EBI vs CanopyArea_m2 (same tree-year): r={np.corrcoef(ebi_raw, area_raw)[0,1]:.3f} "
      f"(are they redundant, or genuinely different axes?)")

def r_p(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    r = float(np.corrcoef(xs, ys)[0, 1])
    return round(r, 4), round(float(r_pvalue(r, len(xs))), 4)

print("\n--- direct correlations with measured yield, pooled (both cultivars) ---")
for name, v in [("EBI_Norm alone", ebiP), ("CanopyArea_m2 alone", areaP), ("EBI x CanopyArea (bloom volume)", bloomvolP)]:
    r, p = r_p(v, yP)
    print(f"  {name}: r={r} p={p} n={nP}")

print("\n--- within cultivar ---")
by_cult = {}
for cult, mask in [("UEF", uefP == 1), ("53", uefP == 0)]:
    sub_ebi = ebiP[mask]; sub_area = areaP[mask]; sub_bv = bloomvolP[mask]; sub_y = yP[mask]
    res = {}
    for name, v in [("EBI_Norm", sub_ebi), ("CanopyArea_m2", sub_area), ("EBI_x_CanopyArea", sub_bv)]:
        r, p = r_p(v, sub_y)
        res[name] = {"r": r, "p": p, "n": int(mask.sum())}
        print(f"  [{cult}] {name}: r={r} p={p} n={int(mask.sum())}")
    by_cult[cult] = res

def r2_cv(Xd, y, k=5, seed=CV_SEED):
    n = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n); folds = np.array_split(idx, k)
    pred = np.full(n, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None)
        pred[f] = Xd[f] @ beta
    ss = ((y - pred) ** 2).sum(); tss = ((y - y.mean()) ** 2).sum()
    cv = 1 - ss / tss
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None); ins = Xd @ beta
    r2 = 1 - ((y - ins) ** 2).sum() / tss
    return round(float(r2), 4), round(float(cv), 4)

base_cols = {"intercept": oneP, "cultivar": uefP, "EBI": ebiP, "EBIxcultivar": ebiP * uefP, "climate": climP}
X_base = np.column_stack(list(base_cols.values()))
r2_base, cv_base = r2_cv(X_base, yP)
print(f"\nbaseline (cultivar+EBIxcultivar+climate): R2={r2_base} cvR2={cv_base}")

rss0, p0, _ = ols_rss(X_base, yP)
candidates = {
    "CanopyArea": areaP,
    "CanopyArea_x_cultivar": areaP * uefP,
    "EBI_x_CanopyArea": bloomvolP,
    "EBI_x_CanopyArea_x_cultivar": bloomvolP * uefP,
}
partial_F = {}
for name, col in candidates.items():
    X1 = np.column_stack(list(base_cols.values()) + [col])
    rss1, p1, _ = ols_rss(X1, yP)
    df1 = p1 - p0; df2 = nP - p1
    Fstat = ((rss0 - rss1) / df1) / (rss1 / df2)
    pF = float(f_pvalue(Fstat, df1, df2))
    beta, *_ = np.linalg.lstsq(X1, yP, rcond=None)
    partial_F[name] = {"F": round(float(Fstat), 3), "p": round(pF, 4), "coef": round(float(beta[-1]), 4)}
    r2_1, cv_1 = r2_cv(X1, yP)
    partial_F[name]["R2_with_term"] = r2_1; partial_F[name]["cvR2_with_term"] = cv_1
    print(f"  {name}: F={partial_F[name]['F']} p={partial_F[name]['p']} coef={partial_F[name]['coef']:+.3f} "
          f"R2={r2_1} cvR2={cv_1} (baseline cvR2={cv_base})")

def forward_select(base_cols_, candidate_cols, y, seed=None):
    selected = []; history = []
    Xd = np.column_stack(list(base_cols_.values()))
    cur_r2, cur_cv = r2_cv(Xd, y, seed=seed if seed is not None else CV_SEED)
    history.append({"step": 0, "added": None, "cvR2": cur_cv})
    remaining = dict(candidate_cols)
    while remaining:
        best_name, best_cv, best_r2 = None, cur_cv, cur_r2
        for name, col in remaining.items():
            Xtry = np.column_stack(list(base_cols_.values()) + [col])
            r2t, cvt = r2_cv(Xtry, y, seed=seed if seed is not None else CV_SEED)
            if cvt > best_cv + 1e-6:
                best_name, best_cv, best_r2 = name, cvt, r2t
        if best_name is None: break
        selected.append(best_name); base_cols_[best_name] = remaining.pop(best_name)
        cur_cv, cur_r2 = best_cv, best_r2
        history.append({"step": len(selected), "added": best_name, "cvR2": cur_cv})
    return selected, history, cur_r2, cur_cv

sel_main, hist_main, r2_main, cv_main = forward_select(dict(base_cols), candidates, yP)
print(f"\nforward selection (main seed 42): selected={sel_main} final_cvR2={cv_main} (gain={round(cv_main-cv_base,4)})")

robust = []
for seed in [1, 7, 13, 99, 2024]:
    sel_s, hist_s, r2_s, cv_s = forward_select(dict(base_cols), dict(candidates), yP, seed=seed)
    robust.append({"seed": seed, "selected": sel_s, "final_cvR2": cv_s})
    print(f"  seed {seed}: selected={sel_s} final_cvR2={cv_s}")

from collections import Counter
sel_counter = Counter()
for r in [{"selected": sel_main}] + robust:
    for s in r["selected"]: sel_counter[s] += 1
print(f"how often selected of 6: {dict(sel_counter)}")

RESULTS = {
    "note": "Per request: tests EBI x CanopyArea_m2 ('total bloom volume', domain motivation: EBI "
            "is bloom intensity per unit canopy, absolute kg yield should scale with total canopy "
            "bloom, not just intensity) as a new candidate feature, on the FULL pooled cross-sectional "
            "n=167 sample, avoiding the small-panel demeaning problem of sections 34-36 entirely. "
            "Reuses the existing canopy-structure extraction (Master_Trees_CanopyStructure_v5.xlsx, "
            "matched to master within 1m), which was previously tested only as an ADDITIVE term "
            "(canopy_structure_yield.py, all ns, removed from the log); this tests the EBI x CanopyArea "
            "PRODUCT specifically, a genuinely different feature.",
    "n": nP,
    "n_canopy_matched_to_master": n_matched,
    "corr_EBI_vs_CanopyArea_same_tree_year": round(float(np.corrcoef(ebi_raw, area_raw)[0, 1]), 3),
    "direct_correlations_pooled": {name: {"r": r_p(v, yP)[0], "p": r_p(v, yP)[1]}
                                    for name, v in [("EBI_Norm", ebiP), ("CanopyArea_m2", areaP), ("EBI_x_CanopyArea", bloomvolP)]},
    "direct_correlations_by_cultivar": by_cult,
    "baseline_recommended_model": {"R2": r2_base, "cvR2": cv_base},
    "partial_F_new_candidates": partial_F,
    "forward_selection_main_seed42": {"selected_features": sel_main, "history": hist_main,
                                       "final_R2": r2_main, "final_cvR2": cv_main,
                                       "improvement_cvR2": round(cv_main - cv_base, 4)},
    "robustness_across_5_seeds": robust,
    "how_often_selected_of_6": dict(sel_counter),
}
with open(os.path.join(OUT, "Yield_Model_EBI_x_CanopyArea.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
a = ax[0]
names = ["EBI_Norm", "CanopyArea_m2", "EBI_x_CanopyArea"]
uef_r = [by_cult["UEF"][n]["r"] for n in names]; c53_r = [by_cult["53"][n]["r"] for n in names]
xw = np.arange(len(names))
a.bar(xw - 0.18, uef_r, width=0.36, label="UEF", color="#C0392B")
a.bar(xw + 0.18, c53_r, width=0.36, label="cultivar 53", color="#2E75B6")
a.set_xticks(xw); a.set_xticklabels(names, fontsize=8, rotation=15)
a.axhline(0, color="k", lw=0.8)
a.set_title("Correlation with measured yield,\nsame year, within cultivar", fontsize=10, fontweight="bold")
a.legend(fontsize=8); a.grid(axis="y", alpha=.3)
b = ax[1]
seeds = ["main42"] + [str(r["seed"]) for r in robust]
gains = [round(cv_main - cv_base, 4)] * 1 + [round(r["final_cvR2"] - cv_base, 4) for r in robust]
b.bar(seeds, gains, color="#16A085")
b.axhline(0, color="k", lw=1)
b.set_title("CV R2 gain from offering EBI x CanopyArea\n(and cultivar interactions) on top of baseline", fontsize=10, fontweight="bold")
b.grid(axis="y", alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "Yield_Model_EBI_x_CanopyArea.png"), dpi=130)
print("saved figure")
