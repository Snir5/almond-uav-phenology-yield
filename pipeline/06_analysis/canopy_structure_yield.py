#!/usr/bin/env python3
"""
canopy_structure_yield.py: test canopy AREA and the two light-interception
proxies (CanopyCoverFraction, ShadowFraction) from apply_canopy_structure_v5.py
against measured yield. Companion script; needs no rasterio and runs in the
analysis sandbox once Master_Trees_CanopyStructure_v5.xlsx exists.

Three questions, in order:
  1. Same-year, within-cultivar correlation with measured yield (same battery
     style as sections 12.1/19), FDR across the 3 candidates in 2023.
  2. ANCOVA candidate x cultivar interaction, compared to EBI's own (F=11.58).
  3. THE KEY QUESTION for these features specifically: since canopy structure
     is a genuinely different axis from bloom colour (literature motivation,
     section 20.4), does adding it ON TOP OF EBI improve the model, unlike the
     colour reformulations in section 19 which added nothing? Tested with a
     nested partial-F test, pooled and within each cultivar.

Requires 4band_mosaic/Master_Trees_CanopyStructure_v5.xlsx (produced by
apply_canopy_structure_v5.py on a machine with rasterio+scipy). If missing,
prints instructions and exits cleanly.
-> Results_Analysis/08_UEF53_Rerun/canopy_structure_yield.json + Canopy_Structure_Yield.png
"""
import os, sys, csv, math, json
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import r_pvalue, f_pvalue, ols_rss, sig_stars

BASE = find_thesis_root()
V5_PATH = os.path.join(BASE, "4band_mosaic", "Master_Trees_CanopyStructure_v5.xlsx")
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
YEARS = [2021, 2022, 2023, 2024]
FIELDS = ["CanopyArea_m2", "CanopyCoverFraction", "ShadowFraction"]

if not os.path.exists(V5_PATH):
    print(f"NOT FOUND: {V5_PATH}")
    print("Run this on a machine with rasterio + scipy first:")
    print("  python3 pipeline/03_ebi_extraction/apply_canopy_structure_v5.py")
    print("then copy the output xlsx back into 4band_mosaic/ and re-run this script.")
    sys.exit(0)


def star(p): return sig_stars(p) if p == p else ""


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~np.isnan(x) & ~np.isnan(y); x, y = x[m], y[m]; n = len(x)
    if n < 4: return {"r": None, "n": n}
    r = float(np.corrcoef(x, y)[0, 1]); p = float(r_pvalue(r, n))
    return {"r": round(r, 3), "p": round(p, 4), "n": n, "sig": star(p)}


def bh_fdr(pvals):
    idx = [i for i, p in enumerate(pvals) if p == p]
    ps = sorted((pvals[i], i) for i in idx); m = len(ps)
    adj = {}; prev = 1.0
    for rank in range(m - 1, -1, -1):
        p, i = ps[rank]; val = min(prev, p * m / (rank + 1)); adj[i] = round(val, 4); prev = val
    return [adj.get(i, np.nan) for i in range(len(pvals))]


import openpyxl
wb = openpyxl.load_workbook(V5_PATH, read_only=True, data_only=True)
ws = wb.active; raw = list(ws.iter_rows(values_only=True)); H = raw[0]
v5rows = {int(r[H.index("Zone_Value")]): dict(zip(H, r)) for r in raw[1:] if r[H.index("Zone_Value")] is not None}
wb.close()
print(f"v5 canopy-structure rows: {len(v5rows)}")
buf_r = [v.get("Buffer_Radius_m") for v in v5rows.values() if v.get("Buffer_Radius_m") is not None]
print(f"Buffer radius (m): median={np.median(buf_r):.2f}, min={min(buf_r):.2f}, max={max(buf_r):.2f}")

KEEP = {"UEF", "53"}
rows = [r for r in load_master() if r.get("cultivar") in KEEP]
mlat = np.array([r.get("Latitude") for r in rows], float)
mlon = np.array([r.get("Longitude") for r in rows], float)
zone = [r.get("Zone_Value") for r in rows]
cultivar = [r.get("cultivar") for r in rows]

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
cand.sort(); uy = set(); um = set(); mp = {}
for d, t, j in cand:
    if t in uy or j in um: continue
    mp[t] = j; uy.add(t); um.add(j)
print(f"yield trees matched to master (<=8m): {len(mp)}")

per_tree = {}
for j, z in enumerate(zone):
    v5 = v5rows.get(int(z)) if z is not None else None
    if v5 is None: continue
    for y in YEARS:
        d = {f: v5.get(f"{f}_{y}") for f in FIELDS}
        if all(v is None for v in d.values()): continue
        per_tree[(j, y)] = {k: (float(v) if v is not None else np.nan) for k, v in d.items()}

RESULTS = {"note": "CanopyArea_m2 (structural, colour-independent), CanopyCoverFraction and "
                   "ShadowFraction (light-interception proxies, Zarate-Valdez-style; "
                   "ShadowFraction assumes near-solar-noon acquisition, unverified, more "
                   "experimental) vs measured yield, same season, within cultivar."}

# ---------------- Q1: same-year correlation ----------------
by_cultivar = {}
for C in ["UEF", "53"]:
    by_cultivar[C] = {}
    for yr in [2022, 2023, 2024]:
        cell = {}; pvals = []
        for f in FIELDS:
            xs = []; ys = []
            for t, j in mp.items():
                if cultivar[j] != C: continue
                d = per_tree.get((j, yr)); yv = ymeas[t].get(yr)
                if d is None or yv is None or np.isnan(d[f]): continue
                xs.append(d[f]); ys.append(yv)
            res = pearson(xs, ys); cell[f] = res; pvals.append(res.get("p", np.nan))
        if yr == 2023:
            adj = bh_fdr(pvals)
            for f, a in zip(FIELDS, adj): cell[f]["p_fdr"] = a
        by_cultivar[C][str(yr)] = cell
RESULTS["Q1_same_year_by_cultivar"] = by_cultivar
print("\nQ1: same-year correlations:")
print(json.dumps(by_cultivar, indent=2, default=str))


# ---------------- Q2: ANCOVA candidate x cultivar interaction, 2023 ----------------
def ancova(field, yr=2023):
    E = []; Yv = []; CU = []
    for t, j in mp.items():
        d = per_tree.get((j, yr)); yv = ymeas[t].get(yr)
        if d is None or yv is None or np.isnan(d[field]): continue
        E.append(d[field]); Yv.append(yv); CU.append(cultivar[j])
    E = np.array(E); Yv = np.array(Yv); CU = np.array(CU); n = len(Yv)
    if n < 10: return {"n": n, "note": "too few paired observations"}
    def design(cols): return np.column_stack([np.ones(n)] + cols)
    dum = [(CU == "UEF").astype(float)]; inter = [((CU == "UEF").astype(float)) * E]
    rss_full, kf, _ = ols_rss(design([E] + dum), Yv)
    rss_null, kn, _ = ols_rss(design([]), Yv)
    rss_int, ki, _ = ols_rss(design([E] + dum + inter), Yv)
    def Ft(rss_r, dr, rss_f, dfk):
        F = ((rss_r - rss_f) / (dfk - dr)) / (rss_f / (n - dfk))
        return round(float(F), 2), round(float(f_pvalue(F, dfk - dr, n - dfk)), 4)
    Fi, pi = Ft(rss_full, kf, rss_int, ki)
    return {"n": n, "interaction": {"F": Fi, "p": pi, "sig": star(pi)},
            "R2": round(1 - rss_full / rss_null, 3), "R2_with_interaction": round(1 - rss_int / rss_null, 3)}


RESULTS["Q2_ANCOVA_2023"] = {f: ancova(f) for f in FIELDS}
RESULTS["Q2_ANCOVA_baseline_EBI_reference"] = {"note": "EBIxcultivar interaction F=11.58, p=0.0009, R2 0.079->0.147."}
print("\nQ2: ANCOVA 2023:", json.dumps(RESULTS["Q2_ANCOVA_2023"], indent=2, default=str))


# ---------------- Q3: THE key question, does canopy structure add anything ON TOP OF EBI? ----------------
def get_ebi(j, yr):
    r = rows[j]; v = r.get(f"EBI_Norm_{yr}")
    return float(v) if v is not None else None


def nested_pooled(extra_name, yr=2023):
    E = []; X = []; Yv = []; CU = []
    for t, j in mp.items():
        d = per_tree.get((j, yr)); yv = ymeas[t].get(yr); eb = get_ebi(j, yr)
        if d is None or yv is None or eb is None or np.isnan(d[extra_name]): continue
        E.append(eb); X.append(d[extra_name]); Yv.append(yv); CU.append(cultivar[j])
    E = np.array(E); X = np.array(X); Yv = np.array(Yv); CU = np.array(CU); n = len(Yv)
    if n < 12: return {"n": n, "note": "too few paired observations"}
    isUEF = (CU == "UEF").astype(float)
    def design(cols): return np.column_stack([np.ones(n)] + cols)
    base_cols = [E, isUEF, E * isUEF]
    full_cols = base_cols + [X, X * isUEF]
    rss_base, kb, _ = ols_rss(design(base_cols), Yv)
    rss_full, kf, _ = ols_rss(design(full_cols), Yv)
    rss_null, kn, _ = ols_rss(design([]), Yv)
    F = ((rss_base - rss_full) / (kf - kb)) / (rss_full / (n - kf))
    p = float(f_pvalue(F, kf - kb, n - kf))
    return {"n": n, "R2_EBI_only": round(1 - rss_base / rss_null, 3), "R2_EBI_plus_extra": round(1 - rss_full / rss_null, 3),
            "partial_F_extra_terms": round(float(F), 2), "p": round(p, 4), "sig": star(p), "df": [kf - kb, n - kf]}


def nested_within_cultivar(extra_name, C, yr=2023):
    E = []; X = []; Yv = []
    for t, j in mp.items():
        if cultivar[j] != C: continue
        d = per_tree.get((j, yr)); yv = ymeas[t].get(yr); eb = get_ebi(j, yr)
        if d is None or yv is None or eb is None or np.isnan(d[extra_name]): continue
        E.append(eb); X.append(d[extra_name]); Yv.append(yv)
    E = np.array(E); X = np.array(X); Yv = np.array(Yv); n = len(Yv)
    if n < 8: return {"n": n, "note": "too few paired observations"}
    def design(cols): return np.column_stack([np.ones(n)] + cols)
    rss_base, kb, _ = ols_rss(design([E]), Yv)
    rss_full, kf, _ = ols_rss(design([E, X]), Yv)
    rss_null, kn, _ = ols_rss(design([]), Yv)
    F = ((rss_base - rss_full) / (kf - kb)) / (rss_full / (n - kf))
    p = float(f_pvalue(F, kf - kb, n - kf))
    return {"n": n, "R2_EBI_only": round(1 - rss_base / rss_null, 3), "R2_EBI_plus_extra": round(1 - rss_full / rss_null, 3),
            "partial_F_extra_term": round(float(F), 2), "p": round(p, 4), "sig": star(p)}


RESULTS["Q3_does_it_add_to_EBI_pooled_2023"] = {f: nested_pooled(f) for f in FIELDS}
RESULTS["Q3_does_it_add_to_EBI_within_cultivar_2023"] = {
    f: {C: nested_within_cultivar(f, C) for C in ["UEF", "53"]} for f in FIELDS}
print("\nQ3: does canopy structure add value beyond EBI?")
print(json.dumps(RESULTS["Q3_does_it_add_to_EBI_pooled_2023"], indent=2, default=str))
print(json.dumps(RESULTS["Q3_does_it_add_to_EBI_within_cultivar_2023"], indent=2, default=str))

json.dump(RESULTS, open(os.path.join(OUT, "canopy_structure_yield.json"), "w"), indent=2, default=str)

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
a = ax[0]
x = np.arange(len(FIELDS)); w = 0.35
for i, (C, color) in enumerate([("UEF", "#2E86C1"), ("53", "#C0392B")]):
    rs = [by_cultivar[C]["2023"][f]["r"] if by_cultivar[C]["2023"][f]["r"] is not None else 0 for f in FIELDS]
    a.bar(x + (i - 0.5) * w, rs, width=w, color=color, edgecolor="k", label=f"cv {C}")
a.axhline(0, color="k", lw=.8); a.set_xticks(x); a.set_xticklabels(FIELDS, rotation=20, ha="right", fontsize=9)
a.set_ylabel("Pearson r vs measured yield 2023"); a.legend(fontsize=9); a.grid(alpha=.3)
a.set_title("A. Canopy structure vs yield, 2023\n(compare to EBI: UEF r=+0.29, cv-53 r=-0.26)", fontsize=10, fontweight="bold")

for k, f in enumerate(["CanopyArea_m2", "CanopyCoverFraction"]):
    a = ax[k + 1]
    for C, color in [("UEF", "#2E86C1"), ("53", "#C0392B")]:
        xs = []; ys = []
        for t, j in mp.items():
            if cultivar[j] != C: continue
            d = per_tree.get((j, 2023)); yv = ymeas[t].get(2023)
            if d is None or yv is None or np.isnan(d[f]): continue
            xs.append(d[f]); ys.append(yv)
        a.scatter(xs, ys, s=40, color=color, edgecolors="k", alpha=.75, label=f"cv {C}")
    r_uef = by_cultivar["UEF"]["2023"][f]; r_53 = by_cultivar["53"]["2023"][f]
    add_uef = RESULTS["Q3_does_it_add_to_EBI_within_cultivar_2023"][f]["UEF"]
    add_53 = RESULTS["Q3_does_it_add_to_EBI_within_cultivar_2023"][f]["53"]
    a.set_xlabel(f); a.set_ylabel("measured yield 2023 (kg)"); a.legend(fontsize=8)
    a.set_title(f"{'B' if k == 0 else 'C'}. {f}\nUEF r={r_uef['r']}, 53 r={r_53['r']}", fontsize=9.5, fontweight="bold")
    a.grid(alpha=.3)
fig.suptitle("Canopy area and light-interception proxies vs measured yield (structural, not colour)", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Canopy_Structure_Yield.png"), dpi=140, bbox_inches="tight")
print("\nsaved figure")
