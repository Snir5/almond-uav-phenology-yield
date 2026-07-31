#!/usr/bin/env python3
"""
bloom_fraction_yield.py — test the pixel-level BloomFraction / BrightFraction
feature (from apply_bloom_fraction_v3.py, run separately with rasterio on a
machine that has it) against measured yield, same battery as
color_indices_yield.py (THESIS_PROGRESS_LOG.md section 19), so the two are
directly comparable: same-year, within-cultivar Pearson r, FDR across
candidates in the 2023 inferential test, ANCOVA interaction, and a nested test
of whether BloomFraction adds anything on top of EBI.

Requires 4band_mosaic/Master_Trees_BloomFraction_v3.xlsx to exist (produced by
apply_bloom_fraction_v3.py on a machine with rasterio; this script itself needs
no rasterio and runs fine in the analysis sandbox). If the file is missing,
prints instructions and exits cleanly instead of failing.
-> Results_Analysis/08_UEF53_Rerun/bloom_fraction_yield.json + Bloom_Fraction_Yield.png
"""
import os, sys, csv, math, json
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import r_pvalue, f_pvalue, ols_rss, sig_stars

BASE = find_thesis_root()
V3_PATH = os.path.join(BASE, "4band_mosaic", "Master_Trees_BloomFraction_v3.xlsx")
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
YEARS = [2021, 2022, 2023, 2024]

if not os.path.exists(V3_PATH):
    print(f"NOT FOUND: {V3_PATH}")
    print("This feature needs a pixel-level extraction that requires rasterio, which")
    print("this analysis sandbox does not have. Run this on a machine with rasterio:")
    print("  python3 pipeline/03_ebi_extraction/apply_bloom_fraction_v3.py")
    print("(same workflow already used for Master_Trees_EBIofMeans_v2.xlsx). It reads")
    print("the four Final_Orthomosaic_4Band.tif mosaics directly, so it must run where")
    print("those files and rasterio are both available, then copy the output xlsx back")
    print("into 4band_mosaic/ and re-run this script.")
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


IDX_NAMES = ["BloomFraction", "BrightFraction"]

import openpyxl
wb = openpyxl.load_workbook(V3_PATH, read_only=True, data_only=True)
ws = wb.active; raw = list(ws.iter_rows(values_only=True)); H = raw[0]
v3rows = {int(r[H.index("Zone_Value")]): dict(zip(H, r)) for r in raw[1:] if r[H.index("Zone_Value")] is not None}
wb.close()
print(f"v3 bloom-fraction rows: {len(v3rows)}")

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
    v3 = v3rows.get(int(z)) if z is not None else None
    if v3 is None: continue
    for y in YEARS:
        bf, brf = v3.get(f"BloomFraction_{y}"), v3.get(f"BrightFraction_{y}")
        if bf is None or brf is None: continue
        per_tree[(j, y)] = {"BloomFraction": float(bf), "BrightFraction": float(brf)}

RESULTS = {"note": "Pixel-level BloomFraction/BrightFraction (Otsu-thresholded crown pixel fractions, "
                   "apply_bloom_fraction_v3.py) vs measured yield, same season, within cultivar. "
                   "Compare directly to color_indices_yield.json (crown-mean candidates, all negative)."}
by_cultivar = {}
for C in ["UEF", "53"]:
    by_cultivar[C] = {}
    for yr in [2022, 2023, 2024]:
        vals = {name: ([], []) for name in IDX_NAMES}
        for t, j in mp.items():
            if cultivar[j] != C: continue
            yv = ymeas[t].get(yr); d = per_tree.get((j, yr))
            if yv is None or d is None: continue
            for name in IDX_NAMES:
                vals[name][0].append(d[name]); vals[name][1].append(yv)
        cell = {}; pvals = []
        for name in IDX_NAMES:
            res = pearson(vals[name][0], vals[name][1]); cell[name] = res; pvals.append(res.get("p", np.nan))
        if yr == 2023:
            adj = bh_fdr(pvals)
            for name, a in zip(IDX_NAMES, adj): cell[name]["p_fdr"] = a
        by_cultivar[C][str(yr)] = cell
RESULTS["by_cultivar_same_year"] = by_cultivar


def ancova(name, yr=2023):
    E = []; Yv = []; CU = []
    for t, j in mp.items():
        d = per_tree.get((j, yr)); yv = ymeas[t].get(yr)
        if d is None or yv is None: continue
        E.append(d[name]); Yv.append(yv); CU.append(cultivar[j])
    E = np.array(E); Yv = np.array(Yv); CU = np.array(CU); n = len(Yv)
    def design(cols): return np.column_stack([np.ones(n)] + cols)
    dum = [(CU == "UEF").astype(float)]; inter = [((CU == "UEF").astype(float)) * E]
    rss_full, kf, _ = ols_rss(design([E] + dum), Yv)
    rss_null, kn, _ = ols_rss(design([]), Yv)
    rss_int, ki, _ = ols_rss(design([E] + dum + inter), Yv)
    def Ft(rss_r, dr, rss_f, dfk):
        F = ((rss_r - rss_f) / (dfk - dr)) / (rss_f / (n - dfk))
        return round(float(F), 2), round(float(f_pvalue(F, dfk - dr, n - dfk)), 4)
    Fi, pi = Ft(rss_full, kf, rss_int, ki)
    return {"n": n, "EBIxcultivar_interaction_analog": {"F": Fi, "p": pi, "sig": star(pi)},
            "R2": round(1 - rss_full / rss_null, 3), "R2_with_interaction": round(1 - rss_int / rss_null, 3)}


RESULTS["ANCOVA_2023"] = {name: ancova(name) for name in IDX_NAMES}
RESULTS["ANCOVA_baseline_EBI_reference"] = {"note": "EBIxcultivar interaction F=11.58, p=0.0009, R2 0.079->0.147."}


# ---------- does BloomFraction add anything ON TOP OF EBI? pooled (with interaction) AND within cultivar ----------
def get_ebi(j, yr):
    r = rows[j]; v = r.get(f"EBI_Norm_{yr}")
    return float(v) if v is not None else None


def nested_pooled(extra_name, yr=2023):
    E = []; X = []; Yv = []; CU = []
    for t, j in mp.items():
        d = per_tree.get((j, yr)); yv = ymeas[t].get(yr); eb = get_ebi(j, yr)
        if d is None or yv is None or eb is None: continue
        E.append(eb); X.append(d[extra_name]); Yv.append(yv); CU.append(cultivar[j])
    E = np.array(E); X = np.array(X); Yv = np.array(Yv); CU = np.array(CU); n = len(Yv)
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
        if d is None or yv is None or eb is None: continue
        E.append(eb); X.append(d[extra_name]); Yv.append(yv)
    E = np.array(E); X = np.array(X); Yv = np.array(Yv); n = len(Yv)
    def design(cols): return np.column_stack([np.ones(n)] + cols)
    rss_base, kb, _ = ols_rss(design([E]), Yv)
    rss_full, kf, _ = ols_rss(design([E, X]), Yv)
    rss_null, kn, _ = ols_rss(design([]), Yv)
    F = ((rss_base - rss_full) / (kf - kb)) / (rss_full / (n - kf))
    p = float(f_pvalue(F, kf - kb, n - kf))
    return {"n": n, "R2_EBI_only": round(1 - rss_base / rss_null, 3), "R2_EBI_plus_extra": round(1 - rss_full / rss_null, 3),
            "partial_F_extra_term": round(float(F), 2), "p": round(p, 4), "sig": star(p)}


RESULTS["does_BloomFraction_add_to_EBI_pooled_2023"] = nested_pooled("BloomFraction")
RESULTS["does_BloomFraction_add_to_EBI_within_cultivar_2023"] = {
    C: nested_within_cultivar("BloomFraction", C) for C in ["UEF", "53"]}

json.dump(RESULTS, open(os.path.join(OUT, "bloom_fraction_yield.json"), "w"), indent=2, default=str)
print(json.dumps(RESULTS, indent=2, default=str))
print("\nDone. Compare to color_indices_yield.json for the crown-mean baseline comparison.")

# ---------- figure: r by index/cultivar 2023 + best scatter + nested-model comparison ----------
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
a = ax[0]
x = np.arange(len(IDX_NAMES)); w = 0.35
for i, (C, color) in enumerate([("UEF", "#2E86C1"), ("53", "#C0392B")]):
    rs = [by_cultivar[C]["2023"][n]["r"] if by_cultivar[C]["2023"][n]["r"] is not None else 0 for n in IDX_NAMES]
    a.bar(x + (i - 0.5) * w, rs, width=w, color=color, edgecolor="k", label=f"cv {C}")
a.axhline(0, color="k", lw=.8)
a.set_xticks(x); a.set_xticklabels(IDX_NAMES, rotation=20, ha="right", fontsize=9)
a.set_ylabel("Pearson r vs measured yield 2023"); a.legend(fontsize=9); a.grid(alpha=.3)
a.set_title("A. Pixel-level bloom fraction vs yield, 2023\n(compare to EBI: UEF r=+0.29, cv-53 r=-0.26)", fontsize=10, fontweight="bold")

for k, C in enumerate(["UEF", "53"]):
    a = ax[k + 1]
    xs = []; ys = []
    for t, j in mp.items():
        if cultivar[j] != C: continue
        d = per_tree.get((j, 2023)); yv = ymeas[t].get(2023)
        if d is None or yv is None: continue
        xs.append(d["BloomFraction"]); ys.append(yv)
    a.scatter(xs, ys, s=45, color="#2E86C1" if C == "UEF" else "#C0392B", edgecolors="k", alpha=.8)
    r = by_cultivar[C]["2023"]["BloomFraction"]
    if len(xs) > 2:
        b1, b0 = np.polyfit(xs, ys, 1)
        xr = np.linspace(min(xs), max(xs), 20); a.plot(xr, b0 + b1 * xr, "k--", lw=1.2)
    nadd = RESULTS["does_BloomFraction_add_to_EBI_within_cultivar_2023"][C]
    a.set_xlabel("BloomFraction (Otsu-thresholded)"); a.set_ylabel("measured yield 2023 (kg)")
    a.set_title(f"{'B' if k == 0 else 'C'}. {C}: r={r['r']} (p={r.get('p')})\nadds to EBI? partial F={nadd['partial_F_extra_term']}, p={nadd['p']} ({nadd['sig'] or 'ns'})",
                fontsize=9.5, fontweight="bold")
    a.grid(alpha=.3)
fig.suptitle("Pixel-level bloom fraction (Otsu-thresholded) vs measured yield, 2023", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Bloom_Fraction_Yield.png"), dpi=140, bbox_inches="tight")
print("saved figure")
