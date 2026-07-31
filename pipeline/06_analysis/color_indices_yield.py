#!/usr/bin/env python3
"""
color_indices_yield.py — test whether additional RGB-derived color indices (beyond
EBI and NGRDI) improve the bloom-to-yield relationship, within each cultivar.

Motivation (literature): Chen, Jin & Brown 2019 (ISPRS J. Photogramm. Remote Sens.,
the EBI paper) build EBI as Brightness / (Greenness . Soil-signature) and note in
their Discussion (S5, Fig. 12) that a *customized* index, swapping which channel
combination stands for "flower colour", can outperform a single generic formula for
differently-coloured flowers. Standard RGB vegetation-index literature (Woebbecke
et al. 1995 Excess Green; Gitelson et al. 2002 VARI; Louhaichi et al. 2001 GLI;
Kataoka et al. 2003 CIVE; Bendig et al. 2015 RGBVI) offers several other channel
combinations that were never tried on this data. This script computes them all
from crown-mean R,G,B already extracted per tree per year (no new image processing
needed) and re-runs the same same-year, within-cultivar yield battery already used
for EBI (see THESIS_PROGRESS_LOG.md section 12.1), so the comparison is apples to
apples. Multiple candidate indices -> FDR-controlled within each cultivar's 2023
test (n=64-87, the inferential year); 2022/2024 (n=9) stay directional as before.

Source of crown-mean R,G,B: 4band_mosaic/Master_Trees_EBIofMeans_v2.xlsx (the
EBI-of-means re-extraction, section 13.5 of the log), joined by Zone_Value to the
canonical master (for cultivar) and by nearest-neighbour (<=8 m) to the measured-
yield CSV (same join already used throughout, e.g. hypothesis_tests_uef53.py).
-> Results_Analysis/08_UEF53_Rerun/color_indices_yield.json + Color_Indices_Yield.png
"""
import os, sys, csv, math, json
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import r_pvalue, f_pvalue, ols_rss, sig_stars

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
SC = os.environ.get("SCRATCH_DIR", OUT); os.makedirs(OUT, exist_ok=True)
YEARS = [2021, 2022, 2023, 2024]
EPS = 1e-6


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


# ---------- candidate color indices from crown-mean R, G, B (0-255 scale) ----------
def indices(R, G, B):
    R = np.asarray(R, float); G = np.asarray(G, float); B = np.asarray(B, float)
    mx = np.maximum(np.maximum(R, G), B); mn = np.minimum(np.minimum(R, G), B)
    S = (mx - mn) / (mx + EPS)                       # HSV saturation (low = achromatic/white)
    Bright = R + G + B
    out = {
        "ExG": 2 * G - R - B,                                          # Woebbecke et al. 1995
        "ExGR": (2 * G - R - B) - (1.4 * R - G),                       # Excess Green minus Excess Red
        "VARI": (G - R) / (G + R - B + EPS),                           # Gitelson et al. 2002
        "GLI": (2 * G - R - B) / (2 * G + R + B + EPS),                # Louhaichi et al. 2001
        "CIVE": 0.441 * R - 0.881 * G + 0.385 * B + 18.78745,          # Kataoka et al. 2003
        "RGBVI": (G ** 2 - R * B) / (G ** 2 + R * B + EPS),            # Bendig et al. 2015
        "Saturation_inv": 1 - S,                                       # whiteness (achromatic-ness)
        "WhitenessBrightness": Bright * (1 - S),                       # bright AND achromatic (customized)
    }
    return out


IDX_NAMES = ["ExG", "ExGR", "VARI", "GLI", "CIVE", "RGBVI", "Saturation_inv", "WhitenessBrightness"]

# ---------- load crown-mean R,G,B (v2 extraction) keyed by Zone_Value ----------
import openpyxl
wb = openpyxl.load_workbook(os.path.join(BASE, "4band_mosaic", "Master_Trees_EBIofMeans_v2.xlsx"),
                            read_only=True, data_only=True)
ws = wb.active; raw = list(ws.iter_rows(values_only=True)); H = raw[0]
v2rows = {int(r[H.index("Zone_Value")]): dict(zip(H, r)) for r in raw[1:] if r[H.index("Zone_Value")] is not None}
wb.close()
print(f"v2 crown-mean RGB rows: {len(v2rows)}")

# ---------- load canonical master (UEF+53), for cultivar + coords ----------
KEEP = {"UEF", "53"}
rows = [r for r in load_master() if r.get("cultivar") in KEEP]
mlat = np.array([r.get("Latitude") for r in rows], float)
mlon = np.array([r.get("Longitude") for r in rows], float)
zone = [r.get("Zone_Value") for r in rows]
cultivar = [r.get("cultivar") for r in rows]

# ---------- join measured yield (nearest-neighbour <=8 m, same method as hypothesis_tests_uef53.py) ----------
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

# ---------- compute candidate indices per tree per year from v2 crown-mean RGB ----------
per_tree_idx = {}  # (master_row_idx, year) -> {index_name: value}
for j, z in enumerate(zone):
    v2 = v2rows.get(int(z)) if z is not None else None
    if v2 is None: continue
    for y in YEARS:
        Rv, Gv, Bv = v2.get(f"meanR_{y}"), v2.get(f"meanG_{y}"), v2.get(f"meanB_{y}")
        if Rv is None or Gv is None or Bv is None: continue
        d = indices([Rv], [Gv], [Bv])
        per_tree_idx[(j, y)] = {k: float(v[0]) for k, v in d.items()}

# ---------- battery: same-year color index -> measured yield, within cultivar ----------
RESULTS = {"note": "Candidate RGB color indices (crown-mean R,G,B, v2 extraction) vs measured yield, "
                   "same season, within each cultivar. Baseline EBI/NGRDI results are in section 12.1 "
                   "of THESIS_PROGRESS_LOG.md (not recomputed here). FDR (Benjamini-Hochberg) applied "
                   "across the 8 candidate indices within each cultivar's 2023 test (n=64/87, the "
                   "inferential year); 2022/2024 (n=9) are directional only, as with EBI."}
by_cultivar = {}
for C in ["UEF", "53"]:
    by_cultivar[C] = {}
    for yr in [2022, 2023, 2024]:
        vals = {name: ([], []) for name in IDX_NAMES}
        for t, j in mp.items():
            if cultivar[j] != C: continue
            yv = ymeas[t].get(yr)
            d = per_tree_idx.get((j, yr))
            if yv is None or d is None: continue
            for name in IDX_NAMES:
                vals[name][0].append(d[name]); vals[name][1].append(yv)
        cell = {}
        pvals = []
        for name in IDX_NAMES:
            res = pearson(vals[name][0], vals[name][1])
            cell[name] = res; pvals.append(res.get("p", np.nan))
        if yr == 2023:
            adj = bh_fdr(pvals)
            for name, a in zip(IDX_NAMES, adj): cell[name]["p_fdr"] = a
        by_cultivar[C][str(yr)] = cell
RESULTS["by_cultivar_same_year"] = by_cultivar

# ---------- best candidate per cultivar (largest |r| in 2023) ----------
best = {}
for C in ["UEF", "53"]:
    cell = by_cultivar[C]["2023"]
    ranked = sorted(IDX_NAMES, key=lambda n: -abs(cell[n]["r"]) if cell[n]["r"] is not None else 0)
    best[C] = ranked[0]
RESULTS["best_candidate_2023"] = best

# ---------- ANCOVA index x cultivar interaction, for the best candidate per cultivar + EBI baseline ----------
def ancova(field_vals_by_tree, yr=2023):
    E = []; Yv = []; CU = []
    for t, j in mp.items():
        d = per_tree_idx.get((j, yr)); yv = ymeas[t].get(yr)
        if d is None or yv is None: continue
        E.append(d[field_vals_by_tree]); Yv.append(yv); CU.append(cultivar[j])
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


ANCOVA = {}
tested = set(best.values()) | {"WhitenessBrightness"}
for name in tested:
    ANCOVA[name] = ancova(name)
RESULTS["ANCOVA_candidates_2023"] = ANCOVA
RESULTS["ANCOVA_baseline_EBI_reference"] = {"note": "EBIxcultivar interaction F=11.58, p=0.0009, R2 0.079->0.147 "
                                             "(THESIS_PROGRESS_LOG.md section 7/11), compare to ANCOVA_candidates_2023 above."}

# ---------- does the best candidate add anything ON TOP OF EBI (not instead of)? ----------
# yield ~ EBI_ofMeans*cultivar (base) vs yield ~ EBI_ofMeans*cultivar + WhitenessBrightness*cultivar (full)
def nested_add(extra_name, yr=2023):
    E = []; X = []; Yv = []; CU = []
    for t, j in mp.items():
        d = per_tree_idx.get((j, yr)); yv = ymeas[t].get(yr)
        if d is None or yv is None: continue
        v2 = v2rows.get(int(zone[j]))
        eb = v2.get(f"EBI_Norm_ofMeans_{yr}") if v2 else None
        if eb is None: continue
        E.append(float(eb)); X.append(d[extra_name]); Yv.append(yv); CU.append(cultivar[j])
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
            "partial_F_extra_terms": round(float(F), 2), "p": round(p, 4), "sig": star(p),
            "df": [kf - kb, n - kf]}


RESULTS["does_WhitenessBrightness_add_to_EBI_2023"] = nested_add("WhitenessBrightness")
print("\nEBI + WhitenessBrightness nested test:", json.dumps(RESULTS["does_WhitenessBrightness_add_to_EBI_2023"], indent=2))

json.dump(RESULTS, open(os.path.join(OUT, "color_indices_yield.json"), "w"), indent=2, default=str)
json.dump(RESULTS, open(os.path.join(SC, "color_indices_yield.json"), "w"), indent=2, default=str)
print(json.dumps({"best_candidate_2023": best, "ANCOVA_candidates_2023": ANCOVA}, indent=2, default=str))

# ---------- figure: r by index, by cultivar, 2023 (the inferential year) + best-candidate scatter ----------
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
a = ax[0]
x = np.arange(len(IDX_NAMES)); w = 0.35
for i, (C, color) in enumerate([("UEF", "#2E86C1"), ("53", "#C0392B")]):
    rs = [by_cultivar[C]["2023"][n]["r"] if by_cultivar[C]["2023"][n]["r"] is not None else 0 for n in IDX_NAMES]
    a.bar(x + (i - 0.5) * w, rs, width=w, color=color, edgecolor="k", label=f"cv {C}")
a.axhline(0, color="k", lw=.8)
a.set_xticks(x); a.set_xticklabels(IDX_NAMES, rotation=35, ha="right", fontsize=8)
a.set_ylabel("Pearson r vs measured yield 2023"); a.legend(fontsize=9); a.grid(alpha=.3)
a.set_title("A. Candidate color indices vs yield, 2023\n(compare to EBI: UEF r=+0.29, cv-53 r=-0.26)", fontsize=10, fontweight="bold")

for k, C in enumerate(["UEF", "53"]):
    a = ax[k + 1]
    name = best[C]
    xs = []; ys = []
    for t, j in mp.items():
        if cultivar[j] != C: continue
        d = per_tree_idx.get((j, 2023)); yv = ymeas[t].get(2023)
        if d is None or yv is None: continue
        xs.append(d[name]); ys.append(yv)
    a.scatter(xs, ys, s=45, color="#2E86C1" if C == "UEF" else "#C0392B", edgecolors="k", alpha=.8)
    r = by_cultivar[C]["2023"][name]
    if len(xs) > 2:
        b1, b0 = np.polyfit(xs, ys, 1)
        xr = np.linspace(min(xs), max(xs), 20); a.plot(xr, b0 + b1 * xr, "k--", lw=1.2)
    a.set_xlabel(name); a.set_ylabel("measured yield 2023 (kg)")
    a.set_title(f"{'B' if k == 0 else 'C'}. Best candidate for {C}: {name}\nr={r['r']} (n={r['n']}, p={r.get('p')})",
                fontsize=10, fontweight="bold")
    a.grid(alpha=.3)
fig.suptitle("Additional RGB color indices vs measured yield, same season, within cultivar (2023)", fontweight="bold", fontsize=12)
fig.tight_layout()
for d in (OUT, SC): fig.savefig(os.path.join(d, "Color_Indices_Yield.png"), dpi=140, bbox_inches="tight")
print("saved figure")
