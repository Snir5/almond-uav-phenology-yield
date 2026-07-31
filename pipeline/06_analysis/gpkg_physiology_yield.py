#!/usr/bin/env python3
"""
gpkg_physiology_yield.py: uses previously untapped fields from the 2022/2023
Yield_with_clustering GPKGs (Trees_Data_Survey/Field_Data_Yield_GPKGs/), which carry
much more than predicted_Yield (the only field extracted into the master so far).
Seven ground-measured physiological candidates, never analysed before in this
project: SWP_April, Growth_April, SWP_MayJune, Growth_MayJune, CNC_June, SWP_June,
Growth_June (stem water potential and canopy growth at three points in the season;
CNC_June's exact meaning is undocumented anywhere in the project, likely a canopy
nitrogen/nutrient content metric, reported here without over-interpreting the unit).
These are genuinely orthogonal to every RGB/multispectral index tried so far (ground
physiological measurement, not remote sensing), so unlike sections 19-20.5 this is a
real new information source, not another reformulation of the same imagery.

GPKGs have no rasterio dependency (SQLite-based), read with the stdlib sqlite3 and a
minimal inline GeoPackage-WKB centroid parser (no shapely available in the sandbox).
Both GPKGs cover Plots A, B and C; this project's scope is Plot A only (project
convention), so rows are filtered to Plot="A" and spatially matched to the canonical
master (<=3 m, cultivar-verified: 100% cultivar agreement up to 5 m, so 3 m is a safe,
conservative threshold). Matching is validated against the master's own
predicted_Yield_2023 (mean abs diff 0.0015 against the GPKG's predicted_Yield for the
same trees), confirming the spatial join is correct.

Same three-part battery used throughout sections 19-20.5:
  Q1. Same-year, within-cultivar correlation with measured yield, FDR across the 7
      candidates in 2023 (the inferential year, n~85-140 depending on field
      completeness); 2022 (n~9 per cultivar) stays directional.
  Q2. ANCOVA candidate x cultivar interaction on 2023 yield, compared to EBI's own
      (F=11.58).
  Q3. THE decisive question: does the candidate add anything ON TOP OF EBI (nested
      partial-F test), pooled and within each cultivar.
Bonus: does the candidate correlate with EBI itself (does water stress or growth
covary with bloom intensity)?

-> Results_Analysis/08_UEF53_Rerun/gpkg_physiology_yield.json + GPKG_Physiology_Yield.png
"""
import os, sys, csv, math, json, sqlite3, struct
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import r_pvalue, f_pvalue, ols_rss, sig_stars

BASE = find_thesis_root()
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
FIELDS = ["SWP_April", "Growth_April", "SWP_MayJune", "Growth_MayJune", "CNC_June", "SWP_June", "Growth_June"]
GPKG_COLS = ["SWP_April", "Growth__April", "SWP_May.June", "Growth_May.June", "CNC_June", "SWP_June", "Growth_June"]
MATCH_THRESHOLD_M = 3.0


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


# ---------------- GeoPackage geometry -> centroid (no shapely in the sandbox) ----------------
def gpkg_geom_centroid(blob):
    assert blob[0:2] == b"GP", "not a GeoPackage geometry blob"
    flags = blob[3]
    envelope_ind = (flags >> 1) & 0x07
    env_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[envelope_ind]
    wkb = blob[8 + env_bytes:]
    endian = "<" if wkb[0] == 1 else ">"
    pos = 1
    geom_type = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
    base_type = geom_type % 1000
    xs, ys = [], []
    if base_type == 6:  # MULTIPOLYGON
        npoly = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
        for _ in range(npoly):
            pos += 1
            struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4  # sub geom type
            nrings = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
            for r in range(nrings):
                npts = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
                for _p in range(npts):
                    x, y = struct.unpack_from(endian + "dd", wkb, pos); pos += 16
                    if r == 0: xs.append(x); ys.append(y)
    elif base_type == 3:  # POLYGON
        nrings = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
        for r in range(nrings):
            npts = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
            for _p in range(npts):
                x, y = struct.unpack_from(endian + "dd", wkb, pos); pos += 16
                if r == 0: xs.append(x); ys.append(y)
    if not xs: return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def load_gpkg_plotA(year):
    path = os.path.join(GDIR, f"Yield_with_clustering_{year}.gpkg")
    con = sqlite3.connect(path); cur = con.cursor()
    cols_sql = ", ".join(f'"{c}"' for c in GPKG_COLS)
    cur.execute(f'SELECT fid, geom, Plot, cultivar, {cols_sql} FROM "Yield_with_clustering_{year}" WHERE Plot="A"')
    out = []
    for row in cur.fetchall():
        fid, geom, plot, cultivar = row[0], row[1], row[2], row[3]
        c = gpkg_geom_centroid(geom)
        if c is None: continue
        vals = dict(zip(FIELDS, row[4:]))
        out.append({"fid": fid, "x": c[0], "y": c[1], "cultivar": cultivar, **vals})
    con.close()
    return out


M = load_master()
mx = np.array([r.get("X_UTM") for r in M], float)
my = np.array([r.get("Y_UTM") for r in M], float)


def match_to_master(gpkg_rows):
    matched = {}  # tree_idx(j) -> physiology dict
    for r in gpkg_rows:
        d = np.sqrt((mx - r["x"]) ** 2 + (my - r["y"]) ** 2)
        j = int(np.argmin(d))
        if d[j] > MATCH_THRESHOLD_M: continue
        if M[j].get("cultivar") != r["cultivar"]: continue  # extra safety, should already agree
        matched[j] = {f: (float(r[f]) if r[f] is not None else np.nan) for f in FIELDS}
    return matched


g22 = load_gpkg_plotA(2022); g23 = load_gpkg_plotA(2023)
m22 = match_to_master(g22); m23 = match_to_master(g23)
print(f"Plot A rows: 2022={len(g22)} matched={len(m22)} | 2023={len(g23)} matched={len(m23)}")
PHYS = {2022: m22, 2023: m23}

# ---------------- measured yield + spatial match (same convention as other scripts) ----------------
yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
ymeas = defaultdict(dict); coord = {}
for r in yc:
    ymeas[r["tree_id"]][int(r["year"])] = float(r["net_kernel_yield_per_tree_kg"])
    coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))
mlat = np.array([r.get("Latitude") for r in M], float); mlon = np.array([r.get("Longitude") for r in M], float)


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
print(f"yield trees matched to master (<=8m): {len(ymap)}")

cultivar = [r.get("cultivar") for r in M]


def get_ebi(j, yr):
    v = M[j].get(f"EBI_Norm_{yr}")
    return float(v) if v is not None else None


RESULTS = {"note": "Ground-measured physiological candidates from the 2022/2023 Yield_with_clustering "
                   "GPKGs (Plot A only, <=3m matched to master, cultivar-verified), tested against "
                   "measured yield and against EBI. Genuinely orthogonal to every RGB/multispectral "
                   "candidate tried in sections 19-20.5 (ground physiology, not remote sensing).",
           "match_diagnostics": {"2022": {"plotA_rows": len(g22), "matched_to_master": len(m22)},
                                  "2023": {"plotA_rows": len(g23), "matched_to_master": len(m23)}}}

# ---------------- Q0: does the candidate correlate with EBI (bonus) ----------------
q0 = {}
for yr in [2022, 2023]:
    q0[str(yr)] = {}
    for C in ["UEF", "53"]:
        cell = {}
        for f in FIELDS:
            xs, ys = [], []
            for j, d in PHYS[yr].items():
                if cultivar[j] != C or np.isnan(d[f]): continue
                eb = get_ebi(j, yr)
                if eb is None: continue
                xs.append(d[f]); ys.append(eb)
            cell[f] = pearson(xs, ys)
        q0[str(yr)][C] = cell
RESULTS["Q0_candidate_vs_EBI_same_year"] = q0

# ---------------- Q1: same-year correlation vs measured yield ----------------
by_cultivar = {}
for C in ["UEF", "53"]:
    by_cultivar[C] = {}
    for yr in [2022, 2023]:
        cell = {}; pvals = []
        for f in FIELDS:
            xs, ys = [], []
            for t, j in ymap.items():
                if cultivar[j] != C: continue
                d = PHYS[yr].get(j); yv = ymeas[t].get(yr)
                if d is None or yv is None or np.isnan(d[f]): continue
                xs.append(d[f]); ys.append(yv)
            res = pearson(xs, ys); cell[f] = res; pvals.append(res.get("p", np.nan))
        if yr == 2023:
            adj = bh_fdr(pvals)
            for f, a in zip(FIELDS, adj): cell[f]["p_fdr"] = a
        by_cultivar[C][str(yr)] = cell
RESULTS["Q1_same_year_by_cultivar"] = by_cultivar
print("\nQ1: same-year correlations vs yield:"); print(json.dumps(by_cultivar, indent=2, default=str))


# ---------------- Q2: ANCOVA candidate x cultivar interaction, 2023 ----------------
def ancova(field, yr=2023):
    E, Yv, CU = [], [], []
    for t, j in ymap.items():
        d = PHYS[yr].get(j); yv = ymeas[t].get(yr)
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


# ---------------- Q3: does the candidate add anything ON TOP OF EBI? ----------------
def nested_pooled(extra_name, yr=2023):
    E, X, Yv, CU = [], [], [], []
    for t, j in ymap.items():
        d = PHYS[yr].get(j); yv = ymeas[t].get(yr); eb = get_ebi(j, yr)
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
    E, X, Yv = [], [], []
    for t, j in ymap.items():
        if cultivar[j] != C: continue
        d = PHYS[yr].get(j); yv = ymeas[t].get(yr); eb = get_ebi(j, yr)
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
print("\nQ3: does it add value beyond EBI?")
print(json.dumps(RESULTS["Q3_does_it_add_to_EBI_pooled_2023"], indent=2, default=str))
print(json.dumps(RESULTS["Q3_does_it_add_to_EBI_within_cultivar_2023"], indent=2, default=str))

json.dump(RESULTS, open(os.path.join(OUT, "gpkg_physiology_yield.json"), "w"), indent=2, default=str)

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
a = ax[0]
x = np.arange(len(FIELDS)); w = 0.35
for i, (C, color) in enumerate([("UEF", "#2E86C1"), ("53", "#C0392B")]):
    rs = [by_cultivar[C]["2023"][f]["r"] if by_cultivar[C]["2023"][f]["r"] is not None else 0 for f in FIELDS]
    a.bar(x + (i - 0.5) * w, rs, width=w, color=color, edgecolor="k", label=f"cv {C}")
a.axhline(0, color="k", lw=.8); a.set_xticks(x); a.set_xticklabels(FIELDS, rotation=25, ha="right", fontsize=8.5)
a.set_ylabel("Pearson r vs measured yield 2023"); a.legend(fontsize=9); a.grid(alpha=.3)
a.set_title("A. Physiology (SWP, Growth, CNC) vs yield, 2023\n(compare to EBI: UEF r=+0.29, cv-53 r=-0.26)", fontsize=9.5, fontweight="bold")

best_field = max(FIELDS, key=lambda f: abs(by_cultivar["UEF"]["2023"][f]["r"] or 0) + abs(by_cultivar["53"]["2023"][f]["r"] or 0))
for k, f in enumerate([best_field, "CNC_June" if best_field != "CNC_June" else "SWP_June"]):
    a = ax[k + 1]
    for C, color in [("UEF", "#2E86C1"), ("53", "#C0392B")]:
        xs, ys = [], []
        for t, j in ymap.items():
            if cultivar[j] != C: continue
            d = PHYS[2023].get(j); yv = ymeas[t].get(2023)
            if d is None or yv is None or np.isnan(d[f]): continue
            xs.append(d[f]); ys.append(yv)
        a.scatter(xs, ys, s=40, color=color, edgecolors="k", alpha=.75, label=f"cv {C}")
    r_uef = by_cultivar["UEF"]["2023"][f]; r_53 = by_cultivar["53"]["2023"][f]
    a.set_xlabel(f); a.set_ylabel("measured yield 2023 (kg)"); a.legend(fontsize=8)
    a.set_title(f"{'B' if k == 0 else 'C'}. {f}\nUEF r={r_uef['r']}, 53 r={r_53['r']}", fontsize=9.5, fontweight="bold")
    a.grid(alpha=.3)
fig.suptitle("Ground-measured physiology (stem water potential, growth, CNC) vs measured yield", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "GPKG_Physiology_Yield.png"), dpi=140, bbox_inches="tight")
print("\nsaved figure")
