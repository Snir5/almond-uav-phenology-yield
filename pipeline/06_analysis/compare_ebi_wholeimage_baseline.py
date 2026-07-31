#!/usr/bin/env python3
"""
compare_ebi_wholeimage_baseline.py: sandbox-safe (no rasterio needed). Run
AFTER apply_fixed_zones_yearly_v3_wholeimage_baseline.py has produced
4band_mosaic/Master_Trees_WholeImageBaseline_v3.xlsx on your machine.

Compares three variants, keeping the canonical results untouched throughout:
  1. CANONICAL: EBI_Raw/Norm/Z_{year} in the canonical master (canopy-only
     radiometric baseline, the pipeline used everywhere in this thesis).
  2. WHOLEIMG: EBI_Raw/Norm/Z_wholeimg_{year} in the new v3 file (whole-image
     radiometric baseline, the Oren-style scope).
  3. COMBINED: the simple average of the two (z-scored consistently first),
     to test whether ensembling the two baselines gains anything neither
     has alone, per the user's request ("maybe they will gain something
     separate or together").

Three checks, in order of how much they matter for the thesis:
  A. Within-year structure: does cultivar still explain EBI the same way
     under each variant (rank correlation + cultivar ANOVA F, per year)?
     This is the section-13.5-style check that a change to the correction
     doesn't break the sections-11-12 findings.
  B. Cross-year trend: does the "mean flat/rises, SD halves" trajectory
     (the disputed part, per section 13.6) look different under each
     variant?
  C. Yield-model value: swapping each variant in as the pooled 2022-2023
     yield model's bloom term (cultivar+EBIxcultivar+climate+Growth_April
     base, same pipeline as sections 22/28/29), does CV R^2 change?

-> Results_Analysis/08_UEF53_Rerun/EBI_WholeImageBaseline_Comparison.json + figure.
"""
import os, sys, csv, math, json, sqlite3, struct
from collections import defaultdict
import numpy as np
import openpyxl
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import f_pvalue, r_pvalue

BASE = find_thesis_root()
MOS = os.path.join(BASE, "4band_mosaic")
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
WHOLEIMG_PATH = os.path.join(MOS, "Master_Trees_WholeImageBaseline_v3.xlsx")
YEARS = [2021, 2022, 2023, 2024]
PHYS_FIELDS = ["SWP_April", "Growth_April", "SWP_MayJune", "Growth_MayJune", "CNC_June", "SWP_June", "Growth_June"]
PHYS_GPKG_COLS = ["SWP_April", "Growth__April", "SWP_May.June", "Growth_May.June", "CNC_June", "SWP_June", "Growth_June"]
MATCH_THRESHOLD_M = 3.0
CV_SEED = 42

if not os.path.exists(WHOLEIMG_PATH):
    print(f"NOT FOUND: {WHOLEIMG_PATH}")
    print("Run apply_fixed_zones_yearly_v3_wholeimage_baseline.py on a machine with rasterio first.")
    sys.exit(1)

M = load_master()
cultivar = {r["Tree_ID"]: r.get("cultivar") for r in M}
by_tid = {r["Tree_ID"]: r for r in M}

# ---------------- load the new whole-image-baseline workbook ----------------
wb = openpyxl.load_workbook(WHOLEIMG_PATH, read_only=True)
ws = wb[wb.sheetnames[0]]
rows = list(ws.iter_rows(values_only=True))
header = rows[0]; idx = {h: i for i, h in enumerate(header)}
wholeimg = {}
for r in rows[1:]:
    tid = r[idx["Tree_ID"]]
    wholeimg[tid] = {h: r[idx[h]] for h in header if h not in ("Tree_ID", "Zone_Value")}

print(f"canonical master: {len(M)} trees; wholeimg file: {len(wholeimg)} trees")


def zc(v):
    v = np.asarray(v, float); s = np.nanstd(v); return (v - np.nanmean(v)) / (s if s > 0 else 1)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    r = float(np.corrcoef(ra, rb)[0, 1])
    return r, float(r_pvalue(r, len(a)))


def anova_1way_2groups(vals_a, vals_b):
    n1, n2 = len(vals_a), len(vals_b)
    grand = np.concatenate([vals_a, vals_b])
    ss_between = n1 * (vals_a.mean() - grand.mean()) ** 2 + n2 * (vals_b.mean() - grand.mean()) ** 2
    ss_within = ((vals_a - vals_a.mean()) ** 2).sum() + ((vals_b - vals_b.mean()) ** 2).sum()
    df1, df2 = 1, n1 + n2 - 2
    F = (ss_between / df1) / max(ss_within / df2, 1e-12)
    p = float(f_pvalue(F, df1, df2))
    eta2 = ss_between / (ss_between + ss_within)
    return round(F, 3), round(p, 4), round(eta2, 4)


# ================= PART A: within-year structure (rank corr + cultivar ANOVA) =================
partA = {}
for y in YEARS:
    tids = [tid for tid in wholeimg if by_tid.get(tid, {}).get(f"EBI_Raw_{y}") is not None
            and wholeimg[tid].get(f"EBI_Raw_wholeimg_{y}") is not None
            and cultivar.get(tid) in ("UEF", "53")]
    canon = np.array([by_tid[tid][f"EBI_Raw_{y}"] for tid in tids], float)
    whole = np.array([wholeimg[tid][f"EBI_Raw_wholeimg_{y}"] for tid in tids], float)
    cult = np.array([cultivar[tid] for tid in tids])
    rho, p_rho = spearman(canon, whole)

    F_canon, p_canon, eta2_canon = anova_1way_2groups(canon[cult == "UEF"], canon[cult == "53"])
    F_whole, p_whole, eta2_whole = anova_1way_2groups(whole[cult == "UEF"], whole[cult == "53"])
    partA[y] = {"n": len(tids), "spearman_canonical_vs_wholeimg": round(rho, 4), "p": round(p_rho, 6),
                "cultivar_ANOVA_canonical": {"F": F_canon, "p": p_canon, "eta2": eta2_canon},
                "cultivar_ANOVA_wholeimg": {"F": F_whole, "p": p_whole, "eta2": eta2_whole}}
    print(f"{y}: n={len(tids)} spearman(canon,wholeimg)={rho:.4f}  "
          f"cultivar-ANOVA F: canon={F_canon} wholeimg={F_whole}")

# ================= PART B: cross-year mean/SD trajectory =================
partB = {}
for variant, key_fmt in [("canonical", "EBI_Norm_{y}"), ("wholeimg", "EBI_Norm_wholeimg_{y}")]:
    means, sds = [], []
    for y in YEARS:
        tids = [tid for tid in wholeimg if cultivar.get(tid) in ("UEF", "53")]
        if variant == "canonical":
            vals = np.array([by_tid[tid].get(key_fmt.format(y=y)) for tid in tids
                              if by_tid[tid].get(key_fmt.format(y=y)) is not None], float)
        else:
            vals = np.array([wholeimg[tid].get(key_fmt.format(y=y)) for tid in tids
                              if wholeimg[tid].get(key_fmt.format(y=y)) is not None], float)
        means.append(round(float(vals.mean()), 4)); sds.append(round(float(vals.std()), 4))
    partB[variant] = {"mean_by_year": dict(zip(YEARS, means)), "sd_by_year": dict(zip(YEARS, sds))}

# combined = average of the two, both z-scored within-year first (so neither dominates by scale)
combined_by_year = {}
for y in YEARS:
    tids = [tid for tid in wholeimg if by_tid.get(tid, {}).get(f"EBI_Norm_{y}") is not None
            and wholeimg[tid].get(f"EBI_Norm_wholeimg_{y}") is not None
            and cultivar.get(tid) in ("UEF", "53")]
    canon = zc([by_tid[tid][f"EBI_Norm_{y}"] for tid in tids])
    whole = zc([wholeimg[tid][f"EBI_Norm_wholeimg_{y}"] for tid in tids])
    combo = (canon + whole) / 2.0
    combined_by_year[y] = {"mean": round(float(combo.mean()), 4), "sd": round(float(combo.std()), 4), "n": len(tids)}
partB["combined_avg_zscored"] = {"mean_by_year": {y: combined_by_year[y]["mean"] for y in YEARS},
                                  "sd_by_year": {y: combined_by_year[y]["sd"] for y in YEARS}}
print("\ncross-year SD by variant:")
for variant in ["canonical", "wholeimg", "combined_avg_zscored"]:
    print(f"  {variant}: {partB[variant]['sd_by_year']}")

# ================= PART C: yield-model value (pooled 2022-2023, same pipeline as §22/28/29) =================
def gpkg_geom_centroid(blob):
    flags = blob[3]; envelope_ind = (flags >> 1) & 0x07
    env_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[envelope_ind]
    wkb = blob[8 + env_bytes:]; endian = "<" if wkb[0] == 1 else ">"
    pos = 1
    geom_type = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
    base_type = geom_type % 1000
    xs, ys = [], []
    if base_type == 6:
        npoly = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
        for _ in range(npoly):
            pos += 1; struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
            nrings = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
            for r in range(nrings):
                npts = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
                for _p in range(npts):
                    x, y = struct.unpack_from(endian + "dd", wkb, pos); pos += 16
                    if r == 0: xs.append(x); ys.append(y)
    elif base_type == 3:
        nrings = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
        for r in range(nrings):
            npts = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
            for _p in range(npts):
                x, y = struct.unpack_from(endian + "dd", wkb, pos); pos += 16
                if r == 0: xs.append(x); ys.append(y)
    return sum(xs) / len(xs), sum(ys) / len(ys)


def load_phys(year):
    path = os.path.join(GDIR, f"Yield_with_clustering_{year}.gpkg")
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT table_name FROM gpkg_geometry_columns")
    tbl = cur.fetchone()[0]
    cols = ",".join(f'"{c}"' for c in PHYS_GPKG_COLS)
    cur.execute(f'SELECT "Row","Plot",cultivar,{cols},geom FROM "{tbl}"')
    rows_ = cur.fetchall(); con.close()
    out = []
    for row_ in rows_:
        plot = row_[1]; vals = row_[3:3 + len(PHYS_FIELDS)]; blob = row_[-1]
        if plot != "A" or any(v is None for v in vals): continue
        x, y = gpkg_geom_centroid(blob)
        d = {"x": x, "y": y}
        for f, v in zip(PHYS_FIELDS, vals): d[f] = float(v)
        out.append(d)
    return out


PHYS = {y: load_phys(y) for y in [2022, 2023]}
mx = np.array([r.get("X_UTM") for r in M], float); my = np.array([r.get("Y_UTM") for r in M], float)
phys_match = {}
for y in [2022, 2023]:
    cand = []
    for i, r in enumerate(PHYS[y]):
        d = np.sqrt((mx - r["x"]) ** 2 + (my - r["y"]) ** 2)
        j = int(np.argmin(d))
        if d[j] <= MATCH_THRESHOLD_M: cand.append((d[j], i, j))
    cand.sort(); ui = set(); uj = set(); m = {}
    for d, i, j in cand:
        if i in ui or j in uj: continue
        m[j] = PHYS[y][i]; ui.add(i); uj.add(j)
    phys_match[y] = m

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

rowsP = [(t, j, yr) for t, j in ymap.items() for yr in (2022, 2023)
         if ymeas[t].get(yr) is not None and M[j].get(f"EBI_Norm_{yr}") is not None and j in phys_match[yr]
         and M[j]["Tree_ID"] in wholeimg and wholeimg[M[j]["Tree_ID"]].get(f"EBI_Norm_wholeimg_{yr}") is not None]
nP = len(rowsP)
print(f"\npooled yield-model sample: n={nP}")

yP = np.array([ymeas[t][yr] for t, j, yr in rowsP])
uefP = np.array([1.0 if cultivar[M[j]["Tree_ID"]] == "UEF" else 0.0 for t, j, yr in rowsP])
ebiP_canon = zc([M[j].get(f"EBI_Norm_{yr}") for t, j, yr in rowsP])
ebiP_whole = zc([wholeimg[M[j]["Tree_ID"]].get(f"EBI_Norm_wholeimg_{yr}") for t, j, yr in rowsP])
ebiP_combo = (ebiP_canon + ebiP_whole) / 2.0
CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
climP = zc([CP[yr] for t, j, yr in rowsP])
oneP = np.ones(nP)
phys_vals = {}
for f in PHYS_FIELDS:
    phys_vals[f] = zc([phys_match[yr][j][f] for t, j, yr in rowsP])
gaP = phys_vals["Growth_April"]


def r2_cv(Xd, y, k=5, seed=CV_SEED):
    n = len(y); rng = np.random.default_rng(seed)
    idx_ = rng.permutation(n); folds = np.array_split(idx_, k)
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


partC = {}
for label, ebi_variant in [("canonical_EBI_Norm", ebiP_canon), ("wholeimg_EBI_Norm", ebiP_whole),
                           ("combined_avg", ebiP_combo)]:
    cols = {"intercept": oneP, "cultivar": uefP, "EBI": ebi_variant, "EBIxcultivar": ebi_variant * uefP,
            "climate": climP, "Growth_April": gaP}
    X = np.column_stack(list(cols.values()))
    r2, cv = r2_cv(X, yP)
    robust = [r2_cv(X, yP, seed=s)[1] for s in [1, 7, 13, 99, 2024]]
    partC[label] = {"R2": r2, "cvR2_seed42": cv, "cvR2_5seeds": robust, "cvR2_mean_5seeds": round(float(np.mean(robust)), 4)}
    print(f"{label}: R2={r2} cvR2(seed42)={cv} cvR2(mean of 5 robustness seeds)={partC[label]['cvR2_mean_5seeds']}")

RESULTS = {
    "note": "Compares the canonical (canopy-only radiometric baseline) EBI against the new "
            "whole-image-baseline variant (Master_Trees_WholeImageBaseline_v3.xlsx, produced by "
            "apply_fixed_zones_yearly_v3_wholeimage_baseline.py on a machine with rasterio), and a "
            "combined (average, z-scored) variant. Neither the canonical master nor v1/v2 files were "
            "modified; this only reads them.",
    "partA_within_year_structure": partA,
    "partB_cross_year_trajectory": partB,
    "partC_yield_model_value": partC,
}
with open(os.path.join(OUT, "EBI_WholeImageBaseline_Comparison.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
for variant, color in [("canonical", "#C0392B"), ("wholeimg", "#16A085"), ("combined_avg_zscored", "#8E44AD")]:
    sd = [partB[variant]["sd_by_year"][y] for y in YEARS]
    ax.plot(YEARS, sd, "o-", label=variant, color=color)
ax.set_xlabel("Year"); ax.set_ylabel("EBI_Norm SD across trees")
ax.set_title("Cross-year EBI spread:\ncanonical vs whole-image vs combined", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1]
labels = list(partC.keys())
vals = [partC[l]["cvR2_mean_5seeds"] for l in labels]
ax.bar(range(len(labels)), vals, color=["#C0392B", "#16A085", "#8E44AD"])
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8, rotation=15)
ax.set_ylabel("mean CV R^2 across 5 robustness seeds")
ax.set_title("Yield-model value:\ncanonical vs whole-image vs combined", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "EBI_WholeImageBaseline_Comparison.png"), dpi=130)
print("saved figure")
