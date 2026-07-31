#!/usr/bin/env python3
"""
yield_model_canopy_feature_bank.py: per request, builds a bank of creative new
canopy/bloom features from the two existing pixel-level extractions already on
disk (Master_Trees_CanopyStructure_v5.xlsx: CanopyArea_m2, CanopyCoverFraction,
ShadowFraction; Master_Trees_BloomFraction_v3.xlsx: BloomFraction, BrightFraction,
the Otsu-threshold-crossing-percentage feature, the closest already-extracted
analog to a literal "EBI threshold, % of canopy pixels above it" feature; a
genuinely NEW EBI-specific threshold would need rasterio + the raw mosaics, not
available in this sandbox, noted below), then tests them:
  (A) same-year 2023, ALL matched 2023 measured-yield trees (not the smaller
      2022+2023 intersection used in earlier sections), standalone + ANCOVA +
      pairwise combinations among the new features
  (B) pooled 2022-2023, added into the full existing 12-feature candidate pool
      (section 30) via forward selection + 5-seed robustness, i.e. combined with
      every other feature already in the model
New engineered features (all derived from already-extracted data, no new
rasterio needed):
  EBI_density        = EBI_Norm / CanopyArea_m2 (bloom intensity per unit canopy)
  BloomVolume        = EBI_Norm * CanopyArea_m2 (tested in section 37, included
                        again here for completeness in the combined sweep)
  BloomPixelVolume   = BloomFraction * CanopyPixelCount (literal count of
                        Otsu-classified "bloom" pixels, not just an intensity x
                        area proxy)
  EBI_x_BloomFraction (two distinct bloom channels: continuous intensity vs
                        thresholded pixel classification)
  EBI_x_CanopyCover
  BrightFraction_x_CanopyArea
-> Results_Analysis/08_UEF53_Rerun/Yield_Model_Canopy_Feature_Bank.json + figure
"""
import os, sys, csv, math, json, itertools
from collections import defaultdict, Counter
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


def load_xlsx_by_id(path):
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]; hidx = {h: i for i, h in enumerate(header)}
    out = {}
    for r in rows[1:]:
        d = {h: r[hidx[h]] for h in header}
        out[d["Tree_ID"]] = d
    return out


canopy_by_id = load_xlsx_by_id(os.path.join(BASE, "4band_mosaic", "Master_Trees_CanopyStructure_v5.xlsx"))
bloomfrac_by_id = load_xlsx_by_id(os.path.join(BASE, "4band_mosaic", "Master_Trees_BloomFraction_v3.xlsx"))
print(f"canopy structure file: {len(canopy_by_id)} trees; bloom fraction file: {len(bloomfrac_by_id)} trees")

canopy_for_master = [canopy_by_id.get(r.get("Tree_ID")) for r in M]
bloomfrac_for_master = [bloomfrac_by_id.get(r.get("Tree_ID")) for r in M]
n_canopy_matched = sum(1 for c in canopy_for_master if c is not None)
n_bloomfrac_matched = sum(1 for c in bloomfrac_for_master if c is not None)
print(f"matched to master by Tree_ID: canopy={n_canopy_matched}/{len(M)}, bloomfrac={n_bloomfrac_matched}/{len(M)}")


def build_features(j, yr):
    """Returns a dict of every canopy/bloom feature for master row j, year yr, or None if missing."""
    c = canopy_for_master[j]; b = bloomfrac_for_master[j]
    ebi = M[j].get(f"EBI_Norm_{yr}")
    if c is None or b is None or ebi is None: return None
    area = c.get(f"CanopyArea_m2_{yr}"); cover = c.get(f"CanopyCoverFraction_{yr}")
    shadow = c.get(f"ShadowFraction_{yr}"); pix = c.get(f"CanopyPixelCount_{yr}")
    bf = b.get(f"BloomFraction_{yr}"); brf = b.get(f"BrightFraction_{yr}")
    if any(v is None for v in [area, cover, shadow, pix, bf, brf]): return None
    ebi = float(ebi); area = float(area); cover = float(cover); shadow = float(shadow); pix = float(pix)
    bf = float(bf); brf = float(brf)
    return {
        "EBI": ebi, "CanopyArea": area, "CanopyCover": cover, "ShadowFraction": shadow,
        "BloomFraction": bf, "BrightFraction": brf,
        "EBI_density": ebi / area if area > 0 else np.nan,
        "BloomVolume": ebi * area,
        "BloomPixelVolume": bf * pix,
        "EBI_x_BloomFraction": ebi * bf,
        "EBI_x_CanopyCover": ebi * cover,
        "BrightFraction_x_CanopyArea": brf * area,
    }


FEATURE_NAMES = ["CanopyArea", "CanopyCover", "ShadowFraction", "BloomFraction", "BrightFraction",
                  "EBI_density", "BloomVolume", "BloomPixelVolume", "EBI_x_BloomFraction",
                  "EBI_x_CanopyCover", "BrightFraction_x_CanopyArea"]

# ================= PART A: all matched 2023 trees, same-year =================
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

rows2023 = []
for t, j in ymap.items():
    if ymeas[t].get(2023) is None or cultivar[j] not in ("UEF", "53"): continue
    feats = build_features(j, 2023)
    if feats is None: continue
    rows2023.append((t, j, ymeas[t][2023], feats))
n2023 = len(rows2023)
n2023_yield_total = sum(1 for t in ymap if ymeas[t].get(2023) is not None)
print(f"\n=== PART A: 2023, all matched trees ===")
print(f"2023 measured-yield trees matched to master (any cultivar): {n2023_yield_total}")
print(f"...of those, with EBI + canopy + bloomfraction features: {n2023}")

yv = np.array([r[2] for r in rows2023])
uef = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rows2023])


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


feat_arrays = {name: zc([r[3][name] for r in rows2023]) for name in FEATURE_NAMES}
ebi_arr = zc([r[3]["EBI"] for r in rows2023])


def r_p(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if xs.std() == 0: return None, None
    r = float(np.corrcoef(xs, ys)[0, 1])
    return round(r, 4), round(float(r_pvalue(r, len(xs))), 4)


print("\n--- standalone correlations with 2023 measured yield, pooled + by cultivar ---")
standalone = {}
pvals_pooled = []
names_order = ["EBI"] + FEATURE_NAMES
for name in names_order:
    arr = ebi_arr if name == "EBI" else feat_arrays[name]
    r_pool, p_pool = r_p(arr, yv)
    r_uef, p_uef = r_p(arr[uef == 1], yv[uef == 1])
    r_53, p_53 = r_p(arr[uef == 0], yv[uef == 0])
    standalone[name] = {"pooled": {"r": r_pool, "p": p_pool, "n": n2023},
                         "UEF": {"r": r_uef, "p": p_uef, "n": int((uef == 1).sum())},
                         "cv53": {"r": r_53, "p": p_53, "n": int((uef == 0).sum())}}
    pvals_pooled.append(p_pool)
    print(f"  {name}: pooled r={r_pool} p={p_pool} | UEF r={r_uef} p={p_uef} | cv53 r={r_53} p={p_53}")


def bh_fdr(pvals):
    idx = [i for i, p in enumerate(pvals) if p is not None]
    ps = sorted((pvals[i], i) for i in idx); m = len(ps)
    adj = {}; prev = 1.0
    for rank in range(m - 1, -1, -1):
        p, i = ps[rank]; val = min(prev, p * m / (rank + 1)); adj[i] = round(val, 4); prev = val
    return [adj.get(i) for i in range(len(pvals))]


fdr_pooled = bh_fdr(pvals_pooled)
print("\n  BH-FDR corrected (pooled, 12 candidates incl. EBI):")
for name, pf in zip(names_order, fdr_pooled):
    flag = "**survives FDR<0.05**" if (pf is not None and pf < 0.05) else ""
    print(f"    {name}: p_fdr={pf} {flag}")

# ---- ANCOVA-style: does EBI x cultivar interaction hold, and do new features have their own interaction? ----
one = np.ones(n2023)
X_ebi_cult = np.column_stack([one, uef, ebi_arr, ebi_arr * uef])
rss_ebi, p_ebi, _ = ols_rss(X_ebi_cult, yv)
X_cult_only = np.column_stack([one, uef])
rss0, p0, _ = ols_rss(X_cult_only, yv)
df1 = p_ebi - p0; df2 = n2023 - p_ebi
F_ebi_int = ((rss0 - rss_ebi) / df1) / (rss_ebi / df2)
print(f"\nEBI x cultivar interaction (2023 only, all matched trees, n={n2023}): "
      f"F={F_ebi_int:.3f} p={f_pvalue(F_ebi_int, df1, df2):.4f}")

# ================= PART B: pairwise combinations among the new features (2023) =================
print(f"\n--- pairwise combos among the 11 new canopy features (+EBI), full-F test, 2023 only ---")
all_feats = dict(feat_arrays); all_feats["EBI"] = ebi_arr
pair_names = list(itertools.combinations(all_feats.keys(), 2))
X_base23 = np.column_stack([one, uef])
rss0b, p0b, _ = ols_rss(X_base23, yv)
pairwise_F = {}
for a, b in pair_names:
    col = all_feats[a] * all_feats[b]
    X1 = np.column_stack([one, uef, col])
    rss1, p1, _ = ols_rss(X1, yv)
    d1 = p1 - p0b; d2 = n2023 - p1
    if d1 <= 0 or d2 <= 0: continue
    Fs = ((rss0b - rss1) / d1) / (rss1 / d2)
    pf = float(f_pvalue(Fs, d1, d2))
    pairwise_F[f"{a}_x_{b}"] = {"F": round(float(Fs), 3), "p": round(pf, 4)}
names_sorted = sorted(pairwise_F.keys(), key=lambda n: pairwise_F[n]["p"])
pv = [pairwise_F[n]["p"] for n in names_sorted]
fdr = bh_fdr(pv)
for n, pf in zip(names_sorted, fdr): pairwise_F[n]["p_fdr"] = pf
n_survive = sum(1 for n in pairwise_F if pairwise_F[n]["p_fdr"] is not None and pairwise_F[n]["p_fdr"] < 0.05)
print(f"{n_survive} of {len(pairwise_F)} pairwise combos survive FDR<0.05. Top 5 by raw p:")
for n in names_sorted[:5]:
    print(f"  {n}: F={pairwise_F[n]['F']} p={pairwise_F[n]['p']} p_fdr={pairwise_F[n]['p_fdr']}")

# ================= PART C: pooled 2022-2023, combined with ALL other features (extends section 30) =================
print(f"\n=== PART C: pooled 2022-2023, new canopy features combined with the full existing feature set ===")
rowsP = []
for t, j in ymap.items():
    for yr in (2022, 2023):
        if ymeas[t].get(yr) is None or cultivar[j] not in ("UEF", "53"): continue
        feats = build_features(j, yr)
        if feats is None: continue
        rowsP.append((t, j, yr, ymeas[t][yr], feats))
nP = len(rowsP)
print(f"pooled 2022-2023 n={nP}")

yP = np.array([r[3] for r in rowsP])
uefP = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rowsP])
ebiP = zc([r[4]["EBI"] for r in rowsP])
CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
climP = zc([CP[yr] for t, j, yr, y, f in rowsP])
oneP = np.ones(nP)
new_feat_arrays = {name: zc([r[4][name] for r in rowsP]) for name in FEATURE_NAMES}

base_cols = {"intercept": oneP, "cultivar": uefP, "EBI": ebiP, "EBIxcultivar": ebiP * uefP, "climate": climP}
X_base = np.column_stack(list(base_cols.values()))


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


r2_base, cv_base = r2_cv(X_base, yP)
print(f"baseline (cultivar+EBIxcultivar+climate): R2={r2_base} cvR2={cv_base}")

candidates = {}
for name, arr in new_feat_arrays.items():
    candidates[name] = arr
    candidates[f"{name}_x_cultivar"] = arr * uefP


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
print(f"forward selection over {len(candidates)} new candidates (main seed 42): "
      f"selected={sel_main} final_cvR2={cv_main} (gain={round(cv_main-cv_base,4)})")

robust = []
for seed in [1, 7, 13, 99, 2024]:
    sel_s, hist_s, r2_s, cv_s = forward_select(dict(base_cols), dict(candidates), yP, seed=seed)
    robust.append({"seed": seed, "selected": sel_s, "final_cvR2": cv_s})
    print(f"  seed {seed}: selected={sel_s} final_cvR2={cv_s}")
sel_counter = Counter()
for r in [{"selected": sel_main}] + robust:
    for s in r["selected"]: sel_counter[s] += 1
print(f"how often selected of 6: {dict(sel_counter)}")

# also: combine with the FULL section-30 12-feature base (Growth_April, DEM, NGRDI, physiology) if available
import sqlite3, struct
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")
PHYS_FIELDS = ["SWP_April", "Growth_April", "SWP_MayJune", "Growth_MayJune", "CNC_June", "SWP_June", "Growth_June"]
PHYS_GPKG_COLS = ["SWP_April", "Growth__April", "SWP_May.June", "Growth_May.June", "CNC_June", "SWP_June", "Growth_June"]


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
    rowsg = cur.fetchall(); con.close()
    out = []
    for row_ in rowsg:
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
    cand2 = []
    for i, r in enumerate(PHYS[y]):
        d = np.sqrt((mx - r["x"]) ** 2 + (my - r["y"]) ** 2)
        j = int(np.argmin(d))
        if d[j] <= 3.0: cand2.append((d[j], i, j))
    cand2.sort(); ui = set(); uj = set(); mm = {}
    for d, i, j in cand2:
        if i in ui or j in uj: continue
        mm[j] = PHYS[y][i]; ui.add(i); uj.add(j)
    phys_match[y] = mm

rowsFull = [r for r in rowsP if r[1] in phys_match[r[2]]]
nF = len(rowsFull)
print(f"\nsubset with physiology match too (for full-feature combination): n={nF}")
if nF > 30:
    yF = np.array([r[3] for r in rowsFull])
    uefF = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rowsFull])
    ebiF = zc([r[4]["EBI"] for r in rowsFull])
    climF = zc([CP[r[2]] for r in rowsFull])
    gaF = zc([phys_match[r[2]][r[1]]["Growth_April"] for r in rowsFull])
    oneF = np.ones(nF)
    baseF = {"intercept": oneF, "cultivar": uefF, "EBI": ebiF, "EBIxcultivar": ebiF * uefF, "climate": climF, "Growth_April": gaF}
    X_baseF = np.column_stack(list(baseF.values()))
    r2_baseF, cv_baseF = r2_cv(X_baseF, yF)
    print(f"full recommended-model baseline on this subsample: R2={r2_baseF} cvR2={cv_baseF}")
    new_feat_F = {name: zc([r[4][name] for r in rowsFull]) for name in FEATURE_NAMES}
    candF = {}
    for name, arr in new_feat_F.items():
        candF[name] = arr; candF[f"{name}_x_cultivar"] = arr * uefF
    sel_mainF, hist_mainF, r2_mainF, cv_mainF = forward_select(dict(baseF), candF, yF)
    print(f"forward selection, canopy features on top of FULL recommended model: "
          f"selected={sel_mainF} final_cvR2={cv_mainF} (gain={round(cv_mainF-cv_baseF,4)})")
    robustF = []
    for seed in [1, 7, 13, 99, 2024]:
        sel_s, hist_s, r2_s, cv_s = forward_select(dict(baseF), dict(candF), yF, seed=seed)
        robustF.append({"seed": seed, "selected": sel_s, "final_cvR2": cv_s})
        print(f"  seed {seed}: selected={sel_s} final_cvR2={cv_s}")
    selF_counter = Counter()
    for r in [{"selected": sel_mainF}] + robustF:
        for s in r["selected"]: selF_counter[s] += 1
    print(f"how often selected of 6: {dict(selF_counter)}")
else:
    r2_baseF = cv_baseF = sel_mainF = hist_mainF = r2_mainF = cv_mainF = robustF = selF_counter = None
    print("too few rows for the full-feature combination test")

RESULTS = {
    "note": "Per request: creative new canopy/bloom features (all derived from the two existing "
            "pixel-level extractions on disk, CanopyStructure_v5 and BloomFraction_v3, no new "
            "rasterio extraction was possible in this sandbox; BloomFraction/BrightFraction are the "
            "already-extracted Otsu-threshold-crossing-percentage features, the closest existing "
            "analog to a literal EBI-specific pixel threshold, which would need a new rasterio "
            "extraction to build directly). Tested standalone, in same-year 2023 ANCOVA, in pairwise "
            "combination with each other, and combined with the full existing feature set.",
    "feature_definitions": {
        "EBI_density": "EBI_Norm / CanopyArea_m2",
        "BloomVolume": "EBI_Norm * CanopyArea_m2",
        "BloomPixelVolume": "BloomFraction * CanopyPixelCount (literal Otsu-classified bloom pixel count)",
        "EBI_x_BloomFraction": "EBI_Norm * BloomFraction (two distinct bloom channels)",
        "EBI_x_CanopyCover": "EBI_Norm * CanopyCoverFraction",
        "BrightFraction_x_CanopyArea": "BrightFraction * CanopyArea_m2",
    },
    "partA_2023_all_matched_trees": {
        "n_2023_yield_trees_matched_to_master": n2023_yield_total,
        "n_with_full_feature_set": n2023,
        "standalone_correlations": standalone,
        "bh_fdr_pooled": {name: pf for name, pf in zip(names_order, fdr_pooled)},
        "EBI_x_cultivar_interaction_2023_all_trees": {"F": round(float(F_ebi_int), 3), "p": round(float(f_pvalue(F_ebi_int, df1, df2)), 4)},
    },
    "partB_pairwise_combos_2023": {
        "n_candidates": len(pairwise_F), "n_survive_FDR": n_survive,
        "top10_by_raw_p": {n: pairwise_F[n] for n in names_sorted[:10]},
    },
    "partC_pooled_2022_2023_vs_full_feature_set": {
        "n": nP, "baseline_cvR2": cv_base,
        "forward_selection_main_seed42": {"selected": sel_main, "final_cvR2": cv_main, "gain": round(cv_main - cv_base, 4)},
        "robustness_5_seeds": robust, "how_often_selected_of_6": dict(sel_counter),
    },
    "partC2_combined_with_full_recommended_model_incl_physiology": {
        "n": nF, "baseline_cvR2": cv_baseF,
        "forward_selection_main_seed42": ({"selected": sel_mainF, "final_cvR2": cv_mainF, "gain": round(cv_mainF - cv_baseF, 4)} if nF > 30 else None),
        "robustness_5_seeds": robustF, "how_often_selected_of_6": (dict(selF_counter) if nF > 30 else None),
    },
}
with open(os.path.join(OUT, "Yield_Model_Canopy_Feature_Bank.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
a = ax[0]
names_plot = names_order
vals = [standalone[n]["pooled"]["r"] or 0 for n in names_plot]
colors = ["#C0392B" if (fdr_pooled[i] is not None and fdr_pooled[i] < 0.05) else "#95a5a6" for i in range(len(names_plot))]
a.barh(range(len(names_plot)), vals, color=colors)
a.set_yticks(range(len(names_plot))); a.set_yticklabels(names_plot, fontsize=8); a.invert_yaxis()
a.axvline(0, color="k", lw=0.8)
a.set_title(f"2023 standalone correlation with yield\n(n={n2023}, red=survives FDR<0.05)", fontsize=10, fontweight="bold")
b = ax[1]
seeds = ["main42"] + [str(r["seed"]) for r in robust]
gains = [round(cv_main - cv_base, 4)] + [round(r["final_cvR2"] - cv_base, 4) for r in robust]
b.bar(seeds, gains, color="#16A085")
b.axhline(0, color="k", lw=1)
b.set_title("CV R2 gain, canopy feature bank\non top of cultivar+EBIxcultivar+climate", fontsize=10, fontweight="bold")
b.grid(axis="y", alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "Yield_Model_Canopy_Feature_Bank.png"), dpi=130)
print("saved figure")
