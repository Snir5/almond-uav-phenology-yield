#!/usr/bin/env python3
"""
all_features_no_outliers_recheck.py: re-runs the full engineered-feature test
(sections 38 canopy bank, 39 EBI pixel-distribution bank, 41 composite bloom
indices, 43 one-by-one comparison, 44 near-miss combination check) on the
actual-yield-anomaly-free sample established in sections 45-46 (both tails
removed: 9 high-side modZ>3.5 outliers + 6 low-side <1.0kg domain-trimmed
trees, from Four_Way_Comparison_No_Outliers_Both_Tails.json).

Same candidate universe as before: 11 canopy-structure features (v5) + 18 true
EBI pixel-distribution features (v6) + 6 composite bloom indices, each also
tested x cultivar, tested against the SAME recommended-model baseline
(cultivar + EBI x cultivar + climate + Growth_April), refit fresh on the
outlier-free rows (not the original n=167 coefficients).

-> Results_Analysis/08_UEF53_Rerun/All_Features_No_Outliers_Recheck.json
"""
import os, sys, csv, math, json, sqlite3, struct
from collections import defaultdict, Counter
import numpy as np
import openpyxl
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import ols_rss, f_pvalue, pca

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")
CV_SEED = 42

M = load_master(); cultivar = [r.get("cultivar") for r in M]
mlat = np.array([r.get("Latitude") for r in M], float); mlon = np.array([r.get("Longitude") for r in M], float)
mx = np.array([r.get("X_UTM") for r in M], float); my = np.array([r.get("Y_UTM") for r in M], float)


def load_xlsx_by_id(path):
    wb = openpyxl.load_workbook(path, read_only=True); ws = wb.active
    rows = list(ws.iter_rows(values_only=True)); header = rows[0]; hidx = {h: i for i, h in enumerate(header)}
    return {r[hidx["Tree_ID"]]: {h: r[hidx[h]] for h in header} for r in rows[1:]}


canopy_by_id = load_xlsx_by_id(os.path.join(BASE, "4band_mosaic", "Master_Trees_CanopyStructure_v5.xlsx"))
bloomfrac_by_id = load_xlsx_by_id(os.path.join(BASE, "4band_mosaic", "Master_Trees_BloomFraction_v3.xlsx"))
v6_by_id = load_xlsx_by_id(os.path.join(BASE, "4band_mosaic", "Master_Trees_EBIPixelFeatures_v6.xlsx"))
canopy_for_master = [canopy_by_id.get(r.get("Tree_ID")) for r in M]
bloomfrac_for_master = [bloomfrac_by_id.get(r.get("Tree_ID")) for r in M]
v6_for_master = [v6_by_id.get(r.get("Tree_ID")) for r in M]

CANOPY_NAMES = ["CanopyArea", "CanopyCover", "ShadowFraction", "BloomFraction", "BrightFraction",
                "EBI_density", "BloomVolume", "BloomPixelVolume", "EBI_x_BloomFraction",
                "EBI_x_CanopyCover", "BrightFraction_x_CanopyArea"]
V6_NAMES = ["EBI_Norm_std", "EBI_Norm_var", "EBI_Norm_median", "EBI_Norm_P10", "EBI_Norm_P25",
            "EBI_Norm_P75", "EBI_Norm_P90", "EBI_Norm_IQR", "EBI_Norm_skew", "EBI_Norm_kurtosis",
            "EBI_Norm_CV", "EBI_OtsuFrac", "HighEBI_frac_055", "HighEBI_density_055",
            "HighEBI_frac_065", "HighEBI_density_065", "HighEBI_frac_075", "HighEBI_density_075"]
COMPOSITE_NAMES = ["TBL", "ICBI", "TICBL", "UBI", "CCAB", "SBS"]


def build_all_features(j, yr):
    c = canopy_for_master[j]; b = bloomfrac_for_master[j]; v6 = v6_for_master[j]
    ebi = M[j].get(f"EBI_Norm_{yr}")
    if c is None or b is None or v6 is None or ebi is None: return None
    area = c.get(f"CanopyArea_m2_{yr}"); cover = c.get(f"CanopyCoverFraction_{yr}")
    shadow = c.get(f"ShadowFraction_{yr}"); pix = c.get(f"CanopyPixelCount_{yr}")
    bf = b.get(f"BloomFraction_{yr}"); brf = b.get(f"BrightFraction_{yr}")
    frac065 = v6.get(f"HighEBI_frac_065_{yr}"); ebi_std = v6.get(f"EBI_Norm_std_{yr}"); ebi_cv = v6.get(f"EBI_Norm_CV_{yr}")
    base_needed = [area, cover, shadow, pix, bf, brf, frac065, ebi_std, ebi_cv]
    v6_needed = [v6.get(f"{name}_{yr}") for name in V6_NAMES]
    if any(v is None for v in base_needed) or any(v is None for v in v6_needed): return None
    ebi = float(ebi); area = float(area); cover = float(cover); shadow = float(shadow); pix = float(pix)
    bf = float(bf); brf = float(brf); frac065 = float(frac065); ebi_std = float(ebi_std); ebi_cv = float(ebi_cv)

    canopy_feats = {
        "CanopyArea": area, "CanopyCover": cover, "ShadowFraction": shadow, "BloomFraction": bf,
        "BrightFraction": brf, "EBI_density": ebi / area if area > 0 else np.nan, "BloomVolume": ebi * area,
        "BloomPixelVolume": bf * pix, "EBI_x_BloomFraction": ebi * bf, "EBI_x_CanopyCover": ebi * cover,
        "BrightFraction_x_CanopyArea": brf * area,
    }
    v6_feats = {name: float(v6.get(f"{name}_{yr}")) for name in V6_NAMES}
    composite_feats = {
        "TBL": area * frac065, "ICBI": ebi * (1 - shadow), "TICBL": area * ebi * (1 - shadow),
        "UBI": ebi * (1 - ebi_cv), "CCAB": frac065 * cover,
        "SBS": None,  # filled after z-scoring below (needs population stats)
    }
    out = {}
    out.update(canopy_feats); out.update(v6_feats); out.update(composite_feats)
    out["_raw_area"] = area; out["_raw_frac065"] = frac065; out["_raw_ebi_std"] = ebi_std; out["_raw_shadow"] = shadow
    return out


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
    cur.execute("SELECT table_name FROM gpkg_geometry_columns"); tbl = cur.fetchone()[0]
    cur.execute(f'SELECT "Row","Plot",cultivar,"Growth__April",geom FROM "{tbl}"')
    rowsg = cur.fetchall(); con.close(); out = []
    for row_ in rowsg:
        plot = row_[1]; ga = row_[3]; blob = row_[-1]
        if plot != "A" or ga is None: continue
        x, y = gpkg_geom_centroid(blob); out.append({"x": x, "y": y, "Growth_April": float(ga)})
    return out


PHYS = {y: load_phys(y) for y in [2022, 2023]}
phys_match = {}
for y in [2022, 2023]:
    cand2 = []
    for i, r in enumerate(PHYS[y]):
        d = np.sqrt((mx - r["x"]) ** 2 + (my - r["y"]) ** 2); j = int(np.argmin(d))
        if d[j] <= 3.0: cand2.append((d[j], i, j))
    cand2.sort(); ui = set(); uj = set(); mm = {}
    for d, i, j in cand2:
        if i in ui or j in uj: continue
        mm[j] = PHYS[y][i]; ui.add(i); uj.add(j)
    phys_match[y] = mm

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

# ---------- outlier exclusion set, from sections 45-46 ----------
no_out = json.load(open(os.path.join(OUT, "Four_Way_Comparison_No_Outliers_Both_Tails.json")))
excluded = set()
for entry in no_out["removed_trees_high"] + no_out["removed_trees_low"]:
    excluded.add((entry["tree_id"], entry["year"]))
print(f"excluding {len(excluded)} tree-years (outlier removal, both tails)")

rowsAll = []
for t, j in ymap.items():
    for yr in (2022, 2023):
        if (t, yr) in excluded: continue
        if ymeas[t].get(yr) is None or cultivar[j] not in ("UEF", "53") or j not in phys_match[yr]: continue
        if M[j].get(f"EBI_Norm_{yr}") is None: continue
        feats = build_all_features(j, yr)
        if feats is None: continue
        rowsAll.append((t, j, yr, ymeas[t][yr], feats))
nF = len(rowsAll)
n_excluded_and_matched = sum(1 for t, j in ymap.items() for yr in (2022, 2023) if (t, yr) in excluded)
print(f"n={nF} (with full canopy+v6+physiology feature match, outlier tree-years excluded: {n_excluded_and_matched})")


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


yF = np.array([r[3] for r in rowsAll])
uefF = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rowsAll])
ebiF_raw = np.array([M[r[1]].get(f"EBI_Norm_{r[2]}") for r in rowsAll], float)
ebiF = zc(ebiF_raw)
CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
climF = zc([CP[r[2]] for r in rowsAll])
gaF = zc([phys_match[r[2]][r[1]]["Growth_April"] for r in rowsAll])
oneF = np.ones(nF)
base_cols = {"intercept": oneF, "cultivar": uefF, "EBI": ebiF, "EBIxcultivar": ebiF * uefF, "climate": climF, "Growth_April": gaF}
X_base = np.column_stack(list(base_cols.values()))
rss0, p0, _ = ols_rss(X_base, yF)


def r2_cv(Xd, y, k=5, seed=CV_SEED):
    n = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n); folds = np.array_split(idx, k)
    pred = np.full(n, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None); pred[f] = Xd[f] @ beta
    ss = ((y - pred) ** 2).sum(); tss = ((y - y.mean()) ** 2).sum(); cv = 1 - ss / tss
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None); ins = Xd @ beta
    r2 = 1 - ((y - ins) ** 2).sum() / tss
    return round(float(r2), 4), round(float(cv), 4)


r2_base, cv_base = r2_cv(X_base, yF)
print(f"baseline (recommended model, refit on n={nF} outlier-free rows): R2={r2_base} cvR2={cv_base}")

# ---------- build z-scored candidate arrays (SBS assembled after z-scoring) ----------
raw_area = zc([r[4]["_raw_area"] for r in rowsAll])
raw_frac065 = zc([r[4]["HighEBI_frac_065"] for r in rowsAll])
raw_ebistd = zc([r[4]["EBI_Norm_std"] for r in rowsAll])
raw_shadow_inv = zc([-(r[4]["ShadowFraction"]) for r in rowsAll])
sbs = raw_area + raw_frac065 + raw_ebistd + raw_shadow_inv

feat_arrays = {}
for name in CANOPY_NAMES + V6_NAMES:
    feat_arrays[name] = zc([r[4][name] for r in rowsAll])
for name in ["TBL", "ICBI", "TICBL", "UBI", "CCAB"]:
    feat_arrays[name] = zc([r[4][name] for r in rowsAll])
feat_arrays["SBS"] = zc(sbs)

candidates = {}
for name, arr in feat_arrays.items():
    candidates[name] = arr
    candidates[f"{name}_x_cultivar"] = arr * uefF
print(f"total candidates (feature x [plain, x cultivar]): {len(candidates)}")

partial_F = {}
for name, col in candidates.items():
    X1 = np.column_stack(list(base_cols.values()) + [col])
    rss1, p1, _ = ols_rss(X1, yF)
    d1 = p1 - p0; d2 = nF - p1
    if d1 <= 0 or d2 <= 0: continue
    Fs = ((rss0 - rss1) / d1) / (rss1 / d2)
    pf = float(f_pvalue(Fs, d1, d2))
    beta, *_ = np.linalg.lstsq(X1, yF, rcond=None)
    r2_1, cv_1 = r2_cv(X1, yF)
    partial_F[name] = {"F": round(float(Fs), 3), "p": round(pf, 4), "coef": round(float(beta[-1]), 4),
                        "R2_with_term": r2_1, "cvR2_with_term": cv_1, "cvR2_gain": round(cv_1 - cv_base, 4)}


def bh_fdr(pvals_dict):
    names = list(pvals_dict.keys()); ps = sorted((pvals_dict[n], n) for n in names); m = len(ps)
    adj = {}; prev = 1.0
    for rank in range(m - 1, -1, -1):
        p, n = ps[rank]; val = min(prev, p * m / (rank + 1)); adj[n] = round(val, 4); prev = val
    return adj


fdr = bh_fdr({n: partial_F[n]["p"] for n in partial_F})
for n in partial_F: partial_F[n]["p_fdr"] = fdr[n]
n_survive = sum(1 for n in partial_F if partial_F[n]["p_fdr"] < 0.05)
names_by_p = sorted(partial_F.keys(), key=lambda n: partial_F[n]["p"])
names_by_gain = sorted(partial_F.keys(), key=lambda n: -partial_F[n]["cvR2_gain"])
print(f"\n{n_survive} of {len(partial_F)} candidates survive FDR<0.05 (baseline cvR2={cv_base}). Top 10 by raw p:")
for n in names_by_p[:10]:
    print(f"  {n}: F={partial_F[n]['F']} p={partial_F[n]['p']} p_fdr={partial_F[n]['p_fdr']} coef={partial_F[n]['coef']:+.3f} cvR2gain={partial_F[n]['cvR2_gain']:+.4f}")
print("\nTop 10 by raw cvR2 gain (FDR-ignored, one-by-one):")
for n in names_by_gain[:10]:
    print(f"  {n}: cvR2gain={partial_F[n]['cvR2_gain']:+.4f} p={partial_F[n]['p']} p_fdr={partial_F[n]['p_fdr']}")

# ---------- forward selection + 6-seed robustness across ALL candidates ----------
def forward_select(base_cols_, candidate_cols, y, seed=None):
    selected = []
    Xd = np.column_stack(list(base_cols_.values()))
    cur_r2, cur_cv = r2_cv(Xd, y, seed=seed if seed is not None else CV_SEED)
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
    return selected, cur_r2, cur_cv


sel_main, r2_main, cv_main = forward_select(dict(base_cols), candidates, yF, seed=CV_SEED)
print(f"\nforward selection (main seed 42): selected={sel_main} final_cvR2={cv_main} (gain={round(cv_main-cv_base,4)})")
robust = []
for seed in [1, 7, 13, 99, 2024]:
    sel_s, r2_s, cv_s = forward_select(dict(base_cols), dict(candidates), yF, seed=seed)
    robust.append({"seed": seed, "selected": sel_s, "final_cvR2": cv_s})
    print(f"  seed {seed}: selected={sel_s} final_cvR2={cv_s}")
sel_counter = Counter()
for r in [{"selected": sel_main}] + robust:
    for s in r["selected"]: sel_counter[s] += 1
print(f"how often selected of 6 runs: {dict(sel_counter)}")

# ---------- specific re-check of the previously flagged near-misses ----------
NEAR_MISSES = ["ShadowFraction", "BrightFraction", "UBI_x_cultivar", "EBI_Norm_P90_x_cultivar",
               "EBI_Norm_std_x_cultivar", "EBI_Norm_CV_x_cultivar"]
near_miss_report = {n: partial_F.get(n) for n in NEAR_MISSES if n in partial_F}
print("\nPreviously-flagged near-misses, re-checked on outlier-free data:")
for n, v in near_miss_report.items():
    print(f"  {n}: {v}")

RESULTS = {
    "note": "Sections 38/39/41/43/44 re-run after removing the 15 actual-yield "
            "anomaly tree-years (9 high modZ>3.5 + 6 low <1.0kg domain trim, "
            "sections 45-46). Baseline recommended model refit fresh on this n.",
    "n": nF, "n_excluded_tree_years_in_this_join": n_excluded_and_matched,
    "baseline": {"R2": r2_base, "cvR2": cv_base},
    "n_candidates": len(candidates),
    "partial_F_all_candidates_FDR_corrected": partial_F,
    "n_survive_FDR": n_survive,
    "top10_by_raw_p": names_by_p[:10], "top10_by_cvR2_gain": names_by_gain[:10],
    "forward_selection_main_seed42": {"selected": sel_main, "final_cvR2": cv_main, "gain": round(cv_main - cv_base, 4)},
    "robustness_5_seeds": robust, "how_often_selected_of_6": dict(sel_counter),
    "near_miss_recheck": near_miss_report,
}
with open(os.path.join(OUT, "All_Features_No_Outliers_Recheck.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")
