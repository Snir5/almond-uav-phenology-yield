#!/usr/bin/env python3
"""
yield_model_ebi_pixel_features_test.py: tests the new EBI pixel-distribution
feature bank (Master_Trees_EBIPixelFeatures_v6.xlsx, section 39) against measured
yield: standalone (2023, all matched trees, highest power), a redundancy check
(many of these 18 candidates are constructed from overlapping information, e.g.
percentiles, or the same threshold at 3 cutoffs), pairwise combinations (FDR
corrected across the true family size), and combined with the full recommended
model via forward selection + 5-seed robustness, matching sections 29-30/37-38
methodology throughout.
-> Results_Analysis/08_UEF53_Rerun/Yield_Model_EBI_Pixel_Features.json + figure
"""
import os, sys, csv, math, json, itertools, sqlite3, struct
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

M = load_master(); cultivar = [r.get("cultivar") for r in M]
mlat = np.array([r.get("Latitude") for r in M], float); mlon = np.array([r.get("Longitude") for r in M], float)

wb = openpyxl.load_workbook(os.path.join(BASE, "4band_mosaic", "Master_Trees_EBIPixelFeatures_v6.xlsx"), read_only=True)
ws = wb.active
rows_xlsx = list(ws.iter_rows(values_only=True)); header = rows_xlsx[0]; hidx = {h: i for i, h in enumerate(header)}
v6_by_id = {r[hidx["Tree_ID"]]: {h: r[hidx[h]] for h in header} for r in rows_xlsx[1:]}
v6_for_master = [v6_by_id.get(r.get("Tree_ID")) for r in M]
n_matched = sum(1 for v in v6_for_master if v is not None)
print(f"v6 pixel-feature match (by Tree_ID): {n_matched}/{len(M)}")

CAND_FEATURES = ["EBI_Norm_std", "EBI_Norm_var", "EBI_Norm_median", "EBI_Norm_P10", "EBI_Norm_P25",
                  "EBI_Norm_P75", "EBI_Norm_P90", "EBI_Norm_IQR", "EBI_Norm_skew", "EBI_Norm_kurtosis",
                  "EBI_Norm_CV", "EBI_OtsuFrac", "HighEBI_frac_055", "HighEBI_density_055",
                  "HighEBI_frac_065", "HighEBI_density_065", "HighEBI_frac_075", "HighEBI_density_075"]


def get_feat(j, yr, name):
    v = v6_for_master[j]
    if v is None: return None
    val = v.get(f"{name}_{yr}")
    return None if val is None else float(val)


# ================= redundancy check among the 18 candidates (2023) =================
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
    ebi = M[j].get("EBI_Norm_2023")
    feats = {name: get_feat(j, 2023, name) for name in CAND_FEATURES}
    if ebi is None or any(v is None for v in feats.values()): continue
    rows2023.append((t, j, ymeas[t][2023], float(ebi), feats))
n2023 = len(rows2023)
print(f"\n2023, all matched trees with full v6 pixel-feature set: n={n2023}")


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


yv = np.array([r[2] for r in rows2023])
uef = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rows2023])
ebi_arr = zc([r[3] for r in rows2023])
feat_arrays = {name: zc([r[4][name] for r in rows2023]) for name in CAND_FEATURES}

print("\n--- redundancy: correlation matrix among the 18 candidates (2023) ---")
mat = np.column_stack([feat_arrays[n] for n in CAND_FEATURES])
corr_mat = np.corrcoef(mat.T)
high_pairs = []
for i in range(len(CAND_FEATURES)):
    for j2 in range(i + 1, len(CAND_FEATURES)):
        if abs(corr_mat[i, j2]) > 0.9:
            high_pairs.append((CAND_FEATURES[i], CAND_FEATURES[j2], round(float(corr_mat[i, j2]), 3)))
print(f"{len(high_pairs)} pairs with |r|>0.9 (near-redundant):")
for a, b, r in high_pairs[:20]:
    print(f"  {a} <-> {b}: r={r}")


def r_p(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if xs.std() == 0: return None, None
    r = float(np.corrcoef(xs, ys)[0, 1])
    return round(r, 4), round(float(r_pvalue(r, len(xs))), 4)


def bh_fdr(pvals_dict):
    names = list(pvals_dict.keys()); ps = sorted((pvals_dict[n], n) for n in names); m = len(ps)
    adj = {}; prev = 1.0
    for rank in range(m - 1, -1, -1):
        p, n = ps[rank]; val = min(prev, p * m / (rank + 1)); adj[n] = round(val, 4); prev = val
    return adj


print("\n--- standalone correlations with 2023 measured yield ---")
standalone = {}
pvals_pooled = {}
for name in CAND_FEATURES:
    arr = feat_arrays[name]
    r_pool, p_pool = r_p(arr, yv)
    r_uef, p_uef = r_p(arr[uef == 1], yv[uef == 1])
    r_53, p_53 = r_p(arr[uef == 0], yv[uef == 0])
    standalone[name] = {"pooled": {"r": r_pool, "p": p_pool}, "UEF": {"r": r_uef, "p": p_uef}, "cv53": {"r": r_53, "p": p_53}}
    pvals_pooled[name] = p_pool
    print(f"  {name}: pooled r={r_pool} p={p_pool} | UEF r={r_uef} p={p_uef} | cv53 r={r_53} p={p_53}")

fdr_pooled = bh_fdr(pvals_pooled)
n_survive_standalone = sum(1 for n in fdr_pooled if fdr_pooled[n] < 0.05)
print(f"\n{n_survive_standalone} of {len(CAND_FEATURES)} survive FDR<0.05 (pooled, standalone):")
for n in sorted(fdr_pooled, key=lambda n: fdr_pooled[n])[:5]:
    print(f"  {n}: p_fdr={fdr_pooled[n]}")

# ================= pairwise combos among the 18 (2023) =================
one = np.ones(n2023)
X_base23 = np.column_stack([one, uef])
rss0b, p0b, _ = ols_rss(X_base23, yv)
all_feats23 = dict(feat_arrays); all_feats23["EBI"] = ebi_arr
pair_names = list(itertools.combinations(all_feats23.keys(), 2))
pairwise_F = {}
for a, b in pair_names:
    col = all_feats23[a] * all_feats23[b]
    X1 = np.column_stack([one, uef, col])
    rss1, p1, _ = ols_rss(X1, yv)
    d1 = p1 - p0b; d2 = n2023 - p1
    if d1 <= 0 or d2 <= 0: continue
    Fs = ((rss0b - rss1) / d1) / (rss1 / d2)
    pf = float(f_pvalue(Fs, d1, d2))
    pairwise_F[f"{a}_x_{b}"] = {"F": round(float(Fs), 3), "p": round(pf, 4)}
fdr_pairwise = bh_fdr({n: pairwise_F[n]["p"] for n in pairwise_F})
for n in pairwise_F: pairwise_F[n]["p_fdr"] = fdr_pairwise[n]
n_survive_pw = sum(1 for n in pairwise_F if pairwise_F[n]["p_fdr"] < 0.05)
names_sorted = sorted(pairwise_F.keys(), key=lambda n: pairwise_F[n]["p"])
print(f"\n{n_survive_pw} of {len(pairwise_F)} pairwise combos (incl. EBI) survive FDR<0.05. Top 5:")
for n in names_sorted[:5]:
    print(f"  {n}: F={pairwise_F[n]['F']} p={pairwise_F[n]['p']} p_fdr={pairwise_F[n]['p_fdr']}")

# ================= combined with full recommended model (pooled 2022-2023 + physiology) =================
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")
PHYS_FIELDS_MIN = ["Growth_April"]
PHYS_GPKG_COLS_MIN = ["Growth__April"]


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
    cols = ",".join(f'"{c}"' for c in PHYS_GPKG_COLS_MIN)
    cur.execute(f'SELECT "Row","Plot",cultivar,{cols},geom FROM "{tbl}"')
    rowsg = cur.fetchall(); con.close(); out = []
    for row_ in rowsg:
        plot = row_[1]; vals = row_[3:3 + len(PHYS_FIELDS_MIN)]; blob = row_[-1]
        if plot != "A" or any(v is None for v in vals): continue
        x, y = gpkg_geom_centroid(blob); d = {"x": x, "y": y}
        for f, v in zip(PHYS_FIELDS_MIN, vals): d[f] = float(v)
        out.append(d)
    return out


mx = np.array([r.get("X_UTM") for r in M], float); my = np.array([r.get("Y_UTM") for r in M], float)
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

rowsFull = []
for t, j in ymap.items():
    for yr in (2022, 2023):
        if ymeas[t].get(yr) is None or cultivar[j] not in ("UEF", "53") or j not in phys_match[yr]: continue
        feats = {name: get_feat(j, yr, name) for name in CAND_FEATURES}
        if any(v is None for v in feats.values()) or M[j].get(f"EBI_Norm_{yr}") is None: continue
        rowsFull.append((t, j, yr, ymeas[t][yr], feats))
nF = len(rowsFull)
print(f"\npooled 2022-2023, matched with full recommended model features: n={nF}")

yF = np.array([r[3] for r in rowsFull])
uefF = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rowsFull])
ebiF = zc([M[r[1]].get(f"EBI_Norm_{r[2]}") for r in rowsFull])
CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
climF = zc([CP[r[2]] for r in rowsFull])
gaF = zc([phys_match[r[2]][r[1]]["Growth_April"] for r in rowsFull])
oneF = np.ones(nF)
base_cols = {"intercept": oneF, "cultivar": uefF, "EBI": ebiF, "EBIxcultivar": ebiF * uefF, "climate": climF, "Growth_April": gaF}
X_base = np.column_stack(list(base_cols.values()))


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
print(f"baseline (recommended model): R2={r2_base} cvR2={cv_base}")

new_featF = {name: zc([r[4][name] for r in rowsFull]) for name in CAND_FEATURES}
rss0, p0, _ = ols_rss(X_base, yF)
candidates = {}
for name, arr in new_featF.items():
    candidates[name] = arr
    candidates[f"{name}_x_cultivar"] = arr * uefF

partial_F = {}
for name, col in candidates.items():
    X1 = np.column_stack(list(base_cols.values()) + [col])
    rss1, p1, _ = ols_rss(X1, yF)
    d1 = p1 - p0; d2 = nF - p1
    Fs = ((rss0 - rss1) / d1) / (rss1 / d2)
    pf = float(f_pvalue(Fs, d1, d2))
    beta, *_ = np.linalg.lstsq(X1, yF, rcond=None)
    r2_1, cv_1 = r2_cv(X1, yF)
    partial_F[name] = {"F": round(float(Fs), 3), "p": round(pf, 4), "coef": round(float(beta[-1]), 4), "cvR2_with_term": cv_1}
fdr_final = bh_fdr({n: partial_F[n]["p"] for n in partial_F})
for n in partial_F: partial_F[n]["p_fdr"] = fdr_final[n]
n_survive_final = sum(1 for n in partial_F if partial_F[n]["p_fdr"] < 0.05)
names_sorted_final = sorted(partial_F.keys(), key=lambda n: partial_F[n]["p"])
print(f"\n{n_survive_final} of {len(partial_F)} candidates (added to full recommended model) survive FDR<0.05. Top 8:")
for n in names_sorted_final[:8]:
    print(f"  {n}: F={partial_F[n]['F']} p={partial_F[n]['p']} p_fdr={partial_F[n]['p_fdr']} coef={partial_F[n]['coef']:+.3f} cvR2={partial_F[n]['cvR2_with_term']}")


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


sel_main, hist_main, r2_main, cv_main = forward_select(dict(base_cols), candidates, yF)
print(f"\nforward selection over {len(candidates)} candidates (main seed 42): "
      f"selected={sel_main} final_cvR2={cv_main} (gain={round(cv_main-cv_base,4)})")
robust = []
for seed in [1, 7, 13, 99, 2024]:
    sel_s, hist_s, r2_s, cv_s = forward_select(dict(base_cols), dict(candidates), yF, seed=seed)
    robust.append({"seed": seed, "selected": sel_s, "final_cvR2": cv_s})
    print(f"  seed {seed}: selected={sel_s} final_cvR2={cv_s}")
sel_counter = Counter()
for r in [{"selected": sel_main}] + robust:
    for s in r["selected"]: sel_counter[s] += 1
print(f"how often selected of 6: {dict(sel_counter)}")

RESULTS = {
    "note": "Tests the new true-EBI pixel-distribution feature bank (section 39, "
            "Master_Trees_EBIPixelFeatures_v6.xlsx, sanity-checked exact match to canonical "
            "EBI_Norm) standalone, for redundancy among candidates, in pairwise combination, "
            "and combined with the full recommended model, same rigor as sections 29-30/37-38.",
    "n_2023_standalone": n2023,
    "redundancy_high_corr_pairs_gt_0.9": high_pairs,
    "standalone_correlations_2023": standalone,
    "standalone_fdr": fdr_pooled, "n_survive_standalone_FDR": n_survive_standalone,
    "pairwise_2023": {"n_candidates": len(pairwise_F), "n_survive_FDR": n_survive_pw,
                       "top10": {n: pairwise_F[n] for n in names_sorted[:10]}},
    "combined_with_full_model": {"n": nF, "baseline_cvR2": cv_base,
                                  "partial_F_FDR_corrected": partial_F, "n_survive_FDR": n_survive_final,
                                  "forward_selection_main_seed42": {"selected": sel_main, "final_cvR2": cv_main, "gain": round(cv_main - cv_base, 4)},
                                  "robustness_5_seeds": robust, "how_often_selected_of_6": dict(sel_counter)},
}
with open(os.path.join(OUT, "Yield_Model_EBI_Pixel_Features.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
a = ax[0]
vals = [standalone[n]["pooled"]["r"] or 0 for n in CAND_FEATURES]
colors_ = ["#C0392B" if fdr_pooled[n] < 0.05 else "#95a5a6" for n in CAND_FEATURES]
a.barh(range(len(CAND_FEATURES)), vals, color=colors_)
a.set_yticks(range(len(CAND_FEATURES))); a.set_yticklabels(CAND_FEATURES, fontsize=7.5); a.invert_yaxis()
a.axvline(0, color="k", lw=0.8)
a.set_title(f"2023 standalone correlation with yield\n(n={n2023}, red=survives FDR<0.05)", fontsize=10, fontweight="bold")
b = ax[1]
seeds = ["main42"] + [str(r["seed"]) for r in robust]
gains = [round(cv_main - cv_base, 4)] + [round(r["final_cvR2"] - cv_base, 4) for r in robust]
b.bar(seeds, gains, color="#16A085"); b.axhline(0, color="k", lw=1)
b.set_title("CV R2 gain, EBI pixel-distribution features\non top of recommended model", fontsize=10, fontweight="bold")
b.grid(axis="y", alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "Yield_Model_EBI_Pixel_Features.png"), dpi=130)
print("saved figure")
