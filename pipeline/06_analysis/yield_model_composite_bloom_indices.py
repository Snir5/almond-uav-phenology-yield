#!/usr/bin/env python3
"""
yield_model_composite_bloom_indices.py: per request, acting as a canopy/remote-
sensing domain expert, builds a small set of HYPOTHESIS-DRIVEN composite bloom
indices (not another blind combinatorial search), from the features already
extracted (sections 37-39): canopy size (CanopyArea_m2), a real EBI threshold-
crossing fraction (HighEBI_frac_065), a real areal bright-pixel density
(HighEBI_density_065), bloom intensity (EBI_Norm), within-canopy heterogeneity
(EBI_Norm_std, EBI_Norm_CV), and structural channels (ShadowFraction,
CanopyCoverFraction).

IMPORTANT ALGEBRA CHECK done before building anything: the user's suggested
"canopy size x EBI-threshold % x EBI density" literally collapses to just
EBI_Norm x HighEBI_frac_065, because section 38's "EBI_density" was defined as
EBI_Norm / CanopyArea (intensity per unit area), so
CanopyArea x frac x (EBI_Norm/CanopyArea) = EBI_Norm x frac, i.e. canopy area
cancels out algebraically and contributes NO new information in that literal
formula. To genuinely bring canopy size into a composite (not cancel it out),
size must be multiplied by a feature that is NOT itself already divided by
area; section 39's HighEBI_density_065 (count of bright PIXELS per m^2, not
EBI/area) also cancels area when multiplied by area (recovers a raw pixel
count, proportional to CanopyArea x frac up to that year's GSD). So every
composite below is checked to make sure it uses genuinely independent
information (size, threshold-fraction, intensity, patchiness, or structure),
not accidentally-redundant restatements of the same 1-2 underlying quantities.

Composite indices built (with the domain rationale for each):
  TBL  (Total Bloom Load)          = CanopyArea_m2 x HighEBI_frac_065
        -> absolute m^2 of high-intensity bloom, not just its fraction; a big
           tree at 20% bright bloom may carry more total bloom than a small
           tree at 40%.
  ICBI (Illumination-Corrected Bloom Intensity) = EBI_Norm x (1-ShadowFraction)
        -> mean intensity adjusted for how shadowed the canopy is, shadow can
           suppress apparent brightness independent of true bloom state.
  TICBL (Total Illumination-Corrected Bloom Load) = CanopyArea_m2 x EBI_Norm x (1-ShadowFraction)
        -> 3-way: total size x mean intensity x illumination correction.
  UBI  (Uniform Bloom Intensity)   = EBI_Norm x (1 - EBI_Norm_CV)
        -> rewards trees blooming both brightly AND evenly (patchy bloom may
           set fruit less evenly than uniform bloom of the same mean).
  CCAB (Canopy-Completeness-Adjusted Bloom) = HighEBI_frac_065 x CanopyCoverFraction
        -> bright-bloom fraction weighted by how complete/gap-free the canopy
           itself is, distinguishing a dense blooming canopy from a sparse one.
  SBS  (Structural Bloom Score, additive z-composite) =
        z(CanopyArea) + z(HighEBI_frac_065) + z(EBI_Norm_std) + z(1-ShadowFraction)
        -> simple, transparent omnibus score combining size, threshold bloom,
           patchiness, and illumination.
  PC1/PC2 of the full raw feature bank (data-driven alternative to hand-built
  composites, via PCA/SVD, stats_utils.pca).

-> Results_Analysis/08_UEF53_Rerun/Yield_Model_Composite_Bloom_Indices.json
"""
import os, sys, csv, math, json, sqlite3, struct
from collections import defaultdict, Counter
import numpy as np
import openpyxl
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import ols_rss, f_pvalue, r_pvalue, pca

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
CV_SEED = 42

M = load_master(); cultivar = [r.get("cultivar") for r in M]
mlat = np.array([r.get("Latitude") for r in M], float); mlon = np.array([r.get("Longitude") for r in M], float)


def load_xlsx_by_id(path):
    wb = openpyxl.load_workbook(path, read_only=True); ws = wb.active
    rows = list(ws.iter_rows(values_only=True)); header = rows[0]; hidx = {h: i for i, h in enumerate(header)}
    return {r[hidx["Tree_ID"]]: {h: r[hidx[h]] for h in header} for r in rows[1:]}


canopy_by_id = load_xlsx_by_id(os.path.join(BASE, "4band_mosaic", "Master_Trees_CanopyStructure_v5.xlsx"))
v6_by_id = load_xlsx_by_id(os.path.join(BASE, "4band_mosaic", "Master_Trees_EBIPixelFeatures_v6.xlsx"))
canopy_for_master = [canopy_by_id.get(r.get("Tree_ID")) for r in M]
v6_for_master = [v6_by_id.get(r.get("Tree_ID")) for r in M]


def build_composites(j, yr):
    c = canopy_for_master[j]; v6 = v6_for_master[j]
    ebi = M[j].get(f"EBI_Norm_{yr}")
    if c is None or v6 is None or ebi is None: return None
    area = c.get(f"CanopyArea_m2_{yr}"); cover = c.get(f"CanopyCoverFraction_{yr}"); shadow = c.get(f"ShadowFraction_{yr}")
    frac065 = v6.get(f"HighEBI_frac_065_{yr}"); ebi_std = v6.get(f"EBI_Norm_std_{yr}"); ebi_cv = v6.get(f"EBI_Norm_CV_{yr}")
    if any(v is None for v in [area, cover, shadow, frac065, ebi_std, ebi_cv]): return None
    ebi = float(ebi); area = float(area); cover = float(cover); shadow = float(shadow)
    frac065 = float(frac065); ebi_std = float(ebi_std); ebi_cv = float(ebi_cv)
    return {
        "CanopyArea": area, "HighEBI_frac_065": frac065, "EBI_Norm": ebi, "ShadowFraction": shadow,
        "CanopyCoverFraction": cover, "EBI_Norm_std": ebi_std, "EBI_Norm_CV": ebi_cv,
        "TBL": area * frac065,
        "ICBI": ebi * (1 - shadow),
        "TICBL": area * ebi * (1 - shadow),
        "UBI": ebi * (1 - ebi_cv),
        "CCAB": frac065 * cover,
    }


COMPOSITE_NAMES = ["TBL", "ICBI", "TICBL", "UBI", "CCAB", "SBS"]
RAW_FOR_PCA = ["CanopyArea", "HighEBI_frac_065", "EBI_Norm", "ShadowFraction", "CanopyCoverFraction", "EBI_Norm_std"]

# ================= 2023, all matched trees =================
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
    comp = build_composites(j, 2023)
    if comp is None: continue
    rows2023.append((t, j, ymeas[t][2023], comp))
n2023 = len(rows2023)
print(f"2023, all matched trees: n={n2023}")


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


yv = np.array([r[2] for r in rows2023])
uef = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rows2023])

z_raw = {name: zc([r[3][name] for r in rows2023]) for name in RAW_FOR_PCA}
sbs = z_raw["CanopyArea"] + z_raw["HighEBI_frac_065"] + z_raw["EBI_Norm_std"] + (-z_raw["ShadowFraction"])

comp_arrays = {name: zc([r[3][name] for r in rows2023]) for name in ["TBL", "ICBI", "TICBL", "UBI", "CCAB"]}
comp_arrays["SBS"] = zc(sbs)


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


print("\n--- standalone correlations, composite indices, 2023 ---")
standalone = {}; pvals = {}
for name in COMPOSITE_NAMES:
    arr = comp_arrays[name]
    r_pool, p_pool = r_p(arr, yv)
    r_uef, p_uef = r_p(arr[uef == 1], yv[uef == 1])
    r_53, p_53 = r_p(arr[uef == 0], yv[uef == 0])
    standalone[name] = {"pooled": {"r": r_pool, "p": p_pool}, "UEF": {"r": r_uef, "p": p_uef}, "cv53": {"r": r_53, "p": p_53}}
    pvals[name] = p_pool
    print(f"  {name}: pooled r={r_pool} p={p_pool} | UEF r={r_uef} p={p_uef} | cv53 r={r_53} p={p_53}")
fdr_standalone = bh_fdr(pvals)
n_survive_standalone = sum(1 for n in fdr_standalone if fdr_standalone[n] < 0.05)
print(f"{n_survive_standalone} of {len(COMPOSITE_NAMES)} survive FDR<0.05 (small, curated family): "
      f"{ {n: fdr_standalone[n] for n in fdr_standalone} }")

# ---- PCA on the raw feature bank ----
X_pca = np.column_stack([np.array([r[3][name] for r in rows2023], float) for name in RAW_FOR_PCA])
scores, Vt, explained, mu, sd = pca(X_pca, n_components=3)
print(f"\nPCA on {RAW_FOR_PCA}: explained variance ratio PC1-3 = {[round(float(e),3) for e in explained]}")
print(f"PC1 loadings: { {n: round(float(v),3) for n, v in zip(RAW_FOR_PCA, Vt[0])} }")
pc1 = zc(scores[:, 0]); pc2 = zc(scores[:, 1])
r_pc1, p_pc1 = r_p(pc1, yv); r_pc2, p_pc2 = r_p(pc2, yv)
print(f"PC1 vs yield: r={r_pc1} p={p_pc1} | PC2 vs yield: r={r_pc2} p={p_pc2}")

# ================= combined with full recommended model =================
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")


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
        x, y = gpkg_geom_centroid(blob)
        out.append({"x": x, "y": y, "Growth_April": float(ga)})
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
        comp = build_composites(j, yr)
        if comp is None: continue
        rowsFull.append((t, j, yr, ymeas[t][yr], comp))
nF = len(rowsFull)
print(f"\npooled 2022-2023, matched with full recommended model: n={nF}")

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

z_rawF = {name: zc([r[4][name] for r in rowsFull]) for name in RAW_FOR_PCA}
sbsF = z_rawF["CanopyArea"] + z_rawF["HighEBI_frac_065"] + z_rawF["EBI_Norm_std"] + (-z_rawF["ShadowFraction"])
comp_arraysF = {name: zc([r[4][name] for r in rowsFull]) for name in ["TBL", "ICBI", "TICBL", "UBI", "CCAB"]}
comp_arraysF["SBS"] = zc(sbsF)

rss0, p0, _ = ols_rss(X_base, yF)
candidates = {}
for name, arr in comp_arraysF.items():
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
print(f"\n{n_survive_final} of {len(partial_F)} candidates (added to full recommended model, small "
      f"curated family of {len(COMPOSITE_NAMES)} composites x 2) survive FDR<0.05:")
for n in names_sorted_final:
    print(f"  {n}: F={partial_F[n]['F']} p={partial_F[n]['p']} p_fdr={partial_F[n]['p_fdr']} "
          f"coef={partial_F[n]['coef']:+.3f} cvR2={partial_F[n]['cvR2_with_term']}")


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
print(f"\nforward selection (main seed 42): selected={sel_main} final_cvR2={cv_main} (gain={round(cv_main-cv_base,4)})")
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
    "note": "Domain-motivated composite bloom indices (canopy remote-sensing expert framing), "
            "not a blind search: TBL, ICBI, TICBL, UBI, CCAB, SBS. Algebra check performed first: "
            "the user's literal 'size x threshold% x density' collapses to just intensity x "
            "threshold% since section-38 'EBI_density' = EBI/area, so area cancels; composites "
            "here are checked to combine genuinely independent axes instead.",
    "composite_definitions": {
        "TBL": "CanopyArea_m2 x HighEBI_frac_065 (total m^2 of high-intensity bloom)",
        "ICBI": "EBI_Norm x (1-ShadowFraction) (shadow-corrected mean intensity)",
        "TICBL": "CanopyArea_m2 x EBI_Norm x (1-ShadowFraction) (3-way: size x intensity x illumination)",
        "UBI": "EBI_Norm x (1-EBI_Norm_CV) (uniform bloom intensity)",
        "CCAB": "HighEBI_frac_065 x CanopyCoverFraction (bloom fraction adjusted for canopy completeness)",
        "SBS": "z(CanopyArea)+z(HighEBI_frac_065)+z(EBI_Norm_std)+z(1-ShadowFraction) (additive composite score)",
    },
    "n_2023_standalone": n2023,
    "standalone_correlations_2023": standalone, "standalone_fdr": fdr_standalone, "n_survive_standalone_FDR": n_survive_standalone,
    "pca_on_raw_feature_bank": {"features_used": RAW_FOR_PCA, "explained_variance_ratio": [round(float(e), 4) for e in explained],
                                 "PC1_loadings": {n: round(float(v), 3) for n, v in zip(RAW_FOR_PCA, Vt[0])},
                                 "PC1_vs_yield": {"r": r_pc1, "p": p_pc1}, "PC2_vs_yield": {"r": r_pc2, "p": p_pc2}},
    "combined_with_full_model": {"n": nF, "baseline_cvR2": cv_base,
                                  "partial_F_FDR_corrected": partial_F, "n_survive_FDR": n_survive_final,
                                  "forward_selection_main_seed42": {"selected": sel_main, "final_cvR2": cv_main, "gain": round(cv_main - cv_base, 4)},
                                  "robustness_5_seeds": robust, "how_often_selected_of_6": dict(sel_counter)},
}
with open(os.path.join(OUT, "Yield_Model_Composite_Bloom_Indices.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
a = ax[0]
vals = [standalone[n]["pooled"]["r"] or 0 for n in COMPOSITE_NAMES]
colors_ = ["#C0392B" if fdr_standalone[n] < 0.05 else "#95a5a6" for n in COMPOSITE_NAMES]
a.barh(range(len(COMPOSITE_NAMES)), vals, color=colors_)
a.set_yticks(range(len(COMPOSITE_NAMES))); a.set_yticklabels(COMPOSITE_NAMES); a.invert_yaxis()
a.axvline(0, color="k", lw=0.8)
a.set_title(f"2023 standalone correlation with yield\n(n={n2023}, curated composite indices)", fontsize=10, fontweight="bold")
b = ax[1]
seeds = ["main42"] + [str(r["seed"]) for r in robust]
gains = [round(cv_main - cv_base, 4)] + [round(r["final_cvR2"] - cv_base, 4) for r in robust]
b.bar(seeds, gains, color="#16A085"); b.axhline(0, color="k", lw=1)
b.set_title("CV R2 gain, composite bloom indices\non top of recommended model", fontsize=10, fontweight="bold")
b.grid(axis="y", alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "Yield_Model_Composite_Bloom_Indices.png"), dpi=130)
print("saved figure")
