#!/usr/bin/env python3
"""
canopy_feature_bank_fdr_check.py: follow-up on yield_model_canopy_feature_bank.py.
ShadowFraction was selected 5/6 (Part C) and 6/6 (Part C2, full recommended-model
baseline) by forward selection, the strongest robustness signal seen anywhere in
this project. Before treating that as adopted, run the same partial-F + FDR
discipline used in sections 29/30: test all 22 candidates (11 canopy features x
[plain, x cultivar]) against the FULL recommended-model baseline
(cultivar+EBIxcultivar+climate+Growth_April, n=167), FDR-correct across all 22,
and separately report ShadowFraction/BrightFraction's own partial-F cleanly.
"""
import os, sys, csv, math, json, sqlite3, struct
from collections import defaultdict
import numpy as np
import openpyxl
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import ols_rss, f_pvalue

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")
PHYS_FIELDS = ["SWP_April", "Growth_April", "SWP_MayJune", "Growth_MayJune", "CNC_June", "SWP_June", "Growth_June"]
PHYS_GPKG_COLS = ["SWP_April", "Growth__April", "SWP_May.June", "Growth_May.June", "CNC_June", "SWP_June", "Growth_June"]

M = load_master(); cultivar = [r.get("cultivar") for r in M]
mlat = np.array([r.get("Latitude") for r in M], float); mlon = np.array([r.get("Longitude") for r in M], float)
mx = np.array([r.get("X_UTM") for r in M], float); my = np.array([r.get("Y_UTM") for r in M], float)


def load_xlsx_by_id(path):
    wb = openpyxl.load_workbook(path, read_only=True); ws = wb.active
    rows = list(ws.iter_rows(values_only=True)); header = rows[0]; hidx = {h: i for i, h in enumerate(header)}
    return {r[hidx["Tree_ID"]]: {h: r[hidx[h]] for h in header} for r in rows[1:]}


canopy_by_id = load_xlsx_by_id(os.path.join(BASE, "4band_mosaic", "Master_Trees_CanopyStructure_v5.xlsx"))
bloomfrac_by_id = load_xlsx_by_id(os.path.join(BASE, "4band_mosaic", "Master_Trees_BloomFraction_v3.xlsx"))
canopy_for_master = [canopy_by_id.get(r.get("Tree_ID")) for r in M]
bloomfrac_for_master = [bloomfrac_by_id.get(r.get("Tree_ID")) for r in M]

FEATURE_NAMES = ["CanopyArea", "CanopyCover", "ShadowFraction", "BloomFraction", "BrightFraction",
                  "EBI_density", "BloomVolume", "BloomPixelVolume", "EBI_x_BloomFraction",
                  "EBI_x_CanopyCover", "BrightFraction_x_CanopyArea"]


def build_features(j, yr):
    c = canopy_for_master[j]; b = bloomfrac_for_master[j]
    ebi = M[j].get(f"EBI_Norm_{yr}")
    if c is None or b is None or ebi is None: return None
    area = c.get(f"CanopyArea_m2_{yr}"); cover = c.get(f"CanopyCoverFraction_{yr}")
    shadow = c.get(f"ShadowFraction_{yr}"); pix = c.get(f"CanopyPixelCount_{yr}")
    bf = b.get(f"BloomFraction_{yr}"); brf = b.get(f"BrightFraction_{yr}")
    if any(v is None for v in [area, cover, shadow, pix, bf, brf]): return None
    ebi = float(ebi); area = float(area); cover = float(cover); shadow = float(shadow); pix = float(pix); bf = float(bf); brf = float(brf)
    return {"CanopyArea": area, "CanopyCover": cover, "ShadowFraction": shadow, "BloomFraction": bf, "BrightFraction": brf,
            "EBI_density": ebi / area if area > 0 else np.nan, "BloomVolume": ebi * area, "BloomPixelVolume": bf * pix,
            "EBI_x_BloomFraction": ebi * bf, "EBI_x_CanopyCover": ebi * cover, "BrightFraction_x_CanopyArea": brf * area}


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
    cols = ",".join(f'"{c}"' for c in PHYS_GPKG_COLS)
    cur.execute(f'SELECT "Row","Plot",cultivar,{cols},geom FROM "{tbl}"')
    rowsg = cur.fetchall(); con.close(); out = []
    for row_ in rowsg:
        plot = row_[1]; vals = row_[3:3 + len(PHYS_FIELDS)]; blob = row_[-1]
        if plot != "A" or any(v is None for v in vals): continue
        x, y = gpkg_geom_centroid(blob); d = {"x": x, "y": y}
        for f, v in zip(PHYS_FIELDS, vals): d[f] = float(v)
        out.append(d)
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

rowsFull = []
for t, j in ymap.items():
    for yr in (2022, 2023):
        if ymeas[t].get(yr) is None or cultivar[j] not in ("UEF", "53") or j not in phys_match[yr]: continue
        feats = build_features(j, yr)
        if feats is None: continue
        rowsFull.append((t, j, yr, ymeas[t][yr], feats))
nF = len(rowsFull)
print(f"n={nF}")


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


yF = np.array([r[3] for r in rowsFull])
uefF = np.array([1.0 if cultivar[r[1]] == "UEF" else 0.0 for r in rowsFull])
ebiF = zc([M[r[1]].get(f"EBI_Norm_{r[2]}") for r in rowsFull])
CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
climF = zc([CP[r[2]] for r in rowsFull])
gaF = zc([phys_match[r[2]][r[1]]["Growth_April"] for r in rowsFull])
oneF = np.ones(nF)
base_cols = {"intercept": oneF, "cultivar": uefF, "EBI": ebiF, "EBIxcultivar": ebiF * uefF, "climate": climF, "Growth_April": gaF}
X_base = np.column_stack(list(base_cols.values()))
rss0, p0, _ = ols_rss(X_base, yF)


def r2_cv(Xd, y, k=5, seed=42):
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
print(f"baseline: R2={r2_base} cvR2={cv_base}")

new_feat = {name: zc([r[4][name] for r in rowsFull]) for name in FEATURE_NAMES}
candidates = {}
for name, arr in new_feat.items():
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
    partial_F[name] = {"F": round(float(Fs), 3), "p": round(pf, 4), "coef": round(float(beta[-1]), 4),
                        "R2_with_term": r2_1, "cvR2_with_term": cv_1}


def bh_fdr(pvals_dict):
    names = list(pvals_dict.keys()); ps = sorted((pvals_dict[n], n) for n in names); m = len(ps)
    adj = {}; prev = 1.0
    for rank in range(m - 1, -1, -1):
        p, n = ps[rank]; val = min(prev, p * m / (rank + 1)); adj[n] = round(val, 4); prev = val
    return adj


fdr = bh_fdr({n: partial_F[n]["p"] for n in partial_F})
for n in sorted(partial_F, key=lambda n: partial_F[n]["p"]):
    partial_F[n]["p_fdr"] = fdr[n]

names_sorted = sorted(partial_F.keys(), key=lambda n: partial_F[n]["p"])
n_survive = sum(1 for n in partial_F if partial_F[n]["p_fdr"] < 0.05)
print(f"\n{n_survive} of {len(partial_F)} candidates survive FDR<0.05 (baseline cvR2={cv_base}):")
for n in names_sorted[:8]:
    print(f"  {n}: F={partial_F[n]['F']} p={partial_F[n]['p']} p_fdr={partial_F[n]['p_fdr']} "
          f"coef={partial_F[n]['coef']:+.3f} cvR2_with_term={partial_F[n]['cvR2_with_term']}")

# both together
X_both = np.column_stack(list(base_cols.values()) + [new_feat["ShadowFraction"], new_feat["BrightFraction"]])
r2_both, cv_both = r2_cv(X_both, yF)
rss_both, p_both, _ = ols_rss(X_both, yF)
d1b = p_both - p0; d2b = nF - p_both
Fboth = ((rss0 - rss_both) / d1b) / (rss_both / d2b)
print(f"\nShadowFraction + BrightFraction together: R2={r2_both} cvR2={cv_both}, joint F={Fboth:.3f} p={f_pvalue(Fboth, d1b, d2b):.4f}")
print(f"correlation(ShadowFraction, BrightFraction) = {np.corrcoef(new_feat['ShadowFraction'], new_feat['BrightFraction'])[0,1]:.3f}")

RESULTS = {"n": nF, "baseline": {"R2": r2_base, "cvR2": cv_base},
           "partial_F_all_22_FDR_corrected": partial_F,
           "n_survive_FDR": n_survive,
           "shadow_plus_bright_together": {"R2": r2_both, "cvR2": cv_both, "joint_F": round(float(Fboth), 3), "joint_p": round(float(f_pvalue(Fboth, d1b, d2b)), 4)},
           "corr_shadow_vs_bright": round(float(np.corrcoef(new_feat['ShadowFraction'], new_feat['BrightFraction'])[0, 1]), 3)}
with open(os.path.join(OUT, "Canopy_Feature_Bank_FDR_Check.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")
