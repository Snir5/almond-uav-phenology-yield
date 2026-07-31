#!/usr/bin/env python3
"""
yield_model_2024_only_test.py: completes section 33 by testing 2024 alone.
Measured yield exists for 2024 (n=20, kedma_plot_a_yield.csv) but there is no
Yield_with_clustering_2024.gpkg (confirmed absent, only 2022/2023 exist), so
Growth_April cannot be tested for 2024; only cultivar+EBIxcultivar is fit,
directional only given n=20.
-> Results_Analysis/08_UEF53_Rerun/Yield_Model_2024_Only_Test.json
"""
import os, sys, csv, math, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
CV_SEED = 42

M = load_master()
cultivar = [r.get("cultivar") for r in M]
mlat = np.array([r.get("Latitude") for r in M], float); mlon = np.array([r.get("Longitude") for r in M], float)

yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
ymeas = {}; coord = {}
for r in yc:
    if int(r["year"]) == 2024:
        ymeas[r["tree_id"]] = float(r["net_kernel_yield_per_tree_kg"])
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

rows = [(t, j) for t, j in ymap.items() if M[j].get("EBI_Norm_2024") is not None and cultivar[j] in ("UEF", "53")]
n = len(rows)
print(f"2024-only matched sample: n={n}")

def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)

def r2_cv(Xd, y, k, seed=CV_SEED):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); folds = np.array_split(idx, k)
    pred = np.full(len(y), np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(len(y)), f)
        if len(tr) <= Xd.shape[1]: continue
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None)
        pred[f] = Xd[f] @ beta
    valid = ~np.isnan(pred)
    ss = ((y[valid] - pred[valid]) ** 2).sum(); tss = ((y - y.mean()) ** 2).sum()
    cv = 1 - ss / tss
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None); ins = Xd @ beta
    r2 = 1 - ((y - ins) ** 2).sum() / tss
    return round(float(r2), 4), round(float(cv), 4), int(valid.sum())

if n >= 6:
    yv = np.array([ymeas[t] for t, j in rows])
    uef = np.array([1.0 if cultivar[j] == "UEF" else 0.0 for t, j in rows])
    ebi = zc([M[j].get("EBI_Norm_2024") for t, j in rows])
    one = np.ones(n)
    k = max(2, min(5, n // 4))
    X0 = np.column_stack([one, uef]); r2_0, cv_0, k0 = r2_cv(X0, yv, k)
    X1 = np.column_stack([one, uef, ebi, ebi * uef]); r2_1, cv_1, k1 = r2_cv(X1, yv, k)
    print(f"cultivar-only floor: R2={r2_0} cvR2={cv_0} (folds={k})")
    print(f"+EBIxcultivar: R2={r2_1} cvR2={cv_1} (folds={k})")
    out = {"n": n, "k_folds": k, "cultivar_only": {"R2": r2_0, "cvR2": cv_0},
           "plus_EBIxcultivar": {"R2": r2_1, "cvR2": cv_1},
           "note": "n=20 total in the yield CSV for 2024, matched sample here is smaller after "
                   "requiring EBI_Norm_2024 and UEF/53 cultivar; no Growth_April available, no "
                   "Yield_with_clustering_2024.gpkg exists. Directional only given n."}
else:
    print(f"n={n} too small to fit, skipping")
    out = {"n": n, "note": "too few matched trees to fit any model"}

with open(os.path.join(OUT, "Yield_Model_2024_Only_Test.json"), "w") as f:
    json.dump(out, f, indent=2, default=str)
print("saved json")
