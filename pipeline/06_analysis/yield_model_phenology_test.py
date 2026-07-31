#!/usr/bin/env python3
"""
yield_model_phenology_test.py: does swapping the phenology-timing-normalized EBI
(section 16.3) into the pooled multi-season yield model (yield_model_prototype.py)
improve its cross-validated fit?

Why this is a DIFFERENT question from "does it change the yield-EBI correlation"
(already answered no, mathematically, for any single cultivar-year): the pooled model
z-scores EBI across the WHOLE 2022-2024 sample before fitting, so it compares EBI levels
ACROSS cultivar-years directly. That cross-group comparability is exactly what the
phenology correction targets, so here, unlike the within-cultivar-year case, the
correction is not mathematically forced to be a no-op.

Two honest versions are run, since only 3 of 6 cultivar-year cells in 2022-2024 have a
reliable correction (UEF 2022/2023, cultivar 53 2022; cultivar 53 2023 and UEF 2024 are
flagged unreliable, cultivar 53 2024 has no field survey at all, section 16.3):
  (A) "practical" swap: use the corrected EBI where reliable, fall back to raw EBI
      elsewhere (what you would actually deploy).
  (B) "clean" comparison: restrict to only the 3 reliable cultivar-year cells, comparing
      raw vs corrected EBI on the identical, fully-normalizable subset.

-> Results_Analysis/08_UEF53_Rerun/Yield_Model_Phenology_Test.json
"""
import os, sys, csv, math, json
import numpy as np
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
BASE = find_thesis_root(); OUT = os.path.join(BASE, "Results_Analysis")
FIG = os.path.join(OUT, "08_UEF53_Rerun")
rng = np.random.default_rng(42)

M = load_master()
def col(c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in M], float)
CP = {y: float(np.nanmean(col(f"Chill_Portions_{y}"))) for y in [2021, 2022, 2023, 2024]}
mlat, mlon = col("Latitude"), col("Longitude")
yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
ymeas = defaultdict(dict); coord = {}
for r in yc: ymeas[r["tree_id"]][int(r["year"])] = float(r["net_kernel_yield_per_tree_kg"]); coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))
def hav(a, b, d, e):
    p = math.pi / 180; x = math.sin((d - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(d * p) * math.sin((e - b) * p / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(max(0, x)))
cand = []
for t, (la, lo) in coord.items():
    ds = np.array([hav(la, lo, a, b) for a, b in zip(mlat, mlon)])
    for j in np.where(ds <= 8)[0]: cand.append((float(ds[j]), t, int(j)))
cand.sort(); uy = set(); um = set(); mp = {}
for d, t, j in cand:
    if t in uy or j in um: continue
    mp[t] = j; uy.add(t); um.add(j)

# --- phenology correction factors, per cultivar-year, reliable only ---
PBN = json.load(open(os.path.join(FIG, "Phenology_Bloom_Normalization.json")))
FACTOR = {}
for r in PBN["per_cultivar_year"]:
    if r["reliable"]:
        FACTOR[(r["year"], r["cultivar"])] = r["correction_factor"]


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


def r2_cv(Xd, y, k=5):
    n = len(y); idx = rng.permutation(n); folds = np.array_split(idx, k)
    pred = np.full(n, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None)
        pred[f] = Xd[f] @ beta
    ss = ((y - pred) ** 2).sum(); tss = ((y - y.mean()) ** 2).sum()
    cv = 1 - ss / tss
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None); ins = Xd @ beta
    r2 = 1 - ((y - ins) ** 2).sum() / tss
    p = Xd.shape[1] - 1; adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return round(float(r2), 3), round(float(adj), 3), round(float(cv), 3)


def build_records(years, require_reliable):
    recs = []
    for t, j in mp.items():
        for yr in years:
            v = ymeas[t].get(yr); e = M[j].get(f"EBI_Norm_{yr}"); c = M[j].get("cultivar")
            if v is None or e is None or c not in ("UEF", "53"):
                continue
            key = (yr, c)
            if require_reliable and key not in FACTOR:
                continue
            f = FACTOR.get(key, 1.0)
            recs.append((v, c, float(e), float(e) * f, yr))
    return recs


def run(recs, label):
    yv = np.array([a for a, _, _, _, _ in recs]); n = len(yv)
    uef = np.array([1.0 if c == "UEF" else 0.0 for _, c, _, _, _ in recs])
    ebi_raw = zc([e for _, _, e, _, _ in recs])
    ebi_norm = zc([en for _, _, _, en, _ in recs])
    yr_arr = np.array([yr for _, _, _, _, yr in recs])
    cp_arr = zc([CP[yr] for yr in yr_arr])
    one = np.ones(n)
    out = {"n": n, "label": label}
    for tag, ebi in [("raw_EBI", ebi_raw), ("phenology_normalized_EBI", ebi_norm)]:
        m_noclim = np.column_stack([one, uef, ebi, ebi * uef])
        m_clim = np.column_stack([one, uef, ebi, ebi * uef, cp_arr])
        out[tag] = {
            "+EBI×cultivar": dict(zip(["R2", "adjR2", "cvR2"], r2_cv(m_noclim, yv))),
            "+EBI×cultivar+climate": dict(zip(["R2", "adjR2", "cvR2"], r2_cv(m_clim, yv))),
        }
    return out


recs_practical = build_records([2022, 2023, 2024], require_reliable=False)
recs_clean = build_records([2022, 2023, 2024], require_reliable=True)

out = {
    "note": "Tests whether the phenology-timing-normalized EBI (section 16.3) improves the "
            "pooled multi-season yield model (yield_model_prototype.py), unlike the "
            "within-cultivar-year yield correlations, which are mathematically invariant to "
            "this correction, the pooled model z-scores EBI across cultivar-years, so cross-group "
            "comparability (what the correction targets) can genuinely matter here.",
    "practical_swap_reliable_where_available_else_raw": run(recs_practical, "all 6 cultivar-year cells, 3 corrected + 3 left raw (practical/deployable version)"),
    "clean_comparison_reliable_cells_only": run(recs_clean, "restricted to the 3 reliable cultivar-year cells only (UEF 2022, 53 2022, UEF 2023): raw vs corrected on an identical subset"),
    "reliable_cells_used": [f"{y}-{c}" for (y, c) in sorted(FACTOR)],
    "excluded_unreliable_or_missing": ["2023-53 (factor 17.5x, unreliable)", "2024-UEF (factor 8.0x, unreliable)", "2024-53 (no field survey)"],
}
json.dump(out, open(os.path.join(FIG, "Yield_Model_Phenology_Test.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
