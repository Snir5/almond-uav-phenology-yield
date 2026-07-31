#!/usr/bin/env python3
"""
yield_model_ebi_interactions.py: per request, tests EBI in combination with every other
feature already validated or available in the pooled yield model, i.e. explicit
INTERACTION terms (EBI x X), not just additive main effects (every additive combination
has already been tried across sections 19-28). An additive model assumes each feature's
effect on yield is constant regardless of the others; an interaction tests whether EBI's
own effect is modified by season, physiology, topography, or a second color index.

Candidates, all on top of the recommended base (cultivar+EBI x cultivar+climate+
Growth_April, CV R^2=0.456):
  - EBI x Growth_April: does bloom's yield relationship depend on how much the tree grew
    in April (a vegetative-reproductive trade-off already suggested in section 22)?
  - EBI x climate: does bloom matter more or less depending on the season's severity?
  - EBI x DEM: does bloom's effect on yield depend on where the tree sits (elevation)?
  - EBI x each remaining physiology field (SWP_April, SWP_MayJune, SWP_June,
    Growth_MayJune, Growth_June, CNC_June): does water status or CNC modify the bloom
    signal?
  - EBI x NGRDI_Norm: do the two RGB color indices reinforce or cancel each other?
  - EBI^2: a simple non-linearity check (is the bloom-yield relationship curved rather
    than a straight ANCOVA slope)?
  - Three-way: EBI x cultivar x climate, EBI x cultivar x Growth_April (does the already-
    established EBI x cultivar reversal itself depend on season or April growth?)

Same rigor throughout: forward selection on 5-fold CV R^2, a 5-independent-seed
robustness check, and an explicit collinearity check on whatever is selected (the
section-28 EBI_Raw/EBI_Norm trap is exactly the failure mode to guard against here,
since several of these candidates, e.g. EBI^2 and EBI x Growth_April, could be highly
correlated with EBI or Growth_April's own main effect).
-> Results_Analysis/08_UEF53_Rerun/Yield_Model_EBI_Interactions.json + figure.
"""
import os, sys, csv, math, json, sqlite3, struct
from collections import defaultdict, Counter
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import ols_rss, f_pvalue

BASE = find_thesis_root()
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
CV_SEED = 42
PHYS_FIELDS = ["SWP_April", "Growth_April", "SWP_MayJune", "Growth_MayJune", "CNC_June", "SWP_June", "Growth_June"]
PHYS_GPKG_COLS = ["SWP_April", "Growth__April", "SWP_May.June", "Growth_May.June", "CNC_June", "SWP_June", "Growth_June"]
MATCH_THRESHOLD_M = 3.0

M = load_master()
cultivar = [r.get("cultivar") for r in M]


def gpkg_geom_centroid(blob):
    flags = blob[3]
    envelope_ind = (flags >> 1) & 0x07
    env_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[envelope_ind]
    wkb = blob[8 + env_bytes:]
    endian = "<" if wkb[0] == 1 else ">"
    pos = 1
    geom_type = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
    base_type = geom_type % 1000
    xs, ys = [], []
    if base_type == 6:
        npoly = struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
        for _ in range(npoly):
            pos += 1
            struct.unpack_from(endian + "I", wkb, pos)[0]; pos += 4
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
    rows = cur.fetchall(); con.close()
    out = []
    for row_ in rows:
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
         if ymeas[t].get(yr) is not None and M[j].get(f"EBI_Norm_{yr}") is not None and j in phys_match[yr]]
nP = len(rowsP)
print(f"pooled 2022-2023 sample: n={nP}")

yP = np.array([ymeas[t][yr] for t, j, yr in rowsP])
uefP = np.array([1.0 if cultivar[j] == "UEF" else 0.0 for t, j, yr in rowsP])


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


ebiP = zc([M[j].get(f"EBI_Norm_{yr}") for t, j, yr in rowsP])
CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
climP = zc([CP[yr] for t, j, yr in rowsP])
oneP = np.ones(nP)

phys_vals = {}
for f in PHYS_FIELDS:
    phys_vals[f] = zc([phys_match[yr][j][f] for t, j, yr in rowsP])
gaP = phys_vals["Growth_April"]

dem_raw = np.array([M[j].get("DEM_Jul2024") for t, j, yr in rowsP], float)
n_dem_missing = int(np.isnan(dem_raw).sum())
dem_raw[np.isnan(dem_raw)] = np.nanmean(dem_raw)
demP = zc(dem_raw)
print(f"DEM: {n_dem_missing}/{nP} missing, mean-imputed")

ngrdiP = zc([M[j].get(f"NGRDI_Norm_{yr}") for t, j, yr in rowsP])
ebi2P = zc(ebiP ** 2)

CV_SEED_ = 42


def r2_cv(Xd, y, k=5, seed=None):
    n = len(y); rng = np.random.default_rng(seed if seed is not None else CV_SEED_)
    idx = rng.permutation(n); folds = np.array_split(idx, k)
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


base_cols = {"intercept": oneP, "cultivar": uefP, "EBI": ebiP, "EBIxcultivar": ebiP * uefP,
             "climate": climP, "Growth_April": gaP}
X_base = np.column_stack(list(base_cols.values()))
baseline_r2, baseline_cv = r2_cv(X_base, yP)
print(f"baseline (recommended model): R2={baseline_r2} cvR2={baseline_cv}")

candidates = {
    "EBI_x_GrowthApril": ebiP * gaP,
    "EBI_x_climate": ebiP * climP,
    "EBI_x_DEM": ebiP * demP,
    "EBI_x_SWP_April": ebiP * phys_vals["SWP_April"],
    "EBI_x_SWP_MayJune": ebiP * phys_vals["SWP_MayJune"],
    "EBI_x_SWP_June": ebiP * phys_vals["SWP_June"],
    "EBI_x_Growth_MayJune": ebiP * phys_vals["Growth_MayJune"],
    "EBI_x_Growth_June": ebiP * phys_vals["Growth_June"],
    "EBI_x_CNC_June": ebiP * phys_vals["CNC_June"],
    "EBI_x_NGRDI": ebiP * ngrdiP,
    "EBI_squared": ebi2P,
    "EBI_x_cultivar_x_climate": ebiP * uefP * climP,
    "EBI_x_cultivar_x_GrowthApril": ebiP * uefP * gaP,
}
print(f"\ncandidates tested: {list(candidates.keys())}")


def forward_select(base_cols, candidate_cols, y, seed=None):
    selected = []; history = []
    Xd = np.column_stack(list(base_cols.values()))
    cur_r2, cur_cv = r2_cv(Xd, y, seed=seed)
    history.append({"step": 0, "added": None, "cvR2": cur_cv})
    remaining = dict(candidate_cols)
    while remaining:
        best_name, best_cv, best_r2 = None, cur_cv, cur_r2
        for name, col in remaining.items():
            Xtry = np.column_stack(list(base_cols.values()) + [col])
            r2t, cvt = r2_cv(Xtry, y, seed=seed)
            if cvt > best_cv + 1e-6:
                best_name, best_cv, best_r2 = name, cvt, r2t
        if best_name is None: break
        selected.append(best_name); base_cols[best_name] = remaining.pop(best_name)
        cur_cv, cur_r2 = best_cv, best_r2
        history.append({"step": len(selected), "added": best_name, "cvR2": cur_cv})
    return selected, history, cur_r2, cur_cv


base_for_search = dict(base_cols)
selected, history, final_r2, final_cv = forward_select(base_for_search, candidates, yP)
print(f"\nforward selection: selected={selected}")
print(f"final: R2={final_r2} cvR2={final_cv} (gain={round(final_cv-baseline_cv,4)})")

robust = []
for seed in [1, 7, 13, 99, 2024]:
    base_s = dict(base_cols)
    sel_s, hist_s, r2_s, cv_s = forward_select(base_s, dict(candidates), yP, seed=seed)
    robust.append({"seed": seed, "selected": sel_s, "baseline_cvR2": hist_s[0]["cvR2"], "final_cvR2": cv_s})
    print(f"  seed {seed}: selected={sel_s} final_cvR2={cv_s} (gain={round(cv_s-hist_s[0]['cvR2'],4)})")

sel_counter = Counter()
for r in robust:
    for s in r["selected"]: sel_counter[s] += 1
print(f"\nhow often each feature selected of 5: {dict(sel_counter)}")

# nested partial-F test for whatever was selected at the main seed (if anything)
partial_F_result = None
if selected:
    X_full = np.column_stack(list(base_cols.values()) + [candidates[s] for s in selected])
    rss0, p0, _ = ols_rss(X_base, yP)
    rss1, p1, _ = ols_rss(X_full, yP)
    df1 = p1 - p0; df2 = nP - p1
    Fstat = ((rss0 - rss1) / df1) / (rss1 / df2)
    pF = float(f_pvalue(Fstat, df1, df2))
    partial_F_result = {"added_terms": selected, "F": round(float(Fstat), 3), "df": [df1, df2], "p": round(pF, 4)}
    print(f"\npartial F test for {selected}: F={partial_F_result['F']} df={df1},{df2} p={partial_F_result['p']}")

# collinearity check: correlate every selected candidate against EBI and Growth_April main effects
collinearity = {}
for name in candidates:
    r_ebi = float(np.corrcoef(candidates[name], ebiP)[0, 1])
    r_ga = float(np.corrcoef(candidates[name], gaP)[0, 1])
    collinearity[name] = {"corr_with_EBI": round(r_ebi, 3), "corr_with_Growth_April": round(r_ga, 3)}
print("\ncollinearity check (candidate vs EBI main effect, vs Growth_April main effect):")
for name, c in collinearity.items():
    print(f"  {name}: r_EBI={c['corr_with_EBI']:+.3f} r_GrowthApril={c['corr_with_Growth_April']:+.3f}")

RESULTS = {
    "note": "Tests EBI in explicit INTERACTION with every other available feature "
            "(Growth_April, climate, DEM, remaining physiology fields, NGRDI, EBI^2, "
            "and two three-way interactions), on top of the recommended model "
            "(cultivar+EBIxcultivar+climate+Growth_April, CV R^2=0.456).",
    "n": nP,
    "baseline": {"R2": baseline_r2, "cvR2": baseline_cv},
    "candidates_tested": list(candidates.keys()),
    "forward_selection": {"selected_features": selected, "history": history,
                           "final_R2": final_r2, "final_cvR2": final_cv,
                           "improvement_cvR2": round(final_cv - baseline_cv, 4)},
    "robustness_across_5_seeds": robust,
    "how_often_selected_of_5": dict(sel_counter),
    "partial_F_test_main_seed_selection": partial_F_result,
    "collinearity_check": collinearity,
}
with open(os.path.join(OUT, "Yield_Model_EBI_Interactions.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axes[0]
names = list(candidates.keys())
# single-addition CV R^2 for each candidate alone (not the greedy path), for a clean overview panel
single_cv = {}
for name, col in candidates.items():
    Xt = np.column_stack(list(base_cols.values()) + [col])
    _, cvt = r2_cv(Xt, yP)
    single_cv[name] = cvt
vals = [single_cv[n] for n in names]
ax.barh(range(len(names)), vals, color="#8E44AD")
ax.axvline(baseline_cv, color="k", ls="--", lw=1, label="baseline (no interaction)")
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("CV R^2 (each candidate added alone)"); ax.set_title("EBI interaction candidates,\nadded one at a time", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1]
steps = [h["step"] for h in history]; cvh = [h["cvR2"] for h in history]
ax.plot(steps, cvh, "o-", color="#16A085")
for h in history:
    if h["added"]: ax.annotate(h["added"], (h["step"], h["cvR2"]), fontsize=7, rotation=20, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("forward selection step"); ax.set_ylabel("CV R^2")
ax.set_title("Forward selection across all\nEBI interaction candidates", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "Yield_Model_EBI_Interactions.png"), dpi=130)
print("saved figure")
