#!/usr/bin/env python3
"""
yield_model_all_pairwise_interactions.py: per request, extends section 29 (EBI-anchored
interactions only) to a full pairwise interaction sweep across every available feature,
not just EBI. 12 base features (cultivar, EBI, climate, DEM, Growth_April, Growth_MayJune,
Growth_June, SWP_April, SWP_MayJune, SWP_June, CNC_June, NGRDI) give C(12,2)=66 possible
pairwise products; EBI x cultivar is already a baseline model term, so 65 genuinely new
candidates are tested here (11 of the 65 duplicate individual EBI-anchored pairs already
reported in section 29, e.g. EBI x climate, EBI x CNC_June; they are re-tested here inside
the SAME FDR family as the 54 new non-EBI-anchored pairs, which is the statistically correct
way to search the full 65-candidate space rather than testing subsets under separate
corrections).

Same rigor as section 29: partial-F test for each candidate alone (full n=167 sample, not
CV-fold-dependent), BH-FDR correction across all 65, then for whatever survives, a 5
independent-seed CV robustness check plus a redundancy check against any already-known
finding (EBI x CNC_June, EBI x Growth_MayJune).

-> Results_Analysis/08_UEF53_Rerun/Yield_Model_All_Pairwise_Interactions.json + figure.
"""
import os, sys, csv, math, json, sqlite3, struct, itertools
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


def bh_fdr(pvals):
    idx = [i for i, p in enumerate(pvals) if p == p]
    ps = sorted((pvals[i], i) for i in idx); m = len(ps)
    adj = {}; prev = 1.0
    for rank in range(m - 1, -1, -1):
        p, i = ps[rank]; val = min(prev, p * m / (rank + 1)); adj[i] = round(val, 4); prev = val
    return [adj.get(i, np.nan) for i in range(len(pvals))]


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

# ---------------- full pairwise sweep across every base feature, not just EBI ----------------
features = {
    "cultivar": uefP,
    "EBI": ebiP,
    "climate": climP,
    "DEM": demP,
    "Growth_April": gaP,
    "Growth_MayJune": phys_vals["Growth_MayJune"],
    "Growth_June": phys_vals["Growth_June"],
    "SWP_April": phys_vals["SWP_April"],
    "SWP_MayJune": phys_vals["SWP_MayJune"],
    "SWP_June": phys_vals["SWP_June"],
    "CNC_June": phys_vals["CNC_June"],
    "NGRDI": ngrdiP,
}
feat_names = list(features.keys())
all_pairs = list(itertools.combinations(feat_names, 2))
print(f"\n{len(feat_names)} base features -> {len(all_pairs)} possible pairs")

candidates = {}
already_in_baseline = []
for a, b in all_pairs:
    name = f"{a}_x_{b}"
    if {a, b} == {"EBI", "cultivar"}:
        already_in_baseline.append(name)
        continue
    candidates[name] = features[a] * features[b]
print(f"{len(already_in_baseline)} pair(s) already a baseline term (skipped): {already_in_baseline}")
print(f"{len(candidates)} genuinely tested candidates")

# ---------------- partial-F test for every candidate alone, full n=167 sample ----------------
rss0, p0, _ = ols_rss(X_base, yP)
partial_F_all = {}
for name, col in candidates.items():
    X1 = np.column_stack(list(base_cols.values()) + [col])
    rss1, p1, _ = ols_rss(X1, yP)
    df1 = p1 - p0; df2 = nP - p1
    Fstat = ((rss0 - rss1) / df1) / (rss1 / df2)
    pF = float(f_pvalue(Fstat, df1, df2))
    beta, *_ = np.linalg.lstsq(X1, yP, rcond=None)
    partial_F_all[name] = {"F": round(float(Fstat), 3), "p": round(pF, 4), "coef": round(float(beta[-1]), 4)}

names_sorted = sorted(partial_F_all.keys(), key=lambda n: partial_F_all[n]["p"])
pvals = [partial_F_all[n]["p"] for n in names_sorted]
p_fdr = bh_fdr(pvals)
for n, pf in zip(names_sorted, p_fdr):
    partial_F_all[n]["p_fdr"] = pf
    partial_F_all[n]["survives_FDR"] = bool(pf < 0.05)

n_survive = sum(1 for n in partial_F_all if partial_F_all[n]["survives_FDR"])
print(f"\n{n_survive} of {len(candidates)} candidates survive FDR<0.05:")
for n in names_sorted:
    if partial_F_all[n]["survives_FDR"]:
        print(f"  {n}: F={partial_F_all[n]['F']} p={partial_F_all[n]['p']} p_fdr={partial_F_all[n]['p_fdr']} coef={partial_F_all[n]['coef']:+.3f}")

survivors = [n for n in names_sorted if partial_F_all[n]["survives_FDR"]]


# ---------------- forward selection across all 65 candidates, main seed + 5 robustness seeds ----------------
def forward_select(base_cols_, candidate_cols, y, seed=None):
    selected = []; history = []
    Xd = np.column_stack(list(base_cols_.values()))
    cur_r2, cur_cv = r2_cv(Xd, y, seed=seed)
    history.append({"step": 0, "added": None, "cvR2": cur_cv})
    remaining = dict(candidate_cols)
    while remaining:
        best_name, best_cv, best_r2 = None, cur_cv, cur_r2
        for name, col in remaining.items():
            Xtry = np.column_stack(list(base_cols_.values()) + [col])
            r2t, cvt = r2_cv(Xtry, y, seed=seed)
            if cvt > best_cv + 1e-6:
                best_name, best_cv, best_r2 = name, cvt, r2t
        if best_name is None: break
        selected.append(best_name); base_cols_[best_name] = remaining.pop(best_name)
        cur_cv, cur_r2 = best_cv, best_r2
        history.append({"step": len(selected), "added": best_name, "cvR2": cur_cv})
    return selected, history, cur_r2, cur_cv


sel_main, hist_main, r2_main, cv_main = forward_select(dict(base_cols), candidates, yP)
print(f"\nforward selection (main seed 42): selected={sel_main} final_cvR2={cv_main} (gain={round(cv_main-baseline_cv,4)})")

robust = []
for seed in [1, 7, 13, 99, 2024]:
    sel_s, hist_s, r2_s, cv_s = forward_select(dict(base_cols), dict(candidates), yP, seed=seed)
    robust.append({"seed": seed, "selected": sel_s, "baseline_cvR2": hist_s[0]["cvR2"], "final_cvR2": cv_s})
    print(f"  seed {seed}: selected={sel_s} final_cvR2={cv_s} (gain={round(cv_s-hist_s[0]['cvR2'],4)})")

sel_counter = Counter()
for r in [{"seed": "main42", "selected": sel_main}] + robust:
    for s in r["selected"]: sel_counter[s] += 1
print(f"\nhow often each feature selected of 6 (main+5 robustness): {dict(sel_counter)}")

# ---------------- redundancy check: survivors vs each other, and vs section-29's known findings ----------------
redundancy = {}
known = {"EBI_x_CNC_June": ebiP * phys_vals["CNC_June"], "EBI_x_Growth_MayJune": ebiP * phys_vals["Growth_MayJune"]}
for s in survivors:
    row = {}
    for kn, kv in known.items():
        if s == kn: continue
        row[f"corr_vs_{kn}"] = round(float(np.corrcoef(candidates[s], kv)[0, 1]), 3)
    for s2 in survivors:
        if s2 == s: continue
        row[f"corr_vs_{s2}"] = round(float(np.corrcoef(candidates[s], candidates[s2])[0, 1]), 3)
    redundancy[s] = row

RESULTS = {
    "note": "Full pairwise interaction sweep across all 12 base features (not just EBI-anchored, "
            "extending section 29). C(12,2)=66 pairs, EBI x cultivar excluded as already a "
            "baseline term, 65 candidates tested, each via a full-sample partial-F test, "
            "BH-FDR corrected across all 65 as a single family.",
    "n": nP,
    "n_base_features": len(feat_names),
    "base_features": feat_names,
    "n_pairs_possible": len(all_pairs),
    "already_in_baseline_model": already_in_baseline,
    "n_candidates_tested": len(candidates),
    "baseline": {"R2": baseline_r2, "cvR2": baseline_cv},
    "partial_F_all_65_candidates_FDR_corrected": partial_F_all,
    "survivors_FDR_lt_0.05": survivors,
    "forward_selection_main_seed42": {"selected_features": sel_main, "history": hist_main,
                                       "final_R2": r2_main, "final_cvR2": cv_main,
                                       "improvement_cvR2": round(cv_main - baseline_cv, 4)},
    "robustness_across_5_seeds": robust,
    "how_often_selected_of_6_incl_main": dict(sel_counter),
    "redundancy_check_survivors_vs_known_and_each_other": redundancy,
}
with open(os.path.join(OUT, "Yield_Model_All_Pairwise_Interactions.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
ax = axes[0]
names_top = names_sorted[:20]
vals = [-math.log10(max(partial_F_all[n]["p"], 1e-12)) for n in names_top]
colors = ["#C0392B" if partial_F_all[n]["survives_FDR"] else "#95a5a6" for n in names_top]
ax.barh(range(len(names_top)), vals, color=colors)
ax.set_yticks(range(len(names_top))); ax.set_yticklabels(names_top, fontsize=7)
ax.invert_yaxis()
ax.axvline(-math.log10(0.05), color="k", ls="--", lw=1, label="p=0.05 (uncorrected)")
ax.set_xlabel("-log10(p), partial F test"); ax.set_title("Top 20 of 65 pairwise interaction\ncandidates (red = survives FDR<0.05)", fontsize=10)
ax.legend(fontsize=7)

ax = axes[1]
steps = [h["step"] for h in hist_main]; cvh = [h["cvR2"] for h in hist_main]
ax.plot(steps, cvh, "o-", color="#16A085")
for h in hist_main:
    if h["added"]: ax.annotate(h["added"], (h["step"], h["cvR2"]), fontsize=7, rotation=20, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("forward selection step"); ax.set_ylabel("CV R^2")
ax.set_title("Forward selection across all 65\npairwise interaction candidates (seed 42)", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "Yield_Model_All_Pairwise_Interactions.png"), dpi=130)
print("saved figure")
