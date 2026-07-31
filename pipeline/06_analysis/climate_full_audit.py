#!/usr/bin/env python3
"""
climate_full_audit.py: a complete audit of every station-level agro-climate field in the
master (per request: "check all the climate fields"), in three parts.

Part A. Complete the climate-vs-EBI table (section 6 of the log only tested 5 of 13
available fields). All 13 are confirmed station-level (std=0 across all 1523 trees within
a year, i.e. one value per season), so this stays n=4, directional only, same convention
as Chill Portions throughout this project.

Part B. The honest answer to "does swapping in a different climate field improve the
yield model": with only 2 distinct measured-yield years (2022, 2023) in the pooled model,
ANY single station-level field is an affine rescaling of a 2-point year indicator, so it
is mathematically guaranteed (not just empirically likely) to give identical CV R^2 to
whichever climate field is already in the model. This is verified computationally (not
just asserted) by swapping in several different fields and confirming the CV R^2 is
identical to numerical precision, and by showing a second simultaneous climate feature is
rank-deficient (no unique coefficient split, no CV R^2 gain).

Part C. The one thing that IS new information: DEM_Jul2024 (per-tree elevation, already
tested as a main effect in section 23 with no gain) interacted with climate. Since DEM
varies by tree and climate varies by year, DEM x climate is not reducible to a pure year
effect or a pure DEM effect, it tests whether topography's effect on yield is season-
dependent (e.g. cold-air drainage / frost pooling mattering more in some years). Forward-
selected onto the current best model (cultivar+EBIxcultivar+climate+Growth_April), with
the same 5-independent-fold-split robustness check used throughout sections 22-23.
Frost_Hrs is excluded from the interaction candidates: it is exactly 0 in all 4 years, so
any interaction with it would be uniformly zero.

-> Results_Analysis/08_UEF53_Rerun/Climate_Full_Audit.json + Climate_Full_Audit.png
"""
import os, sys, csv, math, json
from collections import defaultdict, Counter
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import r_pvalue, f_pvalue, ols_rss, sig_stars

BASE = find_thesis_root()
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
CV_SEED = 42
CLIMATE_FIELDS = ["Chill_Portions", "Late_Chill", "Chill_Hrs", "GDD_Feb", "GDD_Jan_Feb",
                   "Avg_DTR_Feb", "Rad_Feb", "Max_Dry_Hrs", "Trans_Ratio", "Frost_Hrs",
                   "Heat_Stress_Feb", "Rain_Feb", "High_Wind_Hrs"]
LEGACY = {"Chill_Hrs"}  # secondary legacy chilling metric per project convention, not primary

M = load_master()
cultivar = [r.get("cultivar") for r in M]


def star(p): return sig_stars(p) if p == p else ""


# ================= Part A: complete climate-vs-EBI table (n=4) =================
YEARS = [2021, 2022, 2023, 2024]
station_val = {}
for f in CLIMATE_FIELDS:
    station_val[f] = {}
    for y in YEARS:
        vals = [r.get(f"{f}_{y}") for r in M if r.get(f"{f}_{y}") is not None]
        assert len(set(round(v, 6) for v in vals)) == 1, f"{f}_{y} is not station-constant!"
        station_val[f][y] = vals[0]

ebi_mean = {}; ebi_sd = {}
for y in YEARS:
    col = f"EBI_Norm_{y}"
    # UEF + 53 only, matching this project's restricted-cultivar convention (section 6);
    # the master itself still carries all 1,523 trees including cultivar 54.
    vals = np.array([r.get(col) for r in M if r.get(col) is not None and r.get("cultivar") in ("UEF", "53")], float)
    ebi_mean[y] = float(vals.mean()); ebi_sd[y] = float(vals.std())

partA = {}
for f in CLIMATE_FIELDS:
    xs = np.array([station_val[f][y] for y in YEARS], float)
    ym = np.array([ebi_mean[y] for y in YEARS], float)
    ys = np.array([ebi_sd[y] for y in YEARS], float)
    novariance = float(xs.std()) == 0.0
    entry = {"values_by_year": {y: station_val[f][y] for y in YEARS}, "no_variance_across_years": novariance,
              "legacy_metric": f in LEGACY}
    if novariance:
        entry["r_vs_EBI_mean"] = None; entry["r_vs_EBI_sd"] = None
        entry["note"] = "identical in all 4 years, zero cross-year variance, cannot correlate with anything"
    else:
        r1 = float(np.corrcoef(xs, ym)[0, 1]); r2 = float(np.corrcoef(xs, ys)[0, 1])
        entry["r_vs_EBI_mean"] = round(r1, 3); entry["p_vs_EBI_mean"] = round(float(r_pvalue(r1, 4)), 4)
        entry["r_vs_EBI_sd"] = round(r2, 3); entry["p_vs_EBI_sd"] = round(float(r_pvalue(r2, 4)), 4)
    partA[f] = entry

print("=== Part A: climate-vs-EBI, all 13 fields, n=4, directional ===")
for f, e in partA.items():
    if e["no_variance_across_years"]:
        print(f"  {f}: NO VARIANCE across years ({e['values_by_year']})")
    else:
        print(f"  {f}: r_vs_mean={e['r_vs_EBI_mean']:+.3f} r_vs_sd={e['r_vs_EBI_sd']:+.3f}  values={e['values_by_year']}")

# ================= shared data assembly for Parts B & C =================
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


def r2_cv(Xd, y, k=5, seed=None):
    n = len(y); local_rng = np.random.default_rng(seed if seed is not None else CV_SEED)
    idx = local_rng.permutation(n); folds = np.array_split(idx, k)
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


# ---- reuse the section-21/22 GPKG parser to get Growth_April (the section-22 recommended addition) ----
import struct


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


import sqlite3


def load_phys_growth_april(year):
    path = os.path.join(GDIR, f"Yield_with_clustering_{year}.gpkg")
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT table_name FROM gpkg_geometry_columns")
    tbl = cur.fetchone()[0]
    cur.execute(f'SELECT "Row","Plot",cultivar,"Growth__April",geom FROM "{tbl}"')
    rows = cur.fetchall(); con.close()
    out = []
    for row_, plot, cult, ga, blob in rows:
        if plot != "A" or ga is None: continue
        x, y = gpkg_geom_centroid(blob)
        out.append({"x": x, "y": y, "cultivar": cult, "Growth_April": float(ga)})
    return out


PHYS = {y: load_phys_growth_april(y) for y in [2022, 2023]}
mx = np.array([r.get("X_UTM") for r in M], float); my = np.array([r.get("Y_UTM") for r in M], float)
MATCH_THRESHOLD_M = 3.0
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
        m[j] = PHYS[y][i]["Growth_April"]; ui.add(i); uj.add(j)
    phys_match[y] = m

# measured yield + haversine match (same pattern as yield_model_v2/v3)
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
yP = np.array([ymeas[t][yr] for t, j, yr in rowsP])
uefP = np.array([1.0 if cultivar[j] == "UEF" else 0.0 for t, j, yr in rowsP])
ebiP = zc([M[j].get(f"EBI_Norm_{yr}") for t, j, yr in rowsP])
gaP = zc([phys_match[yr][j] for t, j, yr in rowsP])
oneP = np.ones(nP)
print(f"\npooled 2022-2023 sample: n={nP}")

# ================= Part B: climate-swap invariance =================
print("\n=== Part B: swapping the climate field (pooled 2022-2023, n={}) ===".format(nP))
base_no_climate = {"intercept": oneP, "cultivar": uefP, "EBI": ebiP, "EBIxcultivar": ebiP * uefP, "Growth_April": gaP}
partB = {"n": nP, "swap_results": {}}
swap_fields = ["Chill_Portions", "GDD_Feb", "Heat_Stress_Feb", "Rain_Feb", "High_Wind_Hrs", "Avg_DTR_Feb", "Max_Dry_Hrs"]
for f in swap_fields:
    cvals = zc([station_val[f][yr] for t, j, yr in rowsP])
    Xd = np.column_stack(list(base_no_climate.values()) + [cvals])
    r2, cv = r2_cv(Xd, yP)
    partB["swap_results"][f] = {"R2": r2, "cvR2": cv}
    print(f"  climate={f:16s} R2={r2}  cvR2={cv}")

identical = len(set(v["cvR2"] for v in partB["swap_results"].values())) == 1
partB["all_identical_cvR2"] = identical
print(f"  -> all identical cvR2 across every climate field tried: {identical}")

# two climate features simultaneously: rank-deficiency check
c1 = zc([station_val["Chill_Portions"][yr] for t, j, yr in rowsP])
c2 = zc([station_val["GDD_Feb"][yr] for t, j, yr in rowsP])
Xd2 = np.column_stack(list(base_no_climate.values()) + [c1, c2])
r2_2, cv_2 = r2_cv(Xd2, yP)
rank_full = np.linalg.matrix_rank(Xd2)
partB["two_climate_features_simultaneously"] = {
    "R2": r2_2, "cvR2": cv_2, "design_matrix_rank": int(rank_full), "design_matrix_cols": Xd2.shape[1],
    "note": "rank < columns confirms the two climate columns carry no independent information "
            "at n=2 distinct years; cvR2 should match the single-climate-field result above"
}
print(f"  two climate fields at once: R2={r2_2} cvR2={cv_2}  design rank={rank_full}/{Xd2.shape[1]}")

# ================= Part C: DEM x climate interaction =================
print("\n=== Part C: DEM x climate interaction, forward selection ===")
dem_raw = np.array([M[j].get("DEM_Jul2024") for t, j, yr in rowsP], float)
n_dem_missing = int(np.isnan(dem_raw).sum())
dem_raw[np.isnan(dem_raw)] = np.nanmean(dem_raw)
dem = zc(dem_raw)
print(f"DEM: {n_dem_missing}/{nP} missing, mean-imputed")

base_cols_C = {"intercept": oneP, "cultivar": uefP, "EBI": ebiP, "EBIxcultivar": ebiP * uefP,
               "climate": zc([station_val["Chill_Portions"][yr] for t, j, yr in rowsP]), "Growth_April": gaP}
X_base = np.column_stack(list(base_cols_C.values()))
baseline_r2, baseline_cv = r2_cv(X_base, yP)
print(f"baseline (with Growth_April, Chill_Portions climate): R2={baseline_r2} cvR2={baseline_cv}")

interact_fields = ["Chill_Portions", "GDD_Feb", "Heat_Stress_Feb", "Rain_Feb", "High_Wind_Hrs", "Avg_DTR_Feb", "Max_Dry_Hrs", "Trans_Ratio"]
engineered = {}
for f in interact_fields:
    cvals = zc([station_val[f][yr] for t, j, yr in rowsP])
    engineered[f"DEM_x_{f}"] = dem * cvals
engineered["DEM"] = dem  # re-test main effect too, for completeness alongside the interactions


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


base_for_search = dict(base_cols_C)
selected, history, final_r2, final_cv = forward_select(base_for_search, engineered, yP)
print(f"selected: {selected}")
print(f"final: R2={final_r2} cvR2={final_cv}  (gain={round(final_cv-baseline_cv,4)})")

# robustness across 5 seeds
robust = []
for seed in [1, 7, 13, 99, 2024]:
    base_s = dict(base_cols_C)
    sel_s, hist_s, r2_s, cv_s = forward_select(base_s, dict(engineered), yP, seed=seed)
    robust.append({"seed": seed, "selected": sel_s, "baseline_cvR2": hist_s[0]["cvR2"], "final_cvR2": cv_s})
    print(f"  seed {seed}: selected={sel_s} final_cvR2={cv_s}")

sel_counter = Counter()
for r in robust:
    for s in r["selected"]: sel_counter[s] += 1

partC = {
    "n": nP, "dem_missing_imputed": n_dem_missing,
    "baseline_features": list(base_cols_C.keys()), "baseline_R2": baseline_r2, "baseline_cvR2": baseline_cv,
    "candidates_tested": list(engineered.keys()),
    "forward_selection_history": history, "selected_features": selected,
    "final_R2": final_r2, "final_cvR2": final_cv, "improvement_cvR2": round(final_cv - baseline_cv, 4),
    "robustness_across_5_seeds": robust,
    "how_often_selected_of_5": dict(sel_counter),
}

RESULTS = {
    "note": "Complete audit of all 13 station-level agro-climate fields, per request "
            "'check all the climate fields / think about more data to improve the model'.",
    "partA_climate_vs_EBI_all_13_fields_n4": partA,
    "partB_climate_swap_invariance": partB,
    "partC_DEM_x_climate_interaction": partC,
    "raw_climate_data_date_range": "mid-October (prior year) through 31 March each year; "
                                    "NO April-August coverage, so no fruit-development/growing-season "
                                    "climate feature can be engineered from existing raw station data",
}

with open(os.path.join(OUT, "Climate_Full_Audit.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

ax = axes[0]
names = [f for f in CLIMATE_FIELDS if not partA[f]["no_variance_across_years"]]
rmean = [partA[f]["r_vs_EBI_mean"] for f in names]
rsd = [partA[f]["r_vs_EBI_sd"] for f in names]
yidx = np.arange(len(names))
ax.barh(yidx - 0.2, rmean, height=0.4, label="r vs mean EBI", color="#4aa3df")
ax.barh(yidx + 0.2, rsd, height=0.4, label="r vs EBI SD", color="#E67E22")
ax.set_yticks(yidx); ax.set_yticklabels(names, fontsize=8); ax.axvline(0, color="k", lw=0.8)
ax.set_title("Part A: all climate fields vs EBI (n=4, directional)", fontsize=10); ax.legend(fontsize=8)

ax = axes[1]
fs = list(partB["swap_results"].keys())
cvs = [partB["swap_results"][f]["cvR2"] for f in fs]
ax.bar(range(len(fs)), cvs, color="#8E44AD")
ax.set_xticks(range(len(fs))); ax.set_xticklabels(fs, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("CV R^2"); ax.set_title("Part B: swapping the climate field\n(all identical, n=2 years)", fontsize=10)
ax.set_ylim(0, max(cvs) * 1.2)

ax = axes[2]
steps = [h["step"] for h in history]; cvh = [h["cvR2"] for h in history]
ax.plot(steps, cvh, "o-", color="#16A085")
for h in history:
    if h["added"]: ax.annotate(h["added"], (h["step"], h["cvR2"]), fontsize=7, rotation=20, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("forward selection step"); ax.set_ylabel("CV R^2")
ax.set_title("Part C: DEM x climate forward selection", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "Climate_Full_Audit.png"), dpi=130)
print("saved figure")
