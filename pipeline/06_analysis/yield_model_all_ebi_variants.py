#!/usr/bin/env python3
"""
yield_model_all_ebi_variants.py: per request, uses every EBI/NGRDI variant already in the
master, not just EBI_Norm (the one used as "EBI" throughout sections 11-25), as a
candidate for the yield model's bloom term.

The master carries three scalings of each index per year: *_Raw, *_Norm, *_Z, for both
EBI and NGRDI. Checked directly (not assumed): within a single year, Raw/Norm/Z are all
near-perfect affine transforms of each other (r>0.9998), so swapping between them WITHIN
one season changes nothing (same logic as section 25's climate-swap-invariance). But
*_Z is a PER-YEAR z-score (mean/sd computed separately each season), while *_Raw/*_Norm
carry the raw between-year level shift, so in the POOLED 2022-2023 model these are NOT
interchangeable, *_Z isolates within-year relative bloom rank and could behave
differently once climate already carries the between-year signal. This is a real,
motivated reason to test all six variants properly rather than assuming EBI_Norm (a
somewhat arbitrary historical choice) is already the best one.

Design: base model is cultivar+climate+Growth_April (the parts of the section-22
recommended model that are NOT in question), then forward selection is offered EVERY
bloom-index variant and its cultivar interaction (12 candidates: EBI_Raw, EBI_Norm,
EBI_Z, NGRDI_Raw, NGRDI_Norm, NGRDI_Z, each alone and x cultivar), scored on 5-fold CV
R^2, same rigor as every prior test (sections 22-23), plus a 5-independent-seed
robustness check.
-> Results_Analysis/08_UEF53_Rerun/Yield_Model_All_EBI_Variants.json + figure.
"""
import os, sys, csv, math, json, sqlite3, struct
from collections import defaultdict, Counter
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master

BASE = find_thesis_root()
GDIR = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
CV_SEED = 42

M = load_master()
cultivar = [r.get("cultivar") for r in M]

# ---------------- check the affine-transform relationships directly (not assumed) ----------------
print("=== checking Raw/Norm/Z relationships, within-year and pooled ===")
check = {}
for y in [2022, 2023]:
    for base in ["EBI", "NGRDI"]:
        raw = np.array([r.get(f"{base}_Raw_{y}") for r in M if r.get(f"{base}_Raw_{y}") is not None and r.get("cultivar") in ("UEF", "53")])
        norm = np.array([r.get(f"{base}_Norm_{y}") for r in M if r.get(f"{base}_Norm_{y}") is not None and r.get("cultivar") in ("UEF", "53")])
        z = np.array([r.get(f"{base}_Z_{y}") for r in M if r.get(f"{base}_Z_{y}") is not None and r.get("cultivar") in ("UEF", "53")])
        r1 = float(np.corrcoef(raw, norm)[0, 1]); r2 = float(np.corrcoef(raw, z)[0, 1])
        print(f"{base} {y}: corr(Raw,Norm)={r1:.6f} corr(Raw,Z)={r2:.6f} | Raw mean/sd={raw.mean():.4f}/{raw.std():.4f}")
        check[f"{base}_{y}"] = {"corr_Raw_Norm": round(r1, 6), "corr_Raw_Z": round(r2, 6),
                                  "Raw_mean": round(float(raw.mean()), 4), "Raw_sd": round(float(raw.std()), 4)}

# ---------------- reuse section-21/22 GPKG parser for Growth_April ----------------
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
        out.append({"x": x, "y": y, "Growth_April": float(ga)})
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

# ---------------- measured yield + haversine match (identical pattern to yield_model_v2/v3) ----------------
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
print(f"\npooled 2022-2023 sample: n={nP}")

yP = np.array([ymeas[t][yr] for t, j, yr in rowsP])
uefP = np.array([1.0 if cultivar[j] == "UEF" else 0.0 for t, j, yr in rowsP])


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


CP = {y: float(np.nanmean([r.get(f"Chill_Portions_{y}") for r in M if r.get(f"Chill_Portions_{y}") is not None])) for y in [2022, 2023]}
climP = zc([CP[yr] for t, j, yr in rowsP])
gaP = zc([phys_match[yr][j] for t, j, yr in rowsP])
oneP = np.ones(nP)

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


base_cols = {"intercept": oneP, "cultivar": uefP, "climate": climP, "Growth_April": gaP}
X_base = np.column_stack(list(base_cols.values()))
baseline_r2, baseline_cv = r2_cv(X_base, yP)
print(f"base (cultivar+climate+Growth_April, no bloom index at all): R2={baseline_r2} cvR2={baseline_cv}")

# ---- what the CURRENT model gives, for reference ----
ebi_norm = zc([M[j].get(f"EBI_Norm_{yr}") for t, j, yr in rowsP])
X_current = np.column_stack(list(base_cols.values()) + [ebi_norm, ebi_norm * uefP])
cur_r2, cur_cv = r2_cv(X_current, yP)
print(f"current model (+ EBI_Norm + EBI_Norm x cultivar): R2={cur_r2} cvR2={cur_cv}")

# ---- all 6 variants as candidates, each alone and x cultivar ----
variants = {}
for base in ["EBI", "NGRDI"]:
    for scale in ["Raw", "Norm", "Z"]:
        v = zc([M[j].get(f"{base}_{scale}_{yr}") for t, j, yr in rowsP])
        variants[f"{base}_{scale}"] = v
        variants[f"{base}_{scale}_x_cultivar"] = v * uefP

candidates = dict(variants)


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
print(f"\nforward selection across all 12 EBI/NGRDI-variant candidates:")
print(f"  selected: {selected}")
print(f"  final: R2={final_r2} cvR2={final_cv} (gain over no-bloom base: {round(final_cv-baseline_cv,4)})")

# robustness across 5 seeds
robust = []
for seed in [1, 7, 13, 99, 2024]:
    base_s = dict(base_cols)
    sel_s, hist_s, r2_s, cv_s = forward_select(base_s, dict(candidates), yP, seed=seed)
    robust.append({"seed": seed, "selected": sel_s, "baseline_cvR2": hist_s[0]["cvR2"], "final_cvR2": cv_s})
    print(f"  seed {seed}: selected={sel_s} final_cvR2={cv_s}")

sel_counter = Counter()
for r in robust:
    for s in r["selected"]: sel_counter[s] += 1
print(f"\nhow often each feature selected of 5: {dict(sel_counter)}")

# also: single-variant-swap test (replace EBI_Norm with each variant one at a time, same as climate-swap logic)
print("\n=== single-variant swap (EBI x cultivar term only, one variant at a time) ===")
swap_results = {}
for name in ["EBI_Raw", "EBI_Norm", "EBI_Z", "NGRDI_Raw", "NGRDI_Norm", "NGRDI_Z"]:
    v = variants[name]
    Xd = np.column_stack(list(base_cols.values()) + [v, v * uefP])
    r2t, cvt = r2_cv(Xd, yP)
    swap_results[name] = {"R2": r2t, "cvR2": cvt}
    print(f"  {name} + {name}xcultivar: R2={r2t} cvR2={cvt}")

RESULTS = {
    "note": "Tests every EBI/NGRDI scaling variant (Raw/Norm/Z) already in the master as "
            "the yield model's bloom term, not just EBI_Norm (the variant used throughout "
            "sections 11-25). Pooled 2022-2023, n=167, same sample as section 22.",
    "n": nP,
    "raw_norm_z_relationship_check": check,
    "base_no_bloom_index": {"R2": baseline_r2, "cvR2": baseline_cv},
    "current_model_EBI_Norm": {"R2": cur_r2, "cvR2": cur_cv},
    "single_variant_swap": swap_results,
    "forward_selection_all_12_candidates": {
        "selected_features": selected, "history": history, "final_R2": final_r2, "final_cvR2": final_cv,
    },
    "robustness_across_5_seeds": robust,
    "how_often_selected_of_5": dict(sel_counter),
}
with open(os.path.join(OUT, "Yield_Model_All_EBI_Variants.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
names = list(swap_results.keys())
cvs = [swap_results[n]["cvR2"] for n in names]
colors = ["#16A085" if n == "EBI_Norm" else "#4aa3df" if n.startswith("EBI") else "#E67E22" for n in names]
ax.bar(range(len(names)), cvs, color=colors)
ax.axhline(baseline_cv, color="k", ls="--", lw=1, label="no bloom index at all")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("CV R^2"); ax.set_title("Swapping in each EBI/NGRDI variant\n(cultivar+climate+Growth_April base)", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1]
steps = [h["step"] for h in history]; cvh = [h["cvR2"] for h in history]
ax.plot(steps, cvh, "o-", color="#8E44AD")
for h in history:
    if h["added"]: ax.annotate(h["added"], (h["step"], h["cvR2"]), fontsize=7, rotation=20, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("forward selection step"); ax.set_ylabel("CV R^2")
ax.set_title("Forward selection across all 12\nEBI/NGRDI-variant candidates", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "Yield_Model_All_EBI_Variants.png"), dpi=130)
print("saved figure")
