#!/usr/bin/env python3
"""
yield_model_same_trees_panel.py: per request, checks whether the pooled EBI-yield
result is an artifact of 2023 contributing many more trees (149) than 2022/2024
(18-20 each), by isolating the exact set of trees measured in ALL THREE years
(a balanced panel) and testing the EBI-yield relationship WITHIN those same trees
across years, which removes both the year-count imbalance and any between-tree
confound (cultivar mix, tree vigor, position), since each tree is compared only to
its own other years.
-> Results_Analysis/08_UEF53_Rerun/Yield_Model_Same_Trees_Panel.json
"""
import os, sys, csv, math, json
from collections import defaultdict, Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import r_pvalue

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")

M = load_master()
cultivar = [r.get("cultivar") for r in M]
mlat = np.array([r.get("Latitude") for r in M], float); mlon = np.array([r.get("Longitude") for r in M], float)

yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
by_tree_years = defaultdict(dict); coord = {}
for r in yc:
    by_tree_years[r["tree_id"]][int(r["year"])] = float(r["net_kernel_yield_per_tree_kg"])
    coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))

panel_trees = [t for t, yrs in by_tree_years.items() if set(yrs.keys()) == {2022, 2023, 2024}]
print(f"trees measured in ALL 3 years (balanced panel): {len(panel_trees)}")
print(f"trees measured in only one year: {sum(1 for t,yrs in by_tree_years.items() if len(yrs)==1)}")

def hav(a, b, d, e):
    p = math.pi / 180
    x = math.sin((d - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(d * p) * math.sin((e - b) * p / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(max(0, x)))

cand = []
for t in panel_trees:
    la, lo = coord[t]
    ds = np.array([hav(la, lo, a, b) for a, b in zip(mlat, mlon)])
    for j in np.where(ds <= 8)[0]: cand.append((float(ds[j]), t, int(j)))
cand.sort(); uy = set(); um = set(); ymap = {}
for d, t, j in cand:
    if t in uy or j in um: continue
    ymap[t] = j; uy.add(t); um.add(j)

rows = []  # (tree_id, master_idx, year, yield, ebi_norm)
for t, j in ymap.items():
    for yr in (2022, 2023, 2024):
        e = M[j].get(f"EBI_Norm_{yr}")
        if e is None: continue
        rows.append((t, j, yr, by_tree_years[t][yr], float(e)))

n_matched_trees = len(ymap)
print(f"matched to master with EBI in all 3 years: {n_matched_trees} trees, {len(rows)} tree-year rows")

cult_counts = Counter(cultivar[j] for t, j in ymap.items())
print(f"cultivar composition of the 20-tree panel: {dict(cult_counts)}")

# ---------------- within-tree (de-meaned) analysis ----------------
tree_ids = sorted(set(t for t, j, yr, y, e in rows))
tree_yield_mean = {t: np.mean([y for tt, j, yr, y, e in rows if tt == t]) for t in tree_ids}
tree_ebi_mean = {t: np.mean([e for tt, j, yr, y, e in rows if tt == t]) for t in tree_ids}

dev_yield = np.array([y - tree_yield_mean[t] for t, j, yr, y, e in rows])
dev_ebi = np.array([e - tree_ebi_mean[t] for t, j, yr, y, e in rows])
n_rows = len(rows)

def corr(xs, ys):
    xs = np.array(xs, float); ys = np.array(ys, float)
    if xs.std() == 0 or ys.std() == 0: return None, None
    r = float(np.corrcoef(xs, ys)[0, 1])
    return round(r, 4), round(float(r_pvalue(r, len(xs))), 4)

r_within, p_within_naive = corr(dev_ebi, dev_yield)
print(f"\nWITHIN-TREE (de-meaned) EBI vs yield: r={r_within}, naive p={p_within_naive} (n={n_rows} tree-years, "
      f"but only {len(tree_ids)} independent trees, naive p ignores clustering, see below)")

# cluster-robust check: block bootstrap by tree (resample trees with replacement, keep all 3 years together)
rng = np.random.default_rng(42)
boot_r = []
tree_arr = np.array([t for t, j, yr, y, e in rows])
for _ in range(2000):
    samp_trees = rng.choice(tree_ids, size=len(tree_ids), replace=True)
    idxs = []
    for st in samp_trees:
        idxs.extend(list(np.where(tree_arr == st)[0]))
    if len(idxs) < 4: continue
    de = dev_ebi[idxs]; dy = dev_yield[idxs]
    if de.std() == 0 or dy.std() == 0: continue
    boot_r.append(float(np.corrcoef(de, dy)[0, 1]))
boot_r = np.array(boot_r)
ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
print(f"cluster (tree) block-bootstrap 95% CI for within-tree r: [{ci_lo:.3f}, {ci_hi:.3f}] over {len(boot_r)} resamples")

# ---------------- pooled cross-sectional (same panel, not de-meaned), for comparison ----------------
yv_all = np.array([y for t, j, yr, y, e in rows]); ev_all = np.array([e for t, j, yr, y, e in rows])
r_pooled_panel, p_pooled_panel = corr(ev_all, yv_all)
print(f"\nPOOLED cross-sectional (same 20-tree panel, all 3 years, not de-meaned): "
      f"r={r_pooled_panel}, p={p_pooled_panel}, n={n_rows}")

# ---------------- per-year, within this same panel only (n=20 each) ----------------
per_year = {}
for yr in (2022, 2023, 2024):
    yv = np.array([y for t, j, y2, y, e in rows if y2 == yr])
    ev = np.array([e for t, j, y2, y, e in rows if y2 == yr])
    r, p = corr(ev, yv)
    per_year[yr] = {"n": len(yv), "r": r, "p": p}
    print(f"  {yr} only (same panel, n={len(yv)}): r={r}, p={p}")

# ---------------- for context: full unbalanced sample result (2023-heavy) ----------------
full_rows = []
for tt, yrs in by_tree_years.items():
    la, lo = coord[tt]
    ds = np.array([hav(la, lo, a, b) for a, b in zip(mlat, mlon)])
    jbest = int(np.argmin(ds))
    if ds[jbest] > 8: continue
    for yr, y in yrs.items():
        e = M[jbest].get(f"EBI_Norm_{yr}")
        if e is not None:
            full_rows.append((yr, y, float(e)))
yv_full = np.array([y for yr, y, e in full_rows]); ev_full = np.array([e for yr, y, e in full_rows])
r_full, p_full = corr(ev_full, yv_full)
print(f"\nFor context, full unbalanced sample (all {len(full_rows)} tree-year rows, 2023-heavy, "
      f"naive nearest-match not the project's dedup'd matcher, illustrative only): r={r_full}, p={p_full}")

RESULTS = {
    "note": "Per request: checks whether the pooled EBI-yield result is biased by 2023 "
            "contributing far more trees (149) than 2022/2024 (18-20), by isolating the exact "
            "20 trees measured in ALL THREE years and testing the relationship WITHIN those same "
            "trees across years (each tree as its own control), which removes both the year-count "
            "imbalance and between-tree confounds (cultivar mix, vigor, position).",
    "n_trees_measured_in_all_3_years": len(panel_trees),
    "n_trees_measured_in_only_one_year": sum(1 for t, yrs in by_tree_years.items() if len(yrs) == 1),
    "n_matched_to_master_with_EBI": n_matched_trees,
    "cultivar_composition_of_panel": dict(cult_counts),
    "within_tree_demeaned": {"n_tree_years": n_rows, "n_independent_trees": len(tree_ids),
                             "r": r_within, "naive_p_ignores_clustering": p_within_naive,
                             "cluster_bootstrap_95CI": [round(float(ci_lo), 3), round(float(ci_hi), 3)],
                             "n_bootstrap_resamples": len(boot_r)},
    "pooled_cross_sectional_same_panel": {"n": n_rows, "r": r_pooled_panel, "p": p_pooled_panel},
    "per_year_same_panel_only": per_year,
    "full_unbalanced_sample_for_context": {"n": len(full_rows), "r": r_full, "p": p_full},
}
with open(os.path.join(OUT, "Yield_Model_Same_Trees_Panel.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")
