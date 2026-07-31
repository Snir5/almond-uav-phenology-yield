#!/usr/bin/env python3
"""
check_mean_reliability.py: quantifies the user's concern directly. Sections 34/35
demean each tree by its own mean over just 3 years (and, in section 35, also
demean by each year's mean over just 9 trees). With T=3, a "tree mean" is a very
imprecise estimate: this script computes, for both EBI and yield, the standard
error of each tree's own 3-year mean (sd_within_tree / sqrt(3)) and compares it to
the between-tree spread of those means, to check whether the means being subtracted
are themselves reliable enough to trust the resulting residual correlations.
"""
import os, sys, csv, math, json
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
M = load_master(); cultivar = [r.get("cultivar") for r in M]
mlat = np.array([r.get("Latitude") for r in M], float); mlon = np.array([r.get("Longitude") for r in M], float)

yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
by_tree_years = defaultdict(dict); coord = {}
for r in yc:
    by_tree_years[r["tree_id"]][int(r["year"])] = float(r["net_kernel_yield_per_tree_kg"])
    coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))
panel_trees = [t for t, yrs in by_tree_years.items() if set(yrs.keys()) == {2022, 2023, 2024}]

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

rows = []
for t, j in ymap.items():
    for yr in (2022, 2023, 2024):
        e = M[j].get(f"EBI_Norm_{yr}")
        if e is None: continue
        rows.append((t, j, yr, by_tree_years[t][yr], float(e), cultivar[j]))

tree_ids = sorted(set(r[0] for r in rows))
print(f"n trees with complete 3-year EBI+yield: {len(tree_ids)}")

for varname, idx in [("yield_kg", 3), ("EBI_Norm", 4)]:
    tree_means = []; tree_se = []
    for t in tree_ids:
        vals = np.array([r[idx] for r in rows if r[0] == t])
        tree_means.append(vals.mean())
        tree_se.append(vals.std(ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else np.nan)
    tree_means = np.array(tree_means); tree_se = np.array(tree_se)
    between_tree_sd = tree_means.std(ddof=1)
    mean_se = np.nanmean(tree_se)
    ratio = mean_se / between_tree_sd
    print(f"\n[{varname}] mean SE of each tree's own 3-yr mean: {mean_se:.4f}")
    print(f"  between-tree SD of those means: {between_tree_sd:.4f}")
    print(f"  ratio (SE of the mean / between-tree spread): {ratio:.3f}  "
          f"({'a large fraction of the between-tree spread is just noise' if ratio > 0.3 else 'means look reasonably precise relative to spread'})")
