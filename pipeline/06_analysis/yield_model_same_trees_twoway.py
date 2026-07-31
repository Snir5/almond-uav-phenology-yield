#!/usr/bin/env python3
"""
yield_model_same_trees_twoway.py: follow-up check on yield_model_same_trees_by_cultivar.py.
That script's "within-tree" test only removes each TREE's own 3-year mean, not each
YEAR's common mean across the panel. That leaves a real confound: if all cultivar-53
panel trees happened to bloom brighter AND yield more in the same season (a common
climate/year effect, not an idiosyncratic tree-level signal), it would still show up
as a "within-tree" correlation. This script adds the year-mean removal too (two-way,
tree + year, fixed effects), to isolate genuinely tree-idiosyncratic covariation,
free of any effect shared by the whole cultivar-panel in a given year.
-> Results_Analysis/08_UEF53_Rerun/Yield_Model_Same_Trees_TwoWay.json
"""
import os, sys, csv, math, json
from collections import defaultdict
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

def corr(xs, ys):
    xs = np.array(xs, float); ys = np.array(ys, float)
    if len(xs) < 3 or xs.std() == 0 or ys.std() == 0: return None, None
    r = float(np.corrcoef(xs, ys)[0, 1])
    return round(r, 4), round(float(r_pvalue(r, len(xs))), 4)

out = {}
for cult in ("UEF", "53"):
    crows = [r for r in rows if r[5] == cult]
    tree_ids = sorted(set(r[0] for r in crows))
    n_trees = len(tree_ids)
    # year means WITHIN this cultivar panel (common season effect for this cultivar's panel trees)
    year_yield_mean = {yr: np.mean([y for t, j, y2, y, e, c in crows if y2 == yr]) for yr in (2022, 2023, 2024)}
    year_ebi_mean = {yr: np.mean([e for t, j, y2, y, e, c in crows if y2 == yr]) for yr in (2022, 2023, 2024)}
    print(f"\n[{cult}] panel-year means: yield={ {yr: round(v,3) for yr,v in year_yield_mean.items()} }, "
          f"EBI={ {yr: round(v,3) for yr,v in year_ebi_mean.items()} }")

    # step 1: remove year effect first (common to all panel trees of this cultivar, that season)
    y_dy = np.array([y - year_yield_mean[yr] for t, j, yr, y, e, c in crows])
    e_dy = np.array([e - year_ebi_mean[yr] for t, j, yr, y, e, c in crows])
    # step 2: then remove each tree's own mean of THOSE year-demeaned residuals (two-way FE)
    tree_arr = np.array([r[0] for r in crows])
    y_final = y_dy.copy(); e_final = e_dy.copy()
    for t in tree_ids:
        m = tree_arr == t
        y_final[m] -= y_dy[m].mean()
        e_final[m] -= e_dy[m].mean()

    r_2way, p_2way_naive = corr(e_final, y_final)

    # cluster bootstrap by tree
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(2000):
        samp = rng.choice(tree_ids, size=n_trees, replace=True)
        idxs = []
        for st in samp: idxs.extend(list(np.where(tree_arr == st)[0]))
        if len(idxs) < 4: continue
        de = e_final[idxs]; dyv = y_final[idxs]
        if de.std() == 0 or dyv.std() == 0: continue
        boot.append(float(np.corrcoef(de, dyv)[0, 1]))
    boot = np.array(boot)
    ci = np.percentile(boot, [2.5, 97.5]) if len(boot) > 10 else [None, None]

    print(f"  TWO-WAY (tree + year) de-meaned r={r_2way} (naive p={p_2way_naive}), "
          f"bootstrap 95% CI=[{ci[0]},{ci[1]}]" if ci[0] is not None else f"  r={r_2way}")

    out[cult] = {
        "n_trees": n_trees, "n_rows": len(crows),
        "panel_year_means": {"yield": {str(k): round(v, 3) for k, v in year_yield_mean.items()},
                              "EBI": {str(k): round(v, 3) for k, v in year_ebi_mean.items()}},
        "two_way_demeaned_tree_and_year": {"r": r_2way, "naive_p": p_2way_naive,
                                            "bootstrap_95CI": [round(float(ci[0]), 3), round(float(ci[1]), 3)] if ci[0] is not None else None},
    }

RESULTS = {
    "note": "Follow-up on yield_model_same_trees_by_cultivar.py: that script's within-tree test "
            "only removed each tree's OWN 3-year mean, not each YEAR's common mean across the "
            "panel, leaving a real confound (a shared season/climate effect hitting all panel "
            "trees of a cultivar in the same year would look like a within-tree correlation even "
            "with no idiosyncratic tree-level signal). This adds year-mean removal too (two-way, "
            "tree+year fixed effects), isolating genuinely tree-idiosyncratic covariation.",
    "by_cultivar": out,
    "prior_tree_only_demeaned_result_for_comparison": "UEF r=+0.079 (ns), cv-53 r=+0.532, p=0.004 (Yield_Model_Same_Trees_By_Cultivar.json)",
}
with open(os.path.join(OUT, "Yield_Model_Same_Trees_TwoWay.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")
