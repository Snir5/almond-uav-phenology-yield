#!/usr/bin/env python3
"""
yield_model_same_trees_by_cultivar.py: per request, splits section 34's same-tree
panel (18 trees with EBI in all 3 years) BY CULTIVAR, to check whether the pooled
within-panel r=+0.31/+0.33 is diluted by mixing UEF and cultivar 53, which the
existing ANCOVA (EBI x cultivar interaction, F=11.58, p=0.0009) already shows have
OPPOSITE-signed EBI-yield relationships (UEF r=+0.29, cv-53 r=-0.26, pooled full
sample). If that reversal holds within this same-tree panel too, splitting should
show a LARGER |r| per cultivar than the pooled 0.31, since opposite signs cancel
when mixed.
-> Results_Analysis/08_UEF53_Rerun/Yield_Model_Same_Trees_By_Cultivar.json
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
    n_trees = len(tree_ids); n_rows = len(crows)
    tree_yield_mean = {t: np.mean([y for tt, j, yr, y, e, c in crows if tt == t]) for t in tree_ids}
    tree_ebi_mean = {t: np.mean([e for tt, j, yr, y, e, c in crows if tt == t]) for t in tree_ids}
    dev_y = np.array([y - tree_yield_mean[t] for t, j, yr, y, e, c in crows])
    dev_e = np.array([e - tree_ebi_mean[t] for t, j, yr, y, e, c in crows])
    r_within, p_within = corr(dev_e, dev_y)
    yv = np.array([y for t, j, yr, y, e, c in crows]); ev = np.array([e for t, j, yr, y, e, c in crows])
    r_pooled, p_pooled = corr(ev, yv)
    # cluster bootstrap for within-tree r
    rng = np.random.default_rng(42)
    boot = []
    tree_arr = np.array([r[0] for r in crows])
    for _ in range(2000):
        samp = rng.choice(tree_ids, size=n_trees, replace=True)
        idxs = []
        for st in samp: idxs.extend(list(np.where(tree_arr == st)[0]))
        if len(idxs) < 4: continue
        de = dev_e[idxs]; dyv = dev_y[idxs]
        if de.std() == 0 or dyv.std() == 0: continue
        boot.append(float(np.corrcoef(de, dyv)[0, 1]))
    boot = np.array(boot)
    ci = np.percentile(boot, [2.5, 97.5]) if len(boot) > 10 else [None, None]
    per_year = {}
    for yr in (2022, 2023, 2024):
        yvv = np.array([y for t, j, y2, y, e, c in crows if y2 == yr])
        evv = np.array([e for t, j, y2, y, e, c in crows if y2 == yr])
        rr, pp = corr(evv, yvv)
        per_year[yr] = {"n": len(yvv), "r": rr, "p": pp}
    print(f"\n[{cult}] n_trees={n_trees}, n_rows={n_rows}")
    print(f"  within-tree r={r_within} (naive p={p_within}), bootstrap 95% CI=[{ci[0]},{ci[1]}]" if ci[0] is not None else f"  within-tree r={r_within}")
    print(f"  pooled (not de-meaned) r={r_pooled} p={p_pooled}")
    for yr, d in per_year.items(): print(f"    {yr} only: n={d['n']} r={d['r']} p={d['p']}")
    out[cult] = {
        "n_trees": n_trees, "n_rows": n_rows,
        "within_tree_demeaned": {"r": r_within, "naive_p": p_within,
                                  "bootstrap_95CI": [round(float(ci[0]), 3), round(float(ci[1]), 3)] if ci[0] is not None else None},
        "pooled_not_demeaned": {"r": r_pooled, "p": p_pooled},
        "per_year": per_year,
    }

print(f"\n[POOLED both cultivars, for reference, same as section 34] see Yield_Model_Same_Trees_Panel.json")

RESULTS = {
    "note": "Splits the section-34 same-tree panel (18 trees, EBI in all 3 years) by cultivar, "
            "to test whether pooling UEF and cultivar 53 together dilutes the effect, given the "
            "existing ANCOVA finding that EBI-yield reverses sign by cultivar (UEF r=+0.29, "
            "cv-53 r=-0.26, full sample). If the reversal holds within this same-tree panel too, "
            "splitting should show a LARGER |r| per cultivar than the pooled +0.31/+0.33.",
    "by_cultivar": out,
    "pooled_both_cultivars_reference": "r_within=+0.327, r_pooled=+0.311 (section 34, Yield_Model_Same_Trees_Panel.json)",
}
with open(os.path.join(OUT, "Yield_Model_Same_Trees_By_Cultivar.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")
