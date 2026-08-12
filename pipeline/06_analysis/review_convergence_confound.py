#!/usr/bin/env python3
"""
review_convergence_confound.py

Tests whether the reported convergence of bloom intensity between cultivars can be
separated from the change of camera between seasons.

Background. The four bloom flights used four different sensors: a Sony DSC body in
2021, an IMG-series camera in 2022, a DJI Zenmuse in 2023 and a DJI wide camera in
2024. Over those same four seasons the thesis reports that the between-cultivar
share of EBI variance falls from 69 percent to 1 percent and that the spread of EBI
roughly halves. A monotonic decline that tracks an instrument change every year is
the alternative explanation raised in Tarin Paz-Kagan's comments 442 and 596.

Four diagnostics, all computed from the master tree table:

  1. Variance decomposition per year: total, within-cultivar and between-cultivar.
     If a biological convergence is occurring, the between-cultivar component should
     collapse while the within-cultivar spread persists. If the whole distribution
     is being compressed, both fall together, which points at radiometry.

  2. The same decomposition for NGRDI. NGRDI is a different RGB index that does not
     specifically track flowers. If NGRDI shows the same collapse, the effect is not
     bloom-specific and a common radiometric cause is likely.

  3. Coefficient of variation per year, which removes any pure change of scale
     between sensors and asks whether relative spread still declines.

  4. Cross-year rank agreement between trees. If the same trees stay relatively
     bright from season to season, the spatial bloom signal persists and only its
     amplitude changes, which is the signature of radiometric compression rather
     than of trees becoming biologically alike.

Writes Results_Analysis/review_convergence_confound.json
"""
import json
import os
import sys

import numpy as np
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis")
YEARS = [2021, 2022, 2023, 2024]
SENSOR = {2021: "Sony DSC", 2022: "IMG series", 2023: "DJI Zenmuse", 2024: "DJI wide"}


def load():
    ws = openpyxl.load_workbook(
        os.path.join(BASE, "4band_mosaic", "Master_Trees_Extended.xlsx"),
        read_only=True).active
    rows = list(ws.iter_rows(values_only=True))
    hdr = list(rows[0])
    data = {h: [] for h in hdr if h}
    for r in rows[1:]:
        for h, v in zip(hdr, r):
            if h:
                data[h].append(v)
    return data


def num(data, name):
    return np.array([v if isinstance(v, (int, float)) else np.nan
                     for v in data[name]], float)


def decompose(values, groups):
    """Total, within-group and between-group variance for one season."""
    ok = ~np.isnan(values)
    v, g = values[ok], np.asarray(groups)[ok]
    if len(v) < 10:
        return None
    grand = v.mean()
    total = v.var(ddof=1)
    between = 0.0
    within_ss = 0.0
    n = len(v)
    for lab in np.unique(g):
        sub = v[g == lab]
        if len(sub) < 2:
            continue
        between += len(sub) * (sub.mean() - grand) ** 2
        within_ss += ((sub - sub.mean()) ** 2).sum()
    between_var = between / (n - 1)
    within_var = within_ss / (n - 1)
    return dict(n=int(n), mean=round(float(grand), 4),
                sd_total=round(float(np.sqrt(total)), 4),
                sd_within=round(float(np.sqrt(within_var)), 4),
                sd_between=round(float(np.sqrt(between_var)), 4),
                between_share_pct=round(100 * float(between_var / total), 2),
                cv_pct=round(100 * float(np.sqrt(total) / grand), 2) if grand else None)


def spearman(a, b):
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 10:
        return None
    def rank(x):
        order = np.argsort(x)
        r = np.empty(len(x), float)
        r[order] = np.arange(len(x), dtype=float)
        return r
    ra, rb = rank(a[ok]), rank(b[ok])
    ra -= ra.mean(); rb -= rb.mean()
    return round(float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb))), 3), int(ok.sum())


def main():
    d = load()
    cult = np.array([str(c) for c in d["cultivar"]])

    res = {"sensor_by_year": SENSOR, "indices": {}}
    for index in ("EBI_Norm", "EBI_Raw", "NGRDI_Norm", "NGRDI_Raw"):
        per_year = {}
        for y in YEARS:
            col = f"{index}_{y}"
            if col not in d:
                continue
            out = decompose(num(d, col), cult)
            if out:
                out["sensor"] = SENSOR[y]
                per_year[str(y)] = out
        res["indices"][index] = per_year

    # cross-year rank agreement, consecutive pairs and the full span
    ranks = {}
    for index in ("EBI_Norm", "NGRDI_Norm"):
        pairs = {}
        for a, b in [(2021, 2022), (2022, 2023), (2023, 2024), (2021, 2024)]:
            ca, cb = f"{index}_{a}", f"{index}_{b}"
            if ca in d and cb in d:
                r = spearman(num(d, ca), num(d, cb))
                if r:
                    pairs[f"{a}_vs_{b}"] = dict(rho=r[0], n=r[1])
        ranks[index] = pairs
    res["cross_year_rank_agreement"] = ranks

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "review_convergence_confound.json"), "w") as f:
        json.dump(res, f, indent=2)

    for index, per_year in res["indices"].items():
        if not per_year:
            continue
        print(f"\n=== {index} ===")
        print(f"  {'year':6s} {'sensor':13s} {'n':>5s} {'mean':>8s} {'SD tot':>8s} "
              f"{'SD within':>10s} {'SD betw':>8s} {'betw %':>7s} {'CV %':>7s}")
        for y in YEARS:
            v = per_year.get(str(y))
            if not v:
                continue
            print(f"  {y:<6d} {v['sensor']:13s} {v['n']:5d} {v['mean']:8.3f} "
                  f"{v['sd_total']:8.3f} {v['sd_within']:10.3f} {v['sd_between']:8.3f} "
                  f"{v['between_share_pct']:7.1f} {str(v['cv_pct']):>7s}")

    print("\n=== cross-year rank agreement between trees ===")
    for index, pairs in ranks.items():
        cells = "   ".join(f"{k}: rho {v['rho']:+.3f} (n={v['n']})" for k, v in pairs.items())
        print(f"  {index:11s} {cells}")

    print(f"\nwrote {os.path.join(OUT, 'review_convergence_confound.json')}")


if __name__ == "__main__":
    main()
