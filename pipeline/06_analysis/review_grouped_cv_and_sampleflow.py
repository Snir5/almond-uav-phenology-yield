#!/usr/bin/env python3
"""
review_grouped_cv_and_sampleflow.py

Answers two of Tarin Paz-Kagan's review comments on the measured-yield model:

  [610] "Yield sample sizes are inconsistent and need a flow diagram."
        -> a sample-flow table showing how many records survive each join and
           exclusion, from the 202 raw harvest records to the modelling sample.

  [611] "If the same tree appears in multiple years, random record-level
         cross-validation can place the same tree in training and testing.
         Use grouped cross-validation by tree ID and, ideally, leave-one-year-out."
        -> the same model evaluated under random 5-fold CV (as reported),
           grouped 5-fold CV by tree ID, and leave-one-year-out.

The model is the one reported in Section 6.6.1: ordinary least squares on
cultivar, mean bloom (EBI) with its cultivar interaction, winter chill
(Chill Portions) and April canopy growth.

Writes Results_Analysis/review_grouped_cv_and_sampleflow.json
"""
import csv
import json
import math
import os
import sqlite3
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master, utm36n_to_latlon

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis")
YEARS = [2022, 2023]          # the two seasons pooled in Section 6.6.1
MAX_JOIN_M = 8.0

flow = []                     # (step, n_records, n_trees, note)


def haversine(a, b, d, e):
    R = 6371000.0
    p = math.pi / 180
    x = (math.sin((d - a) * p / 2) ** 2
         + math.cos(a * p) * math.cos(d * p) * math.sin((e - b) * p / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(max(0.0, x)))


def envelope_centre(blob):
    order = '<' if (blob[3] & 1) else '>'
    mnx, mxx, mny, mxy = struct.unpack(order + 'dddd', blob[8:40])
    return (mnx + mxx) / 2, (mny + mxy) / 2


# ----------------------------------------------------------------- load inputs

M = load_master()


def mcol(name):
    return np.array([r.get(name) if r.get(name) is not None else np.nan
                     for r in M], float)


mlat, mlon = mcol("Latitude"), mcol("Longitude")
mcult = [r.get("cultivar") for r in M]

rows = list(csv.DictReader(
    open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"),
         encoding="utf-8-sig")))
flow.append(("Raw harvest records in kedma_plot_a_yield.csv",
             len(rows), len({r["tree_id"] for r in rows}), "all years"))

rows = [r for r in rows if int(r["year"]) in YEARS]
flow.append((f"Restricted to the modelled seasons {YEARS[0]} and {YEARS[1]}",
             len(rows), len({r["tree_id"] for r in rows}),
             "2024 has 20 records and no matching bloom season"))

coord = {}
ymeas = {}
for r in rows:
    coord.setdefault(r["tree_id"], (float(r["latitude"]), float(r["longitude"])))
    ymeas[(r["tree_id"], int(r["year"]))] = float(r["net_kernel_yield_per_tree_kg"])

# ------------------------------------------------ greedy 1:1 spatial join
cand = []
for t, (la, lo) in coord.items():
    d = np.array([haversine(la, lo, a, b) for a, b in zip(mlat, mlon)])
    for j in np.where(d <= MAX_JOIN_M)[0]:
        cand.append((float(d[j]), t, int(j)))
cand.sort()
used_y, used_m, match = set(), set(), {}
for d, t, j in cand:
    if t in used_y or j in used_m:
        continue
    match[t] = (j, d)
    used_y.add(t)
    used_m.add(j)

n_rec = sum(1 for (t, y) in ymeas if t in match)
flow.append((f"Matched to a master tree within {MAX_JOIN_M:.0f} m (greedy 1:1)",
             n_rec, len(match),
             f"median match distance {np.median([d for _, d in match.values()]):.2f} m"))

# ------------------------------------- cultivar-consistency quality filter
con = sqlite3.connect(os.path.join(BASE, "Trees_Data_Survey",
                                   "Field_Data_Yield_GPKGs",
                                   "Yield_with_clustering_2023.gpkg"))
gp = [(cv, *utm36n_to_latlon(*envelope_centre(g)))
      for plot, cv, g in con.execute(
          'SELECT Plot,cultivar,geom FROM Yield_with_clustering_2023')
      if plot == 'A' and g]
con.close()
gla = np.array([p[1] for p in gp])
glo = np.array([p[2] for p in gp])


def gpkg_cultivar(t):
    la, lo = coord[t]
    k = int(np.array([haversine(la, lo, a, b)
                      for a, b in zip(gla, glo)]).argmin())
    return gp[k][0]


clean = {t: v for t, v in match.items() if mcult[v[0]] == gpkg_cultivar(t)}
n_rec = sum(1 for (t, y) in ymeas if t in clean)
flow.append(("Dropped trees whose master cultivar disagrees with the crown polygon",
             n_rec, len(clean),
             "join-quality flag; these sit on pollinizer-row boundaries"))

# --------------------------------------------------- build the model sample
recs = []
for (t, yr), y in sorted(ymeas.items()):
    if t not in clean:
        continue
    j = clean[t][0]
    cult = mcult[j]
    feats = {
        "EBI": M[j].get(f"EBI_Norm_{yr}"),
        "Chill": M[j].get(f"Chill_Portions_{yr}"),
        "Growth": M[j].get(f"Growth_April_{yr}"),
    }
    if cult not in ("UEF", "53"):
        continue
    if any(v is None or (isinstance(v, float) and math.isnan(v))
           for v in feats.values()):
        continue
    recs.append(dict(tree=t, year=yr, cult=cult, y=y, **feats))

flow.append(("Complete cases on EBI, Chill Portions and April growth, cultivars UEF and 53",
             len(recs), len({r["tree"] for r in recs}), "modelling sample"))

# ------------------------------------------------------------------ the model
y = np.array([r["y"] for r in recs], float)
is53 = np.array([1.0 if r["cult"] == "53" else 0.0 for r in recs])
ebi = np.array([r["EBI"] for r in recs], float)
X = np.column_stack([
    np.ones(len(recs)),
    is53,                      # cultivar
    ebi,                       # mean bloom
    ebi * is53,                # bloom x cultivar interaction
    [r["Chill"] for r in recs],
    [r["Growth"] for r in recs],
])
groups = np.array([r["tree"] for r in recs])
yearsv = np.array([r["year"] for r in recs])


def r2(obs, pred):
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_predict(tr, te):
    beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
    return X[te] @ beta


def cv_r2(folds):
    pred = np.full(len(y), np.nan)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(tr) < X.shape[1] + 2 or len(te) == 0:
            continue
        pred[te] = fit_predict(tr, te)
    ok = ~np.isnan(pred)
    return r2(y[ok], pred[ok]), int(ok.sum()), float(np.mean(np.abs(y[ok] - pred[ok])))


beta_full, *_ = np.linalg.lstsq(X, y, rcond=None)
in_sample = r2(y, X @ beta_full)

# 1. random record-level 5-fold, averaged over seeds (as currently reported)
rand_vals = []
for seed in (42, 1, 7, 13, 99, 2024):
    idx = np.random.default_rng(seed).permutation(len(y))
    rand_vals.append(cv_r2(np.array_split(idx, 5))[0])
rand_mean = float(np.mean(rand_vals))

# 2. grouped 5-fold by tree ID: every record of a tree stays in one fold
uniq = np.unique(groups)
grouped_vals = []
for seed in (42, 1, 7, 13, 99, 2024):
    perm = np.random.default_rng(seed).permutation(len(uniq))
    chunks = np.array_split(perm, 5)
    folds = [np.where(np.isin(groups, uniq[c]))[0] for c in chunks]
    grouped_vals.append(cv_r2(folds)[0])
grouped_mean = float(np.mean(grouped_vals))
_, n_used, grouped_mae = cv_r2(
    [np.where(np.isin(groups, uniq[c]))[0]
     for c in np.array_split(np.random.default_rng(42).permutation(len(uniq)), 5)])

# 3. leave-one-year-out
loyo = {}
for yr in YEARS:
    te = np.where(yearsv == yr)[0]
    tr = np.where(yearsv != yr)[0]
    if len(tr) < X.shape[1] + 2 or len(te) == 0:
        loyo[str(yr)] = None
        continue
    pred = fit_predict(tr, te)
    loyo[str(yr)] = dict(n_test=int(len(te)), n_train=int(len(tr)),
                         r2=round(r2(y[te], pred), 4),
                         mae=round(float(np.mean(np.abs(y[te] - pred))), 4))

# how much repetition is there at all?
counts = {}
for g in groups:
    counts[g] = counts.get(g, 0) + 1
repeated = sum(1 for v in counts.values() if v > 1)

res = dict(
    sample_flow=[dict(step=s, records=n, trees=t, note=note)
                 for s, n, t, note in flow],
    model="OLS: yield ~ cultivar + EBI * cultivar + Chill_Portions + Growth_April",
    n_records=len(y), n_trees=int(len(uniq)),
    trees_appearing_in_more_than_one_year=repeated,
    in_sample_r2=round(in_sample, 4),
    cv_random_record_level=dict(r2=round(rand_mean, 4),
                                per_seed=[round(v, 4) for v in rand_vals]),
    cv_grouped_by_tree=dict(r2=round(grouped_mean, 4),
                            mae=round(grouped_mae, 4),
                            per_seed=[round(v, 4) for v in grouped_vals]),
    leave_one_year_out=loyo,
)

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "review_grouped_cv_and_sampleflow.json"), "w") as f:
    json.dump(res, f, indent=2)

print("SAMPLE FLOW")
for s, n, t, note in flow:
    print(f"  {n:5d} records / {t:4d} trees   {s}")
    if note:
        print(f"                              ({note})")

print(f"\nMODEL  {res['model']}")
print(f"  n = {len(y)} tree-years on {len(uniq)} trees; "
      f"{repeated} trees appear in more than one year")
print(f"  in-sample R2                 {in_sample:.4f}")
print(f"  CV, random record level      {rand_mean:.4f}   <- as currently reported")
print(f"  CV, grouped by tree ID       {grouped_mean:.4f}")
print("  leave-one-year-out:")
for yr, d in loyo.items():
    print(f"    hold out {yr}: " + ("not estimable" if d is None else
          f"R2 {d['r2']:.4f}  MAE {d['mae']:.3f}  (train {d['n_train']}, test {d['n_test']})"))
