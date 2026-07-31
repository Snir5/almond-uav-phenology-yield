#!/usr/bin/env python3
"""
deep_yield_master_audit.py: final, row-by-row deep audit that every yield
measurement in kedma_plot_a_yield.csv (202 rows, 162 distinct tree_ids) is
matched to the CORRECT, unique tree in the master, not just an aggregate
sample check. For every row, cross-checks:
  1. Haversine match distance (flag anything above 3 m, well past the 0.88 m
     median, for individual scrutiny)
  2. tree_position grid row/col agreement (148/202 rows have it)
  3. Same-tree-id, cross-year self-consistency: does tree_id X always map to
     the SAME master tree in every year it appears? (internal consistency,
     independent of any external field)
  4. Uniqueness: is any single master tree claimed by two DIFFERENT yield
     tree_ids? (collision check, would indicate a many-to-one mismatch)
  5. Reports the matched cultivar for every row (for a final by-eye sanity
     read, even though the CSV carries no cultivar field to check against)
"""
import os, sys, csv, math, re
from collections import defaultdict, Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master

BASE = find_thesis_root()
M = load_master()
mlat = np.array([r.get("Latitude") for r in M], float); mlon = np.array([r.get("Longitude") for r in M], float)
cultivar = [r.get("cultivar") for r in M]
master_row2023 = [r.get("Row_2023") for r in M]
master_treeid = [r.get("Tree_ID") for r in M]

yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
print(f"total yield rows: {len(yc)}, distinct tree_ids: {len(set(r['tree_id'] for r in yc))}")

def hav(a, b, d, e):
    p = math.pi / 180
    x = math.sin((d - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(d * p) * math.sin((e - b) * p / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(max(0, x)))

# ---- reproduce the EXACT project-standard matching: dedup by tree_id (not by row) ----
coord = {}
for r in yc:
    coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))

cand = []
for t, (la, lo) in coord.items():
    ds = np.array([hav(la, lo, a, b) for a, b in zip(mlat, mlon)])
    for j in np.where(ds <= 8)[0]: cand.append((float(ds[j]), t, int(j)))
cand.sort(); uy = set(); um = set(); ymap = {}; ymap_dist = {}
for d, t, j in cand:
    if t in uy or j in um: continue
    ymap[t] = j; ymap_dist[t] = d; uy.add(t); um.add(j)

n_unmatched = len(coord) - len(ymap)
print(f"matched: {len(ymap)} / {len(coord)} distinct tree_ids ({n_unmatched} unmatched)")
if n_unmatched:
    for t in coord:
        if t not in ymap:
            print(f"  UNMATCHED tree_id={t} at {coord[t]}")

# ---- check 1: distance distribution, flag > 3m ----
dists = list(ymap_dist.values())
print(f"\ndistance stats: median={np.median(dists):.3f} mean={np.mean(dists):.3f} max={np.max(dists):.3f}")
far = [(t, d) for t, d in ymap_dist.items() if d > 3.0]
print(f"trees with match distance > 3m ({len(far)}):")
for t, d in sorted(far, key=lambda x: -x[1]):
    j = ymap[t]
    print(f"  tree_id={t} dist={d:.2f}m -> master {master_treeid[j]} (cultivar={cultivar[j]})")

# ---- check 2: tree_position row/col agreement, EVERY row not a sample ----
n_pos_checked = 0; n_pos_agree = 0; pos_mismatches = []
for r in yc:
    t = r["tree_id"]; tp = r["tree_position"].strip()
    m = re.match(r"A_r_(\d+)_c_(\d+)", tp)
    if not m or t not in ymap: continue
    csv_row = int(m.group(1)); j = ymap[t]
    n_pos_checked += 1
    mrow = master_row2023[j]
    agree = (mrow is not None and abs(float(mrow) - csv_row) <= 1)
    if agree: n_pos_agree += 1
    else: pos_mismatches.append({"tree_id": t, "year": r["year"], "csv_row": csv_row, "master_row": mrow, "master_tree": master_treeid[j], "dist_m": round(ymap_dist[t], 2)})

print(f"\ntree_position check (every row with position, not sampled): {n_pos_agree}/{n_pos_checked} agree")
for mm in pos_mismatches:
    print(f"  MISMATCH: {mm}")

# ---- check 3: same tree_id, cross-year self-consistency (uses the dedup match, so
#      trivially consistent by construction since ymap is per-tree_id, not per-row;
#      report anyway for completeness/transparency) ----
by_tid_years = defaultdict(list)
for r in yc:
    by_tid_years[r["tree_id"]].append(r["year"])
multi_year = {t: yrs for t, yrs in by_tid_years.items() if len(yrs) > 1}
print(f"\ntree_ids measured in >1 year: {len(multi_year)} (all necessarily map to one master "
      f"tree each, by construction of the per-tree_id match; this just confirms none of "
      f"these were dropped or split)")
n_dropped = sum(1 for t in multi_year if t not in ymap)
print(f"  of these, {n_dropped} failed to match at all")

# ---- check 4: uniqueness / collisions (two DIFFERENT tree_ids -> same master tree) ----
master_claims = defaultdict(list)
for t, j in ymap.items():
    master_claims[j].append(t)
collisions = {j: ts for j, ts in master_claims.items() if len(ts) > 1}
print(f"\ncollision check: {len(collisions)} master trees claimed by >1 yield tree_id "
      f"(should be 0, matching is bijective by construction, i.e. dedup already forbids this)")
for j, ts in collisions.items():
    print(f"  master {master_treeid[j]} claimed by {ts}")

# ---- check 5: cultivar breakdown of matched trees, final sanity read ----
cult_counts = Counter(cultivar[j] for j in ymap.values())
print(f"\ncultivar breakdown of the {len(ymap)} matched trees: {dict(cult_counts)}")
n_not_uef53 = sum(1 for j in ymap.values() if cultivar[j] not in ("UEF", "53"))
print(f"matched trees NOT UEF/53 (cultivar 54 or missing): {n_not_uef53}")
if n_not_uef53:
    for t, j in ymap.items():
        if cultivar[j] not in ("UEF", "53"):
            print(f"  tree_id={t} -> master {master_treeid[j]} cultivar={cultivar[j]}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"total distinct yield trees: {len(coord)}")
print(f"matched: {len(ymap)} ({n_unmatched} unmatched)")
print(f"match distance: median={np.median(dists):.3f}m, all <= {np.max(dists):.3f}m")
print(f"trees >3m: {len(far)}")
print(f"tree_position agreement: {n_pos_agree}/{n_pos_checked} ({100*n_pos_agree/max(n_pos_checked,1):.1f}%)")
print(f"collisions (many-to-one): {len(collisions)}")
print(f"non-UEF/53 matches: {n_not_uef53}")
