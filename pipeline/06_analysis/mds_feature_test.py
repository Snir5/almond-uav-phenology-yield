#!/usr/bin/env python3
"""
mds_feature_test.py: tests Maximum Daily Shrinkage (MDS), a genuinely new physiological
variable, found in the newly-added RF_models_portable_package (user-provided). Source:
Trees_Data_Survey/RF_models_portable_package/models/Noam_after_revisions/
DataSet_KDM_ALL(11_09_24)_Filter_2_without_garbage.csv, a per-flight-date raw table
(not previously used anywhere in this project).

Scope and honesty note: this file's Plot-A tree panel is a small FIXED set of 20 trees
monitored across both 2022 and 2023 (verified: 20 unique Plot-A TreeIDs each year, full
overlap). This is much smaller than the 149-167 trees used in sections 21-23 (which draw
on the separate, larger Yield_with_clustering GPKGs), so MDS cannot be pooled into that
larger model.  Every result here is n<=20 per year (n<=40 pooled, repeated-measures on
the same 20 trees), directional only per project convention, not a substitute for the
inferential 2023 census tests.

Join validation: this file's TreeID -> tree_code join against kedma_plot_a_yield.csv
(the "KEDMA-A-{TreeID}" convention) was checked exactly: 20/20 matched for 2022 and
20/20 for 2023, zero mismatches (net_kernel_yield_per_tree_kg identical to 1e-4). This
is a cleaner, exact-string join, not the usual distance-threshold spatial match, because
this file happens to preserve the TreeID exactly (the OTHER RF-package CSVs,
yield_data_predicted_final.csv and Yield_prediction_data.csv, do NOT: their 2023 TreeID
numbering does not correspond to the census tree_code at all, checked and rejected).
Master tree assignment then reuses the existing kedma-to-master haversine match
(<=8 m, first-come dedup) already validated and used throughout yield_model_v2/v3.

MDS itself is measured in April (2022 only, no April flight in 2023) and June (both
years), analogous to the existing SWP_April/SWP_June structure (section 21). Same Q0-Q3
style questions, at this much smaller n:
  Q0. MDS vs EBI same year/cultivar.
  Q1. MDS vs measured yield same year/cultivar.
  Q3. Does MDS add anything on top of EBI (nested partial F), pooled.
-> Results_Analysis/08_UEF53_Rerun/MDS_Feature_Test.json + MDS_Feature_Test.png
"""
import os, sys, csv, math, json
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import r_pvalue, f_pvalue, ols_rss, sig_stars

BASE = find_thesis_root()
KDM = os.path.join(BASE, "Trees_Data_Survey", "RF_models_portable_package", "models",
                    "Noam_after_revisions", "DataSet_KDM_ALL(11_09_24)_Filter_2_without_garbage.csv")
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")


def star(p): return sig_stars(p) if p == p else ""


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~np.isnan(x) & ~np.isnan(y); x, y = x[m], y[m]; n = len(x)
    if n < 4: return {"r": None, "n": n}
    r = float(np.corrcoef(x, y)[0, 1]); p = float(r_pvalue(r, n))
    return {"r": round(r, 3), "p": round(p, 4), "n": n, "sig": star(p)}


# ---------------- load master + cultivar ----------------
M = load_master()
cultivar = [r.get("cultivar") for r in M]

# ---------------- load DataSet_KDM_ALL, Plot A, MDS by TreeID/year/month ----------------
rows = list(csv.DictReader(open(KDM, encoding="utf-8-sig")))
plotA = [r for r in rows if r.get("Plot") == "A"]
uniq_ids = sorted(set(r["TreeID"] for r in plotA))
print(f"DataSet_KDM_ALL Plot-A unique TreeIDs: {len(uniq_ids)}")

# MDS_April_2022 (month==4), MDS_June_2022/2023 (month==6)
mds = defaultdict(dict)  # tree_code -> {"MDS_April_2022":v, "MDS_June_2022":v, "MDS_June_2023":v}
cult_kdm = {}
for r in plotA:
    tid = r["TreeID"]; yr = r.get("Year"); mo = r.get("Month"); v = r.get("MDS")
    if not v: continue
    code = f"KEDMA-A-{tid}"
    cult_kdm[code] = r.get("cultivar")
    if yr == "2022" and mo == "4":
        mds[code]["MDS_April_2022"] = float(v)
    if yr == "2022" and mo == "6":
        mds[code]["MDS_June_2022"] = float(v)
    if yr == "2023" and mo == "6":
        mds[code]["MDS_June_2023"] = float(v)

print(f"trees with MDS_April_2022: {sum('MDS_April_2022' in d for d in mds.values())}")
print(f"trees with MDS_June_2022: {sum('MDS_June_2022' in d for d in mds.values())}")
print(f"trees with MDS_June_2023: {sum('MDS_June_2023' in d for d in mds.values())}")

# ---------------- join kedma_plot_a_yield.csv (tree_code -> lat/lon, measured yield) ----------------
yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
ymeas = defaultdict(dict); coord = {}
for r in yc:
    ymeas[r["tree_code"]][int(r["year"])] = float(r["net_kernel_yield_per_tree_kg"])
    coord[r["tree_code"]] = (float(r["latitude"]), float(r["longitude"]))

mlat = np.array([r.get("Latitude") for r in M], float)
mlon = np.array([r.get("Longitude") for r in M], float)


def hav(a, b, d, e):
    p = math.pi / 180
    x = math.sin((d - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(d * p) * math.sin((e - b) * p / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(max(0, x)))


# match only the tree_codes that have MDS data, to master, via the same haversine <=8m rule
cand = []
for code in mds:
    if code not in coord: continue
    la, lo = coord[code]
    ds = np.array([hav(la, lo, a, b) for a, b in zip(mlat, mlon)])
    for j in np.where(ds <= 8)[0]: cand.append((float(ds[j]), code, int(j)))
cand.sort(); uy = set(); um = set(); jmap = {}
for d, code, j in cand:
    if code in uy or j in um: continue
    jmap[code] = j; uy.add(code); um.add(j)
print(f"matched to master: {len(jmap)} / {len(mds)}")

# cultivar cross-check
agree = sum(1 for code, j in jmap.items() if cultivar[j] == cult_kdm.get(code))
print(f"cultivar agreement: {agree}/{len(jmap)}")

RESULTS = {
    "note": "MDS (maximum daily shrinkage) from the newly-added RF_models_portable_package. "
            "Fixed 20-tree Plot-A panel, 2022+2023, much smaller than sections 21-23's "
            "149-167 tree sample; every result here is n<=20/year, directional only.",
    "provenance": "DataSet_KDM_ALL(11_09_24)_Filter_2_without_garbage.csv, "
                   "Trees_Data_Survey/RF_models_portable_package/models/Noam_after_revisions/",
    "n_unique_PlotA_TreeIDs_in_file": len(uniq_ids),
    "n_matched_to_master": len(jmap),
    "cultivar_agreement": f"{agree}/{len(jmap)}",
}

# ---------------- build per-tree table ----------------
recs = []
for code, j in jmap.items():
    rec = {"code": code, "j": j, "cultivar": cultivar[j]}
    rec.update(mds[code])
    rec["EBI_2022"] = M[j].get("EBI_Norm_2022")
    rec["EBI_2023"] = M[j].get("EBI_Norm_2023")
    rec["yield_2022"] = ymeas.get(code, {}).get(2022)
    rec["yield_2023"] = ymeas.get(code, {}).get(2023)
    recs.append(rec)

MDS_FIELDS_BY_YEAR = {"MDS_April_2022": ("EBI_2022", "yield_2022"),
                      "MDS_June_2022": ("EBI_2022", "yield_2022"),
                      "MDS_June_2023": ("EBI_2023", "yield_2023")}

Q0 = {}  # MDS vs EBI, same year, by cultivar
Q1 = {}  # MDS vs measured yield, same year, by cultivar
for field, (ebi_col, yield_col) in MDS_FIELDS_BY_YEAR.items():
    Q0[field] = {}
    Q1[field] = {}
    for cv in ["UEF", "53"]:
        sub = [r for r in recs if r["cultivar"] == cv and field in r]
        x = [r[field] for r in sub]
        ebi = [r[ebi_col] for r in sub]
        yld = [r[yield_col] for r in sub]
        Q0[field][cv] = pearson(x, ebi)
        Q1[field][cv] = pearson(x, yld)

RESULTS["Q0_MDS_vs_EBI_same_year_by_cultivar"] = Q0
RESULTS["Q1_MDS_vs_measured_yield_same_year_by_cultivar"] = Q1

# pooled (both cultivars, cultivar as covariate) nested partial-F: does MDS_June add to EBI?
# Only MDS_June_2022 and MDS_June_2023 can be pooled across years (April is 2022-only).
pooled_rows = []
for r in recs:
    for field, yr in [("MDS_June_2022", 2022), ("MDS_June_2023", 2023)]:
        if field in r and r[f"EBI_{yr}"] is not None and r[f"yield_{yr}"] is not None:
            pooled_rows.append({"mds": r[field], "ebi": r[f"EBI_{yr}"], "yield": r[f"yield_{yr}"],
                                 "cultivar": r["cultivar"], "year": yr})

npool = len(pooled_rows)
if npool >= 8:
    y = np.array([r["yield"] for r in pooled_rows])
    ebi = np.array([r["ebi"] for r in pooled_rows])
    uef = np.array([1.0 if r["cultivar"] == "UEF" else 0.0 for r in pooled_rows])
    mds_v = np.array([r["mds"] for r in pooled_rows])
    one = np.ones(npool)
    X_ebi = np.column_stack([one, uef, ebi, ebi * uef])
    X_full = np.column_stack([one, uef, ebi, ebi * uef, mds_v])
    rss0, p0, _ = ols_rss(X_ebi, y)
    rss1, p1, _ = ols_rss(X_full, y)
    df1 = p1 - p0; df2 = npool - p1
    if rss1 > 0 and df1 > 0 and df2 > 0:
        Fstat = ((rss0 - rss1) / df1) / (rss1 / df2)
        pF = float(f_pvalue(Fstat, df1, df2))
    else:
        Fstat, pF = None, None
    RESULTS["Q3_pooled_MDS_June_adds_to_EBI"] = {
        "n": npool, "R2_EBI_only": round(1 - rss0 / ((y - y.mean()) ** 2).sum(), 3),
        "R2_EBI_plus_MDS": round(1 - rss1 / ((y - y.mean()) ** 2).sum(), 3),
        "partial_F": round(Fstat, 3) if Fstat is not None else None,
        "p": round(pF, 4) if pF is not None else None, "df": [df1, df2],
    }
else:
    RESULTS["Q3_pooled_MDS_June_adds_to_EBI"] = {"note": f"n={npool} too small to test"}

print(json.dumps(RESULTS, indent=2, default=str))

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
colors = {"UEF": "#2C3E50", "53": "#C0392B"}
panels = [("MDS_April_2022", "yield_2022", "MDS, April 2022 vs 2022 measured yield"),
          ("MDS_June_2022", "yield_2022", "MDS, June 2022 vs 2022 measured yield"),
          ("MDS_June_2023", "yield_2023", "MDS, June 2023 vs 2023 measured yield")]
for ax, (field, ycol, title) in zip(axes, panels):
    for cv in ["UEF", "53"]:
        sub = [r for r in recs if r["cultivar"] == cv and field in r and r[ycol] is not None]
        if not sub: continue
        xs = [r[field] for r in sub]; ys = [r[ycol] for r in sub]
        ax.scatter(xs, ys, label=f"{cv} (n={len(sub)})", color=colors[cv], alpha=0.75)
    ax.set_xlabel(field); ax.set_ylabel("measured yield (kg)"); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "MDS_Feature_Test.png"), dpi=130)
print("saved figure")

with open(os.path.join(OUT, "MDS_Feature_Test.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("saved json")
