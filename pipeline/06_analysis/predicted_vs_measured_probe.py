#!/usr/bin/env python3
"""predicted_vs_measured_probe.py — careful re-examination of the model's
predicted_Yield vs the measured kernel-yield census, joined DIRECTLY from the
Yield_with_clustering GPKGs (no chained joins through the master). Tests agreement
at tree, row, and cultivar level. -> Results_Analysis/08_UEF53_Rerun/Predicted_vs_Measured_Probe.png"""
import os, sys, sqlite3, struct, math, csv, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, utm36n_to_latlon
BASE = find_thesis_root()
FIG = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun"); os.makedirs(FIG, exist_ok=True)
SC = os.environ.get("SCRATCH_DIR", FIG)
CCOL = {"53": "#2E75B6", "UEF": "#C0392B"}
GP = os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs")


def env_c(b):
    bo = '<' if (b[3] & 1) else '>'; mnx, mxx, mny, mxy = struct.unpack(bo + 'dddd', b[8:40]); return (mnx + mxx) / 2, (mny + mxy) / 2


def read_pred(yr):
    f = os.path.join(GP, f"Yield_with_clustering_{yr}.gpkg"); con = sqlite3.connect(f); cur = con.cursor()
    tab = f"Yield_with_clustering_{yr}"
    R = cur.execute(f'SELECT cultivar,Row,predicted_Yield,geom FROM "{tab}" WHERE Plot="A"').fetchall(); con.close()
    out = []
    for cv, row, py, g in R:
        if g is None or py is None: continue
        x, y = env_c(g); lat, lon = utm36n_to_latlon(x, y); out.append((cv, row, float(py), lat, lon))
    return out


yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))


def hav(a, b, d, e):
    R = 6371000; p = math.pi / 180; x = math.sin((d - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(d * p) * math.sin((e - b) * p / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0, x)))


def join(yr):
    pred = read_pred(yr); plat = np.array([p[3] for p in pred]); plon = np.array([p[4] for p in pred])
    meas = [(float(r["net_kernel_yield_per_tree_kg"]), float(r["latitude"]), float(r["longitude"])) for r in yc if int(r["year"]) == yr]
    mv = []; pv = []; cu = []; rw = []
    for yv, la, lo in meas:
        ds = np.array([hav(la, lo, a, b) for a, b in zip(plat, plon)]); j = int(ds.argmin())
        if ds[j] <= 8: mv.append(yv); pv.append(pred[j][2]); cu.append(pred[j][0]); rw.append(pred[j][1])
    return np.array(mv), np.array(pv), np.array(cu), np.array(rw)


res = {}
mv23, pv23, cu23, rw23 = join(2023)
res["2023"] = {"n": len(mv23), "pearson": round(float(np.corrcoef(mv23, pv23)[0, 1]), 3),
               "cultivar_measured": {c: round(float(mv23[cu23 == c].mean()), 2) for c in ["UEF", "53"]},
               "cultivar_predicted": {c: round(float(pv23[cu23 == c].mean()), 2) for c in ["UEF", "53"]}}

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
a = ax[0]
for c in ["UEF", "53"]:
    m = cu23 == c; a.scatter(mv23[m], pv23[m], s=42, alpha=.75, color=CCOL[c], edgecolors="white", lw=.5, label=c)
lim = [0, max(mv23.max(), pv23.max()) * 1.05]; a.plot(lim, lim, "k--", lw=1.2, label="1:1")
a.set_title(f"A. Direct join (median 0.5 m), 2023\nper-tree r={res['2023']['pearson']:+.2f} (predicted range compressed)", fontweight="bold", fontsize=9.5)
a.set_xlabel("Measured kernel yield (kg/tree)"); a.set_ylabel("Model predicted yield (kg/tree)"); a.legend(fontsize=8); a.set_xlim(lim); a.set_ylim(lim)
a = ax[1]; w = .35; xb = np.arange(2)
mm = [mv23[cu23 == c].mean() for c in ["UEF", "53"]]; pm = [pv23[cu23 == c].mean() for c in ["UEF", "53"]]
a.bar(xb - w / 2, mm, width=w, color="#2ecc71", label="measured")
a.bar(xb + w / 2, pm, width=w, color="#9b59b6", label="model predicted")
a.set_xticks(xb); a.set_xticklabels(["UEF", "cv 53"]); a.set_ylabel("Mean yield 2023 (kg/tree)")
a.set_title("B. Model INVERTS the cultivar ranking\nmeasured UEF>53; predicted 53>UEF", fontweight="bold", fontsize=9.5); a.legend(fontsize=8)
a = ax[2]
rows = sorted(set(rw23)); rm = []; rp = []
for r in rows:
    m = rw23 == r
    if m.sum() >= 2: rm.append(mv23[m].mean()); rp.append(pv23[m].mean())
rm = np.array(rm); rp = np.array(rp); rr = float(np.corrcoef(rm, rp)[0, 1])
a.scatter(rm, rp, s=70, color="#e67e22")
a.set_title(f"C. Even row-mean aggregate disagrees\nrow-level r={rr:+.2f} (n={len(rm)} rows)", fontweight="bold", fontsize=9.5)
a.set_xlabel("Row-mean measured (kg/tree)"); a.set_ylabel("Row-mean predicted (kg/tree)")
res["2023"]["row_level_r"] = round(rr, 3)
fig.suptitle("Model predicted_Yield vs measured kernel yield: disagrees at tree, row and cultivar level (2023)", fontweight="bold", fontsize=12)
fig.tight_layout()
for d in (FIG, SC): fig.savefig(os.path.join(d, "Predicted_vs_Measured_Probe.png"), dpi=140, bbox_inches="tight")
json.dump(res, open(os.path.join(FIG, "Predicted_vs_Measured_Probe.json"), "w"), indent=2)
print(json.dumps(res, indent=1))
