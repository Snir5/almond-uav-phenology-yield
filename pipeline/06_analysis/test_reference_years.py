#!/usr/bin/env python3
"""
test_reference_years.py — does the radiometric REFERENCE YEAR change the EBI vs
true-yield relation, and does any reference beat 2024?

Reads the small Raw_Crown_Bands.xlsx (+ Raw_Crown_ChannelStats.json) produced once
by extract_raw_crown_bands.py, applies each candidate reference year's shared-scale
affine (identical formula to apply_fixed_zones_yearly.py), recomputes per-crown
EBI-of-means, and correlates it with measured yield WITHIN each cultivar, for every
reference in {2021, 2022, 2023, 2024}. No re-extraction needed.
-> Results_Analysis/reference_year_test.json + figure.
"""
import os, sys, csv, math, json
import numpy as np, openpyxl
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import r_pvalue, sig_stars
BASE = find_thesis_root(); OUT = os.path.join(BASE, "Results_Analysis")
FIG = os.path.join(OUT, "08_UEF53_Rerun"); SC = os.environ.get("SCRATCH_DIR", FIG)
EBI_EPS = 256.0; MEAS = [2022, 2023, 2024]; REFS = [2021, 2022, 2023, 2024]; CCOL = {"53": "#2E75B6", "UEF": "#C0392B"}
RAW = os.path.join(BASE, "4band_mosaic", "Raw_Crown_Bands.xlsx")
STATS = os.path.join(BASE, "4band_mosaic", "Raw_Crown_ChannelStats.json")
if not os.path.exists(RAW):
    sys.exit(f"Run extract_raw_crown_bands.py first (missing {RAW}). It produces the raw per-crown bands "
             "this test needs; that step requires rasterio + the orthomosaics on your machine.")


def star(p): return sig_stars(p) if (p == p) else ""


cs = json.load(open(STATS))
wb = openpyxl.load_workbook(RAW, read_only=True, data_only=True); ws = wb.active
raw = list(ws.iter_rows(values_only=True)); H = raw[0]; T = [dict(zip(H, r)) for r in raw[1:]]; wb.close()
M = load_master(); cultby = {r["Zone_Value"]: r.get("cultivar") for r in M}


def ebi_of(rawR, rawG, rawB, year, ref):
    sy = (cs[str(year)]["sR"] + cs[str(year)]["sG"] + cs[str(year)]["sB"]) / 3
    sr = (cs[str(ref)]["sR"] + cs[str(ref)]["sG"] + cs[str(ref)]["sB"]) / 3
    scale = sr / sy
    R = (rawR - cs[str(year)]["mR"]) * scale + cs[str(ref)]["mR"]
    G = (rawG - cs[str(year)]["mG"]) * scale + cs[str(ref)]["mG"]
    B = (rawB - cs[str(year)]["mB"]) * scale + cs[str(ref)]["mB"]
    R = max(R, 0); G = max(G, 0); B = max(B, 0)
    den = (G / (B + EBI_EPS)) * (R - B + EBI_EPS)
    return (R + G + B) / den if den != 0 else np.nan


# yield join by lat/lon
yc = [r for r in csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig"))]
coord = {}; ym = {}
for r in yc:
    coord.setdefault(r["tree_id"], (float(r["latitude"]), float(r["longitude"])))
    ym[(r["tree_id"], int(r["year"]))] = float(r["net_kernel_yield_per_tree_kg"])
tlat = np.array([r["Latitude"] for r in T], float); tlon = np.array([r["Longitude"] for r in T], float)
def hav(a, b, d, e):
    R = 6371000; p = math.pi / 180; x = math.sin((d - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(d * p) * math.sin((e - b) * p / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0, x)))
cand = []
for t, (la, lo) in coord.items():
    ds = np.array([hav(la, lo, a, b) for a, b in zip(tlat, tlon)])
    for j in np.where(ds <= 8)[0]: cand.append((float(ds[j]), t, int(j)))
cand.sort(); uy = set(); um = set(); mp = {}
for d, t, j in cand:
    if t in uy or j in um: continue
    mp[t] = j; uy.add(t); um.add(j)

RES = {"note": "EBI-of-means vs measured yield, by cultivar, per radiometric reference year"}
for ref in REFS:
    RES[ref] = {}
    for yr in MEAS:
        for C in ["UEF", "53"]:
            E = []; Y = []
            for t, j in mp.items():
                zv = T[j]["Zone_Value"]
                if cultby.get(zv) != C: continue
                rR, rG, rB = T[j].get(f"rawR_{yr}"), T[j].get(f"rawG_{yr}"), T[j].get(f"rawB_{yr}")
                yv = ym.get((t, yr))
                if None in (rR, rG, rB, yv): continue
                e = ebi_of(rR, rG, rB, yr, ref)
                if e == e: E.append(e); Y.append(yv)
            E = np.array(E); Y = np.array(Y)
            if len(E) >= 5 and E.std() > 0:
                r = float(np.corrcoef(E, Y)[0, 1])
                RES[ref].setdefault(str(yr), {})[C] = {"r": round(r, 3), "p": round(r_pvalue(r, len(E)), 4), "n": len(E), "sig": star(r_pvalue(r, len(E)))}
json.dump(RES, open(os.path.join(OUT, "reference_year_test.json"), "w"), indent=2)
print(json.dumps(RES, indent=1))

# figure: 2023 UEF & 53 r vs reference year
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for k, C in enumerate(["UEF", "53"]):
    a = ax[k]
    rr = [RES[ref].get("2023", {}).get(C, {}).get("r", np.nan) for ref in REFS]
    a.plot(REFS, rr, "o-", color=CCOL[C], lw=2, ms=9)
    for x, y in zip(REFS, rr):
        if y == y: a.annotate(f"{y:+.2f}", (x, y), xytext=(4, 6), textcoords="offset points", fontsize=9)
    a.axhline(0, color="k", lw=.7); a.set_xticks(REFS)
    a.set_title(f"{C}: EBI2023 -> yield vs reference year", fontweight="bold"); a.set_xlabel("radiometric reference year"); a.set_ylabel("Pearson r"); a.grid(alpha=.3)
fig.suptitle("Does the radiometric reference year strengthen the EBI-yield relation? (2023)", fontweight="bold", fontsize=12)
fig.tight_layout()
for d in (FIG, SC): fig.savefig(os.path.join(d, "Reference_Year_Test.png"), dpi=140, bbox_inches="tight")
print("saved figure")
