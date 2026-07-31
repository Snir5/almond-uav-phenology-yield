#!/usr/bin/env python3
"""
clean_join_ebi_yield_allyears.py — re-do the EBI vs true-yield analysis on a
QUALITY-CLEANED join, for all measured years (2022, 2023, 2024), per year and pooled.

Cleaning rule: after the greedy 1:1 spatial join (yield point -> master tree, <=8 m),
drop any tree whose master cultivar disagrees with the nearest GPKG crown-polygon
cultivar (a join-quality flag; these sit at pollinizer-row boundaries). Optionally
also flag far matches (>3 m). Then recompute within-cultivar same-season EBI->yield:
  - per year (2022, 2023, 2024),
  - pooled across all three years, with the season removed (yield and EBI centered
    within cultivar x year) so the pooled slope is the pure within-cultivar effect,
  - and a pooled ANCOVA yield ~ EBI x cultivar + year(climate) for the interaction.
-> Results_Analysis/clean_join_ebi_yield_allyears.json + figure.
"""
import os, sys, csv, math, struct, sqlite3, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master, utm36n_to_latlon
from stats_utils import r_pvalue, t_pvalue, f_pvalue, ols_rss, sig_stars
BASE = find_thesis_root(); OUT = os.path.join(BASE, "Results_Analysis")
FIG = os.path.join(OUT, "08_UEF53_Rerun"); SC = os.environ.get("SCRATCH_DIR", FIG)
MEAS = [2022, 2023, 2024]; CCOL = {"53": "#2E75B6", "UEF": "#C0392B"}


def star(p): return sig_stars(p) if (p == p) else ""


M = load_master()
def col(c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in M], float)
mlat, mlon = col("Latitude"), col("Longitude"); mcult = [r.get("cultivar") for r in M]
CP = {y: float(np.nanmean(col(f"Chill_Portions_{y}"))) for y in MEAS}


def hav(a, b, d, e):
    R = 6371000; p = math.pi / 180; x = math.sin((d - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(d * p) * math.sin((e - b) * p / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0, x)))


# GPKG crown cultivar reference (Plot A)
def envc(b):
    bo = '<' if (b[3] & 1) else '>'; mnx, mxx, mny, mxy = struct.unpack(bo + 'dddd', b[8:40]); return (mnx + mxx) / 2, (mny + mxy) / 2
con = sqlite3.connect(os.path.join(BASE, "Trees_Data_Survey", "Field_Data_Yield_GPKGs", "Yield_with_clustering_2023.gpkg"))
gp = [(cv, *utm36n_to_latlon(*envc(g))) for plot, cv, g in con.execute('SELECT Plot,cultivar,geom FROM Yield_with_clustering_2023') if plot == 'A' and g]
con.close()
gla = np.array([p[1] for p in gp]); glo = np.array([p[2] for p in gp])

yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
coord = {}; ym = {}
for r in yc:
    coord.setdefault(r["tree_id"], (float(r["latitude"]), float(r["longitude"])))
    ym[(r["tree_id"], int(r["year"]))] = float(r["net_kernel_yield_per_tree_kg"])

# greedy 1:1 join (yield tree -> master), plus flags
cand = []
for t, (la, lo) in coord.items():
    ds = np.array([hav(la, lo, a, b) for a, b in zip(mlat, mlon)])
    for j in np.where(ds <= 8)[0]: cand.append((float(ds[j]), t, int(j)))
cand.sort(); uy = set(); um = set(); match = {}
for d, t, j in cand:
    if t in uy or j in um: continue
    match[t] = (j, d); uy.add(t); um.add(j)
# cultivar-consistency flag
def gpkg_cult(t):
    la, lo = coord[t]; k = int(np.array([hav(la, lo, a, b) for a, b in zip(gla, glo)]).argmin()); return gp[k][0]
flag_mismatch = {t for t, (j, d) in match.items() if mcult[j] != gpkg_cult(t)}
clean = {t: (j, d) for t, (j, d) in match.items() if t not in flag_mismatch}

R = {"cleaning_rule": "drop trees whose master cultivar != nearest GPKG crown cultivar",
     "n_matched_total": len(match), "n_dropped_mismatch": len(flag_mismatch), "n_clean": len(clean)}

# per-year join quality + within-cultivar EBI->yield (clean)
def within_year(mp, y, C):
    E = []; Y = []
    for t, (j, d) in mp.items():
        if mcult[j] != C: continue
        yv = ym.get((t, y)); e = M[j].get(f"EBI_Norm_{y}")
        if yv is not None and e is not None: E.append(float(e)); Y.append(yv)
    E = np.array(E); Y = np.array(Y)
    if len(E) < 3 or E.std() == 0: return dict(r=None, n=len(E))
    r = float(np.corrcoef(E, Y)[0, 1]); return dict(r=round(r, 3), p=round(r_pvalue(r, len(E)), 4), n=len(E), sig=star(r_pvalue(r, len(E))))


R["per_year_clean"] = {}
for y in MEAS:
    ny = sum(1 for t, (j, d) in clean.items() if ym.get((t, y)) is not None and M[j].get(f"EBI_Norm_{y}") is not None)
    drop_y = sum(1 for t in flag_mismatch if ym.get((t, y)) is not None)
    R["per_year_clean"][y] = {"dropped_this_year": drop_y,
                              "UEF": within_year(clean, y, "UEF"), "53": within_year(clean, y, "53")}
# also the ORIGINAL (uncleaned) per-year for comparison
R["per_year_original"] = {y: {"UEF": within_year(match, y, "UEF"), "53": within_year(match, y, "53")} for y in MEAS}

# POOLED across all years (clean): remove season by centering yield & EBI within cultivar x year
def pooled(mp):
    recs = []  # (cult, year, ebi, yield)
    for t, (j, d) in mp.items():
        c = mcult[j]
        for y in MEAS:
            yv = ym.get((t, y)); e = M[j].get(f"EBI_Norm_{y}")
            if yv is not None and e is not None: recs.append((c, y, float(e), yv))
    out = {"n": len(recs)}
    # within-cultivar pooled correlation on season-centered values
    for C in ["UEF", "53"]:
        sub = [(y, e, v) for c, y, e, v in recs if c == C]
        ec = []; vc = []
        for yr in MEAS:
            g = [(e, v) for (y, e, v) in sub if y == yr]
            if len(g) < 2: continue
            e = np.array([a for a, _ in g]); v = np.array([b for _, b in g])
            ec += list(e - e.mean()); vc += list(v - v.mean())   # remove that year's mean
        ec = np.array(ec); vc = np.array(vc)
        r = float(np.corrcoef(ec, vc)[0, 1]); out[C] = {"r_within_year_centered": round(r, 3),
            "p": round(r_pvalue(r, len(ec)), 4), "n": len(ec), "sig": star(r_pvalue(r, len(ec)))}
    # ANCOVA: yield ~ EBI + cultivar + EBI:cultivar + year(dummies)
    c = np.array([a for a, _, _, _ in recs]); yr = np.array([b for _, b, _, _ in recs])
    e = np.array([x for _, _, x, _ in recs]); v = np.array([x for _, _, _, x in recs]); n = len(v)
    uef = (c == "UEF").astype(float); d22 = (yr == 2022).astype(float); d24 = (yr == 2024).astype(float)
    one = np.ones(n)
    def dm(cols): return np.column_stack([one] + cols)
    rss_full, kf, _ = ols_rss(dm([e, uef, e * uef, d22, d24]), v)   # with interaction
    rss_noint, kn, _ = ols_rss(dm([e, uef, d22, d24]), v)           # without interaction
    F = ((rss_noint - rss_full) / (kf - kn)) / (rss_full / (n - kf))
    out["interaction_EBIxcultivar"] = {"F": round(float(F), 2), "p": round(float(f_pvalue(F, kf - kn, n - kf)), 4), "sig": star(f_pvalue(F, kf - kn, n - kf))}
    return out


R["pooled_all_years_clean"] = pooled(clean)
R["pooled_all_years_original"] = pooled(match)
json.dump(R, open(os.path.join(OUT, "clean_join_ebi_yield_allyears.json"), "w"), indent=2, default=str)
json.dump(R, open(os.path.join(SC, "clean_join_ebi_yield_allyears.json"), "w"), indent=2, default=str)
print(json.dumps(R, indent=1, default=str))

# ---- figure: per-year (clean) + pooled, by cultivar ----
fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
a = ax[0]; off = {"UEF": 0.12, "53": -0.12}
for C in ["UEF", "53"]:
    for y in MEAS:
        d = R["per_year_clean"][y][C]
        if d.get("r") is None: continue
        a.errorbar(d["r"], y + off[C], fmt="o", color=CCOL[C], ms=10, capsize=4)
        a.text(d["r"], y + off[C] + 0.07, f"{d['r']:+.2f}{'*' if d.get('sig') not in ('ns',None,'') else ''} (n={d['n']})", ha="center", fontsize=8.5)
a.axvline(0, color="k", lw=.9); a.set_yticks(MEAS); a.set_ylim(2021.4, 2024.6); a.set_xlim(-1, 1.05)
a.set_xlabel("Same-season EBI -> true yield  Pearson r"); a.grid(alpha=.3)
from matplotlib.lines import Line2D
a.legend([Line2D([0], [0], color=CCOL['UEF'], marker='o', ls=''), Line2D([0], [0], color=CCOL['53'], marker='o', ls='')], ["UEF", "cv 53"], loc="lower right", fontsize=9)
a.set_title(f"A. Cleaned join, by year (dropped {R['n_dropped_mismatch']} boundary trees)\nUEF positive, cv-53 negative each year", fontweight="bold", fontsize=10)
a = ax[1]
pc = R["pooled_all_years_clean"]; w = .5
vals = [pc["UEF"]["r_within_year_centered"], pc["53"]["r_within_year_centered"]]
a.bar([0, 1], vals, color=[CCOL["UEF"], CCOL["53"]], width=w)
for x, C in zip([0, 1], ["UEF", "53"]):
    d = pc[C]; a.text(x, d["r_within_year_centered"] + (0.01 if d["r_within_year_centered"] >= 0 else -0.03), f"{d['r_within_year_centered']:+.2f}{'*' if d['sig'] not in ('ns','') else ''}\n(n={d['n']})", ha="center", fontsize=9)
a.axhline(0, color="k", lw=.9); a.set_xticks([0, 1]); a.set_xticklabels(["UEF", "cv 53"]); a.grid(axis="y", alpha=.3)
a.set_ylabel("pooled within-cultivar r (season removed)")
a.set_title(f"B. All years pooled 2022-2024 (n={pc['n']}, season removed)\nEBI x cultivar interaction F={pc['interaction_EBIxcultivar']['F']} {pc['interaction_EBIxcultivar']['sig']}", fontweight="bold", fontsize=10)
fig.suptitle("EBI vs true yield on a quality-cleaned join, all measured years", fontweight="bold", fontsize=13)
fig.tight_layout()
for d in (FIG, SC): fig.savefig(os.path.join(d, "Clean_Join_EBI_Yield_AllYears.png"), dpi=140, bbox_inches="tight")
print("saved figure")
