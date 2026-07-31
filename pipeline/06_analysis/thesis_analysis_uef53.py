#!/usr/bin/env python3
"""
thesis_analysis_uef53.py — Restricted rerun of the full thesis statistical flow.

Scope change vs thesis_analysis.py:
  * Cultivars restricted to UEF + 53 (cultivar 54 dropped everywhere). 54 is an
    early, bloom-bright pollinizer row with NO measured yield; dropping it makes
    every analysis comparable to the yield ground truth (which only has 53 & UEF).
  * Chilling quantified ONLY by the Dynamic Model -> Chill Portions (CP).
    Chill_Hrs is excluded from all climate analysis (legacy metric).

Path-robust: resolves the thesis root with geo_guardrails.find_thesis_root()
(no hard-coded /sessions/... paths) and validates coordinate alignment on load.

Order: S1 dataset -> S2 EBI temporal -> S3 cultivar -> S4 spatial(Moran) ->
       S5 climate drivers(CP) -> S6 measured yield -> S7 synthesis.
Emits: outputs/thesis_results_uef53.json and figures in Results_Analysis + outputs.
"""
import csv, math, os, json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master, field_boundary
from stats_utils import f_pvalue, t_pvalue, r_pvalue, split_plot_anova, sig_stars, ols_rss

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis")
# scratch dir: sibling 'outputs' of the mount if present, else Results_Analysis
SC = os.environ.get("SCRATCH_DIR", OUT)
os.makedirs(SC, exist_ok=True)

KEEP = {"UEF", "53"}
YEARS = [2021, 2022, 2023, 2024]
CCOL = {"53": "#2E75B6", "UEF": "#C0392B"}


def star(p):
    return sig_stars(p) if not (isinstance(p, float) and math.isnan(p)) else ""


# ---------- load + FILTER to UEF + 53 ----------
rows_all = load_master()                       # validates coord alignment (raises on shift)
M = [r for r in rows_all if r.get("cultivar") in KEEP]
cults = [r.get("cultivar") for r in M]


def col(c):
    return np.array([r.get(c) if r.get(c) is not None else np.nan for r in M], float)


R = {"scope": {"cultivars_kept": sorted(KEEP), "cultivar_54_dropped": True,
               "chilling_metric": "Chill Portions (Dynamic Model, Fishman et al. 1987)",
               "chill_hours_excluded": True}}

from collections import Counter, defaultdict
R["S1_dataset"] = {"n_trees_all": len(rows_all), "n_trees_kept": len(M),
                   "cultivar_counts": dict(Counter(cults)), "years": YEARS}

# ---------- S2 EBI temporal ----------
ebi = {y: col(f"EBI_Norm_{y}") for y in YEARS}
R["S2_ebi_dist"] = {}
for y in YEARS:
    a = ebi[y]; a = a[~np.isnan(a)]
    R["S2_ebi_dist"][y] = {"n": int(len(a)), "mean": round(float(a.mean()), 4),
                           "sd": round(float(a.std(ddof=1)), 4)}


def paired_t(a, b):
    m = ~np.isnan(a) & ~np.isnan(b); d = a[m] - b[m]; n = len(d)
    if n < 3:
        return np.nan, np.nan, n, np.nan
    md = d.mean(); sd = d.std(ddof=1); t = md / (sd / math.sqrt(n))
    return float(t), float(t_pvalue(abs(t), n - 1)), n, float(md)


R["S2_paired_t"] = {}
for a, b in [(2021, 2022), (2022, 2023), (2023, 2024)]:
    t, p, n, md = paired_t(ebi[a], ebi[b])
    R["S2_paired_t"][f"{a}v{b}"] = {"t": round(t, 3), "p": round(p, 4), "n": n,
                                    "dmean": round(md, 4), "sig": star(p)}

# ---------- S3 cultivar effects (2 groups: UEF vs 53) ----------
def f_oneway(groups):
    groups = [np.asarray(g, float) for g in groups]
    groups = [g[~np.isnan(g)] for g in groups]
    groups = [g for g in groups if len(g) > 0]
    allv = np.concatenate(groups); gm = allv.mean(); k = len(groups); N = len(allv)
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    d1, d2 = k - 1, N - k
    if d1 <= 0 or d2 <= 0 or ssw == 0:
        return np.nan, np.nan, np.nan
    F = (ssb / d1) / (ssw / d2)
    return float(F), float(f_pvalue(F, d1, d2)), float(ssb / (ssb + ssw))


uc = ["53", "UEF"]
R["S3_cultivar_anova_ebi"] = {}
cult_arr = np.array(cults)
for y in YEARS:
    groups = [ebi[y][cult_arr == u] for u in uc]
    F, p, eta = f_oneway(groups)
    means = {u: round(float(np.nanmean(ebi[y][cult_arr == u])), 4) for u in uc}
    R["S3_cultivar_anova_ebi"][y] = {"F": round(F, 2), "p": round(p, 6),
                                     "eta2": round(eta, 3), "sig": star(p), "means": means}

# split-plot cultivar x year
long_rows = []
for i, r in enumerate(M):
    c = r.get("cultivar")
    for y in YEARS:
        v = r.get(f"EBI_Norm_{y}")
        if v is not None:
            long_rows.append({"subj": i, "cult": c, "year": y, "ebi": float(v)})
ldf = pd.DataFrame(long_rows)
sp = split_plot_anova(ldf, "subj", "cult", "year", "ebi")
R["S3_splitplot"] = {k: {"F": round(float(sp[k]["F"]), 2), "df1": sp[k]["df1"], "df2": sp[k]["df2"],
                         "p": float(sp[k]["p"]), "eta2": round(float(sp[k]["eta2"]), 3),
                         "sig": star(sp[k]["p"])}
                     for k in ["between", "within", "interaction"]}

# ---------- S4 spatial (Moran's I, 10 m distance-band weights) ----------
X = col("X_UTM"); Y = col("Y_UTM")


def morans_I(vals, X, Y, band=10.0, perms=199, seed=42):
    m = ~np.isnan(vals) & ~np.isnan(X) & ~np.isnan(Y)
    z = vals[m] - vals[m].mean(); xs = X[m]; ys = Y[m]; n = len(z)
    num = 0.0; S0 = 0.0
    for i in range(n):
        d = np.hypot(xs - xs[i], ys - ys[i]); w = (d > 0) & (d <= band)
        num += np.sum(w * z[i] * z); S0 += np.sum(w)
    den = np.sum(z ** 2)
    I = (n / S0) * (num / den) if S0 > 0 and den > 0 else np.nan
    EI = -1.0 / (n - 1)
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(perms):
        zp = rng.permutation(z); nump = 0.0
        for i in range(n):
            d = np.hypot(xs - xs[i], ys - ys[i]); w = (d > 0) & (d <= band)
            nump += np.sum(w * zp[i] * zp)
        Ip = (n / S0) * (nump / den)
        if Ip >= I:
            cnt += 1
    p = (cnt + 1) / (perms + 1)
    return float(I), float(EI), float(p), n


R["S4_moran"] = {}
for y in YEARS:
    I, EI, p, n = morans_I(ebi[y], X, Y, band=10.0)
    R["S4_moran"][y] = {"I": round(I, 4), "EI": round(EI, 4), "p": round(p, 4), "n": n, "sig": star(p)}
    print(f"Moran {y}: I={I:.4f} p={p:.4f} n={n}")

# ---------- S5 climate drivers (Chill Portions emphasised, Chill_Hrs EXCLUDED) ----------
CLIM = ["Chill_Portions", "Late_Chill", "GDD_Feb", "GDD_Jan_Feb", "Avg_DTR_Feb",
        "Rad_Feb", "Max_Dry_Hrs", "Trans_Ratio", "Frost_Hrs", "Heat_Stress_Feb",
        "Rain_Feb", "High_Wind_Hrs"]   # NOTE: Chill_Hrs intentionally omitted
ebi_year_mean = {y: float(np.nanmean(ebi[y])) for y in YEARS}
ebi_year_sd = {y: float(np.nanstd(ebi[y], ddof=1)) for y in YEARS}
clim_year = {}
for cv in CLIM:
    clim_year[cv] = {}
    for y in YEARS:
        v = col(f"{cv}_{y}"); v = v[~np.isnan(v)]
        clim_year[cv][y] = float(v[0]) if len(v) else np.nan
R["S5_climate_ebi_year"] = {"mean": {}, "sd": {}}
for cv in CLIM:
    xs = np.array([clim_year[cv][y] for y in YEARS])
    for tgt, d in [("mean", ebi_year_mean), ("sd", ebi_year_sd)]:
        ys = np.array([d[y] for y in YEARS]); m = ~np.isnan(xs) & ~np.isnan(ys)
        if m.sum() >= 3 and len(set(xs[m])) > 1:
            r = np.corrcoef(xs[m], ys[m])[0, 1]
            R["S5_climate_ebi_year"][tgt][cv] = {"r": round(float(r), 3),
                                                 "p": round(r_pvalue(r, int(m.sum())), 3),
                                                 "n": int(m.sum())}

# EBI exceedance fractions COMPUTED FROM DATA (not hard-coded)
R["S5_exceedance"] = {}
for y in YEARS:
    a = ebi[y]; a = a[~np.isnan(a)]
    R["S5_exceedance"][y] = {"chill_portions": clim_year["Chill_Portions"][y],
                             "pct_gt_0.60": round(100 * float((a > 0.60).mean()), 2),
                             "pct_gt_0.65": round(100 * float((a > 0.65).mean()), 2),
                             "n": int(len(a))}

# ---------- S6 measured yield (canonical, ground truth) ----------
yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"),
                              encoding="utf-8-sig")))
ymeas = defaultdict(dict); coord = {}
for r in yc:
    ymeas[r["tree_id"]][int(r["year"])] = float(r["net_kernel_yield_per_tree_kg"])
    coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))
mlat = col("Latitude"); mlon = col("Longitude")


def hav(la1, lo1, la2, lo2):
    Rr = 6371000; p = math.pi / 180
    a = math.sin((la2 - la1) * p / 2) ** 2 + math.cos(la1 * p) * math.cos(la2 * p) * math.sin((lo2 - lo1) * p / 2) ** 2
    return 2 * Rr * math.asin(math.sqrt(max(0, a)))


cand = []
for tid, (la, lo) in coord.items():
    ds = np.array([hav(la, lo, a, b) for a, b in zip(mlat, mlon)])
    for j in np.where(ds <= 8.0)[0]:
        cand.append((float(ds[j]), tid, int(j)))
cand.sort(); uy = set(); um = set(); match = {}
for d, tid, j in cand:
    if tid in uy or j in um:
        continue
    match[tid] = (j, d); uy.add(tid); um.add(j)
dists = [d for _, d in match.values()]
R["S6_join"] = {"n_yield_trees": len(coord), "n_matched": len(match),
                "median_dist_m": round(float(np.median(dists)), 3),
                "max_dist_m": round(float(np.max(dists)), 3)}

# same-year EBI x yield 2023, lag, NGRDI
def series(field, year):
    e = []; yv = []; cu = []
    for tid, (j, d) in match.items():
        yy = ymeas[tid].get(2023)
        v = M[j].get(field.replace("YYYY", str(year)))
        if yy is not None and v is not None:
            e.append(float(v)); yv.append(yy); cu.append(M[j].get("cultivar"))
    return np.array(e), np.array(yv), np.array(cu)


R["S6_yield_corr"] = {}
for label, field, yr in [("EBI2023_vs_Y2023", "EBI_Norm_YYYY", 2023),
                         ("EBI2022_lag_Y2023", "EBI_Norm_YYYY", 2022),
                         ("NGRDI2023_vs_Y2023", "NGRDI_Norm_YYYY", 2023)]:
    e, yv, cu = series(field, yr)
    m = ~np.isnan(e) & ~np.isnan(yv)
    r = float(np.corrcoef(e[m], yv[m])[0, 1]); p = r_pvalue(r, int(m.sum()))
    R["S6_yield_corr"][label] = {"r": round(r, 3), "p": round(p, 4), "n": int(m.sum()), "sig": star(p)}

# yield by cultivar
gb = {u: [ymeas[t].get(2023) for t, (j, d) in match.items()
          if M[j].get("cultivar") == u and ymeas[t].get(2023) is not None] for u in uc}
R["S6_yield_by_cultivar"] = {u: {"n": len(gb[u]), "mean": round(float(np.mean(gb[u])), 3),
                                 "sd": round(float(np.std(gb[u], ddof=1)), 3)} for u in uc}
F, p, eta = f_oneway([gb["53"], gb["UEF"]])
R["S6_yield_cultivar_anova"] = {"F": round(F, 2), "p": round(p, 4), "eta2": round(eta, 3), "sig": star(p)}

# ANCOVA yield ~ EBI + cultivar (+ interaction)
E, Yv, CU = series("EBI_Norm_YYYY", 2023)
cats = ["53", "UEF"]; n = len(Yv)
dum = [(CU == "UEF").astype(float)]
inter = [((CU == "UEF").astype(float)) * E]


def design(cols):
    return np.column_stack([np.ones(n)] + cols)


rss_full, k_full, _ = ols_rss(design([E] + dum), Yv)
rss_cult, k_cult, _ = ols_rss(design(dum), Yv)
rss_ebi, k_ebi, _ = ols_rss(design([E]), Yv)
rss_null, k_null, _ = ols_rss(design([]), Yv)
rss_int, k_int, _ = ols_rss(design([E] + dum + inter), Yv)


def Ftest(rss_r, df_r, rss_f, df_f):
    num = (rss_r - rss_f) / (df_f - df_r); den = rss_f / (n - df_f)
    F = num / den if den > 0 else np.nan
    return float(F), float(f_pvalue(F, df_f - df_r, n - df_f))


F_ebi_add, p_ebi_add = Ftest(rss_cult, k_cult, rss_full, k_full)
F_cult_add, p_cult_add = Ftest(rss_ebi, k_ebi, rss_full, k_full)
F_intr, p_intr = Ftest(rss_full, k_full, rss_int, k_int)
r2_full = 1 - rss_full / rss_null; r2_int = 1 - rss_int / rss_null
R["S6_ANCOVA_yield2023"] = {"n": n, "cultivars": cats,
    "EBI_given_cultivar": {"F": round(F_ebi_add, 2), "p": round(p_ebi_add, 4), "sig": star(p_ebi_add)},
    "cultivar_given_EBI": {"F": round(F_cult_add, 2), "p": round(p_cult_add, 4), "sig": star(p_cult_add)},
    "EBIxcultivar_interaction": {"F": round(F_intr, 2), "p": round(p_intr, 4), "sig": star(p_intr)},
    "model_R2": round(float(r2_full), 3), "model_R2_with_interaction": round(float(r2_int), 3)}
R["S6_within_cultivar_slope"] = {}
for k in cats:
    m = CU == k
    if m.sum() >= 5:
        r = float(np.corrcoef(E[m], Yv[m])[0, 1])
        R["S6_within_cultivar_slope"][k] = {"r": round(r, 3), "p": round(r_pvalue(r, int(m.sum())), 3),
                                            "n": int(m.sum())}

json.dump(R, open(os.path.join(SC, "thesis_results_uef53.json"), "w"), indent=2, default=str)
json.dump(R, open(os.path.join(OUT, "thesis_results_uef53.json"), "w"), indent=2, default=str)
print("\n=== KEY RESULTS (UEF + 53) ===")
print(json.dumps(R, indent=1, default=str))
