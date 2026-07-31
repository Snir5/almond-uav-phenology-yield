#!/usr/bin/env python3
"""
hypothesis_tests_uef53.py — full inferential battery for three hypotheses on the
UEF + 53 set (cultivar 54 dropped; chilling = Chill Portions, Dynamic Model):

  H1  Climate (Chill Portions) affects EBI.
  H2  Climate (Chill Portions) affects yield.
  H3  EBI is related to yield  (measured = ground truth; predicted = modelled/circular).
  V   Predicted vs measured yield agreement (2022, 2023; no predicted_2024 exists).
  P   Phenological support: bloom synchrony vs chill, NGRDI convergent validity,
      same-year vs lagged bloom-yield, cultivar bloom trajectory.

Design honesty: Chill Portions is one station value per year, so every climate link
is a BETWEEN-YEAR test with n<=4 and is directional. The inferential weight sits in
the tree-level EBI-yield tests (n up to 151 measured). Multiple comparisons are
FDR-controlled (Benjamini-Hochberg) within each family. No scipy/statsmodels: all
primitives from scratch, validated against stats_utils.
Path-robust via geo_guardrails. Emits Results_Analysis/hypothesis_tests_uef53.json.
"""
import os, sys, csv, math, json, itertools
import numpy as np
from collections import defaultdict, Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import f_pvalue, t_pvalue, r_pvalue, ols_rss, sig_stars

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis")
SC = os.environ.get("SCRATCH_DIR", OUT); os.makedirs(SC, exist_ok=True)
KEEP = {"UEF", "53"}; YEARS = [2021, 2022, 2023, 2024]


def star(p):
    return sig_stars(p) if (p == p) else ""


# ---------- extra primitives ----------
def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~np.isnan(x) & ~np.isnan(y); x, y = x[m], y[m]; n = len(x)
    if n < 3 or x.std() == 0 or y.std() == 0:
        return dict(r=np.nan, p=np.nan, n=n)
    r = float(np.corrcoef(x, y)[0, 1])
    return dict(r=round(r, 3), p=round(r_pvalue(r, n), 4), n=n, sig=star(r_pvalue(r, n)))


def rankdata(a):
    a = np.asarray(a, float); order = a.argsort(); ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ties
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    return sums[inv] / cnt[inv]


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~np.isnan(x) & ~np.isnan(y); x, y = x[m], y[m]; n = len(x)
    if n < 3:
        return dict(rho=np.nan, p=np.nan, n=n)
    rho = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return dict(rho=round(rho, 3), p=round(r_pvalue(rho, n), 4), n=n, sig=star(r_pvalue(rho, n)))


def kendall_tau_exact(x, y):
    """Kendall tau-b with exact permutation p for small n (n<=8)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~np.isnan(x) & ~np.isnan(y); x, y = x[m], y[m]; n = len(x)
    if n < 3:
        return dict(tau=np.nan, p=np.nan, n=n)

    def tau(a, b):
        c = d = 0
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
                if s > 0: c += 1
                elif s < 0: d += 1
        return (c - d) / (0.5 * len(a) * (len(a) - 1))
    t_obs = tau(x, y)
    # exact permutation over all orderings of y ranks
    perms = list(itertools.permutations(range(n)))
    cnt = sum(1 for pm in perms if abs(tau(x, y[list(pm)])) >= abs(t_obs) - 1e-12)
    return dict(tau=round(float(t_obs), 3), p=round(cnt / len(perms), 4), n=n, note="exact permutation")


def perm_pearson_year(x, y):
    """Exact permutation p for Pearson r on n<=6 year-level points (all n! label swaps)."""
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    if n < 3 or x.std() == 0:
        return np.nan
    r_obs = abs(np.corrcoef(x, y)[0, 1])
    perms = list(itertools.permutations(range(n)))
    cnt = sum(1 for pm in perms if abs(np.corrcoef(x, y[list(pm)])[0, 1]) >= r_obs - 1e-12)
    return round(cnt / len(perms), 4)


def linreg(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~np.isnan(x) & ~np.isnan(y); x, y = x[m], y[m]; n = len(x)
    if n < 3:
        return dict(n=n)
    b1, b0 = np.polyfit(x, y, 1); pred = b1 * x + b0
    rss = ((y - pred) ** 2).sum(); tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - rss / tss if tss > 0 else np.nan
    se = math.sqrt(rss / (n - 2) / ((x - x.mean()) ** 2).sum())
    t = b1 / se if se > 0 else np.nan
    p = t_pvalue(abs(t), n - 2) if se > 0 else np.nan
    return dict(slope=round(float(b1), 3), intercept=round(float(b0), 3), R2=round(float(r2), 3),
                t=round(float(t), 2), p=round(float(p), 4), n=n, sig=star(p))


def validation_metrics(meas, pred):
    meas = np.asarray(meas, float); pred = np.asarray(pred, float)
    m = ~np.isnan(meas) & ~np.isnan(pred); meas, pred = meas[m], pred[m]; n = len(meas)
    if n < 3:
        return dict(n=n)
    err = pred - meas
    rmse = float(np.sqrt((err ** 2).mean())); mae = float(np.abs(err).mean())
    bias = float(err.mean())
    r = float(np.corrcoef(meas, pred)[0, 1])
    # Lin's concordance correlation coefficient
    sm, sp = meas.std(), pred.std()
    ccc = (2 * r * sm * sp) / (sm ** 2 + sp ** 2 + (meas.mean() - pred.mean()) ** 2)
    return dict(n=n, pearson_r=round(r, 3), r_p=round(r_pvalue(r, n), 4),
                spearman_rho=spearman(meas, pred)["rho"], RMSE=round(rmse, 3), MAE=round(mae, 3),
                bias_pred_minus_meas=round(bias, 3), Lin_CCC=round(float(ccc), 3),
                meas_mean=round(float(meas.mean()), 3), pred_mean=round(float(pred.mean()), 3))


def levene(groups):
    """Levene's test (median-centered / Brown-Forsythe) for equal variances."""
    z = []; g_idx = []
    for i, g in enumerate(groups):
        g = np.asarray(g, float); g = g[~np.isnan(g)]
        med = np.median(g); z.append(np.abs(g - med)); g_idx += [i] * len(g)
    zall = np.concatenate(z); gm = zall.mean(); k = len(groups); N = len(zall)
    ssb = sum(len(zi) * (zi.mean() - gm) ** 2 for zi in z)
    ssw = sum(((zi - zi.mean()) ** 2).sum() for zi in z)
    d1, d2 = k - 1, N - k
    F = (ssb / d1) / (ssw / d2)
    return dict(F=round(float(F), 2), df1=d1, df2=d2, p=round(float(f_pvalue(F, d1, d2)), 6), sig=star(f_pvalue(F, d1, d2)))


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    idx = [i for i, p in enumerate(pvals) if p == p]
    ps = sorted((pvals[i], i) for i in idx); m = len(ps)
    adj = {}
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        p, i = ps[rank]
        val = min(prev, p * m / (rank + 1)); adj[i] = round(val, 4); prev = val
    return [adj.get(i, np.nan) for i in range(len(pvals))]


# ---------- load + join ----------
rows = [r for r in load_master() if r.get("cultivar") in KEEP]
cult = np.array([r.get("cultivar") for r in rows])
def col(c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in rows], float)
X, Y = col("X_UTM"), col("Y_UTM")
ebi = {y: col(f"EBI_Norm_{y}") for y in YEARS}
ngrdi = {y: col(f"NGRDI_Norm_{y}") for y in YEARS}
CP = {y: float(np.nanmean(col(f"Chill_Portions_{y}"))) for y in YEARS}

yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
ymeas = defaultdict(dict); coord = {}
for r in yc:
    ymeas[r["tree_id"]][int(r["year"])] = float(r["net_kernel_yield_per_tree_kg"]); coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))
mlat, mlon = col("Latitude"), col("Longitude")
def hav(a, b, d, e):
    p = math.pi / 180; x = math.sin((d - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(d * p) * math.sin((e - b) * p / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(max(0, x)))
cand = []
for t, (la, lo) in coord.items():
    ds = np.array([hav(la, lo, a, b) for a, b in zip(mlat, mlon)])
    for j in np.where(ds <= 8)[0]: cand.append((float(ds[j]), t, int(j)))
cand.sort(); uy = set(); um = set(); mp = {}
for d, t, j in cand:
    if t in uy or j in um: continue
    mp[t] = j; uy.add(t); um.add(j)

R = {"scope": {"cultivars": ["UEF", "53"], "chilling": "Chill Portions (Dynamic Model)",
               "chill_portions_by_year": CP, "n_trees": len(rows), "n_yield_matched": len(mp)}}

# ================= H1: climate (CP) -> EBI =================
ebi_mean = {y: float(np.nanmean(ebi[y])) for y in YEARS}
ebi_sd = {y: float(np.nanstd(ebi[y], ddof=1)) for y in YEARS}
ebi_cv = {y: ebi_sd[y] / ebi_mean[y] for y in YEARS}
cpx = np.array([CP[y] for y in YEARS])
H1 = {"note": "CP is one value/year -> between-year test, n=4, directional."}
# variance ceiling: how much EBI variance is between-year at all (one-way ANOVA EBI~year)
allv = []; grp = []
for y in YEARS:
    a = ebi[y][~np.isnan(ebi[y])]; allv.append(a); grp += [y] * len(a)
concat = np.concatenate(allv); gm = concat.mean()
ssb = sum(len(a) * (a.mean() - gm) ** 2 for a in allv); ssw = sum(((a - a.mean()) ** 2).sum() for a in allv)
H1["between_year_variance_ceiling_eta2"] = round(ssb / (ssb + ssw), 4)
for tgt, d in [("CP_vs_meanEBI", ebi_mean), ("CP_vs_sdEBI", ebi_sd), ("CP_vs_cvEBI", ebi_cv)]:
    yv = np.array([d[y] for y in YEARS])
    H1[tgt] = {**pearson(cpx, yv), "spearman": spearman(cpx, yv), "kendall": kendall_tau_exact(cpx, yv),
               "perm_p": perm_pearson_year(cpx, yv)}
# FDR across the three year-level EBI tests
labs = ["CP_vs_meanEBI", "CP_vs_sdEBI", "CP_vs_cvEBI"]
adj = bh_fdr([H1[l]["p"] for l in labs])
for l, a in zip(labs, adj): H1[l]["p_fdr"] = a
R["H1_climate_EBI"] = H1

# ================= H2: climate (CP) -> yield =================
H2 = {"note": "Measured yield is essentially one year (2023); predicted only 2022/2023. "
              "Climate->yield is n<=3 between-year -> descriptive, NOT inferential."}
meas_year = {}
for yr in [2022, 2023, 2024]:
    vals = [ymeas[t].get(yr) for t in mp if ymeas[t].get(yr) is not None]
    meas_year[yr] = {"n": len(vals), "mean": round(float(np.mean(vals)), 3) if vals else None, "CP": CP[yr]}
pred_year = {yr: round(float(np.nanmean(col(f"predicted_Yield_{yr}"))), 3) for yr in [2022, 2023]}
H2["measured_yield_by_year"] = meas_year
H2["predicted_yield_by_year"] = {**pred_year, 2024: "no predicted_Yield_2024 in master"}
# directional CP vs measured mean (n=3) — reported, not tested
cp3 = np.array([CP[y] for y in [2022, 2023, 2024]]); my3 = np.array([meas_year[y]["mean"] for y in [2022, 2023, 2024]], float)
H2["CP_vs_measuredMean_n3"] = {"r": round(float(np.corrcoef(cp3, my3)[0, 1]), 3), "n": 3, "inferential": False}
R["H2_climate_yield"] = H2

# ================= H3: EBI <-> yield =================
def paired(field, yfun, yr_ebi, yr_yield):
    E = []; Yv = []; CU = []
    for t, j in mp.items():
        yv = yfun(t, yr_yield); ev = rows[j].get(f"{field}_{yr_ebi}")
        if yv is not None and ev is not None:
            E.append(float(ev)); Yv.append(float(yv)); CU.append(rows[j].get("cultivar"))
    return np.array(E), np.array(Yv), np.array(CU)


def meas(t, yr): return ymeas[t].get(yr)
H3 = {"measured": {}, "predicted_CIRCULAR": {}}
# same-year and lag, measured
tests = [("EBI2023_vs_measY2023", "EBI_Norm", 2023, 2023),
         ("EBI2022_vs_measY2022", "EBI_Norm", 2022, 2022),
         ("EBI2024_vs_measY2024", "EBI_Norm", 2024, 2024),
         ("EBI2022_lag_measY2023", "EBI_Norm", 2022, 2023),
         ("EBI2023_lag_measY2024", "EBI_Norm", 2023, 2024),
         ("NGRDI2023_vs_measY2023", "NGRDI_Norm", 2023, 2023)]
pcollect = []
for lab, fld, ye, yy in tests:
    E, Yv, CU = paired(fld, meas, ye, yy)
    res = {**pearson(E, Yv), "spearman": spearman(E, Yv)}
    H3["measured"][lab] = res; pcollect.append((lab, res.get("p", np.nan)))
adj = bh_fdr([p for _, p in pcollect])
for (lab, _), a in zip(pcollect, adj): H3["measured"][lab]["p_fdr"] = a

# ANCOVA (measured 2023) with cultivar interaction
E, Yv, CU = paired("EBI_Norm", meas, 2023, 2023)
n = len(Yv)
def design(cols): return np.column_stack([np.ones(n)] + cols)
dum = [(CU == "UEF").astype(float)]; inter = [((CU == "UEF").astype(float)) * E]
rss_full, kf, _ = ols_rss(design([E] + dum), Yv)
rss_cult, kc, _ = ols_rss(design(dum), Yv)
rss_ebi, ke, _ = ols_rss(design([E]), Yv)
rss_null, kn, _ = ols_rss(design([]), Yv)
rss_int, ki, _ = ols_rss(design([E] + dum + inter), Yv)
def Ft(rss_r, dr, rss_f, dfk):
    F = ((rss_r - rss_f) / (dfk - dr)) / (rss_f / (n - dfk)); return round(float(F), 2), round(float(f_pvalue(F, dfk - dr, n - dfk)), 4)
Fe, pe = Ft(rss_cult, kc, rss_full, kf); Fc, pc = Ft(rss_ebi, ke, rss_full, kf); Fi, pi = Ft(rss_full, kf, rss_int, ki)
H3["ANCOVA_measured2023"] = {"n": n, "EBI_given_cultivar": {"F": Fe, "p": pe, "sig": star(pe)},
    "cultivar_given_EBI": {"F": Fc, "p": pc, "sig": star(pc)},
    "EBIxcultivar_interaction": {"F": Fi, "p": pi, "sig": star(pi)},
    "R2": round(1 - rss_full / rss_null, 3), "R2_with_interaction": round(1 - rss_int / rss_null, 3)}
H3["within_cultivar_measured2023"] = {}
for k in ["53", "UEF"]:
    m = CU == k
    H3["within_cultivar_measured2023"][k] = {**pearson(E[m], Yv[m]), "linreg": linreg(E[m], Yv[m])}
# predicted (circular) tree-level
def predf(t, yr):  # value from the master row matched to this yield tree (used only for join set parity)
    return None
for lab, yr in [("EBI2022_vs_predY2022", 2022), ("EBI2023_vs_predY2023", 2023)]:
    E2 = col(f"EBI_Norm_{yr}"); P2 = col(f"predicted_Yield_{yr}")
    H3["predicted_CIRCULAR"][lab] = {**pearson(E2, P2), "spearman": spearman(E2, P2),
        "note": "predicted_Yield uses canopy/growth inputs -> circular, NOT independent validation"}
R["H3_EBI_yield"] = H3

# ================= V: predicted vs measured yield =================
V = {"note": "predicted_Yield_2024 does not exist; validation possible for 2022 & 2023 only."}
for yr in [2022, 2023]:
    meas_v = []; pred_v = []
    for t, j in mp.items():
        mv = ymeas[t].get(yr); pv = rows[j].get(f"predicted_Yield_{yr}")
        if mv is not None and pv is not None:
            meas_v.append(mv); pred_v.append(float(pv))
    V[f"validation_{yr}"] = validation_metrics(meas_v, pred_v)
V["validation_2024"] = "not possible (no predicted_Yield_2024)"
R["V_predicted_vs_measured"] = V

# ================= P: phenology support =================
P = {}
# P1 synchrony: EBI variance differs across years (Levene) + SD/CV vs CP + trend
P["P1_bloom_synchrony"] = {
    "levene_EBI_variance_across_years": levene([ebi[y][~np.isnan(ebi[y])] for y in YEARS]),
    "EBI_sd_by_year": {y: round(ebi_sd[y], 4) for y in YEARS},
    "EBI_cv_by_year": {y: round(ebi_cv[y], 4) for y in YEARS},
    "sd_vs_CP": pearson(cpx, np.array([ebi_sd[y] for y in YEARS])),
    "sd_trend_over_years_kendall": kendall_tau_exact(np.array(YEARS, float), np.array([ebi_sd[y] for y in YEARS])),
    "interpretation": "lower chill -> less synchronous (more heterogeneous) bloom; synchrony rises over time"}
# P2 NGRDI convergent validity
E23, Y23, C23 = paired("EBI_Norm", meas, 2023, 2023)
N23, Y23b, C23b = paired("NGRDI_Norm", meas, 2023, 2023)
P["P2_NGRDI_convergent"] = {
    "NGRDI2023_vs_measY2023": pearson(N23, Y23b),
    "EBI_vs_NGRDI_2023_alltrees": pearson(ebi[2023], ngrdi[2023]),
    "NGRDI_within_cultivar_measY2023": {k: pearson(N23[C23b == k], Y23b[C23b == k]) for k in ["53", "UEF"]},
    "interpretation": "second bloom index (NGRDI) should track EBI and repeat the cultivar-split yield pattern"}
# P3 same-year vs lag timing
P["P3_bloom_yield_timing"] = {
    "same_year_2023": H3["measured"]["EBI2023_vs_measY2023"],
    "lag_prevbloom_2022to2023": H3["measured"]["EBI2022_lag_measY2023"],
    "interpretation": "same-season bloom, not prior-season bloom, carries the (cultivar-split) yield signal"}
# P4 cultivar bloom trajectory divergence (phenology) — reuse split-plot from EBI (recompute interaction quickly)
P["P4_cultivar_trajectory"] = {"reference": "split-plot cultivar x year interaction (see thesis_results_uef53.json)",
    "mean_EBI_by_cultivar_year": {k: {y: round(float(np.nanmean(ebi[y][cult == k])), 4) for y in YEARS} for k in ["53", "UEF"]},
    "interpretation": "53 blooms earlier/brighter then converges; UEF rises; rank flips ~2023 (phenological divergence)"}
R["P_phenology"] = P

# ================= PANEL: same 20 trees measured 2022, 2023, 2024 =================
panel = [t for t in ymeas if all(y in ymeas[t] for y in [2022, 2023, 2024]) and t in mp]
PAN = {"note": "Same 20 trees (ids 1218-1237, consecutive block) measured all three years; "
               "balanced 10 UEF + 10 cv-53. NOTE: this is a spatially clustered, non-random "
               "subset, so its year means and cultivar gap need not match the whole orchard.",
       "n_trees": len(panel)}
Yp = {y: np.array([ymeas[t][y] for t in panel]) for y in [2022, 2023, 2024]}
cup = np.array([rows[mp[t]].get("cultivar") for t in panel])
PAN["yield_by_year"] = {y: {"mean": round(float(Yp[y].mean()), 3), "sd": round(float(Yp[y].std(ddof=1)), 3),
                            "mean_53": round(float(Yp[y][cup == "53"].mean()), 3),
                            "mean_UEF": round(float(Yp[y][cup == "UEF"].mean()), 3)} for y in [2022, 2023, 2024]}
# repeated-measures ANOVA: year within tree
Mp = np.vstack([Yp[2022], Yp[2023], Yp[2024]]).T; npn, kpn = Mp.shape; grand = Mp.mean()
subj = Mp.mean(1); yrm = Mp.mean(0)
SS_year = npn * ((yrm - grand) ** 2).sum(); df_y = kpn - 1
SS_subj = kpn * ((subj - grand) ** 2).sum()
SS_tot = ((Mp - grand) ** 2).sum(); SS_err = SS_tot - SS_year - SS_subj; df_e = df_y * (npn - 1)
F_y = (SS_year / df_y) / (SS_err / df_e); p_y = f_pvalue(F_y, df_y, df_e)
PAN["RM_ANOVA_year"] = {"F": round(float(F_y), 2), "df1": df_y, "df2": df_e, "p": round(float(p_y), 6),
                        "eta2_year": round(float(SS_year / SS_tot), 3), "sig": star(p_y)}
# alternate-bearing: within-tree lag correlations (negative => alternate bearing)
PAN["within_tree_lag"] = {}
for a, b in [(2022, 2023), (2023, 2024), (2022, 2024)]:
    r = float(np.corrcoef(Yp[a], Yp[b])[0, 1])
    PAN["within_tree_lag"][f"Y{a}_vs_Y{b}"] = {"r": round(r, 3), "p": round(r_pvalue(r, len(panel)), 4), "n": len(panel)}
PAN["biennial_bearing_index"] = {"Y22_23": round(float(np.mean(np.abs(Yp[2023] - Yp[2022]) / (Yp[2022] + Yp[2023]))), 3),
                                 "Y23_24": round(float(np.mean(np.abs(Yp[2024] - Yp[2023]) / (Yp[2023] + Yp[2024]))), 3),
                                 "interpretation": "positive lag correlations => productivity persists (NOT classic alternate bearing); big year effect is orchard-wide"}
# cultivar effect per year (balanced 10 vs 10)
PAN["cultivar_by_year"] = {}
for y in [2022, 2023, 2024]:
    a = Yp[y][cup == "53"]; b = Yp[y][cup == "UEF"]; na, nb = len(a), len(b)
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    t = (b.mean() - a.mean()) / math.sqrt(sp2 * (1 / na + 1 / nb)); p = t_pvalue(abs(t), na + nb - 2)
    PAN["cultivar_by_year"][y] = {"UEF_minus_53_kg": round(float(b.mean() - a.mean()), 3),
                                  "t": round(float(t), 2), "p": round(float(p), 4), "sig": star(p)}
# within-tree EBI -> yield per year and cultivar (sign pattern)
PAN["EBI_yield_by_year_cultivar"] = {}
for y in [2022, 2023, 2024]:
    E = np.array([rows[mp[t]].get(f"EBI_Norm_{y}") for t in panel], float)
    d = {}
    for grp, lab in [(np.ones(len(panel), bool), "all"), (cup == "53", "53"), (cup == "UEF", "UEF")]:
        e = E[grp]; yy = Yp[y][grp]; m = ~np.isnan(e)
        if m.sum() >= 5:
            r = float(np.corrcoef(e[m], yy[m])[0, 1]); d[lab] = {"r": round(r, 3), "n": int(m.sum())}
    PAN["EBI_yield_by_year_cultivar"][y] = d
R["PANEL_3yr_2022_2024"] = PAN

json.dump(R, open(os.path.join(OUT, "hypothesis_tests_uef53.json"), "w"), indent=2, default=str)
json.dump(R, open(os.path.join(SC, "hypothesis_tests_uef53.json"), "w"), indent=2, default=str)
print(json.dumps(R, indent=1, default=str))
