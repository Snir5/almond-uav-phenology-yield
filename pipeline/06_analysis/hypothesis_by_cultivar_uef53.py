#!/usr/bin/env python3
"""
hypothesis_by_cultivar_uef53.py — the full battery recomputed ENTIRELY WITHIN each
cultivar (UEF and 53 separately). No pooling across cultivars, no orchard-wide means
as a result, and NO cross-year lag (same-season bloom-to-yield only).

For each cultivar C in {UEF, 53}:
  H1  climate (Chill Portions) -> EBI            (year level, n=4, directional)
  H2  climate -> yield: measured yr means (2022/2023/2024) + predicted (2022/2023)
  H3  EBI -> yield, SAME YEAR ONLY, measured (2022/2023/2024) and predicted (2022/2023)
  V   predicted vs measured yield agreement (2022, 2023; no predicted_2024)
  P   phenology: EBI SD/CV vs CP, Levene across years, EBI vs NGRDI, NGRDI -> yield

Path-robust via geo_guardrails; from-scratch stats; emits
Results_Analysis/hypothesis_by_cultivar_uef53.json.
"""
import os, sys, csv, math, json
import numpy as np
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
from stats_utils import f_pvalue, t_pvalue, r_pvalue, sig_stars
BASE = find_thesis_root(); OUT = os.path.join(BASE, "Results_Analysis")
SC = os.environ.get("SCRATCH_DIR", OUT); os.makedirs(SC, exist_ok=True)
YEARS = [2021, 2022, 2023, 2024]; MEAS_YEARS = [2022, 2023, 2024]; PRED_YEARS = [2022, 2023]
CULTS = ["UEF", "53"]


def star(p): return sig_stars(p) if (p == p) else ""


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~np.isnan(x) & ~np.isnan(y); x, y = x[m], y[m]; n = len(x)
    if n < 3 or x.std() == 0 or y.std() == 0: return dict(r=np.nan, p=np.nan, n=n)
    r = float(np.corrcoef(x, y)[0, 1]); p = r_pvalue(r, n)
    return dict(r=round(r, 3), p=round(p, 4), n=n, sig=star(p))


def rankdata(a):
    a = np.asarray(a, float); order = a.argsort(); ranks = np.empty(len(a)); ranks[order] = np.arange(1, len(a) + 1)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt)); np.add.at(s, inv, ranks); return s[inv] / cnt[inv]


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~np.isnan(x) & ~np.isnan(y); x, y = x[m], y[m]; n = len(x)
    if n < 3: return dict(rho=np.nan, p=np.nan, n=n)
    rho = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return dict(rho=round(rho, 3), p=round(r_pvalue(rho, n), 4), n=n)


def linreg(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~np.isnan(x) & ~np.isnan(y); x, y = x[m], y[m]; n = len(x)
    if n < 3: return dict(n=n)
    b1, b0 = np.polyfit(x, y, 1); pred = b1 * x + b0
    rss = ((y - pred) ** 2).sum(); tss = ((y - y.mean()) ** 2).sum()
    se = math.sqrt(rss / (n - 2) / ((x - x.mean()) ** 2).sum()) if n > 2 and ((x - x.mean()) ** 2).sum() > 0 else np.nan
    t = b1 / se if se and se > 0 else np.nan; p = t_pvalue(abs(t), n - 2) if se and se > 0 else np.nan
    return dict(slope=round(float(b1), 3), R2=round(float(1 - rss / tss), 3) if tss > 0 else np.nan,
                t=round(float(t), 2) if t == t else np.nan, p=round(float(p), 4) if p == p else np.nan, n=n, sig=star(p))


def validation(meas, pred):
    meas = np.asarray(meas, float); pred = np.asarray(pred, float)
    m = ~np.isnan(meas) & ~np.isnan(pred); meas, pred = meas[m], pred[m]; n = len(meas)
    if n < 3: return dict(n=n)
    err = pred - meas; r = float(np.corrcoef(meas, pred)[0, 1]) if meas.std() and pred.std() else np.nan
    sm, sp = meas.std(), pred.std()
    ccc = (2 * r * sm * sp) / (sm ** 2 + sp ** 2 + (meas.mean() - pred.mean()) ** 2) if r == r else np.nan
    return dict(n=n, pearson_r=round(r, 3) if r == r else np.nan, r_p=round(r_pvalue(r, n), 4) if r == r else np.nan,
                RMSE=round(float(np.sqrt((err ** 2).mean())), 3), MAE=round(float(np.abs(err).mean()), 3),
                bias_pred_minus_meas=round(float(err.mean()), 3),
                Lin_CCC=round(float(ccc), 3) if ccc == ccc else np.nan,
                meas_mean=round(float(meas.mean()), 3), pred_mean=round(float(pred.mean()), 3))


def levene(groups):
    z = []
    for g in groups:
        g = np.asarray(g, float); g = g[~np.isnan(g)]; z.append(np.abs(g - np.median(g)))
    zall = np.concatenate(z); gm = zall.mean(); k = len(groups); N = len(zall)
    ssb = sum(len(zi) * (zi.mean() - gm) ** 2 for zi in z); ssw = sum(((zi - zi.mean()) ** 2).sum() for zi in z)
    d1, d2 = k - 1, N - k; F = (ssb / d1) / (ssw / d2)
    return dict(F=round(float(F), 2), df1=d1, df2=d2, p=round(float(f_pvalue(F, d1, d2)), 6), sig=star(f_pvalue(F, d1, d2)))


# ---------- load + join ----------
allrows = [r for r in load_master() if r.get("cultivar") in CULTS]
def col(rs, c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in rs], float)
CP = {y: float(np.nanmean(col(allrows, f"Chill_Portions_{y}"))) for y in YEARS}
yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
ymeas = defaultdict(dict); coord = {}
for r in yc: ymeas[r["tree_id"]][int(r["year"])] = float(r["net_kernel_yield_per_tree_kg"]); coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))
mlat, mlon = col(allrows, "Latitude"), col(allrows, "Longitude")
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
    mp[t] = j; uy.add(t); um.add(j)   # tree_id -> row index in allrows

RES = {"scope": {"design": "all analyses computed within each cultivar; no pooling; same-year bloom-yield only",
                 "chilling": "Chill Portions (Dynamic Model)", "chill_portions_by_year": CP}}

for C in CULTS:
    idx = [i for i, r in enumerate(allrows) if r.get("cultivar") == C]
    rowsC = [allrows[i] for i in idx]
    def cC(c): return col(rowsC, c)
    ebi = {y: cC(f"EBI_Norm_{y}") for y in YEARS}
    ngrdi = {y: cC(f"NGRDI_Norm_{y}") for y in YEARS}
    D = {"n_trees": len(rowsC)}

    # H1 climate -> EBI (year level, n=4)
    ebi_mean = {y: float(np.nanmean(ebi[y])) for y in YEARS}
    ebi_sd = {y: float(np.nanstd(ebi[y], ddof=1)) for y in YEARS}
    cpx = np.array([CP[y] for y in YEARS])
    D["H1_climate_EBI"] = {"note": "n=4 year-level, directional",
        "EBI_mean_by_year": {y: round(ebi_mean[y], 4) for y in YEARS},
        "EBI_sd_by_year": {y: round(ebi_sd[y], 4) for y in YEARS},
        "CP_vs_meanEBI": pearson(cpx, [ebi_mean[y] for y in YEARS]),
        "CP_vs_sdEBI": pearson(cpx, [ebi_sd[y] for y in YEARS])}

    # measured & predicted yield trees for THIS cultivar
    meas_trees = {t: mp[t] for t in mp if allrows[mp[t]].get("cultivar") == C}

    # H2 climate -> yield (measured year means + predicted)
    meas_by_year = {}
    for y in MEAS_YEARS:
        vals = [ymeas[t].get(y) for t in meas_trees if ymeas[t].get(y) is not None]
        meas_by_year[y] = {"n": len(vals), "mean": round(float(np.mean(vals)), 3) if vals else None,
                           "sd": round(float(np.std(vals, ddof=1)), 3) if len(vals) > 1 else None, "CP": CP[y]}
    pred_by_year = {y: {"n": int((~np.isnan(cC(f"predicted_Yield_{y}"))).sum()),
                        "mean": round(float(np.nanmean(cC(f"predicted_Yield_{y}"))), 3)} for y in PRED_YEARS}
    D["H2_climate_yield"] = {"measured_by_year": meas_by_year, "predicted_by_year": pred_by_year,
                             "note": "n<=3 measured years -> descriptive, not inferential"}

    # H3 EBI -> yield SAME YEAR (measured + predicted)
    h3m = {}
    for y in MEAS_YEARS:
        E = []; Yv = []
        for t, j in meas_trees.items():
            v = ymeas[t].get(y); e = allrows[j].get(f"EBI_Norm_{y}")
            if v is not None and e is not None: E.append(float(e)); Yv.append(float(v))
        h3m[f"EBI{y}_vs_measY{y}"] = {**pearson(E, Yv), "spearman": spearman(E, Yv), "linreg": linreg(E, Yv)}
    h3p = {}
    for y in PRED_YEARS:
        E = cC(f"EBI_Norm_{y}"); Pv = cC(f"predicted_Yield_{y}")
        h3p[f"EBI{y}_vs_predY{y}"] = {**pearson(E, Pv), "note": "predicted is modelled/circular"}
    D["H3_EBI_yield_sameyear"] = {"measured": h3m, "predicted_CIRCULAR": h3p}

    # V predicted vs measured (per year)
    v = {}
    for y in PRED_YEARS:
        mv = []; pv = []
        for t, j in meas_trees.items():
            m_ = ymeas[t].get(y); p_ = allrows[j].get(f"predicted_Yield_{y}")
            if m_ is not None and p_ is not None: mv.append(m_); pv.append(float(p_))
        v[f"validation_{y}"] = validation(mv, pv)
    v["validation_2024"] = "no predicted_Yield_2024"
    D["V_predicted_vs_measured"] = v

    # P phenology (within cultivar)
    ng_yield = {}
    for y in MEAS_YEARS:
        N = []; Yv = []
        for t, j in meas_trees.items():
            v_ = ymeas[t].get(y); ngv = allrows[j].get(f"NGRDI_Norm_{y}")
            if v_ is not None and ngv is not None: N.append(float(ngv)); Yv.append(float(v_))
        ng_yield[f"NGRDI{y}_vs_measY{y}"] = pearson(N, Yv)
    D["P_phenology"] = {
        "levene_EBI_var_across_years": levene([ebi[y][~np.isnan(ebi[y])] for y in YEARS]),
        "EBI_cv_by_year": {y: round(ebi_sd[y] / ebi_mean[y], 4) for y in YEARS},
        "EBI_vs_NGRDI_2023": pearson(ebi[2023], ngrdi[2023]),
        "NGRDI_vs_measured_yield_sameyear": ng_yield}

    RES[C] = D

json.dump(RES, open(os.path.join(OUT, "hypothesis_by_cultivar_uef53.json"), "w"), indent=2, default=str)
json.dump(RES, open(os.path.join(SC, "hypothesis_by_cultivar_uef53.json"), "w"), indent=2, default=str)
print(json.dumps(RES, indent=1, default=str))
