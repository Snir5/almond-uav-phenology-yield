#!/usr/bin/env python3
"""
yield_model_prototype.py — prototype a per-tree yield prediction model on the
measured yield (UEF + 53) to quantify which features actually add predictive value.
Fits nested OLS models with 5-fold cross-validated R^2 (honest generalization).
Two settings: within-season (2023, n=151) and pooled multi-season (2022-2024).
-> Results_Analysis/yield_model_prototype.json + figure.
"""
import os, sys, csv, math, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master
BASE = find_thesis_root(); OUT = os.path.join(BASE, "Results_Analysis")
FIG = os.path.join(OUT, "08_UEF53_Rerun"); SC = os.environ.get("SCRATCH_DIR", FIG)
rng = np.random.default_rng(42)

M = load_master()
def col(c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in M], float)
CP = {y: float(np.nanmean(col(f"Chill_Portions_{y}"))) for y in [2021, 2022, 2023, 2024]}
mlat, mlon = col("Latitude"), col("Longitude")
yc = list(csv.DictReader(open(os.path.join(BASE, "Trees_Data_Survey", "kedma_plot_a_yield.csv"), encoding="utf-8-sig")))
ymeas = defaultdict(dict); coord = {}
for r in yc: ymeas[r["tree_id"]][int(r["year"])] = float(r["net_kernel_yield_per_tree_kg"]); coord[r["tree_id"]] = (float(r["latitude"]), float(r["longitude"]))
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


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


def r2_cv(Xd, y, k=5):
    """k-fold CV R^2 (out-of-fold predictions), plus in-sample R^2 and adjusted R^2."""
    n = len(y); idx = rng.permutation(n); folds = np.array_split(idx, k)
    pred = np.full(n, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None)
        pred[f] = Xd[f] @ beta
    ss = ((y - pred) ** 2).sum(); tss = ((y - y.mean()) ** 2).sum()
    cv = 1 - ss / tss
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None); ins = Xd @ beta
    r2 = 1 - ((y - ins) ** 2).sum() / tss
    p = Xd.shape[1] - 1; adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return round(float(r2), 3), round(float(adj), 3), round(float(cv), 3)


RES = {}

# ---------------- within-season model (2023) ----------------
rows2023 = [(ymeas[t][2023], M[j]) for t, j in mp.items() if ymeas[t].get(2023) is not None and M[j].get("EBI_Norm_2023") is not None]
y = np.array([a for a, _ in rows2023]); recs = [b for _, b in rows2023]; n = len(y)
uef = np.array([1.0 if r.get("cultivar") == "UEF" else 0.0 for r in recs])
ebi = zc([r.get("EBI_Norm_2023") for r in recs])
ngrdi = zc([r.get("NGRDI_Norm_2023") for r in recs])
xu = zc([r.get("X_UTM") for r in recs]); yu = zc([r.get("Y_UTM") for r in recs])
one = np.ones(n)
models = {
    "M1 cultivar": np.column_stack([one, uef]),
    "M2 +EBI": np.column_stack([one, uef, ebi]),
    "M3 +EBI×cultivar": np.column_stack([one, uef, ebi, ebi * uef]),
    "M4 +spatial (X,Y)": np.column_stack([one, uef, ebi, ebi * uef, xu, yu]),
    "M5 +NGRDI": np.column_stack([one, uef, ebi, ebi * uef, xu, yu, ngrdi]),
}
RES["within_2023"] = {"n": n, "models": {k: dict(zip(["R2", "adjR2", "cvR2"], r2_cv(X, y))) for k, X in models.items()}}

# ---------------- pooled multi-season model (2022-2024) ----------------
recs = []
for t, j in mp.items():
    for yr in [2022, 2023, 2024]:
        v = ymeas[t].get(yr); e = M[j].get(f"EBI_Norm_{yr}")
        if v is not None and e is not None:
            recs.append((v, M[j].get("cultivar"), float(e), yr))
yv = np.array([a for a, _, _, _ in recs]); n2 = len(yv)
uef2 = np.array([1.0 if c == "UEF" else 0.0 for _, c, _, _ in recs])
ebi2 = zc([e for _, _, e, _ in recs])
yr_arr = np.array([yr for _, _, _, yr in recs])
d22 = (yr_arr == 2022).astype(float); d24 = (yr_arr == 2024).astype(float)
cp_arr = zc([CP[yr] for _, _, _, yr in recs])
one2 = np.ones(n2)
# NOTE: "year" is NOT a usable predictive feature (it just labels a season and stands
# in for that season's climate). The transferable season feature is climate itself
# (Chill Portions). The year-factor model is reported only as a CEILING: the share of
# yield that is seasonal at all, which a climate feature + more seasons could aim for.
models2 = {
    "cultivar only": np.column_stack([one2, uef2]),
    "+EBI×cultivar": np.column_stack([one2, uef2, ebi2, ebi2 * uef2]),
    "+climate (Chill Portions)": np.column_stack([one2, uef2, ebi2, ebi2 * uef2, cp_arr]),
}
ceil_r2, ceil_adj, ceil_cv = r2_cv(np.column_stack([one2, uef2, ebi2, ebi2 * uef2, d22, d24]), yv)
RES["pooled_2022_2024"] = {"n": n2,
    "models": {k: dict(zip(["R2", "adjR2", "cvR2"], r2_cv(X, yv))) for k, X in models2.items()},
    "seasonal_ceiling_year_factor": {"R2": ceil_r2, "cvR2": ceil_cv,
        "note": "year factor is a benchmark of how much yield is seasonal, NOT a usable feature"}}

json.dump(RES, open(os.path.join(OUT, "yield_model_prototype.json"), "w"), indent=2)
json.dump(RES, open(os.path.join(SC, "yield_model_prototype.json"), "w"), indent=2)
print(json.dumps(RES, indent=1))

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
a = ax[0]
mk = list(RES["within_2023"]["models"]); cvs = [RES["within_2023"]["models"][k]["cvR2"] for k in mk]
ins = [RES["within_2023"]["models"][k]["R2"] for k in mk]
xb = np.arange(len(mk)); w = 0.38
a.bar(xb - w / 2, ins, w, color="#bdc3c7", label="in-sample R²")
a.bar(xb + w / 2, cvs, w, color="#E67E22", label="5-fold CV R²")
for x, c in zip(xb, cvs): a.text(x + w / 2, c + .003, f"{c:.2f}", ha="center", fontsize=8.5)
a.set_xticks(xb); a.set_xticklabels(mk, rotation=18, ha="right", fontsize=8.5)
a.set_ylabel("R²  (measured 2023 yield)"); a.axhline(0, color="k", lw=.8)
a.set_title(f"A. Within-season per-tree model (2023, n={RES['within_2023']['n']})\ncultivar + EBI×cultivar carry it; spatial adds a little; NGRDI nothing", fontweight="bold", fontsize=9.5)
a.legend(fontsize=9)
a = ax[1]
mk2 = list(RES["pooled_2022_2024"]["models"]); cv2 = [RES["pooled_2022_2024"]["models"][k]["cvR2"] for k in mk2]
a.bar(range(len(mk2)), cv2, color=["#8E44AD", "#C0392B", "#16A085"])
for x, c in zip(range(len(mk2)), cv2): a.text(x, c + .01, f"{c:.2f}", ha="center", fontsize=9)
ceil = RES["pooled_2022_2024"]["seasonal_ceiling_year_factor"]["cvR2"]
a.axhline(ceil, color="#7f8c8d", ls="--", lw=1.6)
a.text(len(mk2) - 1, ceil + .012, f"seasonal ceiling {ceil:.2f} (year factor, not a usable feature)", ha="right", fontsize=8, color="#555")
a.set_xticks(range(len(mk2))); a.set_xticklabels(mk2, rotation=18, ha="right", fontsize=8.5)
a.set_ylabel("5-fold CV R²  (pooled yield)"); a.set_ylim(-0.05, ceil + 0.08)
a.set_title(f"B. Across seasons (2022-2024, n={RES['pooled_2022_2024']['n']})\nseason enters via climate (Chill Portions), not the year label", fontweight="bold", fontsize=9.5)
fig.suptitle("Per-tree yield model: which features add predictive value", fontweight="bold", fontsize=13)
fig.tight_layout()
for d in (FIG, SC): fig.savefig(os.path.join(d, "Yield_Model_Features.png"), dpi=140, bbox_inches="tight")
print("saved figure")
