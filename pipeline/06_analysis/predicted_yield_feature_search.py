#!/usr/bin/env python3
"""
predicted_yield_feature_search.py: per follow-up request, a full feature
search for the best model of MODELLED predicted_Yield (NOT measured yield,
per project convention this is always labelled "modelled/predicted" and
never conflated with the ground-truth measured-yield model used everywhere
else in this report).

Because predicted_Yield does not require the small measured-yield CSV, this
runs on a much larger sample: every tree with predicted_Yield_2022 or
predicted_Yield_2023 in the master (n roughly 900/year, ~1800 pooled,
UEF + cultivar 53, versus 167 for the measured-yield model).

Candidate pool: cultivar, EBI (+interaction), climate (Chill Portions),
Growth_April, plus the same 34-column remote-sensing engineered bank used
throughout this project (canopy structure, EBI-pixel distribution,
composite indices). Deliberately EXCLUDES SWP_April/MayJune/June,
Growth_MayJune/June, CNC_June, and the cluster fields: those are known,
near-tautological inputs to predicted_Yield's own construction (CNC_June
alone correlates r=+0.95 with predicted_Yield), so including them would not
be a genuine search, it would just reconstruct the label from its own
ingredients. This search asks a narrower, honest question: how well can
remote-sensing-only signals (bloom imagery + climate + cultivar) approximate
the existing model's output.

Method: AIC/CV screening of all 34 candidates (add-one to base), then
forward selection (greedy, CV-driven) building up the best small combined
model, evaluated by 5-fold CV across 6 seeds at each step.

-> Results_Analysis/08_UEF53_Rerun/Predicted_Yield_Feature_Search.json
"""
import os, sys, json
import numpy as np
import openpyxl
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root, load_master

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")

M = load_master(); cultivar = [r.get("cultivar") for r in M]


def load_xlsx_by_id(path):
    wb = openpyxl.load_workbook(path, read_only=True); ws = wb.active
    rowsx = list(ws.iter_rows(values_only=True)); header = rowsx[0]; hidx = {h: i for i, h in enumerate(header)}
    return {r[hidx["Tree_ID"]]: {h: r[hidx[h]] for h in header} for r in rowsx[1:]}


canopy_by_id = load_xlsx_by_id(os.path.join(BASE, "4band_mosaic", "Master_Trees_CanopyStructure_v5.xlsx"))
bloomfrac_by_id = load_xlsx_by_id(os.path.join(BASE, "4band_mosaic", "Master_Trees_BloomFraction_v3.xlsx"))
v6_by_id = load_xlsx_by_id(os.path.join(BASE, "4band_mosaic", "Master_Trees_EBIPixelFeatures_v6.xlsx"))
canopy_for_master = [canopy_by_id.get(r.get("Tree_ID")) for r in M]
bloomfrac_for_master = [bloomfrac_by_id.get(r.get("Tree_ID")) for r in M]
v6_for_master = [v6_by_id.get(r.get("Tree_ID")) for r in M]

CANOPY_NAMES = ["CanopyArea", "CanopyCover", "ShadowFraction", "BloomFraction", "BrightFraction",
                "EBI_density", "BloomVolume", "BloomPixelVolume", "EBI_x_BloomFraction",
                "EBI_x_CanopyCover", "BrightFraction_x_CanopyArea"]
V6_NAMES = ["EBI_Norm_std", "EBI_Norm_var", "EBI_Norm_median", "EBI_Norm_P10", "EBI_Norm_P25",
            "EBI_Norm_P75", "EBI_Norm_P90", "EBI_Norm_IQR", "EBI_Norm_skew", "EBI_Norm_kurtosis",
            "EBI_Norm_CV", "EBI_OtsuFrac", "HighEBI_frac_055", "HighEBI_density_055",
            "HighEBI_frac_065", "HighEBI_density_065", "HighEBI_frac_075", "HighEBI_density_075"]
COMPOSITE_NAMES = ["TBL", "ICBI", "TICBL", "UBI", "CCAB"]
ENG_NAMES = CANOPY_NAMES + V6_NAMES + COMPOSITE_NAMES


def build_all_features(j, yr):
    c = canopy_for_master[j]; b = bloomfrac_for_master[j]; v6 = v6_for_master[j]
    ebi = M[j].get(f"EBI_Norm_{yr}")
    if c is None or b is None or v6 is None or ebi is None: return None
    area = c.get(f"CanopyArea_m2_{yr}"); cover = c.get(f"CanopyCoverFraction_{yr}")
    shadow = c.get(f"ShadowFraction_{yr}"); pix = c.get(f"CanopyPixelCount_{yr}")
    bf = b.get(f"BloomFraction_{yr}"); brf = b.get(f"BrightFraction_{yr}")
    frac065 = v6.get(f"HighEBI_frac_065_{yr}"); ebi_std = v6.get(f"EBI_Norm_std_{yr}"); ebi_cv = v6.get(f"EBI_Norm_CV_{yr}")
    base_needed = [area, cover, shadow, pix, bf, brf, frac065, ebi_std, ebi_cv]
    v6_needed = [v6.get(f"{name}_{yr}") for name in V6_NAMES]
    if any(v is None for v in base_needed) or any(v is None for v in v6_needed): return None
    ebi = float(ebi); area = float(area); cover = float(cover); shadow = float(shadow); pix = float(pix)
    bf = float(bf); brf = float(brf); frac065 = float(frac065); ebi_std = float(ebi_std); ebi_cv = float(ebi_cv)
    canopy_feats = {
        "CanopyArea": area, "CanopyCover": cover, "ShadowFraction": shadow, "BloomFraction": bf,
        "BrightFraction": brf, "EBI_density": ebi / area if area > 0 else np.nan, "BloomVolume": ebi * area,
        "BloomPixelVolume": bf * pix, "EBI_x_BloomFraction": ebi * bf, "EBI_x_CanopyCover": ebi * cover,
        "BrightFraction_x_CanopyArea": brf * area,
    }
    v6_feats = {name: float(v6.get(f"{name}_{yr}")) for name in V6_NAMES}
    composite_feats = {
        "TBL": area * frac065, "ICBI": ebi * (1 - shadow), "TICBL": area * ebi * (1 - shadow),
        "UBI": ebi * (1 - ebi_cv), "CCAB": frac065 * cover,
    }
    out = {}
    out.update(canopy_feats); out.update(v6_feats); out.update(composite_feats)
    return out


def zc(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)


PHYS_NAMES = ["SWP_April", "SWP_May_June", "SWP_June", "Growth_May_June", "Growth_June", "CNC_June"]

rows = []
for j in range(len(M)):
    if cultivar[j] not in ("UEF", "53"): continue
    for yr in (2022, 2023):
        py = M[j].get(f"predicted_Yield_{yr}")
        if py is None: continue
        ga = M[j].get(f"Growth_April_{yr}")
        cp = M[j].get(f"Chill_Portions_{yr}")
        if ga is None or cp is None: continue
        feats = build_all_features(j, yr)
        if feats is None: continue
        phys_vals = {name: M[j].get(f"{name}_{yr}") for name in PHYS_NAMES}
        if any(v is None for v in phys_vals.values()): continue
        feats = dict(feats); feats.update(phys_vals)
        rows.append((j, yr, float(py), feats))
n = len(rows)
print(f"n={n} tree-years with predicted_Yield + full feature bank (UEF + cultivar 53, 2022-2023)")

yv = np.array([r[2] for r in rows])
uef = np.array([1.0 if cultivar[r[0]] == "UEF" else 0.0 for r in rows])
ebi = zc([M[r[0]].get(f"EBI_Norm_{r[1]}") for r in rows])
clim = zc([M[r[0]].get(f"Chill_Portions_{r[1]}") for r in rows])
ga = zc([M[r[0]].get(f"Growth_April_{r[1]}") for r in rows])
one = np.ones(n)
feat_cols = {name: zc([r[3][name] for r in rows]) for name in ENG_NAMES}
feat_cols.update({name: zc([r[3][name] for r in rows]) for name in PHYS_NAMES})

X_base = np.column_stack([one, uef, ebi, ebi * uef, clim, ga])
base_names = ["intercept", "cultivar", "EBI", "EBIxcultivar", "climate", "Growth_April"]

SEEDS_ALL = [42, 1, 7, 13, 99, 2024]


def r2_mse_mae(y_true, y_pred):
    resid = y_true - y_pred
    tss = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - (resid ** 2).sum() / tss), float(np.mean(resid ** 2)), float(np.mean(np.abs(resid)))


def cv_ols(Xd, y, seed, k=5):
    n_ = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n_); folds = np.array_split(idx, k)
    pred = np.full(n_, np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(n_), f)
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None); pred[f] = Xd[f] @ beta
    return r2_mse_mae(y, pred)


def cv_r2_mean(Xd, y, seeds=SEEDS_ALL):
    r2s = [cv_ols(Xd, y, s)[0] for s in seeds]
    return float(np.mean(r2s)), r2s


def fit_ols_metrics(Xd, y):
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta; n_, p_ = Xd.shape
    rss = float((resid ** 2).sum()); k = p_ + 1
    logL = -n_ / 2 * np.log(2 * np.pi) - n_ / 2 * np.log(rss / n_) - n_ / 2
    aic = -2 * logL + 2 * k
    return {"AIC": round(float(aic), 2), "R2_in_sample": round(float(1 - rss / ((y - y.mean()) ** 2).sum()), 4),
            "MAE_in_sample": round(float(np.mean(np.abs(resid))), 4)}


# ---- baseline ----
base_metrics = fit_ols_metrics(X_base, yv)
base_cv_mean, base_cv_all = cv_r2_mean(X_base, yv)
print(f"BASELINE (cultivar+EBIxcultivar+climate+Growth_April): AIC={base_metrics['AIC']} "
      f"R2_in={base_metrics['R2_in_sample']} MAE_in={base_metrics['MAE_in_sample']} cvR2_mean={round(base_cv_mean,4)}")

# ---- stage 1: add-one screening (AIC + CV) across all 34 engineered candidates ----
screen = {}
for name in ENG_NAMES:
    Xt = np.column_stack([X_base, feat_cols[name]])
    m = fit_ols_metrics(Xt, yv)
    cvm, _ = cv_r2_mean(Xt, yv, seeds=[42, 1, 7])  # 3-seed screen, cheap since n is large
    screen[name] = {"dAIC": round(m["AIC"] - base_metrics["AIC"], 2), "cvR2_screen": round(cvm, 4),
                     "d_cvR2_screen": round(cvm - base_cv_mean, 4)}
by_cv = sorted(screen.keys(), key=lambda k: -screen[k]["d_cvR2_screen"])
print("\nTop 10 single-candidate additions by CV R2 gain:")
for name in by_cv[:10]:
    print(f"  {name}: {screen[name]}")

# ---- stage 2: forward selection (greedy, CV-driven, full 6-seed at each accept) ----
selected = []
remaining = set(ENG_NAMES)
current_cols = [X_base]
current_cv, _ = cv_r2_mean(X_base, yv)
history = [{"step": 0, "added": None, "cvR2_mean": round(current_cv, 4), "features": list(base_names)}]
MIN_GAIN = 0.002
for step in range(1, 13):
    best_name, best_cv, best_gain = None, current_cv, 0.0
    for name in remaining:
        Xt = np.column_stack(current_cols + [feat_cols[name]])
        cvm, _ = cv_r2_mean(Xt, yv, seeds=[42, 1, 7])  # cheap screen per candidate
        gain = cvm - current_cv
        if gain > best_gain:
            best_gain, best_cv, best_name = gain, cvm, name
    if best_name is None or best_gain < MIN_GAIN:
        break
    current_cols.append(feat_cols[best_name])
    remaining.discard(best_name)
    selected.append(best_name)
    full_cv, full_cv_all = cv_r2_mean(np.column_stack(current_cols), yv)  # honest 6-seed re-check
    current_cv = full_cv
    history.append({"step": step, "added": best_name, "cvR2_mean": round(full_cv, 4),
                     "cvR2_all_seeds": [round(x, 4) for x in full_cv_all],
                     "features": list(base_names) + selected})
    print(f"step {step}: + {best_name}  ->  cvR2_mean={round(full_cv,4)} (screen said {round(best_cv,4)})")

X_final = np.column_stack(current_cols)
final_metrics = fit_ols_metrics(X_final, yv)
final_cv_mean, final_cv_all = cv_r2_mean(X_final, yv)
print(f"\nFINAL MODEL (remote-sensing only): base + {selected}")
print(f"AIC={final_metrics['AIC']} R2_in={final_metrics['R2_in_sample']} MAE_in={final_metrics['MAE_in_sample']} "
      f"cvR2_mean={round(final_cv_mean,4)} cvR2_range=[{round(min(final_cv_all),4)},{round(max(final_cv_all),4)}]")

# ---- stage 3: what happens if the "circular" physiology fields ARE allowed in ----
print("\n--- now allowing SWP/Growth_May_June/Growth_June/CNC_June back in ---")
screen_phys = {}
for name in PHYS_NAMES:
    Xt = np.column_stack([X_base, feat_cols[name]])
    m = fit_ols_metrics(Xt, yv)
    cvm, _ = cv_r2_mean(Xt, yv, seeds=[42, 1, 7])
    screen_phys[name] = {"dAIC": round(m["AIC"] - base_metrics["AIC"], 2), "cvR2_screen": round(cvm, 4),
                          "d_cvR2_screen": round(cvm - base_cv_mean, 4)}
    print(f"  {name} alone: {screen_phys[name]}")

selected_p = []
remaining_p = set(ENG_NAMES) | set(PHYS_NAMES)
current_cols_p = [X_base]
current_cv_p, _ = cv_r2_mean(X_base, yv)
history_p = [{"step": 0, "added": None, "cvR2_mean": round(current_cv_p, 4), "features": list(base_names)}]
for step in range(1, 13):
    best_name, best_cv, best_gain = None, current_cv_p, 0.0
    for name in remaining_p:
        Xt = np.column_stack(current_cols_p + [feat_cols[name]])
        cvm, _ = cv_r2_mean(Xt, yv, seeds=[42, 1, 7])
        gain = cvm - current_cv_p
        if gain > best_gain:
            best_gain, best_cv, best_name = gain, cvm, name
    if best_name is None or best_gain < MIN_GAIN:
        break
    current_cols_p.append(feat_cols[best_name])
    remaining_p.discard(best_name)
    selected_p.append(best_name)
    full_cv, full_cv_all = cv_r2_mean(np.column_stack(current_cols_p), yv)
    current_cv_p = full_cv
    history_p.append({"step": step, "added": best_name, "cvR2_mean": round(full_cv, 4),
                       "cvR2_all_seeds": [round(x, 4) for x in full_cv_all],
                       "features": list(base_names) + selected_p})
    print(f"step {step}: + {best_name}  ->  cvR2_mean={round(full_cv,4)} (screen said {round(best_cv,4)})")

X_final_p = np.column_stack(current_cols_p)
final_metrics_p = fit_ols_metrics(X_final_p, yv)
final_cv_mean_p, final_cv_all_p = cv_r2_mean(X_final_p, yv)
print(f"\nFINAL MODEL (physiology fields allowed): base + {selected_p}")
print(f"AIC={final_metrics_p['AIC']} R2_in={final_metrics_p['R2_in_sample']} MAE_in={final_metrics_p['MAE_in_sample']} "
      f"cvR2_mean={round(final_cv_mean_p,4)}")

RESULTS = {
    "n": n, "target": "predicted_Yield (MODELLED, not measured, per project convention)",
    "excluded_circular_fields": ["SWP_April", "SWP_MayJune", "SWP_June", "Growth_MayJune", "Growth_June",
                                  "CNC_June", "TG_clusters", "SWP_clusters"],
    "baseline": {"features": base_names, **base_metrics, "cvR2_mean": round(base_cv_mean, 4),
                 "cvR2_all_seeds": [round(x, 4) for x in base_cv_all]},
    "screen_all_34_candidates": screen,
    "forward_selection_history": history,
    "final_model": {"features": base_names + selected, "added_features": selected, **final_metrics,
                     "cvR2_mean": round(final_cv_mean, 4),
                     "cvR2_range": [round(min(final_cv_all), 4), round(max(final_cv_all), 4)],
                     "cvR2_all_seeds": [round(x, 4) for x in final_cv_all]},
    "physiology_fields_tested": PHYS_NAMES,
    "screen_physiology_fields_alone": screen_phys,
    "forward_selection_history_with_physiology": history_p,
    "final_model_with_physiology": {"features": base_names + selected_p, "added_features": selected_p,
                                     **final_metrics_p, "cvR2_mean": round(final_cv_mean_p, 4),
                                     "cvR2_range": [round(min(final_cv_all_p), 4), round(max(final_cv_all_p), 4)],
                                     "cvR2_all_seeds": [round(x, 4) for x in final_cv_all_p]},
}
with open(os.path.join(OUT, "Predicted_Yield_Feature_Search.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nsaved json")
