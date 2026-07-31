#!/usr/bin/env python3
"""
thesis_analysis.py — Consolidated, reproducible statistical flow for the thesis Results.
Recomputes ALL headline numbers from Master_Trees_Extended.xlsx in thesis order.
Emits results.json + corrected yield figures. No scipy (uses stats_utils primitives).
Order: S1 dataset -> S2 EBI temporal -> S3 cultivar -> S4 spatial(Moran) ->
       S5 climate drivers -> S6 measured yield -> S7 synthesis.
"""
import csv, math, os, json
import numpy as np, openpyxl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,"/sessions/trusting-relaxed-gates/mnt/Thesis/pipeline/06_analysis")
from stats_utils import f_pvalue, t_pvalue, r_pvalue, pca, split_plot_anova, sig_stars
import pandas as pd

BASE="/sessions/trusting-relaxed-gates/mnt/Thesis"
OUT=f"{BASE}/Results_Analysis"; SC="/sessions/trusting-relaxed-gates/mnt/outputs"
R={}  # results dict
def star(p): return sig_stars(p) if not (isinstance(p,float) and math.isnan(p)) else ""

wb=openpyxl.load_workbook(f"{BASE}/4band_mosaic/Master_Trees_Extended.xlsx",read_only=True,data_only=True)
ws=wb.active; raw=list(ws.iter_rows(values_only=True)); H=raw[0]
M=[dict(zip(H,r)) for r in raw[1:]]; wb.close()
def col(c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in M],float)
YEARS=[2021,2022,2023,2024]

# ---------- S1 dataset ----------
cults=[r.get("cultivar") for r in M]
from collections import Counter
R["S1_dataset"]={"n_trees":len(M),"cultivar_counts":dict(Counter(c for c in cults if c)),
                 "years":YEARS}

# ---------- S2 EBI temporal ----------
ebi={y:col(f"EBI_Norm_{y}") for y in YEARS}
R["S2_ebi_dist"]={}
for y in YEARS:
    a=ebi[y]; a=a[~np.isnan(a)]
    R["S2_ebi_dist"][y]={"n":int(len(a)),"mean":round(float(a.mean()),4),"sd":round(float(a.std(ddof=1)),4)}

def paired_t(a,b):
    m=~np.isnan(a)&~np.isnan(b); d=a[m]-b[m]; n=len(d)
    if n<3: return np.nan,np.nan,n,np.nan
    md=d.mean(); sd=d.std(ddof=1); t=md/(sd/math.sqrt(n))
    p=t_pvalue(abs(t),n-1); return float(t),float(p),n,float(md)
R["S2_paired_t"]={}
for a,b in [(2021,2022),(2022,2023),(2023,2024)]:
    t,p,n,md=paired_t(ebi[a],ebi[b])
    R["S2_paired_t"][f"{a}v{b}"]={"t":round(t,3),"p":round(p,4),"n":n,"dmean":round(md,4),"sig":star(p)}

# ---------- S3 cultivar effects ----------
def f_oneway(groups):
    groups=[np.asarray(g,float) for g in groups]; groups=[g[~np.isnan(g)] for g in groups]; groups=[g for g in groups if len(g)>0]
    allv=np.concatenate(groups); gm=allv.mean(); k=len(groups); N=len(allv)
    ssb=sum(len(g)*(g.mean()-gm)**2 for g in groups); ssw=sum(((g-g.mean())**2).sum() for g in groups)
    d1,d2=k-1,N-k
    if d1<=0 or d2<=0 or ssw==0: return np.nan,np.nan,np.nan
    F=(ssb/d1)/(ssw/d2); return float(F),float(f_pvalue(F,d1,d2)),float(ssb/(ssb+ssw))
uc=sorted(set(c for c in cults if c))
R["S3_cultivar_anova_ebi"]={}
for y in YEARS:
    groups=[ebi[y][np.array([c==u for c in cults])] for u in uc]
    F,p,eta=f_oneway(groups)
    R["S3_cultivar_anova_ebi"][y]={"F":round(F,2),"p":round(p,6),"eta2":round(eta,3),"sig":star(p)}

# split-plot cultivar x year
long_rows=[]
for i,r in enumerate(M):
    c=r.get("cultivar")
    if not c: continue
    for y in YEARS:
        v=r.get(f"EBI_Norm_{y}")
        if v is not None: long_rows.append({"subj":i,"cult":c,"year":y,"ebi":float(v)})
ldf=pd.DataFrame(long_rows)
sp=split_plot_anova(ldf,"subj","cult","year","ebi")
R["S3_splitplot"]={k:{ "F":round(float(sp[k]["F"]),2),"df1":sp[k]["df1"],"df2":sp[k]["df2"],
                       "p":float(sp[k]["p"]),"eta2":round(float(sp[k]["eta2"]),3),"sig":star(sp[k]["p"])}
                   for k in ["between","within","interaction"]}

# ---------- S4 spatial (Moran's I, distance-band weights) ----------
X=col("X_UTM"); Y=col("Y_UTM")
def morans_I(vals,X,Y,band=10.0):
    m=~np.isnan(vals)&~np.isnan(X)&~np.isnan(Y)
    z=vals[m]-vals[m].mean(); xs=X[m]; ys=Y[m]; n=len(z)
    num=0.0; S0=0.0
    for i in range(n):
        d=np.hypot(xs-xs[i],ys-ys[i]); w=(d>0)&(d<=band)
        num+=np.sum(w*z[i]*z); S0+=np.sum(w)
    den=np.sum(z**2)
    I=(n/S0)*(num/den) if S0>0 and den>0 else np.nan
    EI=-1.0/(n-1)
    # permutation p (approx, 199 perms)
    rng=np.random.default_rng(42); cnt=0; perms=199
    for _ in range(perms):
        zp=rng.permutation(z); nump=0.0
        for i in range(n):
            d=np.hypot(xs-xs[i],ys-ys[i]); w=(d>0)&(d<=band); nump+=np.sum(w*zp[i]*zp)
        Ip=(n/S0)*(nump/den)
        if Ip>=I: cnt+=1
    p=(cnt+1)/(perms+1)
    return float(I),float(EI),float(p),n
R["S4_moran"]={}
for y in YEARS:
    I,EI,p,n=morans_I(ebi[y],X,Y,band=10.0)
    R["S4_moran"][y]={"I":round(I,4),"EI":round(EI,4),"p":round(p,4),"n":n,"sig":star(p)}
    print(f"Moran {y}: I={I:.4f} p={p:.4f} n={n}")

# ---------- S5 climate drivers (year-level EBI vs climate) ----------
CLIM=["Chill_Portions","Late_Chill","Chill_Hrs","GDD_Feb","GDD_Jan_Feb","Avg_DTR_Feb",
      "Rad_Feb","Max_Dry_Hrs","Trans_Ratio","Frost_Hrs","Heat_Stress_Feb","Rain_Feb","High_Wind_Hrs"]
ebi_year_mean={y:float(np.nanmean(ebi[y])) for y in YEARS}
clim_year={}
for cv in CLIM:
    clim_year[cv]={}
    for y in YEARS:
        v=col(f"{cv}_{y}"); v=v[~np.isnan(v)]
        clim_year[cv][y]=float(v[0]) if len(v) else np.nan
R["S5_climate_ebi_year"]={}
for cv in CLIM:
    xs=np.array([clim_year[cv][y] for y in YEARS]); ys=np.array([ebi_year_mean[y] for y in YEARS])
    m=~np.isnan(xs)&~np.isnan(ys)
    if m.sum()>=3 and len(set(xs[m]))>1:
        r=np.corrcoef(xs[m],ys[m])[0,1]
        R["S5_climate_ebi_year"][cv]={"r":round(float(r),3),"n":int(m.sum())}

# ---------- S6 measured yield (canonical) — reload from json produced earlier ----------
try:
    R["S6_measured_yield"]=json.load(open(f"{SC}/measured_yield_results.json"))
except Exception as e:
    R["S6_measured_yield"]={"error":str(e)}

# provenance note
yv=col("Yield_2022"); pv=col("predicted_Yield_2022")
mask=~np.isnan(yv)&~np.isnan(pv)
R["S6_provenance"]={"corr_Yield_vs_predicted_2022":round(float(np.corrcoef(yv[mask],pv[mask])[0,1]),4),
                    "note":"master Yield_2022/2023 == GPKG predicted_Yield (modeled, not measured)"}

json.dump(R,open(f"{SC}/thesis_results.json","w"),indent=2,default=str)
print("\n=== KEY RESULTS ===")
print(json.dumps(R,indent=1,default=str)[:2500])

# ================= corrected yield figures =================
# reload measured join for scatter/boxplot
import sqlite3
with open(f"{BASE}/Trees_Data_Survey/kedma_plot_a_yield.csv",encoding="utf-8-sig") as f:
    yc=list(csv.DictReader(f))
from collections import defaultdict
ymeas=defaultdict(dict);coord={}
for r in yc:
    ymeas[r["tree_id"]][int(r["year"])]=float(r["net_kernel_yield_per_tree_kg"]); coord[r["tree_id"]]=(float(r["latitude"]),float(r["longitude"]))
mlat=col("Latitude");mlon=col("Longitude")
def hav(la1,lo1,la2,lo2):
    Rr=6371000;p=math.pi/180;a=math.sin((la2-la1)*p/2)**2+math.cos(la1*p)*math.cos(la2*p)*math.sin((lo2-lo1)*p/2)**2
    return 2*Rr*math.asin(math.sqrt(max(0,a)))
cand=[]
for tid,(la,lo) in coord.items():
    ds=np.array([hav(la,lo,a,b) for a,b in zip(mlat,mlon)])
    for j in np.where(ds<=8.0)[0]: cand.append((float(ds[j]),tid,int(j)))
cand.sort(); uy=set();um=set();match={}
for d,tid,j in cand:
    if tid in uy or j in um: continue
    match[tid]=(j,d);uy.add(tid);um.add(j)
rows=[]
for tid,(j,d) in match.items():
    m=M[j];rows.append(dict(cult=m.get("cultivar"),EBI=m.get("EBI_Norm_2023"),
        NGRDI=m.get("NGRDI_Norm_2023"),Y=ymeas[tid].get(2023,np.nan)))
ebix=np.array([r["EBI"] if r["EBI"] is not None else np.nan for r in rows],float)
yv=np.array([r["Y"] for r in rows],float)
m=~np.isnan(ebix)&~np.isnan(yv)
r=np.corrcoef(ebix[m],yv[m])[0,1];p=r_pvalue(r,m.sum())
CCOL={"53":"#2E75B6","UEF":"#C0392B","54":"#27AE60"}
fig,axes=plt.subplots(1,2,figsize=(13,5.2))
ax=axes[0]
for c in ["53","UEF"]:
    sel=[i for i in range(len(rows)) if rows[i]["cult"]==c and not np.isnan(ebix[i]) and not np.isnan(yv[i])]
    ax.scatter(ebix[sel],yv[sel],s=45,alpha=.75,c=CCOL.get(c,"gray"),edgecolors="white",lw=.5,label=c)
b1,b0=np.polyfit(ebix[m],yv[m],1);xl=np.linspace(ebix[m].min(),ebix[m].max(),50)
ax.plot(xl,b1*xl+b0,"k--",lw=1.4)
ax.set_title(f"Measured 2023 yield vs spring EBI\n r={r:+.3f} p={p:.3f} {star(p)} (n={m.sum()})",fontweight="bold")
ax.set_xlabel("EBI_Norm 2023");ax.set_ylabel("Measured yield (kg/tree)");ax.legend(title="Cultivar");ax.grid(alpha=.3)
ax=axes[1]
gb=[[rows[i]["Y"] for i in range(len(rows)) if rows[i]["cult"]==c and not np.isnan(rows[i]["Y"])] for c in ["53","UEF"]]
bp=ax.boxplot(gb,patch_artist=True,tick_labels=["53","UEF"],medianprops=dict(color="black",lw=2))
for patch,c in zip(bp["boxes"],["53","UEF"]): patch.set_facecolor(CCOL[c]);patch.set_alpha(.7)
F,pp,eta=f_oneway(gb)
ax.set_title(f"Measured 2023 yield by cultivar\n ANOVA F={F:.2f} p={pp:.4f} {star(pp)} η²={eta:.3f}",fontweight="bold")
ax.set_xlabel("Cultivar");ax.set_ylabel("Measured yield (kg/tree)");ax.grid(axis="y",alpha=.3)
fig.suptitle("Measured (ground-truth) yield — Kedma Plot A, 2023 (n≈151 matched trees)",fontweight="bold",fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/Measured_Yield_EBI_Cultivar.png",dpi=140,bbox_inches="tight")
fig.savefig(f"{SC}/Measured_Yield_EBI_Cultivar.png",dpi=140,bbox_inches="tight")
print("\n✓ Measured_Yield_EBI_Cultivar.png saved")
