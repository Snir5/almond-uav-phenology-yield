#!/usr/bin/env python3
"""Climate x EBI x Yield relationships over years + ANCOVA. Measured yield canonical."""
import csv, math, os, json
import numpy as np, openpyxl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,"/sessions/trusting-relaxed-gates/mnt/Thesis/pipeline/06_analysis")
from stats_utils import f_pvalue, t_pvalue, r_pvalue, ols_rss
BASE="/sessions/trusting-relaxed-gates/mnt/Thesis"; OUT=f"{BASE}/Results_Analysis"; SC="/sessions/trusting-relaxed-gates/mnt/outputs"

wb=openpyxl.load_workbook(f"{BASE}/4band_mosaic/Master_Trees_Extended.xlsx",read_only=True,data_only=True)
ws=wb.active; raw=list(ws.iter_rows(values_only=True)); H=raw[0]; M=[dict(zip(H,r)) for r in raw[1:]]; wb.close()
def c(col): return np.array([r.get(col) if r.get(col) is not None else np.nan for r in M],float)
YEARS=[2021,2022,2023,2024]
def star(p): return "" if (p!=p) else "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"

# measured yield
yc=list(csv.DictReader(open(f"{BASE}/Trees_Data_Survey/kedma_plot_a_yield.csv",encoding="utf-8-sig")))
from collections import defaultdict
ymeas=defaultdict(dict); coord={}
for r in yc: ymeas[r["tree_id"]][int(r["year"])]=float(r["net_kernel_yield_per_tree_kg"]); coord[r["tree_id"]]=(float(r["latitude"]),float(r["longitude"]))
lat=c("Latitude"); lon=c("Longitude")
def hv(a,b,d,e):
    p=math.pi/180; x=math.sin((d-a)*p/2)**2+math.cos(a*p)*math.cos(d*p)*math.sin((e-b)*p/2)**2
    return 2*6371000*math.asin(math.sqrt(max(0,x)))
cand=[]
for t,(la,lo) in coord.items():
    ds=np.array([hv(la,lo,a,b) for a,b in zip(lat,lon)])
    for j in np.where(ds<=8)[0]: cand.append((float(ds[j]),t,int(j)))
cand.sort(); uy=set();um=set();mp={}
for d,t,j in cand:
    if t in uy or j in um: continue
    mp[t]=j; uy.add(t); um.add(j)

R={}
# ---- year-level table: climate, EBI, measured+predicted yield ----
CLIM=["Chill_Portions","Chill_Hrs","GDD_Feb","GDD_Jan_Feb","Trans_Ratio","Rain_Feb","Heat_Stress_Feb","Avg_DTR_Feb"]
clim_year={cv:{y:float(np.nanmean(c(f"{cv}_{y}"))) for y in YEARS} for cv in CLIM}
ebi_year={y:float(np.nanmean(c(f"EBI_Norm_{y}"))) for y in YEARS}
ebi_sd_year={y:float(np.nanstd(c(f"EBI_Norm_{y}"),ddof=1)) for y in YEARS}
# measured yield by year (matched trees)
meas_year={}
for y in [2022,2023,2024]:
    vals=[ymeas[t].get(y) for t in mp if ymeas[t].get(y) is not None]
    meas_year[y]=float(np.mean(vals)) if vals else np.nan
pred_year={2022:float(np.nanmean(c("predicted_Yield_2022"))),2023:float(np.nanmean(c("predicted_Yield_2023")))}
R["year_level"]={"climate":clim_year,"ebi_mean":ebi_year,"ebi_sd":ebi_sd_year,"yield_measured":meas_year,"yield_predicted":pred_year}

# ---- climate vs EBI mean & sd (year level, n=4) ----
R["climate_vs_EBImean"]={}; R["climate_vs_EBIsd"]={}
for cv in CLIM:
    x=np.array([clim_year[cv][y] for y in YEARS])
    for tgt,d in [("climate_vs_EBImean",ebi_year),("climate_vs_EBIsd",ebi_sd_year)]:
        yv=np.array([d[y] for y in YEARS]); m=~np.isnan(x)&~np.isnan(yv)
        if m.sum()>=3 and len(set(x[m]))>1:
            r=np.corrcoef(x[m],yv[m])[0,1]; R[tgt][cv]={"r":round(float(r),3),"p":round(r_pvalue(r,m.sum()),3),"n":int(m.sum())}

# ---- ANCOVA: measured Yield_2023 ~ EBI_2023 + cultivar ----
E=[];Y=[];CU=[]
for t,j in mp.items():
    if ymeas[t].get(2023) is not None and M[j].get("EBI_Norm_2023") is not None:
        E.append(float(M[j]["EBI_Norm_2023"]));Y.append(ymeas[t][2023]);CU.append(M[j].get("cultivar"))
E=np.array(E);Y=np.array(Y);CU=np.array(CU)
cats=sorted(set(CU)); n=len(Y)
# dummy for cultivar (drop first)
def design(cols):
    return np.column_stack([np.ones(n)]+cols)
dum=[ (CU==k).astype(float) for k in cats[1:] ]
inter=[ ((CU==k).astype(float))*E for k in cats[1:] ]  # EBI x cultivar interaction
# Model comparisons (nested F-tests): full has MORE params than reduced
rss_full,k_full,_=ols_rss(design([E]+dum),Y)          # EBI + cultivar
rss_cult,k_cult,_=ols_rss(design(dum),Y)              # cultivar only
rss_ebi,k_ebi,_ =ols_rss(design([E]),Y)               # EBI only
rss_null,k_null,_=ols_rss(design([]),Y)               # intercept
rss_int,k_int,_ =ols_rss(design([E]+dum+inter),Y)     # + EBI x cultivar interaction
def Ftest(rss_r,df_r,rss_f,df_f):
    # reduced (fewer params, df_r) nested in full (more params, df_f); df_f>df_r
    num=(rss_r-rss_f)/(df_f-df_r); den=rss_f/(n-df_f)
    F=num/den if den>0 else np.nan
    return float(F),float(f_pvalue(F,df_f-df_r,n-df_f))
F_ebi_add,p_ebi_add=Ftest(rss_cult,k_cult,rss_full,k_full)   # EBI effect | cultivar
F_cult_add,p_cult_add=Ftest(rss_ebi,k_ebi,rss_full,k_full)   # cultivar effect | EBI
F_intr,p_intr=Ftest(rss_full,k_full,rss_int,k_int)           # EBI x cultivar interaction
r2_full=1-rss_full/rss_null; r2_int=1-rss_int/rss_null
R["ANCOVA_yield2023"]={"n":n,"cultivars":cats,
  "EBI_given_cultivar":{"F":round(F_ebi_add,2),"p":round(p_ebi_add,4),"sig":star(p_ebi_add)},
  "cultivar_given_EBI":{"F":round(F_cult_add,2),"p":round(p_cult_add,4),"sig":star(p_cult_add)},
  "EBIxcultivar_interaction":{"F":round(F_intr,2),"p":round(p_intr,4),"sig":star(p_intr)},
  "model_R2":round(float(r2_full),3),"model_R2_with_interaction":round(float(r2_int),3)}
# within-cultivar EBI-yield slope
R["within_cultivar_EBI_yield"]={}
for k in cats:
    m=CU==k
    if m.sum()>=5:
        r=np.corrcoef(E[m],Y[m])[0,1]; R["within_cultivar_EBI_yield"][k]={"r":round(float(r),3),"p":round(r_pvalue(r,m.sum()),3),"n":int(m.sum())}

print(json.dumps(R["ANCOVA_yield2023"],indent=1)); print("within-cult:",R["within_cultivar_EBI_yield"])
json.dump(R,open(f"{SC}/climate_yield_ebi.json","w"),indent=2,default=str)

# ================= FIGURE: Climate x EBI x Yield over years =================
fig=plt.figure(figsize=(16,10)); gs=fig.add_gridspec(2,3,hspace=.32,wspace=.30)
yrs=YEARS
# A: chill + EBI mean/sd over years (dual axis)
ax=fig.add_subplot(gs[0,0]); ax2=ax.twinx()
ax.bar([y-0.15 for y in yrs],[clim_year["Chill_Portions"][y] for y in yrs],width=.3,color="#4aa3df",label="Chill Portions")
ax2.plot(yrs,[ebi_sd_year[y] for y in yrs],"o-",color="#e07b4a",label="EBI SD")
ax.set_title("A. Chilling vs EBI heterogeneity (SD)",fontweight="bold");ax.set_xlabel("Year")
ax.set_ylabel("Chill Portions",color="#4aa3df");ax2.set_ylabel("EBI_Norm SD",color="#e07b4a");ax.set_xticks(yrs)
# B: GDD vs EBI mean
ax=fig.add_subplot(gs[0,1])
for cv,col2 in [("GDD_Jan_Feb","#2E75B6"),("Chill_Portions","#C0392B")]:
    xs=[clim_year[cv][y] for y in yrs]; ys=[ebi_year[y] for y in yrs]
    ax.scatter(xs,ys,s=70,color=col2,label=cv)
    for x,yy,yr in zip(xs,ys,yrs): ax.annotate(str(yr),(x,yy),fontsize=8,xytext=(4,4),textcoords="offset points")
ax.set_title("B. Climate vs mean EBI (n=4 yrs)",fontweight="bold");ax.set_xlabel("Climate value");ax.set_ylabel("Mean EBI_Norm");ax.legend(fontsize=8)
# C: yield over years (measured vs predicted)
ax=fig.add_subplot(gs[0,2])
my=[meas_year.get(y,np.nan) for y in [2022,2023,2024]]; py=[pred_year.get(y,np.nan) for y in [2022,2023]]
ax.plot([2022,2023,2024],my,"s-",color="#C0392B",label="Measured")
ax.plot([2022,2023],py,"o--",color="#4aa3df",label="Predicted (model)")
ax.set_title("C. Yield over years",fontweight="bold");ax.set_xlabel("Year");ax.set_ylabel("Yield (kg/tree)");ax.set_xticks([2022,2023,2024]);ax.legend(fontsize=8)
# D: ANCOVA — yield vs EBI by cultivar with fitted lines
ax=fig.add_subplot(gs[1,0]); CCOL={"53":"#2E75B6","UEF":"#C0392B","54":"#27AE60"}
for k in cats:
    m=CU==k
    ax.scatter(E[m],Y[m],s=35,alpha=.7,color=CCOL.get(k,"gray"),edgecolors="white",lw=.4,label=k)
    if m.sum()>=5:
        b1,b0=np.polyfit(E[m],Y[m],1);xl=np.linspace(E[m].min(),E[m].max(),20);ax.plot(xl,b1*xl+b0,color=CCOL.get(k,"gray"),lw=1.4)
ax.set_title(f"D. ANCOVA yield~EBI+cultivar (2023)\nEBI|cult F={F_ebi_add:.2f}{star(p_ebi_add)}, cult|EBI F={F_cult_add:.2f}{star(p_cult_add)}",fontweight="bold",fontsize=10)
ax.set_xlabel("EBI_Norm 2023");ax.set_ylabel("Measured yield (kg/tree)");ax.legend(fontsize=8,title="Cultivar")
# E: climate vs EBI SD bar of r
ax=fig.add_subplot(gs[1,1])
items=sorted(R["climate_vs_EBIsd"].items(),key=lambda kv:kv[1]["r"])
names=[k for k,_ in items]; rs=[v["r"] for _,v in items]
ax.barh(names,rs,color=["#C0392B" if v<0 else "#27AE60" for v in rs])
ax.axvline(0,color="k",lw=.8);ax.set_title("E. Climate vs EBI spatial heterogeneity (r, n=4)",fontweight="bold",fontsize=10);ax.set_xlabel("Pearson r")
ax.tick_params(labelsize=8)
# F: EBI exceedance fraction (>0.60,0.65) vs chill
ax=fig.add_subplot(gs[1,2])
exc60={2021:28.8,2022:26.1,2023:16.0,2024:10.0}; exc65={2021:20.8,2022:7.8,2023:2.2,2024:0.6}
chill=[clim_year["Chill_Portions"][y] for y in yrs]
ax.scatter(chill,[exc60[y] for y in yrs],s=70,color="#2E75B6",label="EBI>0.60")
ax.scatter(chill,[exc65[y] for y in yrs],s=70,color="#e07b4a",label="EBI>0.65")
for x,yy,yr in zip(chill,[exc60[y] for y in yrs],yrs): ax.annotate(str(yr),(x,yy),fontsize=8,xytext=(4,4),textcoords="offset points")
ax.set_title("F. Bloom-bright fraction vs chilling",fontweight="bold",fontsize=10);ax.set_xlabel("Chill Portions");ax.set_ylabel("% trees exceeding");ax.legend(fontsize=8)
fig.suptitle("Climate × EBI × Yield Relationships Over Years (2021–2024)",fontsize=15,fontweight="bold")
fig.savefig(f"{OUT}/Climate_EBI_Yield_OverYears.png",dpi=140,bbox_inches="tight")
fig.savefig(f"{SC}/Climate_EBI_Yield_OverYears.png",dpi=140,bbox_inches="tight")
print("\n✓ Climate_EBI_Yield_OverYears.png saved")
