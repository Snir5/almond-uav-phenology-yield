#!/usr/bin/env python3
"""Regenerate spatial map figures that carried the stale (offset) boundary polygon.
Boundary is now the convex hull of the corrected trees -> aligned by construction.
Tree points were always correct; only the overlaid boundary was stale."""
import openpyxl, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BASE="/sessions/trusting-relaxed-gates/mnt/Thesis"; RA=f"{BASE}/Results_Analysis"
def load(fn):
    wb=openpyxl.load_workbook(fn,read_only=True,data_only=True); ws=wb.active
    raw=list(ws.iter_rows(values_only=True)); H=raw[0]; M=[dict(zip(H,r)) for r in raw[1:]]; wb.close(); return M
M=load(f"{BASE}/4band_mosaic/Master_Trees_Extended.xlsx")
TS=load(f"{BASE}/4band_mosaic/Master_Trees_Time_Series_Plot1.xlsx")
def arr(rows,c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in rows],float)
lon=arr(M,"Longitude"); lat=arr(M,"Latitude"); X=arr(M,"X_UTM"); Y=arr(M,"Y_UTM")
cult=np.array([r.get("cultivar") for r in M],object)

def hull(px,py):
    pts=sorted(set(zip(px,py)))
    if len(pts)<3: return pts
    def cr(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lo=[]
    for p in pts:
        while len(lo)>=2 and cr(lo[-2],lo[-1],p)<=0: lo.pop()
        lo.append(p)
    up=[]
    for p in reversed(pts):
        while len(up)>=2 and cr(up[-2],up[-1],p)<=0: up.pop()
        up.append(p)
    h=lo[:-1]+up[:-1]; return h
def hull_xy(px,py):
    g=~np.isnan(px)&~np.isnan(py); h=hull(list(px[g]),list(py[g]))
    hx=[p[0] for p in h]+[h[0][0]]; hy=[p[1] for p in h]+[h[0][1]]; return hx,hy
HLon,HLat=hull_xy(lon,lat)
YEARS=[2021,2022,2023,2024]
CULTCOL={"UEF":"#4C9BD4","53":"#5BA85B","54":"#E8853B"}

def draw_hull(ax): ax.plot(HLon,HLat,color="#555",lw=1.1,ls="--",alpha=.6)

# ---------- 1. Fig4_Spatial_EBI_Maps ----------
fig,axes=plt.subplots(2,2,figsize=(15,15))
for ax,yr in zip(axes.ravel(),YEARS):
    v=arr(M,f"EBI_Norm_{yr}"); m=~np.isnan(v)
    sc=ax.scatter(lon[m],lat[m],c=v[m],cmap="RdYlGn",vmin=0.3,vmax=0.9,s=16,edgecolors="none")
    draw_hull(ax)
    ax.set_title(f"EBI {yr} (n={m.sum()} trees with data)"); ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
    ax.set_aspect("equal"); ax.grid(alpha=.25); plt.colorbar(sc,ax=ax,shrink=.8,label="EBI_Norm")
fig.suptitle("Spatial EBI Distribution — Kedma Plot A (Extended Master, aligned)",fontsize=15,fontweight="bold")
plt.tight_layout(rect=[0,0,1,.99]); fig.savefig(f"{RA}/Fig4_Spatial_EBI_Maps.png",dpi=140,bbox_inches="tight"); plt.close()
print("✓ Fig4_Spatial_EBI_Maps")

# ---------- 2. Cultivar_Crown_Maps_AllYears ----------
fig,axes=plt.subplots(2,2,figsize=(15,13))
for ax,yr in zip(axes.ravel(),YEARS):
    v=arr(M,f"EBI_Norm_{yr}"); has=~np.isnan(v)
    for k in ["UEF","53","54"]:
        mm=has&(cult==k)
        ax.scatter(lon[mm],lat[mm],s=16,c=CULTCOL[k],edgecolors="none",label=f"{k} (n={mm.sum()})")
    draw_hull(ax)
    ax.set_title(f"EBI {yr} — Cultivar Distribution (n={has.sum()})"); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_aspect("equal"); ax.grid(alpha=.25); ax.legend(fontsize=9,loc="upper right")
fig.suptitle("EBI Spatial Maps by Year — Coloured by Cultivar (GPS-Corrected, aligned)",fontsize=14,fontweight="bold")
plt.tight_layout(rect=[0,0,1,.99]); fig.savefig(f"{RA}/Cultivar_Crown_Maps_AllYears.png",dpi=140,bbox_inches="tight"); plt.close()
print("✓ Cultivar_Crown_Maps_AllYears")

# ---------- 3. Cultivar_Segments_Corrected (EBI quartiles) ----------
QC=["#3B6FB0","#9CC3E4","#F2B45C","#C0392B"]; QL=["Q1 (low bloom)","Q2","Q3","Q4 (high bloom)"]
fig,axes=plt.subplots(2,2,figsize=(15,13))
for ax,yr in zip(axes.ravel(),YEARS):
    v=arr(M,f"EBI_Norm_{yr}"); m=~np.isnan(v)
    vv=v[m]; lo_,la_=lon[m],lat[m]
    q=np.quantile(vv,[.25,.5,.75]); qi=np.digitize(vv,q)
    for k in range(4):
        sel=qi==k
        ax.scatter(lo_[sel],la_[sel],s=16,c=QC[k],edgecolors="none",label=f"{QL[k]} (n={sel.sum()})")
    draw_hull(ax)
    ax.set_title(f"EBI {yr} Quartiles"); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_aspect("equal"); ax.grid(alpha=.25); ax.legend(fontsize=8,loc="upper left")
fig.suptitle("EBI Quartile Spatial Distribution by Year (GPS-Corrected, aligned)",fontsize=14,fontweight="bold")
plt.tight_layout(rect=[0,0,1,.99]); fig.savefig(f"{RA}/Cultivar_Segments_Corrected.png",dpi=140,bbox_inches="tight"); plt.close()
print("✓ Cultivar_Segments_Corrected")

# ---------- 4. Orchard_Map_Y0_Corrected (source + cultivar) ----------
# source: trees present in Time-series master (1291) = original; else boundary-recovered
ts_xy=set((round(r["X_UTM"],2),round(r["Y_UTM"],2)) for r in TS if r.get("X_UTM") is not None)
is_orig=np.array([ (round(x,2),round(y,2)) in ts_xy for x,y in zip(X,Y)])
fig,axes=plt.subplots(1,2,figsize=(18,8))
ax=axes[0]
ax.scatter(lon[is_orig],lat[is_orig],s=14,c="#4C9BD4",edgecolors="none",label=f"Original GPS trees (n={is_orig.sum()})")
ax.scatter(lon[~is_orig],lat[~is_orig],s=26,c="#E8853B",marker="*",edgecolors="none",label=f"Boundary-recovered trees (n={(~is_orig).sum()})")
ax.plot(HLon,HLat,color="#222",lw=1.4,label="Field boundary (tree hull)")
ax.set_title(f"By Tree Source (total n={len(M)})"); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_aspect("equal"); ax.grid(alpha=.25); ax.legend(fontsize=9,loc="upper left")
ax=axes[1]
names={"UEF":"UEF (Unknown European)","53":"Cv 53 (Souri-type)","54":"Cv 54 (Barnea-type)"}
for k in ["UEF","53","54"]:
    mm=cult==k
    ax.scatter(lon[mm],lat[mm],s=14,c=CULTCOL[k],edgecolors="none",label=f"{names[k]} (n={mm.sum()})")
ax.plot(HLon,HLat,color="#222",lw=1.4)
ax.set_title("By Cultivar"); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_aspect("equal"); ax.grid(alpha=.25); ax.legend(fontsize=9,loc="upper left")
fig.suptitle("Orchard Spatial Map — Original Trees & Boundary-Recovered Trees (aligned)",fontsize=15,fontweight="bold")
plt.tight_layout(rect=[0,0,1,.98]); fig.savefig(f"{RA}/Orchard_Map_Y0_Corrected.png",dpi=140,bbox_inches="tight"); plt.close()
print(f"✓ Orchard_Map_Y0_Corrected (orig={is_orig.sum()} recovered={(~is_orig).sum()})")
print("done")
