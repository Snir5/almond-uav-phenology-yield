#!/usr/bin/env python3
"""
coord_audit.py — Coordinate alignment + join provenance audit across ALL sources.
No geopandas/rasterio (sandbox has no network). Uses sqlite3 + manual TIFF/UTM.
"""
import sqlite3, struct, os, math, json
import numpy as np
import openpyxl

BASE = "/sessions/trusting-relaxed-gates/mnt/Thesis"
GPKG_DIR = f"{BASE}/Trees_Data_Survey/Field_Data_Yield_GPKGs"
OUT = "/sessions/trusting-relaxed-gates/mnt/outputs"

# ---------- accurate UTM (WGS84) -> lat/lon ----------
def utm_to_latlon(E, N, zone=36, north=True):
    a = 6378137.0; f = 1/298.257223563
    e2 = f*(2-f); e = math.sqrt(e2); k0 = 0.9996
    e1 = (1-math.sqrt(1-e2))/(1+math.sqrt(1-e2))
    x = E - 500000.0; y = N if north else N-10000000.0
    M = y/k0
    mu = M/(a*(1-e2/4-3*e2**2/64-5*e2**3/256))
    ep = (3*e1/2-27*e1**3/32)*math.sin(2*mu)+(21*e1**2/16-55*e1**4/32)*math.sin(4*mu)+(151*e1**3/96)*math.sin(6*mu)
    phi1 = mu+ep
    e_2 = e2/(1-e2)
    C1 = e_2*math.cos(phi1)**2
    T1 = math.tan(phi1)**2
    N1 = a/math.sqrt(1-e2*math.sin(phi1)**2)
    R1 = a*(1-e2)/(1-e2*math.sin(phi1)**2)**1.5
    D = x/(N1*k0)
    lat = phi1-(N1*math.tan(phi1)/R1)*(D**2/2-(5+3*T1+10*C1-4*C1**2-9*e_2)*D**4/24+(61+90*T1+298*C1+45*T1**2-252*e_2-3*C1**2)*D**6/720)
    lon0 = math.radians(zone*6-183)
    lon = lon0+(D-(1+2*T1+C1)*D**3/6+(5-2*C1+28*T1-3*C1**2+8*e_2+24*T1**2)*D**5/120)/math.cos(phi1)
    return math.degrees(lat), math.degrees(lon)

def rng(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return (float(a.min()), float(a.max())) if len(a) else (None,None)

# ---------- load masters ----------
def load_master(fn):
    wb = openpyxl.load_workbook(fn, read_only=True, data_only=True)
    ws = wb.active; raw = list(ws.iter_rows(values_only=True)); h = raw[0]
    rows = [dict(zip(h, r)) for r in raw[1:]]; wb.close()
    return rows, h

report = []
def L(*a):
    s=" ".join(str(x) for x in a); report.append(s); print(s)

L("="*80); L("COORDINATE ALIGNMENT & JOIN PROVENANCE AUDIT"); L("="*80)

ext, hext = load_master(f"{BASE}/4band_mosaic/Master_Trees_Extended.xlsx")
ts,  hts  = load_master(f"{BASE}/4band_mosaic/Master_Trees_Time_Series_Plot1.xlsx")

def col(rows, c): return np.array([r.get(c) if r.get(c) is not None else np.nan for r in rows], float)

for name, rows in [("Extended(1523)", ext), ("TimeSeries(1291)", ts)]:
    xs, ys = col(rows,"X_UTM"), col(rows,"Y_UTM")
    la, lo = col(rows,"Latitude"), col(rows,"Longitude")
    # recompute lat/lon from UTM
    rec = [utm_to_latlon(x,y) for x,y in zip(xs,ys)]
    rla = np.array([r[0] for r in rec]); rlo = np.array([r[1] for r in rec])
    dlat = (la-rla); dlon=(lo-rlo)
    # convert deg error to meters (approx at 31.7N)
    mlat = 111320.0; mlon = 111320.0*math.cos(math.radians(31.7))
    off_e = np.nanmedian(dlon)*mlon; off_n = np.nanmedian(dlat)*mlat
    L(f"\n### MASTER {name}")
    L(f"  X_UTM range: {rng(xs)}   Y_UTM range: {rng(ys)}")
    L(f"  Stored Lat : {rng(la)}   Stored Lon : {rng(lo)}")
    L(f"  Recomputed from UTM -> Lat {rng(rla)}  Lon {rng(rlo)}")
    L(f"  Stored-vs-recomputed median offset:  E={off_e:+.2f} m   N={off_n:+.2f} m")
    L(f"  max |offset| E={np.nanmax(np.abs(dlon))*mlon:.2f}m  N={np.nanmax(np.abs(dlat))*mlat:.2f}m")

# ---------- GPKG centroids via rtree (matches pipeline) ----------
def gpkg_centroids(fn, table):
    con=sqlite3.connect(fn)
    rows=con.execute(f'SELECT id,minx,maxx,miny,maxy FROM "rtree_{table}_geom"').fetchall()
    cent={r[0]:((r[1]+r[2])/2,(r[3]+r[4])/2) for r in rows}
    cols=[c[1] for c in con.execute(f'PRAGMA table_info("{table}")').fetchall()]
    attr=[c for c in cols if c not in ('fid','geom')]
    q=",".join(f'"{c}"' for c in attr)
    data=[]
    for row in con.execute(f'SELECT fid,{q} FROM "{table}"').fetchall():
        fid=row[0]
        if fid in cent:
            cx,cy=cent[fid]; d={'fid':fid,'cx':cx,'cy':cy}; d.update(dict(zip(attr,row[1:]))); data.append(d)
    con.close(); return data, attr

gp = {}
for fn,tab in [("Yield_with_clustering_2022.gpkg","Yield_with_clustering_2022"),
               ("Yield_with_clustering_2023.gpkg","Yield_with_clustering_2023"),
               ("spectral_data_plot_A_31072024.gpkg","spectral_data_plot_A_31072024")]:
    data,attr=gpkg_centroids(os.path.join(GPKG_DIR,fn),tab)
    gp[tab]=data
    cx=np.array([d['cx'] for d in data]); cy=np.array([d['cy'] for d in data])
    plots=set(d.get('Plot') for d in data)
    ll=[utm_to_latlon(x,y) for x,y in zip(cx,cy)]
    la=np.array([p[0] for p in ll]); lo=np.array([p[1] for p in ll])
    L(f"\n### GPKG {tab}  (SRS=EPSG:32636, n={len(data)})  plots={plots}")
    L(f"  cx range: {rng(cx)}  cy range: {rng(cy)}")
    L(f"  -> Lat {rng(la)}  Lon {rng(lo)}")

# ---------- yield CSV ----------
import csv
with open(f"{BASE}/Trees_Data_Survey/kedma_plot_a_yield.csv",encoding="utf-8-sig") as f:
    yc=list(csv.DictReader(f))
ylat=np.array([float(r["latitude"]) for r in yc]); ylon=np.array([float(r["longitude"]) for r in yc])
yyears=sorted(set(int(r["year"]) for r in yc))
L(f"\n### YIELD CSV kedma_plot_a_yield.csv  (WGS84, n={len(yc)} rows, years={yyears})")
L(f"  Lat {rng(ylat)}  Lon {rng(ylon)}  (measured net_kernel_yield_per_tree_kg)")
from collections import Counter
L(f"  rows/year: {dict(Counter(int(r['year']) for r in yc))}")
L(f"  unique tree_id: {len(set(r['tree_id'] for r in yc))}")

# ---------- mosaic TFW footprints ----------
def read_tiff_dims(path):
    with open(path,'rb') as f:
        h=f.read(16)
    bo='<' if h[:2]==b'II' else '>'
    magic=struct.unpack(bo+'H',h[2:4])[0]
    def rd(path,off,n):
        with open(path,'rb') as f: f.seek(off); return f.read(n)
    if magic==42:  # classic
        ifd=struct.unpack(bo+'I',h[4:8])[0]
        cnt=struct.unpack(bo+'H',rd(path,ifd,2))[0]
        entries=rd(path,ifd+2,cnt*12)
        w=ht=None
        for i in range(cnt):
            e=entries[i*12:i*12+12]
            tag,typ=struct.unpack(bo+'HH',e[:4]); val=struct.unpack(bo+'I',e[8:12])[0]
            if typ==3: val=struct.unpack(bo+'H',e[8:10])[0]
            if tag==256:w=val
            if tag==257:ht=val
        return w,ht
    elif magic==43:  # BigTIFF
        ifd=struct.unpack(bo+'Q',h[8:16])[0]
        cnt=struct.unpack(bo+'Q',rd(path,ifd,8))[0]
        entries=rd(path,ifd+8,cnt*20)
        w=ht=None
        for i in range(cnt):
            e=entries[i*20:i*20+20]
            tag,typ=struct.unpack(bo+'HH',e[:4]); val=struct.unpack(bo+'Q',e[12:20])[0]
            if typ==3: val=struct.unpack(bo+'H',e[12:14])[0]
            if typ==4: val=struct.unpack(bo+'I',e[12:16])[0]
            if tag==256:w=val
            if tag==257:ht=val
        return w,ht
    return None,None

mos = {
 2021:("Final_Exports_2021_03_07","Final_Orthomosaic_RGB"),
 2022:("Final_Exports_2022_03_02","Final_Orthomosaic_RGB"),
 2024:("Final_Exports_2024_02_29","Final_Orthomosaic_RGB"),
}
# 2023 dir
d23=os.path.join(BASE,"4band_mosaic","Final_Exports_2023_03_01")
L(f"\n### MOSAIC TFW FOOTPRINTS")
mos_ul={}
for yr,(d,stem) in sorted(mos.items()):
    tfw=os.path.join(BASE,"4band_mosaic",d,stem+".tfw")
    tif=os.path.join(BASE,"4band_mosaic",d,stem+".tif")
    if not os.path.exists(tfw): L(f"  {yr}: TFW missing"); continue
    vals=[float(x) for x in open(tfw).read().split()]
    px,rot1,rot2,py,ulx,uly=vals
    w,ht=read_tiff_dims(tif) if os.path.exists(tif) else (None,None)
    ul=utm_to_latlon(ulx,uly)
    mos_ul[yr]=(ulx,uly)
    ext_txt=""
    if w:
        lrx=ulx+px*w; lry=uly+py*ht
        lr=utm_to_latlon(lrx,lry)
        ext_txt=f" | LR_X={lrx:.1f} LR_Y={lry:.1f} ({w}x{ht}px @ {px:.4f}m)"
    L(f"  {yr}: UL_X={ulx:.1f} UL_Y={uly:.1f}  pix={px:.4f}m  UL_lat={ul[0]:.5f} UL_lon={ul[1]:.5f}{ext_txt}")
# 2023
for stem in ["Final_Orthomosaic_RGB","Final_Orthomosaic_4Band"]:
    tfw=os.path.join(d23,stem+".tfw")
    if os.path.exists(tfw):
        vals=[float(x) for x in open(tfw).read().split()]
        px,_,_,py,ulx,uly=vals; ul=utm_to_latlon(ulx,uly); mos_ul[2023]=(ulx,uly)
        L(f"  2023({stem}): UL_X={ulx:.1f} UL_Y={uly:.1f} pix={px:.4f}m UL_lat={ul[0]:.5f} UL_lon={ul[1]:.5f}")
        break
if len(mos_ul)>=2:
    ref=mos_ul.get(2021)
    if ref:
        L("  Inter-year UL shift vs 2021:")
        for yr in sorted(mos_ul):
            if yr==2021: continue
            L(f"    {yr}: ΔE={mos_ul[yr][0]-ref[0]:+.1f}m  ΔN={mos_ul[yr][1]-ref[1]:+.1f}m")

# ---------- ALIGNMENT: master vs each source (UTM NN) ----------
def nn_stats(ax,ay,bx,by,label,thr=(2,5,8)):
    ax,ay,bx,by=map(np.asarray,(ax,ay,bx,by))
    d=np.full(len(ax),np.inf)
    for i in range(len(ax)):
        dd=np.hypot(bx-ax[i],by-ay[i]); d[i]=dd.min()
    L(f"\n  [{label}]  n_query={len(ax)} vs n_ref={len(bx)}")
    L(f"    NN dist: mean={np.mean(d):.2f} median={np.median(d):.2f} max={np.max(d):.2f} m")
    for t in thr:
        L(f"    within {t}m: {(d<=t).sum()} ({100*(d<=t).mean():.1f}%)")
    return d

ex_x=col(ext,"X_UTM"); ex_y=col(ext,"Y_UTM")
L(f"\n### ALIGNMENT (UTM nearest-neighbour, Extended master as query)")
# yield gpkg plot A
for tab in ["Yield_with_clustering_2022","Yield_with_clustering_2023"]:
    dd=[d for d in gp[tab] if d.get('Plot')=='A']
    nn_stats(ex_x,ex_y,[d['cx'] for d in dd],[d['cy'] for d in dd],f"master vs {tab} (Plot A, n={len(dd)})")
# spectral
sp=gp["spectral_data_plot_A_31072024"]
nn_stats(ex_x,ex_y,[d['cx'] for d in sp],[d['cy'] for d in sp],"master vs spectral_Jul2024")
# yield CSV -> need UTM: convert master lat/lon? CSV is WGS84; compare in lat/lon via haversine done separately
# ---------- YIELD PROVENANCE ----------
L(f"\n"+"="*80); L("YIELD PROVENANCE INVESTIGATION"); L("="*80)
for c in ["Yield_2022","Yield_2023","Yield_2024","predicted_Yield_2022","predicted_Yield_2023"]:
    v=col(ext,c); vv=v[~np.isnan(v)]
    if len(vv):
        L(f"  {c:24} n={len(vv):5d}  mean={vv.mean():.3f}  sd={vv.std():.3f}  range=({vv.min():.2f},{vv.max():.2f})")
    else:
        L(f"  {c:24} n=0")
# GPKG predicted_Yield Plot A distribution
for tab in ["Yield_with_clustering_2022","Yield_with_clustering_2023"]:
    pv=np.array([float(d['predicted_Yield']) for d in gp[tab] if d.get('Plot')=='A' and d.get('predicted_Yield') is not None])
    L(f"  GPKG {tab} predicted_Yield PlotA: n={len(pv)} mean={pv.mean():.3f} sd={pv.std():.3f} range=({pv.min():.2f},{pv.max():.2f})")
# CSV measured distribution per year
for yr in yyears:
    mv=np.array([float(r["net_kernel_yield_per_tree_kg"]) for r in yc if int(r["year"])==yr])
    L(f"  CSV measured net_kernel {yr}: n={len(mv)} mean={mv.mean():.3f} sd={mv.std():.3f} range=({mv.min():.2f},{mv.max():.2f})")

# Does master Yield_* == GPKG predicted (i.e. mislabeled)?  compare distributions
L("\n  --> Compare master Yield_2022 vs predicted_Yield_2022 correlation (paired, same rows):")
y=col(ext,"Yield_2022"); p=col(ext,"predicted_Yield_2022")
m=~np.isnan(y)&~np.isnan(p)
if m.sum()>2:
    yy,pp=y[m],p[m]; r=np.corrcoef(yy,pp)[0,1]
    L(f"     n_paired={m.sum()}  r(Yield_2022, predicted_Yield_2022)={r:.4f}  mean_diff={np.mean(yy-pp):+.3f}")

with open(f"{OUT}/coordinate_audit_report.txt","w") as f:
    f.write("\n".join(report))
L(f"\n✓ Written coordinate_audit_report.txt")
