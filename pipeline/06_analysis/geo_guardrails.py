"""
geo_guardrails.py — shared helpers that PREVENT the two bugs that bit this project:
  (1) stale/shifted coordinates (the 68 m GPS error), and
  (2) a hard-coded, offset plot-boundary polygon overlaid on maps.

USE THESE IN EVERY MAP OR SPATIAL-STATS SCRIPT. Do not hard-code paths or a
boundary polygon. Import from here instead:

    from geo_guardrails import find_thesis_root, load_master, field_boundary, \
        plot_field_boundary, assert_coords_aligned

Rules enforced:
  * Paths are resolved dynamically (no /sessions/<name>/mnt/... literals).
  * The master's stored Latitude/Longitude are validated against X_UTM/Y_UTM
    every load; a residual > 1 m raises — so a shifted master can never be used
    silently again.
  * The field boundary is ALWAYS the convex hull of the actual trees, so it
    cannot be offset from the points.
"""
import os, math
import numpy as np

# ─────────────────────────── path resolution ───────────────────────────
def find_thesis_root(start=None):
    """Walk upward from this file (or `start`, or CWD) until the folder that
    contains 4band_mosaic/Master_Trees_Extended.xlsx is found. Honors the
    THESIS_ROOT env var first. NEVER hard-code a /sessions/.../mnt path."""
    env = os.environ.get("THESIS_ROOT")
    if env and os.path.exists(env):
        return env
    marker = os.path.join("4band_mosaic", "Master_Trees_Extended.xlsx")
    seeds = [start, os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    for seed in filter(None, seeds):
        d = seed
        for _ in range(8):
            if os.path.exists(os.path.join(d, marker)):
                return d
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    raise FileNotFoundError(
        "Thesis root not found. Set THESIS_ROOT env var, or run from inside the "
        "Thesis tree. Do NOT hard-code a session path.")

def master_path(name="Master_Trees_Extended.xlsx"):
    return os.path.join(find_thesis_root(), "4band_mosaic", name)

# ─────────────────────────── UTM 36N → WGS84 ───────────────────────────
def utm36n_to_latlon(E, N):
    a=6378137.0; f=1/298.257223563; e2=f*(2-f); k0=0.9996
    e1=(1-math.sqrt(1-e2))/(1+math.sqrt(1-e2)); x=E-500000.0; M=N/k0
    mu=M/(a*(1-e2/4-3*e2**2/64-5*e2**3/256))
    phi1=mu+(3*e1/2-27*e1**3/32)*math.sin(2*mu)+(21*e1**2/16-55*e1**4/32)*math.sin(4*mu)+(151*e1**3/96)*math.sin(6*mu)
    e_2=e2/(1-e2); C1=e_2*math.cos(phi1)**2; T1=math.tan(phi1)**2
    N1=a/math.sqrt(1-e2*math.sin(phi1)**2); R1=a*(1-e2)/(1-e2*math.sin(phi1)**2)**1.5; D=x/(N1*k0)
    lat=phi1-(N1*math.tan(phi1)/R1)*(D**2/2-(5+3*T1+10*C1-4*C1**2-9*e_2)*D**4/24+(61+90*T1+298*C1+45*T1**2-252*e_2-3*C1**2)*D**6/720)
    lon0=math.radians(36*6-183)
    lon=lon0+(D-(1+2*T1+C1)*D**3/6+(5-2*C1+28*T1-3*C1**2+8*e_2+24*T1**2)*D**5/120)/math.cos(phi1)
    return math.degrees(lat), math.degrees(lon)

# ─────────────────────────── coordinate validation ───────────────────────────
def assert_coords_aligned(X_UTM, Y_UTM, Latitude, Longitude, tol_m=1.0):
    """Raise if stored lat/lon disagree with X_UTM/Y_UTM by more than tol_m.
    This is the tripwire that makes the 68 m shift impossible to use silently."""
    X=np.asarray(X_UTM,float); Y=np.asarray(Y_UTM,float)
    la=np.asarray(Latitude,float); lo=np.asarray(Longitude,float)
    rec=np.array([utm36n_to_latlon(x,y) for x,y in zip(X,Y)])
    mlat=111320.0; mlon=111320.0*math.cos(math.radians(np.nanmean(la)))
    de=np.abs(lo-rec[:,1])*mlon; dn=np.abs(la-rec[:,0])*mlat
    med=float(np.nanmedian(np.hypot(de,dn))); mx=float(np.nanmax(np.hypot(de,dn)))
    if med>tol_m:
        raise ValueError(f"COORDINATE SHIFT DETECTED: stored lat/lon vs UTM median "
                         f"offset={med:.2f} m (max {mx:.2f} m) > {tol_m} m. "
                         f"The master is not GPS-aligned — do NOT run analyses on it.")
    return med, mx

def load_master(name="Master_Trees_Extended.xlsx", validate=True):
    """Load the canonical master and (by default) validate coordinate alignment."""
    import openpyxl
    wb=openpyxl.load_workbook(master_path(name), read_only=True, data_only=True)
    ws=wb.active; raw=list(ws.iter_rows(values_only=True)); H=raw[0]
    rows=[dict(zip(H,r)) for r in raw[1:]]; wb.close()
    if validate:
        col=lambda c:[r.get(c) for r in rows]
        assert_coords_aligned(col("X_UTM"),col("Y_UTM"),col("Latitude"),col("Longitude"))
    return rows

# ─────────────────────────── boundary from trees (never hard-coded) ───────────────────────────
def field_boundary(px, py):
    """Convex hull of the tree points — the ONLY sanctioned field boundary.
    Because it is computed from the same points being plotted, it can never be
    offset. Returns (hull_x, hull_y) closed ring."""
    px=np.asarray(px,float); py=np.asarray(py,float)
    g=~np.isnan(px)&~np.isnan(py); pts=sorted(set(zip(px[g],py[g])))
    if len(pts)<3: return list(px[g]), list(py[g])
    def cr(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lo=[]
    for p in pts:
        while len(lo)>=2 and cr(lo[-2],lo[-1],p)<=0: lo.pop()
        lo.append(p)
    up=[]
    for p in reversed(pts):
        while len(up)>=2 and cr(up[-2],up[-1],p)<=0: up.pop()
        up.append(p)
    h=lo[:-1]+up[:-1]
    return [p[0] for p in h]+[h[0][0]], [p[1] for p in h]+[h[0][1]]

def plot_field_boundary(ax, px, py, **kw):
    """Draw the tree-derived boundary on an axis. NEVER pass literal coordinates."""
    hx,hy=field_boundary(px,py)
    style=dict(color="#555", lw=1.1, ls="--", alpha=.6); style.update(kw)
    ax.plot(hx,hy,**style); return hx,hy

if __name__=="__main__":
    rows=load_master()
    col=lambda c:[r.get(c) for r in rows]
    med,mx=assert_coords_aligned(col("X_UTM"),col("Y_UTM"),col("Latitude"),col("Longitude"))
    print(f"OK: {len(rows)} trees, coord residual median={med:.3f} m max={mx:.3f} m")
    print("Thesis root:", find_thesis_root())
