#!/usr/bin/env python3
"""
check_pipeline.py — pre-flight guard. Run this BEFORE regenerating figures/results.
It fails (non-zero exit) if it finds the patterns that caused past shift bugs:

  1. Hard-coded session paths  (/sessions/<name>/mnt/...)  -> break next session.
  2. Hard-coded boundary coordinate literals (lat ~31.70 / lon ~34.79) used to
     draw a polygon, instead of geo_guardrails.field_boundary().
  3. A master that fails coordinate-alignment validation (the 68 m tripwire).

Usage:  python check_pipeline.py
"""
import os, re, sys, glob
HERE=os.path.dirname(os.path.abspath(__file__))
PIPE=os.path.dirname(HERE)                      # .../pipeline
problems=[]

# --- 1 & 2: scan scripts (skip archive_diagnostics + this file + guardrails) ---
SKIP={"check_pipeline.py","geo_guardrails.py"}
session_re=re.compile(r"/sessions/[A-Za-z0-9_-]+/mnt")
bnd_re=re.compile(r"(31\.70\d|34\.79\d)")
for path in glob.glob(os.path.join(PIPE,"**","*.py"),recursive=True):
    if "archive_diagnostics" in path or os.path.basename(path) in SKIP:
        continue
    txt=open(path,encoding="utf-8",errors="ignore").read()
    rel=os.path.relpath(path,PIPE)
    if session_re.search(txt):
        problems.append(f"[STALE PATH] {rel}: hard-coded /sessions/.../mnt path — "
                        f"use geo_guardrails.find_thesis_root() instead.")
    # boundary literals only matter if the file also plots a line/polygon
    if bnd_re.search(txt) and re.search(r"\.plot\(|Polygon|fill\(|add_patch", txt):
        # allowed if it uses the sanctioned helper
        if "field_boundary" not in txt and "plot_field_boundary" not in txt:
            problems.append(f"[HARD-CODED BOUNDARY] {rel}: coordinate literals near a "
                            f"plotted shape — derive the boundary from the trees via "
                            f"geo_guardrails.field_boundary().")

# --- 3: master coordinate alignment tripwire ---
try:
    sys.path.insert(0,HERE)
    from geo_guardrails import load_master, assert_coords_aligned
    rows=load_master(validate=False)
    col=lambda c:[r.get(c) for r in rows]
    med,mx=assert_coords_aligned(col("X_UTM"),col("Y_UTM"),col("Latitude"),col("Longitude"))
    print(f"master coord alignment: median={med:.3f} m max={mx:.3f} m  (n={len(rows)})  OK")
except Exception as e:
    problems.append(f"[COORD SHIFT] master failed alignment check: {e}")

# --- report ---
print("="*70)
if problems:
    print(f"CHECK FAILED — {len(problems)} issue(s):")
    for p in problems: print("  -",p)
    sys.exit(1)
print("CHECK PASSED — no stale paths, no hard-coded boundaries, master aligned.")
