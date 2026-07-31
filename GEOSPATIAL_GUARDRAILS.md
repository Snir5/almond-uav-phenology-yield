# Geospatial Guardrails — so the coordinate/boundary shift never happens again

**Why this file exists.** Two coordinate bugs bit this project:

1. **The 68 m GPS shift** — the master's stored `Latitude`/`Longitude` were once
   ~68 m east of what `X_UTM`/`Y_UTM` encoded, which quietly corrupted every
   spatial join (e.g. yield matches dropped to 186 instead of ~1,000).
2. **The offset boundary polygon** — several map figures drew a *hard-coded*
   field-boundary outline from the pre-correction survey. The tree points were
   correct, but the outline was drawn shifted, so figures "looked shifted"
   (Analysis18, Fig4, the cultivar crown/segment maps, the orchard map).

Both share the same two bad habits: **trusting coordinates without validating
them**, and **hard-coding geometry/paths instead of deriving them from the data.**
The guardrails below make both bugs impossible to reintroduce silently.

---

## The three rules

**Rule 1 — Never hard-code paths.** No `/sessions/<name>/mnt/...` or machine-
specific absolute paths in scripts. They break the next session. Resolve the
project root dynamically:

```python
from geo_guardrails import find_thesis_root, master_path, load_master
BASE = find_thesis_root()                 # searches upward for the master file
rows = load_master()                      # loads AND validates coordinates
```

**Rule 2 — Never trust the master's coordinates; validate them.** Every load of
the master must confirm that stored `Latitude`/`Longitude` agree with
`X_UTM`/`Y_UTM`. `load_master()` does this automatically and **raises** if the
median residual exceeds 1 m. This is the tripwire that makes the 68 m shift
un-usable: a shifted master can no longer be analysed by accident.

**Rule 3 — Never hard-code a boundary; derive it from the trees.** The only
sanctioned field boundary is the convex hull of the points you are plotting, so
it is aligned by construction and can never be offset:

```python
from geo_guardrails import plot_field_boundary
plot_field_boundary(ax, lon, lat)         # NOT a literal list of lat/lon corners
```

---

## Files that enforce this

| File | Role |
|---|---|
| `pipeline/06_analysis/geo_guardrails.py` | Shared helpers: `find_thesis_root`, `load_master` (validates), `field_boundary`, `plot_field_boundary`, `assert_coords_aligned`. Import these everywhere. |
| `pipeline/06_analysis/check_pipeline.py` | Pre-flight linter. Scans all scripts for stale session paths and hard-coded boundary literals, and runs the master coordinate tripwire. |

### Run the check before regenerating anything

```bash
cd pipeline/06_analysis && python3 check_pipeline.py
```

- Exit 0 + "CHECK PASSED" → safe to regenerate figures/results.
- Exit 1 → it lists every offending script. Fix them (migrate to the helpers)
  before running the analysis.

As of this writing the check **intentionally still flags the legacy analysis
scripts** (they carry old hard-coded session paths). The canonical, current
results come from `thesis_analysis.py` / `climate_yield_ebi.py`; the legacy
scripts should be migrated to `geo_guardrails` the next time each is touched.

---

## Migration pattern (applies to any old script)

```python
# BEFORE  (fragile)
BASE = "/sessions/zealous-elegant-franklin/mnt/Thesis"
master = pd.read_excel(f"{BASE}/4band_mosaic/Master_Trees_Time_Series_Plot1.xlsx")
ax.plot(boundary_lon, boundary_lat, 'r--')     # hard-coded, went stale

# AFTER  (guarded)
from geo_guardrails import find_thesis_root, load_master, plot_field_boundary
BASE = find_thesis_root()
rows = load_master()                            # raises if coordinates are shifted
plot_field_boundary(ax, lon, lat)              # boundary from the trees, always aligned
```

---

## Definition of done for any spatial figure or stat

1. Coordinates come from `load_master()` (validated, aligned).
2. Any boundary/outline is drawn with `plot_field_boundary()` (tree hull).
3. `python3 check_pipeline.py` passes.
4. If it plots yield, it says whether it is **measured** (`kedma_plot_a_yield.csv`)
   or **modeled** (`predicted_Yield`) — never conflate them (see the results log).
