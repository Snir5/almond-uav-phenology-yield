#!/usr/bin/env python3
"""
build_uef53_deck.py — self-contained English HTML deck for the UEF + 53 restricted
rerun (Chill Portions only). Style mirrors the main deck: fixed grouped sidebar,
hero header, colored section icons, a two-part explanation under every figure
("What & how" primer + "Reading it" interpretation), and a final Results Summary of
interpretive cards. Figures are embedded as base64. Path-robust via geo_guardrails.
Output: Analysis_Presentation_UEF53.html at the thesis root.
"""
import os, sys, json, base64
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root
BASE = find_thesis_root()
FIG = os.path.join(BASE, "Results_Analysis", "08_UEF53_Rerun")
R = json.load(open(os.path.join(BASE, "Results_Analysis", "thesis_results_uef53.json")))


def b64(name):
    with open(os.path.join(FIG, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


IMG = {n: b64(n + ".png") for n in
       ["F1_EBI_Temporal_UEF53", "F2_Cultivar_Effects_UEF53", "F3_Moran_Spatial_UEF53",
        "F4_Climate_CP_EBI_UEF53", "F5_Measured_Yield_UEF53",
        "H1_Climate_EBI_UEF53", "H_Validation_PredVsMeas_UEF53", "H3_EBI_Yield_UEF53",
        "H_Phenology_Support_UEF53", "H_Panel_3yr_UEF53",
        "C1_EBI_Yield_byCultivar", "C2_Yield_Validation_byCultivar", "C3_Phenology_byCultivar",
        "Climate_EBI_Yield_Triangle_byCultivar", "Yield_Model_Features", "Tested_Trees_By_Year",
        "Dormancy_Models", "Color_Indices_Yield", "Bloom_Fraction_Yield", "Outlier_Robust_EBI",
        "Jul2024_Spectral_Yield",
        "Phenology_Flight_Timing", "Phenology_Insights", "Phenology_Bloom_Normalization",
        "GPKG_Physiology_Yield", "Yield_Model_v2_AllGPKG", "Yield_Model_v3_FeatureEngineering",
        "MDS_Feature_Test", "Climate_Full_Audit", "Predicted_Yield_Model_Check",
        "Predicted_Yield_Model_FullFeatures",
        "Yield_Model_All_EBI_Variants", "Yield_Model_EBI_Interactions",
        "Yield_Model_All_Pairwise_Interactions", "EBI_Normalization_Scope_Check",
        "EBI_WholeImageBaseline_Comparison", "Phenology_Baseline_Adjudication",
        "EBI_Baseline_TreeLevel_Yield_Effect"]}
HR = json.load(open(os.path.join(BASE, "Results_Analysis", "hypothesis_tests_uef53.json")))
BC = json.load(open(os.path.join(BASE, "Results_Analysis", "hypothesis_by_cultivar_uef53.json")))
TRI = json.load(open(os.path.join(BASE, "Results_Analysis", "climate_ebi_yield_by_cultivar.json")))
YM = json.load(open(os.path.join(BASE, "Results_Analysis", "yield_model_prototype.json")))
DM = json.load(open(os.path.join(FIG, "dormancy_models.json")))
CIX = json.load(open(os.path.join(FIG, "color_indices_yield.json")))
BFY = json.load(open(os.path.join(FIG, "bloom_fraction_yield.json")))
ORY = json.load(open(os.path.join(FIG, "outlier_robust_yield.json")))
J24 = json.load(open(os.path.join(FIG, "jul2024_spectral_yield.json")))
YMPT = json.load(open(os.path.join(FIG, "Yield_Model_Phenology_Test.json")))
GPY = json.load(open(os.path.join(FIG, "gpkg_physiology_yield.json")))
YMv2 = json.load(open(os.path.join(FIG, "Yield_Model_v2_AllGPKG.json")))
YMv3 = json.load(open(os.path.join(FIG, "Yield_Model_v3_FeatureEngineering.json")))
MDSJ = json.load(open(os.path.join(FIG, "MDS_Feature_Test.json")))
CFA = json.load(open(os.path.join(FIG, "Climate_Full_Audit.json")))
PYC = json.load(open(os.path.join(FIG, "Predicted_Yield_Model_Check.json")))
PYF = json.load(open(os.path.join(FIG, "Predicted_Yield_Model_FullFeatures.json")))
AEV = json.load(open(os.path.join(FIG, "Yield_Model_All_EBI_Variants.json")))
AEBI = json.load(open(os.path.join(FIG, "Yield_Model_EBI_Interactions.json")))
APW = json.load(open(os.path.join(FIG, "Yield_Model_All_Pairwise_Interactions.json")))
NSC = json.load(open(os.path.join(FIG, "EBI_Normalization_Scope_Check.json")))
WIC = json.load(open(os.path.join(FIG, "EBI_WholeImageBaseline_Comparison.json")))
PBA = json.load(open(os.path.join(FIG, "Phenology_Baseline_Adjudication.json")))
TLY = json.load(open(os.path.join(FIG, "EBI_Baseline_TreeLevel_Yield_Effect.json")))
PFT = json.load(open(os.path.join(FIG, "Phenology_Flight_Timing.json")))
PIN = json.load(open(os.path.join(FIG, "Phenology_Insights.json")))
PBN = json.load(open(os.path.join(FIG, "Phenology_Bloom_Normalization.json")))

SECTIONS = [
    ("overview", "Overview", "#5b6cff", "&#9635;"),
    ("ebi", "EBI dynamics", "#C0392B", "&#127799;"),
    ("cultivar", "Cultivar", "#8E44AD", "&#129003;"),
    ("spatial", "Spatial", "#16A085", "&#128506;"),
    ("climate", "Climate (CP)", "#4aa3df", "&#10052;"),
    ("dormancy", "Dormancy models", "#4aa3df", "&#127807;"),
    ("phenologynorm", "Bloom timing & EBI", "#4aa3df", "&#128337;"),
    ("yield", "Yield", "#E67E22", "&#127806;"),
    ("hypotheses", "Hypothesis tests", "#C0392B", "&#128300;"),
    ("bycultivar", "By cultivar", "#8E44AD", "&#9878;"),
    ("triangle", "Climate-EBI-Yield", "#16A085", "&#9651;"),
    ("yieldmodel", "Yield model", "#E67E22", "&#128200;"),
    ("colorindices", "Color indices", "#8E44AD", "&#127752;"),
    ("leverage", "Leveraging RGB", "#2C3E50", "&#128269;"),
    ("physiology", "Ground physiology", "#2C3E50", "&#128167;"),
    ("summary", "Results summary", "#2C3E50", "&#9733;"),
]


def figure_block(img, cap, what, reading):
    return f"""
    <figure class="fig">
      <img src="{img}" alt="{cap}"/>
      <figcaption>{cap}</figcaption>
      <div class="explain">
        <div class="what"><span class="tag">What &amp; how</span><p>{what}</p></div>
        <div class="reading"><span class="tag tag2">Reading it</span><p>{reading}</p></div>
      </div>
    </figure>"""


def card(title, body, color):
    return f'<div class="card" style="border-top:4px solid {color}"><h4>{title}</h4><p>{body}</p></div>'


nav = "".join(
    f'<a href="#{sid}"><span class="dot" style="background:{col}">{ic}</span>{label}</a>'
    for sid, label, col, ic in SECTIONS)

d = R["S2_ebi_dist"]
moran = R["S4_moran"]
anc = R["S6_ANCOVA_yield2023"]
wc = R["S6_within_cultivar_slope"]
yc = R["S6_yield_by_cultivar"]
exc = R["S5_exceedance"]

overview = f"""
<section id="overview" class="section">
  <h2><span class="sic" style="background:#5b6cff">&#9635;</span>Overview</h2>
  <p class="lead">This analysis focuses on the two cultivars with field-measured yield,
  <b>UEF (1,045 trees)</b> and <b>cultivar 53 (249 trees)</b>. Cultivar 54 (an early, bloom-bright
  pollinizer row with no harvest data) is dropped everywhere. Chilling is quantified only by
  <b>Chill Portions</b> (Dynamic Model, Fishman et al. 1987); the legacy Chill Hours metric is excluded.</p>
  <div class="grid3">
    {card("Coordinates validated", "Every tree's stored lat/lon is checked against its UTM position on load; the residual is a median 0.000 m across all 1,523 trees.", "#16A085")}
    {card("Yield linked to trees", "Field-measured yield is matched to trees spatially at a median 0.88 m (161 of 162 trees); every matched tree is UEF or cultivar 53.", "#16A085")}
    {card("Cross-year EBI caveat", "EBI values are comparable within a year, but the cross-year mean trend depends materially on the radiometric baseline: a full re-extraction found the canopy-only (canonical) vs whole-image baseline give mean-EBI trajectories nearly 5x apart in range, an open question, not yet resolved.", "#E67E22")}
    {card("Reproducible pipeline", "All analyses resolve their paths dynamically and validate coordinate alignment on every run, so results regenerate cleanly.", "#4aa3df")}
    {card("Data-driven metrics", "Bloom-bright fractions and every summary statistic are computed directly from the tree data rather than fixed in advance.", "#4aa3df")}
    {card("Focus on UEF and 53", "Restricting to the two yield-measured cultivars clarifies the bloom, climate and yield relationships while keeping the results comparable to ground truth.", "#8E44AD")}
  </div>
</section>"""

hyp_block = f"""
  <div style="background:#fff7ef;border-left:5px solid #E67E22;border-radius:10px;padding:16px 20px;margin:8px 0 18px">
    <p style="margin:0;font-size:16px"><b>Hypothesis: Bloom intensity (EBI) affects almond yield in a
    cultivar-specific, opposite-signed way, brighter bloom raises yield in UEF and lowers it in cultivar 53.
    Bloom is itself a genetically structured, spatially organized trait, and the seasonal (climate and
    management) effect sets the orchard-wide yield level.</b></p>
  </div>
  <p class="lead">Seven predictions follow, each tested below (the season/climate prediction is supported
  directionally, since Chill Portions is one value per year). Multiple comparisons are FDR-controlled and
  every statistic was independently re-verified.</p>
  <div class="grid3">
    {card("P1. Cultivar structures bloom", "EBI differs strongly by cultivar and the gap narrows over time: ANOVA F=527 (2021) to ns (2024); split-plot cultivar×year F=447, p&lt;0.001.", "#8E44AD")}
    {card("P2. Bloom is spatially clustered", "Moran's I 0.23 to 0.35, p&lt;0.01 every year, peaking 2023: bloom is organized in space, not random.", "#16A085")}
    {card("P3. Bloom synchronizes over time", "Within-cultivar EBI spread falls significantly (Levene F=56 and 28, p&lt;0.001); the orchard blooms more uniformly each year.", "#16A085")}
    {card("P4. Cultivar drives yield", "Field-measured yield differs by cultivar: ANOVA F=13.46, p=0.0003 (UEF above cultivar 53).", "#E67E22")}
    {card("P5. Bloom affects yield oppositely by cultivar", "EBI×cultivar interaction F=11.58, p=0.0009; within UEF bloom raises yield (r=+0.29, p=0.007) and within cultivar 53 bloom lowers it (r=-0.26, p=0.042).", "#C0392B")}
    {card("P6. The season sets the yield level", "Repeated-measures year effect F=27, p&lt;1e-7; joint-model year/climate effect F=32 to 38, p&lt;0.001 (directional on climate itself, n&le;4).", "#4aa3df")}
    {card("P7. EBI measures bloom, not vigor", "EBI trades off with canopy greenness (NGRDI r=-0.4 to -0.58, p&lt;0.001) and NGRDI does not predict yield, so the signal is bloom-specific.", "#2C3E50")}
  </div>
  <p style="margin-top:14px;font-size:13.5px;color:#555"><b>Stated as not supported:</b> winter chilling does not
  drive tree-level bloom. At most 4.8% of EBI variance is between-year, so any one-value-per-year climate driver
  is bounded to a small share, and the chilling-to-EBI tests are non-significant.</p>"""

ebi = f"""
<section id="ebi" class="section">
  <h2><span class="sic" style="background:#C0392B">&#127799;</span>EBI temporal dynamics</h2>
  {figure_block(IMG["F1_EBI_Temporal_UEF53"], "Figure 1. EBI mean and spread by year, UEF + 53.",
    "The Enhanced Bloom Index (EBI) is a per-tree measure of bloom intensity from the UAV 4-band mosaics, on a normalized 0 to 1 scale. Panel A plots the yearly mean with a standard-deviation whisker; panel B shows the full distribution as a violin. Paired t-tests compare each tree to itself across consecutive years.",
    f"Mean EBI rises modestly and monotonically, {d['2021']['mean']} to {d['2024']['mean']}, and every year-to-year shift is significant (p&lt;0.001). The distribution tightens (SD {d['2021']['sd']} to {d['2023']['sd']}). This differs from the earlier three-cultivar result, where the mean looked flat at ~0.55; that flatness was carried by cultivar 54. Read the rising mean cautiously given the radiometric caveat in the overview.")}

  <h3 class="h3">Checking the radiometric correction itself: canopy-only vs whole-image baseline</h3>
  <p style="font-size:13.5px;color:#555">Oren et al. (the paper behind the dormancy-model comparison) compute their bloom
  index by averaging every pixel across the whole orchard footprint, since Sentinel-2 cannot resolve individual canopies.
  Our pipeline's cross-year radiometric correction, by contrast, derives its baseline statistics from canopy pixels only
  (confirmed by direct code read of <code>apply_fixed_zones_yearly.py</code>). Canopy pixels are not a radiometrically
  neutral population, they ARE the bloom signal, so this was checked directly rather than assumed away.</p>
  {figure_block(IMG["EBI_Normalization_Scope_Check"], "Figure 33. Radiometric correction factor under a canopy-only vs whole-image baseline, by year.",
    "Left: the implied year-to-year contrast-correction factor (shared_scale, relative to the 2024 reference) under the current canopy-only baseline vs an Oren-style whole-image baseline. Right: canopy coverage fraction and the canopy/whole-image color-spread ratio, by year.",
    f"The two baselines diverge by 5-8% in the derived correction factor (2021: {NSC['shared_scale_comparison_canopy_vs_whole_image']['2021']['shared_scale_canopy_only']} vs {NSC['shared_scale_comparison_canopy_vs_whole_image']['2021']['shared_scale_whole_image']}; 2022: {NSC['shared_scale_comparison_canopy_vs_whole_image']['2022']['shared_scale_canopy_only']} vs {NSC['shared_scale_comparison_canopy_vs_whole_image']['2022']['shared_scale_whole_image']}; 2023: {NSC['shared_scale_comparison_canopy_vs_whole_image']['2023']['shared_scale_canopy_only']} vs {NSC['shared_scale_comparison_canopy_vs_whole_image']['2023']['shared_scale_whole_image']}), a real, non-trivial difference, but not a simple &ldquo;more bloom, more distortion&rdquo; story: canopy pixels are actually slightly darker than whole-image in 2021 and only slightly brighter in 2022, while in 2023-2024 they are substantially brighter, suggesting the mosaics differ in more ways than bloom coverage alone (shadow geometry, time of day, soil brightness at acquisition). This sandbox-safe check (no rasterio, no per-tree registration needed) cannot fully resolve which baseline is more correct, that needs re-running the extraction on the full-resolution mosaics, but it confirms the choice is not a negligible methodological detail. <b>The existing cross-year EBI caveat is strengthened, not overturned</b>: within-year and within-cultivar findings (cultivar effects, spatial clustering, EBI-yield ANCOVA) are unaffected, since those compare trees within one already-corrected year, but the cross-year mean/SD/bright-fraction trend should be read as directional pending a whole-image-baseline re-extraction.")}

  <h3 class="h3">The full re-extraction: within-year robust, cross-year mean trend diverges much more than expected</h3>
  <p style="font-size:13.5px;color:#555">The check above only approximated the baseline difference from low-resolution overview frames. A full
  whole-image-baseline re-extraction was then run on the full-resolution mosaics (<code>apply_fixed_zones_yearly_v3_wholeimage_baseline.py</code>,
  a new, separate file, nothing existing was overwritten), producing a real, not approximate, comparison.</p>
  {figure_block(IMG["EBI_WholeImageBaseline_Comparison"], "Figure 34. Canonical vs whole-image-baseline EBI: cross-year spread and yield-model value.",
    "Left: EBI_Norm's cross-year SD under the canonical, whole-image, and combined (ensemble) baselines. Right: mean cross-validated R&sup2; (across the project's 5 established robustness seeds) when each variant replaces the yield model's bloom term.",
    f"Within-year structure survives: canonical and whole-image rank trees almost identically every year (Spearman {WIC['partA_within_year_structure']['2021']['spearman_canonical_vs_wholeimg']}/{WIC['partA_within_year_structure']['2022']['spearman_canonical_vs_wholeimg']}/{WIC['partA_within_year_structure']['2023']['spearman_canonical_vs_wholeimg']}/{WIC['partA_within_year_structure']['2024']['spearman_canonical_vs_wholeimg']}), and the cultivar-collapse story holds under both (F {WIC['partA_within_year_structure']['2021']['cultivar_ANOVA_canonical']['F']}/{WIC['partA_within_year_structure']['2021']['cultivar_ANOVA_wholeimg']['F']} in 2021 down to {WIC['partA_within_year_structure']['2024']['cultivar_ANOVA_canonical']['F']}/{WIC['partA_within_year_structure']['2024']['cultivar_ANOVA_wholeimg']['F']} by 2024), whole-image runs about 13-15% lower in 2021-2022 specifically. <b>The cross-year MEAN trend is where this matters</b>: canonical EBI_Norm's mean spans only {round(WIC['partB_cross_year_trajectory']['canonical']['mean_by_year']['2024']-WIC['partB_cross_year_trajectory']['canonical']['mean_by_year']['2021'],3)} across the 4 years ({WIC['partB_cross_year_trajectory']['canonical']['mean_by_year']['2021']} to {WIC['partB_cross_year_trajectory']['canonical']['mean_by_year']['2024']}), but whole-image's mean spans {round(WIC['partB_cross_year_trajectory']['wholeimg']['mean_by_year']['2023']-WIC['partB_cross_year_trajectory']['wholeimg']['mean_by_year']['2021'],3)} ({WIC['partB_cross_year_trajectory']['wholeimg']['mean_by_year']['2021']} to {WIC['partB_cross_year_trajectory']['wholeimg']['mean_by_year']['2023']}), nearly 5 times the range, concentrated in 2021-2022 before converging with canonical by 2023-2024. Same direction (rising), very different magnitude: this changes &ldquo;EBI is nearly flat across years&rdquo; into &ldquo;EBI was substantially lower in 2021-2022.&rdquo; The SD-decline claim is more robust: it holds under both, and is if anything a little larger under whole-image (45% decline 2021-2023 vs 37% under canonical). <b>The one clear gain is the ensemble</b>: whole-image alone vs canonical alone is a wash in the yield model (mixed sign across seeds), but the simple average of both beats canonical alone in 5 of 5 established seeds (mean CV R&sup2; {WIC['partC_yield_model_value']['canonical_EBI_Norm']['cvR2_mean_5seeds']} to {WIC['partC_yield_model_value']['combined_avg']['cvR2_mean_5seeds']}), a small but genuine, robust improvement. <b>Recommendation:</b> canonical results are not replaced yet, since it is not established which baseline is more externally accurate; the deciding test is to cross-check both against the independent field phenology surveys (unaffected by mosaic radiometric correction), future work. The ensemble EBI is a candidate refinement for the yield model specifically.")}

  <h3 class="h3">Trying to adjudicate: does either baseline track independent field bloom surveys?</h3>
  <p style="font-size:13.5px;color:#555">The recommended next step was run: both EBI variants' year x cultivar means were checked against
  field-measured percent bloom open on the UAV flight date (the same independent ground truth used earlier in this deck's
  bloom-timing section), which does not depend on the mosaic or its correction at all.</p>
  {figure_block(IMG["Phenology_Baseline_Adjudication"], "Figure 35. Canonical, whole-image, and combined EBI vs independent field bloom-open percentage.",
    "Each panel scatters one EBI variant's year x cultivar mean against the field-measured % of flowers open on the UAV flight date, with its own fitted line and correlation.",
    f"<b>Inconclusive, honestly reported.</b> All three variants correlate near zero with the field ground truth (canonical r={PBA['EBI_canonical_vs_field_r']['r']} p={PBA['EBI_canonical_vs_field_r']['p']}; whole-image r={PBA['EBI_wholeimg_vs_field_r']['r']} p={PBA['EBI_wholeimg_vs_field_r']['p']}; combined r={PBA['EBI_combined_vs_field_r']['r']} p={PBA['EBI_combined_vs_field_r']['p']}; n={PBA['n']}), none remotely significant. Canonical is nominally a little higher, but at this n the three are statistically indistinguishable from each other and from zero. This reproduces the already-published finding that EBI does not track field bloom-open percentage (r=0.117 there vs 0.118 here, the same test, a useful internal consistency check), and shows the mismatch is not specific to the canonical baseline. <b>This test cannot adjudicate between the two radiometric baselines</b>, most likely because EBI (either baseline) does not simply track &ldquo;percent open,&rdquo; it responds to bloom color/density in a way that does not reduce to that one field metric, compounded by only 4 flight dates. The choice between baselines remains genuinely open; resolving it further would need a field bloom metric more directly comparable to EBI's own construction, or a ground calibration color panel photographed every year, neither of which exists in this dataset. Canonical results remain the reported baseline throughout this thesis.")}

  <h3 class="h3">What this means for an individual tree, not just the population average</h3>
  <p style="font-size:13.5px;color:#555">The CV R&sup2; comparison summarizes accuracy across the whole sample; it does not say whether any
  specific tree's predicted yield actually changes. Checked directly on the same pooled n=167 tree-years, fitted in-sample so
  individual predictions can be compared.</p>
  {figure_block(IMG["EBI_Baseline_TreeLevel_Yield_Effect"], "Figure 36. Per-tree-year predicted yield: canonical vs whole-image and combined EBI.",
    "Left: each tree-year's fitted predicted yield under canonical EBI vs the alternative baseline, with the 1:1 line. Right: the distribution of per-tree-year prediction shifts.",
    f"Most trees barely move: predicted yield correlates r={TLY['canonical_vs_wholeimg']['pearson_r_of_fitted_predictions']} between canonical and whole-image, and {round(TLY['canonical_vs_wholeimg']['top10pct_predicted_yield_tree_overlap']*100)}%/{round(TLY['canonical_vs_wholeimg']['bottom10pct_predicted_yield_tree_overlap']*100)}% of the top/bottom-10% predicted-yield trees stay flagged under either baseline. Typical shift is small (mean absolute difference {TLY['canonical_vs_wholeimg']['mean_abs_diff_kg']} kg, about {round(TLY['canonical_vs_wholeimg']['mean_abs_diff_kg']/TLY['measured_yield_context']['mean_kg']*100)}% of mean measured yield {TLY['measured_yield_context']['mean_kg']} kg), but not negligible at the extremes (largest single shift {TLY['canonical_vs_wholeimg']['max_abs_diff_kg']} kg). <b>Not random noise</b>: every one of the 10 biggest movers is a 2022 tree-year (2022 has the largest correction divergence of any year), and the direction is cultivar-structured, cultivar 53's predicted yield rises under whole-image, UEF's falls, the signature of the already-established opposite-signed EBI&times;cultivar interaction reacting to systematically lower 2022 EBI under the whole-image baseline. The combined variant sits closer to canonical (r={TLY['canonical_vs_combined']['pearson_r_of_fitted_predictions']}, MAE {TLY['canonical_vs_combined']['mean_abs_diff_kg']} kg), as expected from averaging. <b>Practical reading</b>: switching baselines would not reshuffle most of the orchard's ranking, but for a decision keyed to the model's extremes it would flip the call for roughly 1 in 8 to 1 in 17 trees, concentrated in 2022 and predictable by cultivar direction, not scattered noise.")}
</section>"""

cultivar = f"""
<section id="cultivar" class="section">
  <h2><span class="sic" style="background:#8E44AD">&#129003;</span>Cultivar effects</h2>
  {figure_block(IMG["F2_Cultivar_Effects_UEF53"], "Figure 2. Cultivar effect on EBI over years (UEF vs 53).",
    "Panel A is a one-way ANOVA of EBI by cultivar each year; with two groups the F statistic equals the squared t, and eta-squared is the share of EBI variance explained by cultivar. Panel B tracks each cultivar's mean EBI over the four years.",
    f"The cultivar signal is strong in 2021 (F={R['S3_cultivar_anova_ebi']['2021']['F']}, eta^2={R['S3_cultivar_anova_ebi']['2021']['eta2']}) and collapses to non-significant by 2024 (F={R['S3_cultivar_anova_ebi']['2024']['F']}). Note the rank flip: cultivar 53 blooms brighter early, but UEF overtakes it by 2023. The split-plot cultivar-by-year interaction is large (F={R['S3_splitplot']['interaction']['F']}, p&lt;0.001), confirming the two cultivars follow different trajectories.")}
</section>"""

spatial = f"""
<section id="spatial" class="section">
  <h2><span class="sic" style="background:#16A085">&#128506;</span>Spatial autocorrelation</h2>
  {figure_block(IMG["F3_Moran_Spatial_UEF53"], "Figure 3. Moran's I of EBI by year, and the 2023 spatial pattern.",
    "Moran's I measures whether nearby trees have similar EBI (positive I = clustering) using a 10 m distance band; significance is from 199 spatial permutations. Panel B maps 2023 EBI in UTM coordinates with the tree-derived field boundary.",
    f"Bloom is significantly clustered every year (I={moran['2021']['I']} to {moran['2023']['I']}, all p=0.005), peaking in 2023. Spatial structure is if anything stronger in early years than in the three-cultivar version, because cultivar 54's interleaved rows are removed. Clustering in the most uniform-mean year points to an environmental or management gradient rather than random tree-to-tree variation.")}
</section>"""

climate = f"""
<section id="climate" class="section">
  <h2><span class="sic" style="background:#4aa3df">&#10052;</span>Climate drivers - Chill Portions</h2>
  {figure_block(IMG["F4_Climate_CP_EBI_UEF53"], "Figure 4. Chill Portions vs EBI heterogeneity and bloom-bright fraction.",
    "Chilling is quantified only by the Dynamic Model in Chill Portions (CP), one station value per season. Because there is one climate value per year, climate-to-EBI links use n=4 and are directional hypotheses, not tests. Bloom-bright fractions (share of trees above EBI 0.60 / 0.65) are computed from the data.",
    f"CP across seasons is 18, 35, 25, 10 (non-monotonic). Directionally, higher chill associates with lower mean EBI (r=-0.40) and more heterogeneous bloom (r=+0.43), both non-significant at n=4. The bloom-bright fraction above 0.65 falls a mild {exc['2021']['pct_gt_0.65']}% to {exc['2024']['pct_gt_0.65']}%, not the dramatic 20.8% to 0.6% reported before, which was a cultivar-54 artifact.")}
</section>"""

cp_v_master = DM["CP_computed_vs_master_r"]
mvt = DM["model_vs_bloom_timing"]
dormancy = f"""
<section id="dormancy" class="section">
  <h2><span class="sic" style="background:#4aa3df">&#127807;</span>Dormancy models, compared to the paper</h2>
  <p class="lead">Oren et al. ("Tracking almond bloom patterns using multispectral imagery") benchmark four
  temperature-based dormancy models, Carbohydrate-Temperature (CT), Chill Hours (CH), Utah Chill Units (CU)
  and the Dynamic Model / Chill Portions (DCP), against a remote-sensing peak-bloom date over 3,840
  grid-cell-years, then fit a Random Forest ensemble across the four (leave-one-year-out CV R&sup2;=0.80).
  We have one climate station and four seasons at Kedma, so neither their scale nor the Random Forest is
  identifiable here. Instead this reproduces their per-model evaluation step at our scale: CH, CU and the
  Dynamic Model are computed directly from our 10-minute station temperature (to Feb 28 each season) and
  related to our own bloom benchmark, the field-survey 50% bloom day. The CT model is not implemented, it
  needs physiological parameters beyond what was used here.</p>
  {figure_block(IMG["Dormancy_Models"], "Figure 16. Three dormancy models vs field bloom timing, and a validation.",
    "Panels A-C each plot one model's chill accumulation to Feb 28 (computed from station temperature) against the field 50% bloom day-of-year, one point per season (n=4). Panel D validates the computed Dynamic-Model Chill Portions against the value already stored in the master.",
    f"All three models point the same direction as the paper's dormancy models generally do: more chill by Feb 28 associates with earlier bloom (CH r={mvt['CH_computed'][0]}, CU r={mvt['CU_computed'][0]}, CP r={mvt['CP_computed'][0]}), matching the finer-grained result already in the climate section (t50 vs Chill Portions r=-0.57, p=0.087, n=10 at the cultivar-year level). None reach significance at this n=4 season level, so this stays directional. The Dynamic Model (Chill Portions) tracks bloom timing most strongly of the three, consistent with it being the metric used throughout this thesis. Panel D is a validation, not from the paper: computed Chill Portions reproduces the master's stored value almost exactly (r={cp_v_master[0]}, p={cp_v_master[1]}), confirming the from-scratch Dynamic Model implementation (Fishman et al. 1987) is correctly specified.")}
  <p style="margin-top:10px;font-size:13.5px;color:#555"><b>Scope, stated plainly:</b> this is a small-scale
  sanity check, not a validation at the paper's scale, one station vs their 3,840 grid-cells, four seasons vs
  their multi-year, many-orchard data, a field-survey bloom day vs a satellite-detected peak-bloom date, and
  no CT model or Random Forest (not identifiable with n=4). It shows the Dynamic Model implementation is
  correct and that our data reproduce the paper's general finding directionally, nothing more.</p>
</section>"""

pbn_r = PBN["EBI_vs_corrected_coverage_at_flight"]
pft_r = PFT["EBI_vs_pct_open_at_flight_r"]
t50cp = PIN["t50_vs_ChillPortions"]; t50gd = PIN["t50_vs_GDD"]; border = PIN["cultivar_mean_t50_bloom_order"]
pbn_uef = [r for r in PBN["per_cultivar_year"] if r["cultivar"] == "UEF" and r["mean_EBI_normalized"] is not None]
pbn_53 = [r for r in PBN["per_cultivar_year"] if r["cultivar"] == "53" and r["mean_EBI_normalized"] is not None]
pbn_uef_str = ", ".join(f"{r['year']}: {r['mean_EBI_normalized']:.2f}" for r in pbn_uef)
pbn_53_str = ", ".join(f"{r['year']}: {r['mean_EBI_normalized']:.2f}" for r in pbn_53)
phenologynorm = f"""
<section id="phenologynorm" class="section">
  <h2><span class="sic" style="background:#4aa3df">&#128337;</span>Bloom timing: did the flights catch peak bloom?</h2>
  <p class="lead">The UAV flew on almost the same calendar date every year (late Feb/early March), but bloom
  timing shifts with chilling and heat, so a fixed calendar date does not mean a fixed point on the bloom
  curve. The field phenology survey (the same 0-10 stage scale used nationally for almond, ground-observed
  independently of the drone) lets this be checked directly.</p>
  {figure_block(IMG["Phenology_Flight_Timing"], "Figure 22. Field bloom-progression curves with the UAV flight date marked, per year.",
    "Each panel interpolates the field survey's percent-open curve (cultivar-mean, ground-observed on multiple dates per season) to the exact flight date, for both cultivars.",
    f"Flights caught the orchard at very different points on the bloom curve: 2022 near-peak (~78-85%), 2023 anywhere from under half (cultivar 53, ~45%) to fairly advanced (UEF, ~72%), 2024 mid-bloom (UEF ~60%). Cultivar-mean EBI does not track this field bloom stage across cultivar-years (r={pft_r:+.2f}, ns, n=7): a year/cultivar caught earlier in its own bloom curve does not show lower EBI, and vice versa, so raw EBI's cross-year level comparisons (the rising-mean trend, cultivar convergence) are confounded by flight timing and should be read cautiously.")}

  <h3 class="h3">What the field survey shows that EBI cannot: bloom timing tracks chilling</h3>
  {figure_block(IMG["Phenology_Insights"], "Figure 23. Field bloom timing (50% bloom day) vs Chill Portions, and cultivar bloom order.",
    "The 50% bloom day-of-year is read off each cultivar-year's field bloom curve. Panel A relates it to that season's Chill Portions; panel B ranks the three cultivars by their average 50% bloom day.",
    f"More chilling brings bloom earlier (t50 vs Chill Portions r={t50cp['r']}, p={t50cp['p']}, n={t50cp['n']}, borderline), and low-chill/high-heat seasons bloom significantly later (t50 vs GDD r={t50gd['r']}, p={t50gd['p']}, n={t50gd['n']}), the classic insufficient-chilling delay. Cultivars bloom in a consistent order (54 earliest, then UEF at {border.get('UEF', 0):.1f}, then 53 at {border.get('53', 0):.1f} mean 50%-bloom day-of-year). This is the climate-to-phenology link EBI cannot show on its own, since EBI is a single, possibly mistimed snapshot, while the field survey captures the whole curve.")}

  <h3 class="h3">Correcting the scale: stage 8 is full bloom, not stage 10</h3>
  <p>The field CSVs report a "Percent" column that is simply the 0-10 stage multiplied by ten, silently
  treating stage 10 as the most bloom. That is wrong by the scale's own definition: stage 8 ("80% flowers
  open") is full bloom; stages 9-10 describe petals shedding, i.e. bloom already declining, not increasing.</p>
  {figure_block(IMG["Phenology_Bloom_Normalization"], "Figure 24. Corrected stage-to-coverage function, corrected coverage per cultivar-year, and a phenology-normalized EBI.",
    "Panel A anchors visible bloom coverage at the scale's own labels (10%/50%/80% at stages 6/7/8) and lets it decline through 9-10 as petals fall, instead of the naive linear stage x10 reading. Panel B re-reads each cultivar-year's coverage at the flight date on this corrected curve ('x' marks cultivar-years where the implied correction is too large to trust). Panel C compares raw cultivar-mean EBI to EBI projected to peak bloom (stage 8), where that correction is stable.",
    f"Corrected coverage still does not significantly predict cultivar-mean EBI (r={pbn_r['r']:+.3f}, p={pbn_r['p']}, n={pbn_r['n']}), though the direction is more sensible than the naive reading. Combined with a direct per-tree, same-day test (2021, ground %flowers-open vs that tree's EBI: r=-0.08, n=20, ns), neither version of field bloom coverage explains EBI level at this sample size. The multiplicative correction (EBI x 80/coverage, projecting to peak) is therefore shown as an exploratory sensitivity check, not a validated substitute: it is only computed where coverage at flight is at least 30% (cultivar 53 2023 at ~4.6% coverage and UEF 2024 at ~10% would need 8-17x corrections, too unstable to trust, and cultivar 53 has no 2024 field survey at all). Where stable, UEF's normalized mean EBI ({pbn_uef_str}) keeps roughly the same shape as its raw trend, but cultivar 53's does not: raw EBI looks flat between 2021 and 2022 (0.603 vs 0.597), yet 2021 caught cultivar 53 much further from its own peak (54% coverage) than 2022 did (72.5%), so once corrected, 2021's true bloom intensity is well above 2022's ({pbn_53_str}), sharpening the existing finding that 2021 EBI runs inflated relative to actual bloom (section on radiometric matching).")}
  <p style="margin-top:10px;font-size:13.5px;color:#555"><b>Scope, stated plainly:</b> this correction can only
  change cross-year and cross-cultivar-within-year readings of EBI <i>level</i>. It cannot change any
  within-cultivar-year result already reported (the EBI-yield correlations, the within-year cultivar
  contrast), since every tree in a cultivar-year is scaled by the same constant, which leaves a Pearson
  correlation computed within that group unchanged. No yield finding in this deck is affected by it.</p>
</section>"""

yield_ = f"""
<section id="yield" class="section">
  <h2><span class="sic" style="background:#E67E22">&#127806;</span>Measured yield (ground truth)</h2>
  {figure_block(IMG["F5_Measured_Yield_UEF53"], "Figure 5. Measured 2023 yield vs EBI and by cultivar.",
    "Yield is field-measured net kernel yield per tree (kg), 2023, joined to 151 trees that also have EBI. Panel A fits an EBI-yield line within each cultivar (ANCOVA); panel B compares yield distributions between cultivars.",
    f"Pooled, spring EBI does not significantly predict measured yield (r=+0.137, ns). But the slope reverses by cultivar: UEF r={wc['UEF']['r']:+.2f} (more bloom, more yield) vs cultivar 53 r={wc['53']['r']:+.2f} (more bloom, less yield); the EBI-by-cultivar interaction is F={anc['EBIxcultivar_interaction']['F']} (p&lt;0.001). Cultivar itself drives yield: UEF {yc['UEF']['mean']} kg &gt; 53 {yc['53']['mean']} kg (ANOVA F={R['S6_yield_cultivar_anova']['F']}, p=0.0003). These yield results are unchanged from the full analysis, because measured yield only ever covered 53 and UEF.")}
</section>"""

h1 = HR["H1_climate_EBI"]; ceil = h1["between_year_variance_ceiling_eta2"]
v23 = HR["V_predicted_vs_measured"]["validation_2023"]; v22 = HR["V_predicted_vs_measured"]["validation_2022"]
h3anc = HR["H3_EBI_yield"]["ANCOVA_measured2023"]; h3wc = HR["H3_EBI_yield"]["within_cultivar_measured2023"]
p1 = HR["P_phenology"]["P1_bloom_synchrony"]["levene_EBI_variance_across_years"]
p2 = HR["P_phenology"]["P2_NGRDI_convergent"]["EBI_vs_NGRDI_2023_alltrees"]

hypotheses = f"""
<section id="hypotheses" class="section">
  <h2><span class="sic" style="background:#C0392B">&#128300;</span>Hypothesis and tests</h2>
  {hyp_block}

  <h3 class="h3">H1 - Climate (Chill Portions) &rarr; EBI: not supported</h3>
  {figure_block(IMG["H1_Climate_EBI_UEF53"], "Figure 6. Chill Portions vs EBI at the year level, and the EBI variance partition.",
    "Panels A and B correlate the single yearly Chill Portions value against mean EBI and EBI spread (n=4 years) with Pearson, Spearman, Kendall and an exact permutation p. Panel C partitions total EBI variance into a between-year slice (the most any year-level climate driver could explain) and a within-year slice.",
    f"Nothing is significant at n=4 (CP vs mean EBI r={h1['CP_vs_meanEBI']['r']}, permutation p={h1['CP_vs_meanEBI']['perm_p']}). The decisive result is panel C: only <b>{ceil*100:.1f}%</b> of EBI variance is between-year, so chilling can explain at most ~5% of tree-to-tree bloom variation. Directionally more chill goes with lower, more heterogeneous bloom, but this is a hypothesis, not a test.")}

  <h3 class="h3">H2 - Climate &rarr; yield: not testable, but a rich 3-year panel</h3>
  <p>Climate-yield at the year level is an n&le;3 between-year comparison (higher chill tracks
  higher yield directionally, r=+0.82, n=3), so it stays descriptive. But the 2022 and 2024
  harvests are the <b>same 20 trees</b> as each other and a subset of 2023 (balanced 10 UEF + 10
  cv-53), so all three years form a repeated-measures panel worth analysing on its own.</p>
  {figure_block(IMG["H_Panel_3yr_UEF53"], "Figure 6b. The 20-tree panel measured in 2022, 2023 and 2024.",
    "Panel A traces each tree's yield across the three years (thin lines) with cultivar means (thick); a repeated-measures ANOVA tests the year effect. Panel B shows the cultivar means per year with significance. Panel C is the within-tree EBI-to-yield correlation for each cultivar and year.",
    f"There is a strong orchard-wide year effect (F={HR['PANEL_3yr_2022_2024']['RM_ANOVA_year']['F']}, p&lt;0.001, eta^2={HR['PANEL_3yr_2022_2024']['RM_ANOVA_year']['eta2_year']}): 2022 was a big 'on' year (~6.7 kg), 2023-2024 far lower. Year-to-year within-tree correlations are all positive, so this is a shared season effect, <b>not alternate bearing</b>. The cultivar gap is concentrated in the poor 2023 season, where cv-53 nearly failed (1.1 kg) while UEF held (4.1 kg, p=0.0004); in 2022 and 2024 the cultivars were indistinguishable. Panel C shows the EBI-to-yield reversal recurs: cv-53 negative in 2023-24, UEF positive. Caveat: these 20 trees are a clustered block, so absolute levels need not represent the whole orchard.")}

  <h3 class="h3">H3 - EBI &rarr; yield: opposite-signed effect by cultivar (UEF +, cv 53 -)</h3>
  {figure_block(IMG["H3_EBI_Yield_UEF53"], "Figure 7. Bloom-to-yield correlations (all ns) and the cultivar-conditional ANCOVA.",
    "Panel A is a forest plot of Pearson correlations (95% CI) between each bloom index-year and measured yield. Panel B fits the EBI-yield slope within each cultivar (ANCOVA with an EBI-by-cultivar interaction).",
    f"Bloom acts on yield in <b>both</b> cultivars, with opposite signs: UEF r={h3wc['UEF']['r']:+.2f} (p=0.007, more bloom, more yield) and cultivar 53 r={h3wc['53']['r']:+.2f} (p=0.042, more bloom, less yield); the EBI×cultivar interaction is <b>F={h3anc['EBIxcultivar_interaction']['F']}, p&lt;0.001</b> (R^2 {h3anc['R2']} to {h3anc['R2_with_interaction']}). A pooled, cultivar-blind correlation is near zero only because these two real effects point in opposite directions and cancel (panel A).")}

  <h3 class="h3">Validation - modelled yield does not track measured yield</h3>
  {figure_block(IMG["H_Validation_PredVsMeas_UEF53"], "Figure 8. Modelled 'predicted' yield vs measured yield, 2023 and 2022.",
    "For trees carrying both values, the modelled predicted yield is plotted against ground-truth measured yield with the 1:1 line. Agreement is summarised by Pearson r, Lin's concordance (CCC), RMSE and bias.",
    f"The model has <b>no tree-level skill</b>: 2023 r={v23['pearson_r']} (ns), CCC={v23['Lin_CCC']}, RMSE={v23['RMSE']} kg, and its output is compressed (predicted SD ~0.31 vs measured ~1.19). 2022 is worse (r={v22['pearson_r']}, RMSE={v22['RMSE']}). There is no predicted value for 2024. This confirms quantitatively that predicted yield is not ground truth, and that correlating EBI against it is meaningless.")}

  <h3 class="h3">Phenological support</h3>
  {figure_block(IMG["H_Phenology_Support_UEF53"], "Figure 9. Bloom synchrony, index specificity, and bloom-yield timing.",
    "Panel P1 tracks EBI spread (SD and CV) over years with Levene's test of equal variances. Panel P2 relates the bloom index (EBI) to the greenness index (NGRDI). Panel P3 compares same-season vs prior-season bloom as a yield predictor.",
    f"Bloom becomes more synchronous over time (Levene F={p1['F']}, p&lt;0.001; SD 0.076 to 0.048). EBI and NGRDI are negatively related (r={p2['r']}, p&lt;0.001), the expected flower-vs-leaf trade-off, and NGRDI does not reproduce the cultivar-split, so the yield signal is bloom-specific. Same-season bloom (r=+0.14), not prior-season (r=-0.10), carries the weak signal.")}

  <div class="grid3">
    {card("H1 verdict", "Climate is bounded to &le;4.8% of EBI variance; n=4 links are non-significant. Not supported.", "#C0392B")}
    {card("H2 verdict", "Yield is one usable season; climate-yield is untestable (n&le;3). Directional only.", "#E67E22")}
    {card("H3 verdict", "No simple EBI-yield link, but a robust cultivar-conditional reversal (interaction F=11.58***).", "#2C3E50")}
  </div>
</section>"""

def rrow(C, y):
    d = BC[C]["H3_EBI_yield_sameyear"]["measured"][f"EBI{y}_vs_measY{y}"]
    return f"{d['r']:+.2f}{'' if d['sig'] in ('ns','') else ' '+d['sig']} (n={d['n']})"

bycultivar = f"""
<section id="bycultivar" class="section">
  <h2><span class="sic" style="background:#8E44AD">&#9878;</span>Everything within each cultivar</h2>
  <p class="lead">Computed <b>entirely within UEF and within cultivar 53</b> (never pooled), for all
  measured yield years and the model's predicted yield, using <b>same-season bloom and yield only</b>
  This is the cultivar-level view of the whole battery.</p>

  <h3 class="h3">Bloom &rarr; yield reverses between cultivars, every year</h3>
  {figure_block(IMG["C1_EBI_Yield_byCultivar"], "Figure 10. Same-year EBI&rarr;measured-yield within each cultivar, all years.",
    "Panel A is a coefficient plot of the same-season EBI-to-yield Pearson r for each cultivar and year (95% CI); panels B and C show the 2023 scatter (largest n) for UEF and cultivar 53 with the fitted slope.",
    f"Within <b>UEF</b> the slope is positive every year (2022 {rrow('UEF',2022)}, 2023 {rrow('UEF',2023)}, 2024 {rrow('UEF',2024)}) and significant in the large-n 2023. Within <b>cultivar 53</b> it is negative in 2023 ({rrow('53',2023)}) and 2024 ({rrow('53',2024)}), only weakly positive in 2022 ({rrow('53',2022)}). The n=9 seasons are directional; 2023 carries the inference. The opposing signs are the substantive bloom-yield result, shown without any pooled model.")}

  <h3 class="h3">Yield by year, and the model, within each cultivar</h3>
  {figure_block(IMG["C2_Yield_Validation_byCultivar"], "Figure 11. Measured yield by year and predicted-vs-measured, per cultivar.",
    "Panel A shows each cultivar's mean measured yield for 2022-2024 with Chill Portions overlaid (n=3 seasons, descriptive). Panel B is the predicted-vs-measured Pearson r for each cultivar and year the model covers.",
    "Both cultivars share a high 2022 and low 2023-2024, and UEF out-yields 53 every year (gap largest in 2023). The model's predicted yield has no tree-level skill in either cultivar (all bars near zero, all ns; concordance near zero), and there is no predicted value for 2024, so predicted yield cannot serve as a spectral-yield endpoint.")}

  <h3 class="h3">Phenology within each cultivar</h3>
  {figure_block(IMG["C3_Phenology_byCultivar"], "Figure 12. Bloom synchrony and index specificity, per cultivar.",
    "Panel A tracks EBI coefficient of variation over years for each cultivar with Levene's variance test; panel B relates EBI to NGRDI in 2023 within each cultivar.",
    f"Bloom becomes more synchronous over time in both cultivars (Levene UEF F={BC['UEF']['P_phenology']['levene_EBI_var_across_years']['F']}***, 53 F={BC['53']['P_phenology']['levene_EBI_var_across_years']['F']}***; cultivar 53 ends most uniform). EBI and NGRDI are negatively related in both (UEF r={BC['UEF']['P_phenology']['EBI_vs_NGRDI_2023']['r']}, 53 r={BC['53']['P_phenology']['EBI_vs_NGRDI_2023']['r']}), the flower-vs-leaf trade-off, and NGRDI predicts yield in neither, so the cultivar-specific yield signal is bloom-specific.")}
</section>"""

def tri(C, path):
    d = TRI[C]
    for k in path.split("."):
        d = d[k]
    return d

U = TRI["UEF"]; F = TRI["53"]
triangle = f"""
<section id="triangle" class="section">
  <h2><span class="sic" style="background:#16A085">&#9651;</span>Climate &rarr; EBI &rarr; Yield</h2>
  <p class="lead">The three relationships tested <b>within each cultivar, year by year, same season</b>
  (chilling = Chill Portions, Dynamic Model). Chill Portions is one station value per year,
  so climate links are between-year (n&le;4, directional); EBI&rarr;yield is a tree-level test each year;
  the "all three together" model is a per-cultivar tree-level regression yield ~ EBI + year(climate).</p>
  {figure_block(IMG["Climate_EBI_Yield_Triangle_byCultivar"], "Figure 13. The climate-EBI-yield triangle, per cultivar.",
    "Each panel is one cultivar. Grey arrow = climate to bloom (Chill Portions vs annual mean EBI, n=4). Green arrow = climate to yield (Chill Portions vs annual mean yield, n=3, plus the year effect from the joint model). Coloured arrow = bloom to yield (same-year tree-level, with the climate-adjusted effect from the joint model). The strip below each triangle gives the EBI-to-yield correlation in each year.",
    "Read the two panels side by side: the climate-to-bloom and climate-to-yield arrows behave the same in both cultivars, so the cultivars differ only in the bloom-to-yield arrow, which is positive in UEF and negative in cultivar 53.")}

  <h3 class="h3">UEF</h3>
  <p><b>Climate &rarr; EBI:</b> no effect. Across four years higher chilling is weakly, non-significantly
  linked to lower mean bloom (r={U['climate_to_EBI_year']['r']}, n=4, ns) - Chill Portions do not drive
  UEF bloom intensity. <b>Climate &rarr; yield:</b> the dominant driver. The high-chill 2022 season yielded
  ~7.1 kg/tree vs ~2.9 in 2023-2024; in the joint model the year/climate effect is highly significant
  (F={U['all_three_together']['climate_given_EBI']['F']}, p&lt;0.001). <b>EBI &rarr; yield:</b> positive every
  year (2022 r=+0.52, 2023 r=+0.29*, 2024 r=+0.42), significant in the large-sample 2023. <b>All three
  together:</b> climate sets the yearly yield level and bloom adds independent value on top - adding EBI
  lifts R&sup2; {U['all_three_together']['R2_year_only']} &rarr; {U['all_three_together']['R2_full']} and the
  climate-adjusted EBI effect is significant (F={U['all_three_together']['EBI_given_climate']['F']},
  p={U['all_three_together']['EBI_given_climate']['p']}). For UEF, brighter bloom means higher yield even after climate.</p>

  <h3 class="h3">Cultivar 53</h3>
  <p><b>Climate &rarr; EBI:</b> no effect (r={F['climate_to_EBI_year']['r']}, n=4, ns). <b>Climate &rarr; yield:</b>
  the same dominant year pattern (2022 high; joint-model year effect F={F['all_three_together']['climate_given_EBI']['F']},
  p&lt;0.001). <b>EBI &rarr; yield:</b> it reverses - weakly positive in 2022 (r=+0.17) but negative in 2023
  (r=-0.26*) and 2024 (r=-0.39): more bloom goes with less yield. <b>All three together:</b> climate sets the
  yearly level and bloom adds nothing beyond it (climate-adjusted EBI effect F={F['all_three_together']['EBI_given_climate']['F']},
  {F['all_three_together']['EBI_given_climate']['sig']}, adjusted slope negative; R&sup2;
  {F['all_three_together']['R2_year_only']} &rarr; {F['all_three_together']['R2_full']}). For cultivar 53,
  brighter bloom is associated with <b>lower</b> yield, significantly in 2023 (r=-0.26, p=0.042) and again
  in 2024 (r=-0.39); pooled across seasons the effect stays negative (adjusted slope negative, p=0.09).</p>

  <h3 class="h3">The two cultivars relative to each other, and the three together</h3>
  <p>The climate arrows are shared: in <b>both</b> cultivars Chill Portions do not drive bloom, and the
  season (which carries the climate signal) is the main control on yield. The cultivars diverge on the single
  bloom-to-yield link, and it acts in <b>both</b>, just in opposite directions: <b>positive in UEF</b>
  (more bloom, more yield) and <b>negative in cultivar 53</b> (more bloom, less yield), the difference being
  highly significant (EBI×cultivar interaction F=11.58, p=0.0009). Put the three together and the story is one
  sentence: <b>the year (climate) decides how much the orchard yields, and within a year bloom brightness raises
  yield in UEF and lowers it in cultivar 53.</b> This is why a pooled, cultivar-blind bloom-to-yield correlation
  looks like nothing - the two real cultivar effects point in opposite directions and cancel.</p>
</section>"""

w23 = YM["within_2023"]["models"]; wp = YM["pooled_2022_2024"]["models"]
clim_cv = wp["+climate (Chill Portions)"]["cvR2"]
ceil_cv = YM["pooled_2022_2024"]["seasonal_ceiling_year_factor"]["cvR2"]
ympt_prac = YMPT["practical_swap_reliable_where_available_else_raw"]
ympt_clean = YMPT["clean_comparison_reliable_cells_only"]
yieldmodel = f"""
<section id="yieldmodel" class="section">
  <h2><span class="sic" style="background:#E67E22">&#128200;</span>Toward a per-tree yield prediction model</h2>
  <p class="lead">Using what we have (field-measured yield for UEF and cultivar 53, per-tree EBI and NGRDI,
  cultivar, spatial position, and one climate value per season), a per-tree yield model was prototyped with
  nested regressions and 5-fold cross-validated R&sup2; (honest out-of-sample skill). The result tells us
  which features to keep and why.</p>
  {figure_block(IMG["Tested_Trees_By_Year"], "Figure 15. Yield-tested trees per year, highlighted on the orchard.",
    "Each panel is one measured season; the whole orchard is grey and the trees with field-measured yield are highlighted by cultivar, with counts in the legend.",
    "2023 is a dense central harvest census (151 trees used: 87 UEF + 64 cultivar-53), while 2022 and 2024 are the same 18-tree panel (9 + 9) scattered through the middle rows. So 2023 carries the inference and 2022/2024 form the small repeated-measures panel; the 2-3 trees dropped from matched to used simply lack an EBI value that season.")}
  {figure_block(IMG["Yield_Model_Features"], "Figure 14. Which features add predictive value to a per-tree yield model.",
    "Panel A adds features one at a time within a single season (2023, n=151) and reports both in-sample and cross-validated R&sup2;. Panel B compares cross-validated R&sup2; across the three measured seasons (n=187) as season and climate terms are added.",
    f"Within a season, cultivar plus a cultivar-specific bloom term (EBI×cultivar) carry the model (CV R&sup2; {w23['M1 cultivar']['cvR2']} to {w23['M3 +EBI×cultivar']['cvR2']}); plain EBI without cultivar adds almost nothing, and raw spatial position and NGRDI do not generalize (CV R&sup2; falls or stays flat). Across seasons, the season must enter through <b>climate</b>, not the year label: adding Chill Portions reaches CV R&sup2; {clim_cv}. The dashed line ({ceil_cv}) is a year-factor benchmark showing how much of yield is seasonal, but the year itself is not a usable feature since it just labels a season, so climate is what a deployable model uses.")}

  <h3 class="h3">Which features to use, and why</h3>
  <div class="grid3">
    {card("USE 1 - Season, via climate (not year)", f"The season sets how much the orchard yields, but feed it as <b>climate</b> (Chill Portions, GDD), never as a year label. Year is not a usable feature, it only stands in for that season's climate and cannot predict a future year. Climate reaches CV R&sup2; {clim_cv}, against a seasonal ceiling of {ceil_cv}.", "#16A085")}
    {card("USE 2 - Cultivar", "Required, because it flips the bloom-yield slope. Alone it explains little across years, but without it the bloom term is meaningless. Why: UEF and cultivar 53 respond oppositely.", "#8E44AD")}
    {card("USE 3 - EBI as EBI×cultivar", "Enter bloom only as a cultivar-specific term. Why: pooled EBI adds ~0 (CV R&sup2; barely moves), but EBI×cultivar lifts within-season CV R&sup2; from 0.05 to 0.09 (UEF positive, cultivar 53 negative).", "#C0392B")}
    {card("DROP - NGRDI", "Adds no predictive value (CV R&sup2; does not improve). Why: NGRDI tracks canopy greenness, which trades off with bloom and carries no yield signal.", "#95a5a6")}
    {card("DROP - raw spatial X,Y", "Improves in-sample fit but CV R&sup2; falls (overfitting). Why: a linear north-east trend does not generalize; if spatial structure is used, use management-zone or a neighborhood random effect instead.", "#95a5a6")}
    {card("DO NOT USE - model predicted_Yield", "The physiology-based predicted yield does not match the measured census (r near 0, opposite cultivar ranking), so it cannot be a feature or a target here.", "#95a5a6")}
    {card("TESTED - phenology-normalized EBI", f"Swapping the peak-bloom-normalized EBI (section on bloom timing) into this model does not help once climate is already included: CV R&sup2; {ympt_prac['raw_EBI']['+EBI×cultivar+climate']['cvR2']} (raw) vs {ympt_prac['phenology_normalized_EBI']['+EBI×cultivar+climate']['cvR2']} (normalized), and similarly on the clean reliable-only subset ({ympt_clean['raw_EBI']['+EBI×cultivar+climate']['cvR2']} vs {ympt_clean['phenology_normalized_EBI']['+EBI×cultivar+climate']['cvR2']}). Why: the correction largely duplicates what climate already tells the model, plus its own estimation noise. Keep raw EBI.", "#95a5a6")}
    {card("USE 4 - Growth_April, where a physiology survey exists", f"Ground-measured April canopy growth (from the yield GPKGs) is the one candidate, out of every RGB, spectral and physiology feature tested (sections 19-22), that survives 5 independent cross-validation fold splits (selected in 5 of 5). Adding it lifts the pooled 2022-2023+climate model from CV R&sup2; {YMv2['C_pooled_2022_2023_with_climate_and_physiology']['baseline_cvR2']} to {YMv2['C_pooled_2022_2023_with_climate_and_physiology']['final_cvR2']}. Only usable for seasons with a matching physiology survey (currently 2022-2023).", "#16A085")}
  </div>

  <h3 class="h3">Updated with every GPKG field: forward selection plus a robustness check</h3>
  {figure_block(IMG["Yield_Model_v2_AllGPKG"], "Figure 26. Forward-selected GPKG features and a 5-fold robustness check.",
    "All fourteen physiology candidates (SWP/Growth at 3 timepoints, CNC, each alone and interacted with cultivar) plus the four newly-found July-2024 spectral fields (SR, IPVI, SIPI, area) were offered to a greedy forward selector scored on cross-validated R&sup2;, in three settings: within-2023 alone, pooled 2022-2023 with climate, and 2024 with the new spectral fields. Because forward selection scores every candidate against one fixed fold split (a mild selection-bias risk), the pooled-2022-2023 search was re-run under 5 independent fold splits to check which features keep winning.",
    f"Within 2023 alone the gain is negligible ({YMv2['A_within_2023_physiology']['baseline_cvR2']} to {YMv2['A_within_2023_physiology']['final_cvR2']}, noise-level). The 2024 spectral fields could not be tested (only {YMv2['B_2024_new_spectral_fields']['n']} of the small 18-20 tree yield panel overlap that file's coverage, an unlucky sample intersection, not a registration problem). Pooled 2022-2023 with climate is where it matters: baseline CV R&sup2; {YMv2['C_pooled_2022_2023_with_climate_and_physiology']['baseline_cvR2']} rises to {YMv2['C_pooled_2022_2023_with_climate_and_physiology']['final_cvR2']} adding Growth_April then SWP_MayJune. The robustness check is the deciding evidence: across 5 independent fold splits, <b>Growth_April is selected in 5 of 5</b>, while SWP_MayJune and SWP_June&times;cultivar each appear in only 1 of 5, the signature of a real effect versus fold-specific noise. Growth_April is added to the recommended model for any season with a physiology survey.")}
  <p style="margin-top:10px;font-size:13.5px;color:#555"><b>Not a simple correlation, and the direction reverses.</b>
  Growth_April's raw pooled correlation with yield looks strong (r=+0.371, n=167), but this is confounded:
  Growth_April correlates r=+0.758 with that season's Chill Portions, since 2022 was both the highest-chill,
  highest-spring-growth, and highest-yield season together, so the raw correlation is mostly the shared year
  effect, not a tree-level relationship. Once climate is held constant (already a term in the model), the
  fitted coefficient on Growth_April is <b>negative</b> (-0.58 standardized, larger in magnitude than EBI's
  own coefficient), meaning within a given season, more April vegetative growth associates with <b>less</b>
  yield, a vegetative-reproductive resource trade-off. This matches the weak negative direction already seen
  in the within-2023-only test (section 21); pooling seasons with climate explicitly controlled resolves the
  same relationship more clearly. This is why the effect shows up only inside the model, not as a plain
  correlation, a genuine conditional relationship, not an artefact.</p>

  <h3 class="h3">Feature engineering on the same GPKG fields: nothing beats Growth_April</h3>
  {figure_block(IMG["Yield_Model_v3_FeatureEngineering"], "Figure 27. Engineered features (trajectory, PCA composites, cluster dummies, crown height) vs the Growth_April baseline.",
    "Beyond the raw fields, derived features were engineered from the same physiology data: season-trajectory deltas (June minus April), a 2-component PCA vigor score, an SWP_clusters dummy, and crown height (DEM), each alone and interacted with cultivar, then run through the same forward-selection-plus-robustness-check procedure as Figure 26.",
    f"On their own (pooled 2022-2023+climate baseline, CV R&sup2; {YMv3['pooled_2022_2023_with_climate']['baseline_cvR2']}), one engineered term is selected, Growth_change&times;cultivar, reaching CV R&sup2; {YMv3['pooled_2022_2023_with_climate']['final_cvR2']} ({'+'}{YMv3['pooled_2022_2023_with_climate']['improvement_cvR2']}), a smaller gain than Growth_April's own {YMv2['C_pooled_2022_2023_with_climate_and_physiology']['final_cvR2'] - YMv2['C_pooled_2022_2023_with_climate_and_physiology']['baseline_cvR2']:+.3f}. Its robustness is weaker too: selected in only {YMv3['robustness_across_5_independent_CV_fold_splits']['how_often_each_feature_selected_of_5']['Growth_change_x_cultivar']} of 5 independent fold splits (vs Growth_April's 5 of 5), and it is mechanically related to Growth_April (Growth_change = Growth_June minus Growth_April), so this reads as a noisier echo of the same signal, not an independent one. More decisively: added on top of the already-recommended model (cultivar+EBI&times;cultivar+climate+Growth_April, CV R&sup2; {YMv3['on_top_of_Growth_April']['baseline_cvR2_with_GrowthApril']}), the best engineered feature lifts it only to {YMv3['on_top_of_Growth_April']['final_cvR2']} ({'+'}{YMv3['on_top_of_Growth_April']['improvement_cvR2']}), noise-level. None of the PCA composites, the SWP_clusters dummy, or crown height add real value; the yield-model recommendation from Figure 26 stands unchanged.")}

  <h3 class="h3">A complete audit of every climate field: nothing to swap in, the ceiling is real</h3>
  <p>All 13 station-level agro-climate fields in the master (not just Chill Portions) were checked. Two,
  Rad_Feb and Frost_Hrs, are identical in all 4 years (Frost_Hrs=0 every season is a real physical fact, no
  frost was recorded; Rad_Feb=0 every season looks like an unpopulated field, flagged not corrected). The
  rest extend the section-6 EBI table (n=4, directional): Max_Dry_Hrs (r=+0.858 vs mean EBI) and Avg_DTR_Feb
  (r=-0.828) are the strongest directional associations found, alongside Chill Portions' own -0.404.</p>
  {figure_block(IMG["Climate_Full_Audit"], "Figure 29. Every climate field vs EBI, the climate-swap test, and DEM x climate.",
    "Panel A extends the climate-EBI table to all 13 fields. Panel B refits the yield model swapping in 7 different climate fields one at a time. Panel C forward-selects DEM x climate interaction terms onto the recommended model.",
    f"Panel B is the key result: every climate field gives CV R&sup2;={CFA['partB_climate_swap_invariance']['swap_results']['Chill_Portions']['cvR2']}, identical to 4 decimal places, because the pooled model spans only 2 measured-yield years, so any single station value is an affine rescaling of a 2-point year indicator. This is a mathematical certainty, not an empirical coincidence, verified here rather than assumed: adding a second climate field simultaneously leaves the design matrix rank-deficient ({CFA['partB_climate_swap_invariance']['two_climate_features_simultaneously']['design_matrix_rank']} of {CFA['partB_climate_swap_invariance']['two_climate_features_simultaneously']['design_matrix_cols']} columns) and does not change CV R&sup2; either. Panel C tests the one genuinely new idea, DEM&times;climate (does topography's effect on yield depend on the season): negative, nothing is selected at the main fold split, and only 2 of 5 independent seeds pick anything at all, each a different feature, with gains under 0.005, an order of magnitude below Growth_April's own +0.02 to +0.06. Tenth independent negative result. The structural conclusion: the raw station data stop at 31 March each year (no April-August), so no fruit-development climate feature can be built from what exists, and no amount of feature engineering on the climate side can add real predictive power while only 2-4 distinct seasons exist. Improving the climate term now requires new fieldwork (more measured-yield seasons, or extended-season meteorological records), not more analysis of the current data.")}

  <h3 class="h3">What if the target is predicted_Yield instead of measured yield? A circularity check, not an accuracy result</h3>
  <p style="font-size:13.5px;color:#555">The same model was refit with <code>predicted_Yield</code> (the clustering-model output) as the
  target instead of measured yield, specifically to show why this must never be confused with real accuracy.
  <b>CV R&sup2; jumps to {PYC['A_same_features_vs_predicted_yield']['cvR2']}</b> (vs the real measured-yield model's
  {PYC['measured_yield_model_for_comparison']['cvR2']}), but decomposing it shows why: cultivar+climate alone
  (i.e. just which year and which cultivar) already gives CV R&sup2;={PYC['decomposition_where_does_the_R2_actually_come_from']['cultivar_plus_climate']['cvR2']},
  and EBI adds only {'+'}{round(PYC['B_no_physiology_vs_predicted_yield']['cvR2']-PYC['decomposition_where_does_the_R2_actually_come_from']['cultivar_plus_climate']['cvR2'],4)} on top, negligible.
  Letting the model use the raw physiology fields directly (its own generating ingredients, section 24) pushes CV R&sup2;
  to {PYC['C_full_physiology_forward_selection_vs_predicted_yield']['final_cvR2']} (CNC_June alone correlates r={PYC['raw_correlations_predicted_yield_vs_physiology_fields']['CNC_June']:+.3f}
  with predicted_Yield), mechanical reconstruction of the clustering model's own formula, not predictive skill.
  <b>The clarifying takeaway: EBI barely moves predicted_Yield at all, so EBI's real, modest link to MEASURED yield is not a
  circularity artifact.</b> The one defensible yield-accuracy number in this project remains CV R&sup2;={PYC['measured_yield_model_for_comparison']['cvR2']} against measured yield.</p>

  <h3 class="h3">Side by side: the real model's honest ceiling vs the circular model's inflated one, under the SAME exhaustive search</h3>
  <p style="font-size:13.5px;color:#555">Per request, the predicted_Yield check above is repeated with every feature this thesis uses (12,
  the same set as the full pairwise-interaction sweep) plus all 66 pairwise interactions among them, 78 candidates offered to one
  exhaustive forward selection, exactly the same search already run against MEASURED yield.</p>
  {figure_block(IMG["Predicted_Yield_Model_FullFeatures"], "Figure 37. The same exhaustive feature-and-interaction search, side by side: measured yield (left, real) vs predicted_Yield (right, circular).",
    "Left panel: CV R&sup2; for the real measured-yield model and for the exhaustive search against it (section 30). Right panel: the same exhaustive search's forward-selection path against predicted_Yield.",
    f"<b>The same search that found nothing against measured yield (CV R&sup2; stuck at {PYF['measured_yield_for_comparison']['exhaustive_pairwise_search_cvR2']}, section 30) reconstructs predicted_Yield almost perfectly</b>: the cultivar+climate floor alone already reaches CV R&sup2;={PYF['floor_cultivar_climate_only']['cvR2']}, and the full 78-candidate search ({PYF['n_main_effects']} main effects + {PYF['n_pairwise_interactions']} interactions) pushes it to <b>CV R&sup2;={PYF['forward_selection_main_seed42']['final_cvR2']}</b> (n={PYF['n']}, the full predicted_Yield sample, far larger than the {162} measured-yield trees). Unlike section 30's unstable, seed-dependent search on real yield, this result is tight across every seed tested ({min(r['final_cvR2'] for r in PYF['robustness_across_5_seeds'])}-{max(r['final_cvR2'] for r in PYF['robustness_across_5_seeds'])}), the signature of reconstructing a formula that genuinely depends on these inputs, not fitting noise. <b>This is not a real accuracy number, it is the clearest possible illustration of the contrast this thesis is built on.</b> The real ceiling (CV R&sup2;={PYF['measured_yield_for_comparison']['recommended_model_cvR2']}) is a <i>data</i> problem: only 162 harvest-measured trees across effectively 2 pooled seasons. Section 30 already showed more feature engineering cannot move it. <b>What would move it is more measured-yield seasons</b>: enough seasons to let climate enter as a continuously-estimable predictor instead of a 2-4 point year label, tighter confidence intervals on the cultivar-specific EBI slopes, and a feature search that would not immediately overfit the way section 30's did. The predicted_Yield side of this figure shows what &ldquo;plenty of data, but the wrong (circular) target&rdquo; looks like; closing the gap on the left panel needs the opposite: the right target, with more of it.")}

  <h3 class="h3">Every EBI/NGRDI scaling, not just EBI_Norm</h3>
  <p style="font-size:13.5px;color:#555">The master carries 3 scalings per index (Raw/Norm/Z, both EBI and NGRDI). Checked directly:
  within a year all 3 are near-perfectly interchangeable (r=0.997-1.000), so which one is used barely matters, the swap test
  confirms this in the pooled model too (all 3 EBI scalings land within 0.0007 CV R&sup2; of each other). All 3 NGRDI scalings
  remain clearly worse than any EBI scaling, confirming the original "drop NGRDI" finding holds across every scaling, not just
  the one first tried.</p>
  {figure_block(IMG["Yield_Model_All_EBI_Variants"], "Figure 30. Swapping in every EBI/NGRDI scaling, and a collinearity trap caught along the way.",
    "Left: CV R&sup2; when each of the 6 scalings replaces the model's bloom term one at a time (cultivar+climate+Growth_April base). Right: forward selection offered all 12 candidates (6 scalings, alone and x cultivar) at once.",
    f"The forward search robustly picks (5 of 5 seeds) BOTH EBI_Raw&times;cultivar and EBI_Norm&times;cultivar together, apparently lifting CV R&sup2; well above either alone. This was checked, not trusted: EBI_Raw and EBI_Norm correlate r=0.9998, so together they are two near-duplicate columns exploiting their own tiny, non-generalizable divergence, a collinearity artifact, not real information (confirmed: each alone gives ordinary, unremarkable gains, only the pair together inflates it). <b>Recommendation: use exactly one EBI scaling, never two together.</b> Separately, EBI&times;cultivar's own marginal contribution over the no-bloom-index base is positive at 5 of 6 seeds tested (mean +0.031), the one negative seed is fold-specific noise at n=167, not evidence against the ANCOVA-validated EBI&times;cultivar finding (F=11.58, p=0.0009, sections 11-12). EBI_Norm remains the model's bloom term, nothing here changes the recommendation.")}

  <h3 class="h3">EBI in interaction with every other feature: one new lead, EBI&times;CNC_June</h3>
  <p style="font-size:13.5px;color:#555">Beyond additive combinations, EBI was multiplied against every other candidate feature
  (Growth_April, climate, DEM, the remaining physiology fields, NGRDI, EBI&sup2;, and two three-way terms), 13
  candidates in all, each tested with a full-sample partial-F test and BH-FDR corrected across the family.</p>
  {figure_block(IMG["Yield_Model_EBI_Interactions"], "Figure 31. EBI in interaction with every other feature, FDR-corrected, and the EBI x CNC_June lead.",
    "Left: partial-F significance of each of the 13 EBI-interaction candidates added to the recommended model, BH-FDR corrected. Right: cross-validated gain from adding EBI x CNC_June across 6 seeds (the historical default seed 42 plus the project's 5 established robustness seeds).",
    f"Two candidates survive FDR: EBI&times;CNC_June (F={AEBI['partial_F_all_13_candidates_FDR_corrected']['EBI_x_CNC_June']['F']}, p_fdr={AEBI['partial_F_all_13_candidates_FDR_corrected']['EBI_x_CNC_June']['p_fdr']}) and EBI&times;Growth_MayJune (F={AEBI['partial_F_all_13_candidates_FDR_corrected']['EBI_x_Growth_MayJune']['F']}, p_fdr={AEBI['partial_F_all_13_candidates_FDR_corrected']['EBI_x_Growth_MayJune']['p_fdr']}), but these are not two independent findings: the two interaction terms correlate r={AEBI['EBI_x_CNC_June_deep_dive']['redundancy_check_vs_EBI_x_Growth_MayJune']['corr_between_the_two_interaction_terms']}, because CNC_June and Growth_MayJune themselves correlate r={AEBI['EBI_x_CNC_June_deep_dive']['redundancy_check_vs_EBI_x_Growth_MayJune']['corr_between_CNC_June_and_Growth_MayJune_raw']} (the same general-vigor axis the section-23 PCA already identified), and adding both together is never better than EBI&times;CNC_June alone. At the model's usual seed (42) the CV delta looks slightly negative ({AEBI['EBI_x_CNC_June_deep_dive']['cv_gain_by_seed']['seed_42_main']}), which alone could look like a null result; checked against the project's 5 established robustness seeds it is positive in all 5 (mean {'+'}{AEBI['extended_model_recommendation']['mean_gain_across_5_established_robustness_seeds']}), and the full-sample partial-F test is clearly significant, so seed 42 is read as fold-specific noise, not evidence against the effect (the same pattern already seen for EBI&times;cultivar's own marginal contribution). <b>Recommendation: cultivar+EBI&times;cultivar+climate+Growth_April+EBI&times;CNC_June</b> is the best-supported specification found to date (in-sample R&sup2; {AEBI['extended_model_recommendation']['in_sample_R2']['base']} to {AEBI['extended_model_recommendation']['in_sample_R2']['extended']}), presented as the leading candidate for the next model revision. Caveats: n=167 pooled across only 2 measured-yield years, CNC_June's own field definition is not documented in the source data, and its raw correlation with EBI reverses by cultivar (UEF {AEBI['EBI_x_CNC_June_deep_dive']['raw_correlation_EBI_vs_CNC_June']['UEF']:+.3f}, cultivar 53 {AEBI['EBI_x_CNC_June_deep_dive']['raw_correlation_EBI_vs_CNC_June']['cultivar_53']:+.3f}), so this is flagged as a lead needing replication, not yet substituted as the baseline throughout this log.")}
  <p style="margin-top:10px;font-size:13.5px;color:#c0392b"><b>Update, see below: this recommendation does not survive
  the correctly-sized FDR family.</b> The 13-candidate family above was EBI-anchored only, a family boundary chosen
  before the full pairwise search below existed. Once every pairwise interaction (not just EBI's) is corrected for
  together, EBI&times;CNC_June's own p_fdr rises from 0.031 to 0.062 and no longer survives. The next subsection
  supersedes the recommendation just above.</p>

  <h3 class="h3">All pairwise interactions, every feature against every feature: the honest full search finds nothing</h3>
  <p style="font-size:13.5px;color:#555">The EBI-anchored family above was itself an arbitrary choice, nothing statistically
  privileges pairs involving EBI over any other pair. This repeats the search without that restriction: all
  {APW['n_base_features']} base features already used anywhere in the yield model (cultivar, EBI, climate, DEM,
  Growth_April, Growth_MayJune, Growth_June, SWP_April, SWP_MayJune, SWP_June, CNC_June, NGRDI) multiplied against
  each other in every possible pair, C({APW['n_base_features']},2)={APW['n_pairs_possible']} minus the EBI&times;cultivar
  baseline term = <b>{APW['n_candidates_tested']} candidates</b>, each tested with the same full-sample partial-F test,
  BH-FDR corrected across all {APW['n_candidates_tested']} together as one family.</p>
  {figure_block(IMG["Yield_Model_All_Pairwise_Interactions"], "Figure 32. All pairwise interactions, every feature x every feature, FDR-corrected.",
    "Left: -log10(p) for the top 20 of 65 candidates from the full-sample partial-F test (red = survives FDR&lt;0.05). Right: forward selection across all 65 candidates at the main seed.",
    f"<b>Zero of {APW['n_candidates_tested']} candidates survive FDR&lt;0.05.</b> EBI&times;CNC_June is unchanged in its own raw statistics (F={APW['partial_F_all_65_candidates_FDR_corrected']['EBI_x_CNC_June']['F']}, p={APW['partial_F_all_65_candidates_FDR_corrected']['EBI_x_CNC_June']['p']}, identical test on identical data) but its p_fdr rises to {APW['partial_F_all_65_candidates_FDR_corrected']['EBI_x_CNC_June']['p_fdr']} once corrected against the full, honest count of comparisons actually made. Four other candidates (Growth_April&times;Growth_June, climate&times;DEM, DEM&times;Growth_MayJune, EBI&times;Growth_MayJune) have equally small raw p-values with no more domain justification, which is itself evidence the smaller family understated the real search space. There is some genuine enrichment (15 of 65 candidates have raw p&lt;0.05 against a chance expectation of ~3.2, and 6 have raw p&lt;0.01 against ~0.7 expected), so this is not pure noise, but no single candidate is strong enough alone to trust individually at n=167. Forward selection across 6 seeds (main plus the 5 established robustness seeds) makes the same point differently: 28 distinct features get selected across the 6 runs, no single one more than 4 of 6 times, well short of the 5-of-5 consistency that validated Growth_April and EBI&times;CNC_June under the smaller family, the signature of overfitting via search rather than a real, reproducible effect. <b>No new interaction term is adopted.</b> The recommended model reverts to cultivar+EBI&times;cultivar+climate+Growth_April (CV R&sup2;=0.456); EBI&times;CNC_June is downgraded from &ldquo;leading candidate&rdquo; to a directional, hypothesis-generating lead worth a dedicated test in a future season, not a term used today.")}

  <h3 class="h3">Recommended structure and honest expectation</h3>
  <p>A hierarchical (mixed) model fits the data-generating structure: a <b>season level</b> where climate
  (Chill Portions, GDD) sets the yearly yield magnitude, and a <b>tree level</b> where cultivar and the
  EBI×cultivar interaction adjust each tree around that level, optionally with a spatial (management-zone)
  random effect. Expectation to state plainly: the <b>large, predictable component is the season</b>, but it
  must be captured through <b>climate features</b> (year is not a feature). Climate reaches CV R&sup2; ~0.17
  today against a seasonal ceiling near {ceil_cv}, and <b>per-tree, within-season accuracy is modest</b>
  (CV R&sup2; ~0.09) because single-tree kernel yield is intrinsically noisy. Closing the gap to the ceiling
  needs <b>more than four seasons</b>, so the climate coefficients can be estimated rather than memorized. A full
  pairwise search across every feature found no additional term that survives correction, EBI&times;CNC_June
  remains a directional lead worth re-testing, not an adopted term.</p>
</section>"""

wb_uef = CIX["by_cultivar_same_year"]["UEF"]["2023"]["WhitenessBrightness"]
wb_53 = CIX["by_cultivar_same_year"]["53"]["2023"]["WhitenessBrightness"]
wb_anc = CIX["ANCOVA_candidates_2023"]["WhitenessBrightness"]
wb_add = CIX["does_WhitenessBrightness_add_to_EBI_2023"]
colorindices = f"""
<section id="colorindices" class="section">
  <h2><span class="sic" style="background:#8E44AD">&#127752;</span>Testing more color indices against yield</h2>
  <p class="lead">Chen, Jin &amp; Brown (2019), the paper that originates the EBI formula used throughout this
  thesis, suggests in its discussion that a bloom index can be "customized" by swapping which channel
  combination stands for flower colour. Standard RGB vegetation-index literature (Excess Green, VARI, GLI,
  CIVE, RGBVI) offers other channel combinations never tried on this data. Eight candidates were computed
  from the crown-mean R, G, B already extracted per tree (no new image processing needed) and run through
  the same same-year, within-cultivar yield battery already used for EBI, FDR-corrected across the 8
  candidates in the inferential 2023 test.</p>
  {figure_block(IMG["Color_Indices_Yield"], "Figure 17. Eight candidate RGB color indices vs measured yield, 2023, by cultivar.",
    "Panel A is the same-year Pearson r between each candidate index and measured 2023 yield, within each cultivar (compare to EBI: UEF r=+0.29, cultivar 53 r=-0.26). Panels B-C scatter the best-performing candidate for each cultivar with its fitted line.",
    f"The best candidate in both cultivars is a customized Brightness&times;whiteness term (bright and achromatic pixels): UEF r={wb_uef['r']:+.3f} (p={wb_uef['p']}, n={wb_uef['n']}), cultivar 53 r={wb_53['r']:+.3f} (p={wb_53['p']}, n={wb_53['n']}). Both reproduce EBI's sign reversal, but neither survives FDR correction across the 8 candidates (p_fdr &ge; 0.16 in both). Its ANCOVA interaction (F={wb_anc['EBIxcultivar_interaction_analog']['F']}, p={wb_anc['EBIxcultivar_interaction_analog']['p']}, R&sup2; {wb_anc['R2']} to {wb_anc['R2_with_interaction']}) matches but does not exceed EBI's own (F=11.58, R&sup2; 0.079 to 0.147), and adding it <b>on top of</b> EBI adds nothing (partial F={wb_add['partial_F_extra_terms']}, p={wb_add['p']}, R&sup2; {wb_add['R2_EBI_only']} to {wb_add['R2_EBI_plus_extra']}). The other seven candidates are weaker still.")}
  <p style="margin-top:10px;font-size:13.5px;color:#555"><b>Why this is expected, not a coding error:</b>
  every candidate, like EBI, is built from the crown <i>mean</i> of R, G, B, one averaged colour value per
  tree per year. Averaging discards exactly the pixel-level information the source paper actually validated
  (their EBI-to-bloom-coverage R&sup2;=0.72 compared a pixel-mean index against a pixel-<i>classified</i>
  bloom fraction, not another crown-mean formula).</p>

  <h3 class="h3">The real pixel-level bloom fraction was extracted and tested, same result</h3>
  {figure_block(IMG["Bloom_Fraction_Yield"], "Figure 18. Otsu-thresholded pixel-level bloom fraction vs measured yield, 2023.",
    "BloomFraction is the per-crown percentage of pixels classified bright-and-achromatic by an unsupervised Otsu threshold (computed per year on the radiometrically-corrected mosaic), the direct analogue of the paper's SVM bloom coverage. Panel A compares its same-year Pearson r to EBI's, by cultivar; panels B-C scatter it against 2023 yield with the partial-F test of whether it adds anything on top of EBI.",
    f"BloomFraction reproduces EBI's sign reversal: UEF r={BFY['by_cultivar_same_year']['UEF']['2023']['BloomFraction']['r']:+.3f} (a little weaker than EBI's +0.29), cultivar 53 r={BFY['by_cultivar_same_year']['53']['2023']['BloomFraction']['r']:+.3f} (a little stronger than EBI's -0.26, and this one survives a 2-candidate FDR). Its ANCOVA interaction (F={BFY['ANCOVA_2023']['BloomFraction']['EBIxcultivar_interaction_analog']['F']}, R&sup2; {BFY['ANCOVA_2023']['BloomFraction']['R2']} to {BFY['ANCOVA_2023']['BloomFraction']['R2_with_interaction']}) matches, not exceeds, EBI's own. The decisive test is whether it adds anything <b>on top of</b> EBI: it does not, pooled partial F={BFY['does_BloomFraction_add_to_EBI_pooled_2023']['partial_F_extra_terms']} (p={BFY['does_BloomFraction_add_to_EBI_pooled_2023']['p']}, ns), and non-significant within UEF (p={BFY['does_BloomFraction_add_to_EBI_within_cultivar_2023']['UEF']['p']}) and within cultivar 53 (p={BFY['does_BloomFraction_add_to_EBI_within_cultivar_2023']['53']['p']}) separately.")}
  <h3 class="h3">Per-canopy outlier removal: negligible effect (a reassuring check)</h3>
  {figure_block(IMG["Outlier_Robust_EBI"], "Figure 19. Does removing outlier pixels per crown change EBI or the yield finding?",
    "Since EBI_ofMeans averages every pixel in a crown, extreme pixels (sensor glare, shadow gaps, crown-mask boundary pixels) could in principle pull the mean. Method A drops pixels via a per-crown modified Z-score (adaptive, |z|&gt;3.5 in any channel); method B is a cruder fixed 5-95th percentile brightness trim. Panel A shows how much each method removes per crown by year; panel B compares the yield correlation using plain vs outlier-robust EBI; panel C scatters per-tree EBI, plain vs robust.",
    f"Method A flags almost nothing: {ORY['Q1_how_much_does_it_move_EBI']['2021']['mean_pct_pixels_flagged_outlier_A']}% to {ORY['Q1_how_much_does_it_move_EBI']['2022']['mean_pct_pixels_flagged_outlier_A']}% of pixels per crown across all four years, and plain-vs-robust EBI correlates at r&ge;0.987. Every yield result is essentially unchanged: 2023 UEF r={ORY['Q2_yield_battery_plain_vs_robust_vs_trim']['UEF']['2023']['plain']['r']} (plain) vs {ORY['Q2_yield_battery_plain_vs_robust_vs_trim']['UEF']['2023']['robust']['r']} (robust); cultivar 53 r={ORY['Q2_yield_battery_plain_vs_robust_vs_trim']['53']['2023']['plain']['r']} in both. The ANCOVA interaction is F={ORY['ANCOVA_2023']['plain']['EBIxcultivar_interaction']['F']} (plain) vs F={ORY['ANCOVA_2023']['robust']['EBIxcultivar_interaction']['F']} (robust). There simply is not much of an outlier problem within crowns to begin with, so this was a reasonable concern to test, but the crown mean was not being meaningfully distorted by pixel noise.")}
  <p style="margin-top:10px;font-size:13.5px;color:#555">Three independent checks, reformulating the
  crown-mean colour combination, moving to true pixel-level bloom classification, and removing per-canopy
  pixel outliers, all converge on the same conclusion: the modest EBI-yield effect size (R&sup2; 0.08-0.15)
  is not an artefact of the index formula, the pixel aggregation method, or noisy averaging. It reflects a
  real limit of what spring bloom-season RGB colour predicts about per-tree kernel yield at this sample
  size. EBI is treated as adequate for this thesis's yield analysis going forward.</p>
</section>"""

j24_uef = J24["by_cultivar_year"]["UEF"]["2024"]; j24_53 = J24["by_cultivar_year"]["53"]["2024"]
leverage = f"""
<section id="leverage" class="section">
  <h2><span class="sic" style="background:#2C3E50">&#128269;</span>Leveraging the RGB mosaics further</h2>
  <p class="lead">Sections 19-19.2 (colour reformulation, pixel-level bloom fraction, outlier-robust means)
  all failed to beat plain EBI. This section reviews the wider literature on turning UAV RGB imagery into
  yield predictors in tree crops and tests the one candidate answerable without new extraction.</p>

  <h3 class="h3">What the literature says actually works</h3>
  <div class="grid3">
    {card("Canopy light interception (the key almond driver)", "Lampinen et al. 2012 and Zarate-Valdez et al. 2015 link canopy light interception directly to maximum potential almond yield, and show it can be estimated from plain RGB photographs of canopy shadow (R&sup2;=0.95 vs a ceptometer light bar), an RGB-only method never tried on this data.", "#16A085")}
    {card("Canopy area/volume, R&sup2; 0.71-0.98", "Multiple UAV-RGB fruit-tree studies (apple, olive, strawberry) link segmented canopy size to yield at R&sup2; 0.71-0.98. The signal is canopy SIZE, not colour, a completely different axis from every index tested in section 19.", "#16A085")}
    {card("Texture (GLCM) adds skill on top", "Cotton and rice studies show texture features (contrast, entropy, homogeneity) capture heterogeneity a single mean-colour index misses, usually an incremental addition to a structural/spectral feature rather than a standalone win.", "#4aa3df")}
    {card("Summer/red-edge imagery is the strongest published almond predictor", "Jin et al. 2023 trained a CNN on 30cm, 4-band SUMMER imagery over ~2,000 harvested almond trees, reaching R&sup2;=0.96 (red edge the top feature), the almond-yield-from-imagery ceiling in the literature, and it is canopy state near harvest, not spring bloom colour.", "#E67E22")}
    {card("Deep learning: not recommended here", "That R&sup2;=0.96 result used ~2,000 individually harvested trees. This project has 161 (64-90 per cultivar-season). A CNN here would almost certainly overfit; revisit only if the measured-yield panel grows substantially.", "#95a5a6")}
    {card("Crown area: not yet tried, cheapest to add", "The tree-detection model already computes per-instance pixel area internally as a segmentation filter but never exports it. Extracting it needs no new segmentation, only reusing the existing zone/canopy-mask machinery from sections 19.1-19.2.", "#8E44AD")}
  </div>

  <h3 class="h3">New test: does the July-2024 multispectral acquisition predict measured yield?</h3>
  {figure_block(IMG["Jul2024_Spectral_Yield"], "Figure 20. July-2024 multispectral (NDVI, NDRE, EVI, CWSI, canopy temperature, etc.) vs measured yield.",
    "July 2024 is one date, so only the 2024 comparison is causally sensible (canopy state before that year's harvest); panel A shows all 13 indices, FDR-corrected across them within each cultivar. Panels B-C scatter each cultivar's best candidate.",
    f"Nothing survives FDR at this small n (8-9 per cultivar). The largest raw correlation is cultivar 53's water-stress signal, CWSI and canopy temperature both r=-0.715 (p=0.046 raw, p_fdr=0.299, ns, n=8), direction sensible but not significant after correction. UEF's best is the chlorophyll index CI, r={j24_uef['CI_Jul2024']['r']:+.3f} (ns). Striking, though not itself a significant finding: every one of the 13 indices runs negative for cultivar 53 and positive for UEF (except CWSI/Tc), the same cross-cultivar sign-reversal already seen in bloom EBI (section 11) and pixel-level bloom fraction (section 19.1), now recurring in a third, independent imagery source and season.")}

  <h3 class="h3">Ranked next steps (not yet built)</h3>
  <div class="grid3">
    {card("1. Crown area per tree per year", "Highest confidence, cheapest: reuses the existing zone/mask pipeline, no new segmentation. Canopy size is orthogonal to every colour index tried so far.", "#16A085")}
    {card("2. Canopy-shadow light interception", "Highest literature validation (R&sup2;=0.95, Zarate-Valdez 2015), more effort: needs a ground buffer beyond each crown and a shadow/sunlit classification within it.", "#4aa3df")}
    {card("3. GLCM texture per crown", "Moderate effort, uncertain payoff at this sample size (64-90 per cultivar-season); worth trying after 1-2, not as a standalone feature.", "#8E44AD")}
  </div>
</section>"""

gpy_q1_53 = GPY["Q1_same_year_by_cultivar"]["53"]["2023"]["SWP_June"]
gpy_q2_swp = GPY["Q2_ANCOVA_2023"]["SWP_June"]
gpy_q3_pooled_swp = GPY["Q3_does_it_add_to_EBI_pooled_2023"]["SWP_June"]
gpy_q3_53_swp = GPY["Q3_does_it_add_to_EBI_within_cultivar_2023"]["SWP_June"]["53"]
gpy_q0_53_swp22 = GPY["Q0_candidate_vs_EBI_same_year"]["2022"]["53"]["SWP_June"]
gpy_q0_53_cnc22 = GPY["Q0_candidate_vs_EBI_same_year"]["2022"]["53"]["CNC_June"]
physiology = f"""
<section id="physiology" class="section">
  <h2><span class="sic" style="background:#2C3E50">&#128167;</span>Ground physiology in the yield GPKGs: SWP, growth, CNC</h2>
  <p class="lead">The 2022/2023 <code>Yield_with_clustering</code> GPKGs carry seven ground-measured
  physiological fields beyond <code>predicted_Yield</code>, never analysed before: stem water potential
  (SWP) and canopy growth at three points across the season (April, May-June, June), plus a June metric
  labelled CNC (exact definition undocumented anywhere in the project). Unlike every candidate in the
  sections above, these are ground physiology, not remote sensing, so this is genuinely new information
  rather than another reformulation of the same imagery. Matched to the master within 3&nbsp;m (100%
  cultivar agreement up to 5&nbsp;m; validated against the master's own <code>predicted_Yield_2023</code>,
  mean absolute difference 0.0015&nbsp;kg), Plot A only (the GPKGs also cover Plots B and C, outside this
  thesis's scope).</p>
  {figure_block(IMG["GPKG_Physiology_Yield"], "Figure 25. Stem water potential, growth and CNC vs measured yield, 2023, by cultivar.",
    "Panel A is the same-year Pearson r for each of the 7 candidates within each cultivar, FDR-corrected (compare to EBI: UEF r=+0.29, cultivar 53 r=-0.26). Panels B-C scatter the two most informative candidates against 2023 yield.",
    f"Nothing survives FDR. The closest any candidate has come, across this whole leverage exercise, is SWP_June: cultivar 53's same-year correlation r={gpy_q1_53['r']:+.3f} (p={gpy_q1_53['p']} raw, p_fdr={gpy_q1_53['p_fdr']}, ns, n={gpy_q1_53['n']}); its ANCOVA interaction F={gpy_q2_swp['interaction']['F']} (p={gpy_q2_swp['interaction']['p']}, borderline, R&sup2; {gpy_q2_swp['R2']} to {gpy_q2_swp['R2_with_interaction']}); and the decisive nested test, pooled partial F={gpy_q3_pooled_swp['partial_F_extra_terms']} (p={gpy_q3_pooled_swp['p']}), within cultivar 53 alone partial F={gpy_q3_53_swp['partial_F_extra_term']} (p={gpy_q3_53_swp['p']}, R&sup2; nearly doubling {gpy_q3_53_swp['R2_EBI_only']} to {gpy_q3_53_swp['R2_EBI_plus_extra']}). Right at the conventional significance boundary, and would not survive correcting for 7 candidates, so this stays a negative result by the same standard as sections 19-20.5, but it is the closest, and specific to cultivar 53, the cultivar carrying the bloom-yield reversal.")}
  <p style="margin-top:10px;font-size:13.5px;color:#555"><b>A bonus, better-powered signal: physiology
  correlates with EBI itself.</b> In 2022 (all matched trees, large n), SWP_June and CNC_June both
  correlate negatively with that season's EBI in both cultivars, strongest in cultivar 53 (SWP_June
  r={gpy_q0_53_swp22['r']:+.3f}, p&lt;0.001, n={gpy_q0_53_swp22['n']}; CNC_June r={gpy_q0_53_cnc22['r']:+.3f}, p={gpy_q0_53_cnc22['p']}, n={gpy_q0_53_cnc22['n']}).
  A plausible reading, not a causal claim: SWP_June is measured months after spring bloom, so a brighter
  bloom (heavier crop-load demand) tracking more water stress later the same season is a sensible
  within-season story; if CNC does track a nitrogen/vigour metric, its negative link to bloom would match
  the classic vegetative-reproductive trade-off. 2023 shows a much weaker version of the same pattern, so
  this is suggestive, not established.</p>
  <p style="margin-top:10px;font-size:13.5px;color:#555">Sixth independent negative result for the yield
  question, but the first ground-physiology test, and the nearest anything has come to adding value beyond
  EBI. Flagged as the most promising lead for future work (a larger measured-yield panel, or a tighter
  SWP_June protocol), not a change to the recommended yield model.</p>

  <h3 class="h3">A user-provided RF model package: provenance, and testing MDS</h3>
  <p>Two files added to <code>Trees_Data_Survey/</code> turned out to be exact duplicates of the GPKGs
  already used above (MD5-identical). The accompanying <code>RF_models_portable_package/</code> documents
  five fitted R random-forest models (CNC, Growth, MDS, SWP, Yield) and confirms, in its own README, that
  the Yield model predicts "yield, using predicted physiological measurements", a two-stage pipeline
  (predict physiology from spectral+meteo, then predict yield from that predicted physiology). This is the
  concrete mechanism behind the project's existing circularity caveat on <code>predicted_Yield</code>. No R
  is available in this sandbox, so the fitted <code>.rds</code> models could not be re-run; one exact-value
  check on the training CSVs confirms their "Yield" column is genuine measured ground truth (tree A.1, 2022:
  6.263631838&nbsp;kg, identical to the census). A join hazard was found and avoided: two of the package's
  CSVs use a TreeID that matches the census exactly for 2022 but not at all for 2023 (would have silently
  paired the wrong trees), so only the one file that checked out cleanly (20/20 matched, both years) was used.</p>
  {figure_block(IMG["MDS_Feature_Test"], "Figure 28. Maximum daily shrinkage (MDS), a genuinely new physiology variable, vs measured yield.",
    "MDS is measured for a fixed 20-tree Plot-A panel across 2022-2023 (April 2022 only, June both years), matched to master via the existing kedma haversine join (20/20 matched, 20/20 cultivar agreement).",
    f"Every correlation is non-significant (|r| 0.14-0.73, n=6-10 per cell, small-sample directional only). The decisive pooled test, does MDS_June add anything on top of cultivar+EBI&times;cultivar (n={MDSJ['Q3_pooled_MDS_June_adds_to_EBI']['n']}), gives partial F={MDSJ['Q3_pooled_MDS_June_adds_to_EBI']['partial_F']} (p={MDSJ['Q3_pooled_MDS_June_adds_to_EBI']['p']}), essentially zero, the cleanest negative result of the whole exercise. Ninth independent negative result; no change to the recommended model.")}
</section>"""

summary = f"""
<section id="summary" class="section">
  <h2><span class="sic" style="background:#2C3E50">&#9733;</span>Results summary</h2>
  <div class="grid3">
    {card("Mean EBI rises, spread tightens", "In UEF + 53 the mean climbs 0.513 to 0.551 (all shifts p&lt;0.001) and SD falls; the old flat-mean picture was a cultivar-54 artifact.", "#C0392B")}
    {card("Cultivar effect converges", "EBI differs strongly by cultivar in 2021 (eta^2 0.34) and vanishes by 2024 (F=0.47, ns), with a 53-vs-UEF rank flip near 2023.", "#8E44AD")}
    {card("Bloom is spatially clustered", "Moran's I is positive and significant every year (0.23 to 0.35), peaking 2023, indicating an environmental/management gradient.", "#16A085")}
    {card("Chilling: directional only", "At n=4, higher Chill Portions track lower mean and more heterogeneous bloom, but nothing is significant; tree-level analyses carry inference.", "#4aa3df")}
    {card("Cultivar drives measured yield", "UEF out-yields 53 by ~0.66 kg/tree (F=13.46, p=0.0003); mean spring bloom does not predict yield.", "#E67E22")}
    {card("EBI to yield reverses by cultivar", "UEF r=+0.29, cultivar 53 r=-0.26; interaction F=11.58 (p&lt;0.001). This sign reversal is the substantive bloom-yield finding.", "#2C3E50")}
    {card("Climate to bloom: no effect", "Within both cultivars, Chill Portions do not drive mean EBI (UEF r=-0.35, 53 r=+0.20; n=4, ns). Bloom intensity is not chill-controlled at this scale.", "#4aa3df")}
    {card("Climate to yield: the year rules", "The season carries the yield signal: the joint model's year/climate effect is highly significant in both cultivars (UEF F=31.8, 53 F=37.9, p&lt;0.001); 2022 was the high-yield year.", "#16A085")}
    {card("Three together, one sentence", "The year (climate) sets how much the orchard yields; within a year, bloom raises yield in UEF (r=+0.29, p=0.007) and lowers it in cultivar 53 (r=-0.26, p=0.042).", "#8E44AD")}
  </div>
  <p class="foot">Source data: <code>Master_Trees_Extended.xlsx</code> (1,294 trees kept) and
  <code>kedma_plot_a_yield.csv</code>. Produced by <code>thesis_analysis_uef53.py</code> /
  <code>figures_uef53.py</code>; machine-readable results in <code>thesis_results_uef53.json</code>.</p>
</section>"""

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Kedma Almond Orchard - Bloom, Cultivar, Climate and Yield</title>
<style>
  :root{{--bg:#f6f7fb;--panel:#fff;--ink:#1f2430;--muted:#5a6270;--line:#e6e8ef;--accent:#5b6cff}}
  *{{box-sizing:border-box}}
  body{{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink);line-height:1.55}}
  .layout{{display:flex;min-height:100vh}}
  aside{{position:fixed;top:0;left:0;width:220px;height:100vh;background:var(--panel);border-right:1px solid var(--line);padding:20px 14px;overflow:auto}}
  aside h1{{font-size:15px;margin:0 0 4px}}
  aside .sub{{font-size:11px;color:var(--muted);margin-bottom:16px}}
  aside a{{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:9px;color:var(--ink);text-decoration:none;font-size:13px;margin-bottom:2px}}
  aside a:hover{{background:#eef0f8}}
  .dot{{width:22px;height:22px;border-radius:6px;display:inline-flex;align-items:center;justify-content:center;color:#fff;font-size:12px;flex:none}}
  main{{margin-left:220px;flex:1;padding:0 0 60px}}
  .hero{{background:linear-gradient(120deg,#5b6cff,#8E44AD);color:#fff;padding:38px 40px}}
  .hero h1{{margin:0 0 8px;font-size:26px}}
  .hero p{{margin:0;opacity:.92;max-width:820px}}
  .badge{{display:inline-block;background:rgba(255,255,255,.2);border:1px solid rgba(255,255,255,.35);padding:3px 10px;border-radius:20px;font-size:12px;margin-top:12px}}
  .section{{max-width:980px;margin:34px auto;padding:0 28px}}
  .section h2{{display:flex;align-items:center;gap:12px;font-size:21px;border-bottom:2px solid var(--line);padding-bottom:10px}}
  .sic{{width:34px;height:34px;border-radius:9px;display:inline-flex;align-items:center;justify-content:center;color:#fff;font-size:18px}}
  .lead{{font-size:15px;color:#333}}
  .h3{{font-size:16px;margin:26px 0 4px;padding-left:10px;border-left:4px solid #C0392B;color:#2C3E50}}
  .fig{{margin:18px 0;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 1px 3px rgba(20,24,40,.05)}}
  .fig img{{width:100%;border-radius:8px;display:block}}
  .fig figcaption{{font-size:13px;color:var(--muted);margin:10px 2px 0;font-style:italic}}
  .explain{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}}
  .what,.reading{{background:#f8f9fd;border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
  .reading{{background:#f3f6ff}}
  .tag{{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.03em;text-transform:uppercase;color:#5b6cff;background:#e9ecff;padding:2px 8px;border-radius:20px;margin-bottom:6px}}
  .tag2{{color:#16A085;background:#e4f5f0}}
  .what p,.reading p{{margin:0;font-size:13.5px}}
  .grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:16px}}
  .card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
  .card h4{{margin:0 0 6px;font-size:14px}}
  .card p{{margin:0;font-size:13px;color:#444}}
  .foot{{font-size:12px;color:var(--muted);margin-top:20px}}
  code{{background:#eef0f8;padding:1px 5px;border-radius:5px;font-size:12px}}
  @media(max-width:860px){{aside{{display:none}}main{{margin-left:0}}.explain,.grid3{{grid-template-columns:1fr}}}}
</style></head>
<body><div class="layout">
  <aside>
    <h1>Kedma Plot A</h1>
    <div class="sub">UEF and cultivar 53 &middot; Chill Portions</div>
    {nav}
  </aside>
  <main>
    <div class="hero">
      <h1>Bloom, Cultivar, Climate and Yield</h1>
      <p>UAV-measured bloom phenology in the Kedma almond orchard, linking flowering intensity to
      cultivar, climate and field-measured yield for the two yield-measured cultivars, UEF and cultivar 53.
      Chilling is quantified by Chill Portions (Dynamic Model).</p>
      <span class="badge">1,294 trees &middot; 2021-2024 &middot; field-measured yield</span>
    </div>
    {overview}{ebi}{cultivar}{spatial}{climate}{dormancy}{phenologynorm}{yield_}{hypotheses}{bycultivar}{triangle}{yieldmodel}{colorindices}{leverage}{physiology}{summary}
  </main>
</div></body></html>"""

out = os.path.join(BASE, "Analysis_Presentation_UEF53.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print("Wrote", out, f"({os.path.getsize(out)/1024:.0f} KB)")
