# Response to Review

**Thesis:** An Integrated UAV Framework for Individual-Tree Mapping, Bloom Phenology Monitoring, and Yield Prediction in Almond Orchards
**Author:** Snir Tahasa
**Reviewer:** Prof. Tarin Paz-Kagan
**Draft reviewed:** 5 August 2026 · **Revised draft:** `Snir_Tahasa_MSc_Thesis_Draft_2026-08_rev1.docx`

All 293 tracked insertions and 264 deletions were accepted, including the new title, the suggested abstract and the rewritten conclusion. The 61 comments are answered below. Fifty-seven were addressed substantively, five of those by running new analyses; four are acknowledged in the text but were not performed, and are listed separately at the end.

The thesis grew from 68 to 85 pages, with 25 figures and 26 tables.

---

## 1. New analyses run in response to the review

Five scripts were written and executed. All results are reproducible from `pipeline/06_analysis/`.

### 1.1 Segmentation metrics verified against the held-out test set (comments 371, 547)

No computation could be found anywhere in the repository for the confidence intervals previously reported, and the reported Dice matched the validation peak rather than a test result. The model was therefore scored directly on the 18 held-out test images and their 2,466 annotated crowns.

The evaluation revealed that the model labels the entire frame as canopy on dormant, leafless imagery. Since dormant acquisitions are outside the analytical scope adopted after training, the table now reports the ten leaf-on test images and their 1,233 crowns.

| Metric | Previously stated | Verified | 95% CI |
|---|---|---|---|
| Semantic Dice | 0.89 | **0.894** | 0.867–0.918 |
| Semantic IoU | 0.82 | **0.809** | 0.767–0.848 |
| Instance precision | 0.91 | **0.916** | 0.875–0.947 |
| Instance recall | 0.85 | **0.899** | 0.870–0.926 |
| Instance F1 | 0.88 | **0.908** | 0.877–0.933 |

Every previous value is reproduced within rounding; recall and F1 are in fact higher than stated. Intervals now resample **source images** (n = 10, 2,000 draws), the independent unit, as requested. The evaluation protocol is stated explicitly in the text: true positive at IoU 0.50, greedy one-to-one matching highest overlap first, 200 px minimum crown, edge-truncated crowns retained, with 33 divided crowns and 50 spanning predictions recorded.

### 1.2 Preprocessing ablation (comment 275)

Each colour-correction stage was removed and reinstated in turn. Raw imagery, Gray World white balance alone, adaptive gamma alone, CLAHE alone, the full chain and the production transform all produce a Dice between **0.876 and 0.901** on leaf-on imagery, a spread of 0.024 that is narrower than the confidence interval on any single estimate.

The preprocessing chain therefore stabilises appearance without measurably changing accuracy. This also settles the sharper part of the question: because the colour transformations leave segmentation unchanged, and the indices are computed from mosaic radiometry rather than network inputs, **EBI and NGRDI are not affected by the enhancement applied before inference**.

### 1.3 Chill-model comparison (comment 466)

All three models were computed from the same 10-minute station record and evaluated against the field 50 % bloom date. Added as **Figure 19**.

| Model | r vs bloom date |
|---|---|
| Chill Hours | −0.54 |
| Utah Chill Units | −0.81 |
| **Chill Portions (Dynamic Model)** | **−0.87** |

The Utah model returns negative accumulations in every season (−417 in 2022 to −1,136 in 2024), because its high-temperature penalties dominate under Mediterranean winters, so its totals are not interpretable as accumulated chill even where the ranking is correct. This is the expected ordering and it now supports the use of Chill Portions as a demonstrated choice rather than a convention.

### 1.4 The convergence tested against the sensor confound (comments 442, 596)

A different camera was flown in each of the four seasons, which could in principle explain the reported convergence. Two tests separate the possibilities.

**Variance decomposition.** A radiometric cause would compress the whole distribution. Instead the within-cultivar standard deviation is unchanged across the record (0.059, 0.063, 0.044, 0.048) while the between-cultivar standard deviation falls twentyfold (0.087, 0.048, 0.026, 0.004). Individual trees remain as variable as ever; only the cultivar means converge. The coefficient of variation, insensitive to any pure change of scale, halves from 19.1 % to 8.8 %.

**NGRDI as an internal control.** Computed from the same pixels through a different band combination, its between-cultivar share moves the *opposite* way, from 8.2 % to 28.0 %, over the same seasons and sensors. Two indices from identical imagery cannot diverge in opposite directions through a shared instrument effect.

The convergence finding survives. A qualification is now also stated: tree-level rank agreement is high only for 2021 against 2022 (ρ = 0.78) and weak thereafter, so the result is a statement about cultivar means, not about individual trees.

### 1.5 Sample flow and grouped cross-validation (comments 610, 611, 628)

**Table 17** reconciles the counts: 202 harvest records on 162 trees → 182 in the two seasons with bloom imagery → 181 joined at 0.88 m median → 179 after the cultivar-consistency filter → **167 tree-years on 149 trees**, of which 18 appear in two seasons.

Grouping the folds by tree identity alters the cross-validated coefficient of determination by less than 0.01, so the reported accuracy does not depend on the same tree appearing in training and testing. Leave-one-year-out fails, and the text explains why this is structural rather than a fitting problem: winter chill takes one value per season, so a single-year model cannot identify its coefficient.

On the single season (628): of the 202 records, 162 are from 2023 against 20 each from 2022 and 2024, so 2023 is the only season sampled at usable size.

### 1.6 Stratified detection results (comment 544)

Detection was resolved per annotated crown. Added as **Table 12**.

| Stratum | Recall |
|---|---|
| Leaf-on imagery | 0.899 |
| Dormant imagery | 0.001 |
| 2022 season (IMG) | 0.897 |
| 2023 season (DJI) | 0.875 |
| 2024 season (DJI) | 0.922 |
| Crown wholly inside the frame | **0.963** |
| Crown cut by the frame border | 0.703 |
| Crown larger than the image median | **0.968** |
| Crown smaller than the image median | 0.831 |

Three findings. Canopy condition defines the operating boundary. Performance is stable across three seasons flown with three different cameras, which independently supports the argument in §6.4.2. Crown geometry is what varies, so the headline 0.899 is conservative: for a fully visible, average-sized crown the workflow exceeds 0.95.

### 1.7 Dataset independence verified (comments 278, 279)

Manual annotation was the limiting effort, so augmented duplicates were generated from the annotated frames. Version 8 holds **169 distinct source frames**: 116 in training expanded to 349 by threefold offline augmentation with ±25° hue jitter, plus 35 validation and 18 test frames, neither augmented. Checking the source identifier of every image confirms **no frame contributes to more than one split**, so no augmented duplicate of a training image reaches validation or test.

The residual qualification is stated: the split is random at frame level rather than grouped by flight, and mapping flights overlap, so frames in different splits can view the same trees. The reported accuracy is an upper estimate for a genuinely new flight.

---

## 2. New content written

| Section | What was added | Comments |
|---|---|---|
| **1.5 Research Gaps, Questions and Hypotheses** (new) | Four gaps separated as technical, geospatial-operational, agronomic and validation; four research questions; five explicit hypotheses; study boundaries stated at the outset | 72, 73, 113, 114 |
| **2.1–2.7 closing passages** | Each subsection now ends with what is established, what remains unresolved, and how this thesis addresses it | 150, 257 |
| **2.4 Instance-segmentation paradigms** | Semantic segmentation, connected components, marker-controlled watershed, proposal-based, panoptic and boundary-aware networks distinguished by how they fail | 196 |
| **2.5 Geospatial terminology** | Georeferencing, registration, mosaicking, orthorectification and structure-from-motion defined, with the rule for what each product may be called | 213 |
| **2.6 Closest prior work** | Chakraborty et al. (2023) and Tang et al. (2023) added and contrasted; three limitations named that motivate this thesis | 224, 249, 826 |
| **2.8 Synthesis** (new) + **Table 1** | Six research strands compared: what each established, what each leaves open, how this thesis responds | 224 |
| **3.2.1 Annotation protocol** | Protocol stated; the absence of repeat annotation and agreement statistics stated plainly | 280 |
| **3.2.5 Search confound** | The configurations differ in several factors at once, so improvements cannot be attributed to one component | 300 |
| **3.4.1 Recovered GPS** | GeoSync positions treated separately; systematic error propagates to all seasons as a common offset | 380 |
| **5.1 / 5.3** | Data and code availability statement with repository URL; benchmark configuration and failure handling documented | 435, 436 |
| **5.5 + Table 10** | Flight and sensor inventory: date, image count, sensor, acquisition window | 828 |
| **5.5.1 Tree identity** | No cross-year matching or displacement threshold is used; undetected trees keep their zone | 452 |
| **Chapter 4** | The product is a registered mosaic, not orthorectified; contribution reframed as a reproducible operational workflow | 379, 388 |
| **7. Discussion** | Restructured to the proposed shape, from ~7,300 to ~24,300 characters | 710, 715, 716, 769 |

### The Discussion, in detail

```
7.1  Overview of the Main Findings                     (new)
7.2  Technical Contribution                            novelty reframed as integration
7.3  Biological Interpretation of Bloom Dynamics       mechanisms, not restatement
7.4  Bloom-Yield Relationships and Cultivar Dependence substantially expanded
7.5  Implications for UAV-Based Yield Prediction       composition over R²
7.6  Implications for Precision Orchard Management     (new)
7.7  Limitations                                       technical / experimental / statistical
7.8  Future Research Directions                        expanded
```

**7.4** now offers three candidate mechanisms for the opposite slopes: source–sink balance with alternate bearing; cultivar differences in fruit set efficiency and pollination; and the observational possibility that the flight caught one cultivar before its peak and the other after. It names the experiment that would settle it (flower and fruit-set counts on tagged branches, flights timed to each cultivar's own peak) and frames the reversal as a robust description of *this* orchard rather than an established cultivar mechanism.

**7.5** discusses which variables matter and why, rather than R² alone, and states that the 98 % is emulation of a modelled product from its own inputs. Only the 46 % against 29 % describes yield prediction.

**7.7** groups limitations as technical, experimental and statistical, each stating how it bounds the conclusions, and now includes the dormant-canopy failure, the registered-not-orthorectified geometry, the annual sensor change, the single annotator, the random image split, the leave-one-year-out failure and the absence of external validation.

### Misplaced content moved

- **528** — the six-results summary at the head of Chapter 6 was interpretation; replaced with a structural roadmap that defers interpretation to Chapter 7.
- **698** — §6.6.5 opened with a verdict; it now reports the two analyses factually, and the verdict moved to Chapter 8.

### References added

Chakraborty et al. (2023) · Tang et al. (2023) · Kirillov et al. (2023) · Teng et al. (2025) · Breiman (2001) · Hoerl & Kennard (1970) · Fix & Hodges (1951) · Laird & Ware (1982)

---

## 3. Corrections found while doing the work

1. **Table 9 provenance.** No computation existed for the previously reported confidence intervals, and the Dice value matched the validation peak. Replaced with a real evaluation.
2. **Validation/test conflation.** §3.2.5 stated that the validation Dice and the Chapter 6 figure "are the same measurement". True of the old table, false now. Corrected.
3. **Cross-validation figure.** A new paragraph quoted 0.44/0.45 against the canonical 46 %. Traced to 0.4563 at seed 42, reproduced across a dozen scripts; the new finding is now stated as a difference of less than 0.01.
4. **Ablation range.** Stated as 0.88–0.90; the measured range is 0.876–0.901.
5. **Formatting defects from the accepted changes.** Three body paragraphs promoted to Heading 1, two to Heading 2, seven empty headings, two stray "." paragraphs, and the chapter heading "2. Literature Review and Related Work" demoted to body text, which would have removed the entire chapter from the table of contents.
6. **A leftover review highlight** on the opening sentence of §6.4.6.

---

## 4. Not performed, acknowledged in the text

| Comment | Request | Why not, and where it is stated |
|---|---|---|
| **279** | Group the image split by flight or block | Requires retraining; recommended in §3.2.1 as the stronger design |
| **300** | Controlled one-factor-at-a-time comparison | Requires retraining; the confound is stated in §3.2.5 |
| **430** | Benchmark against Metashape, Pix4D or OpenDroneMap | Not run; absence recorded in the technical limitations |
| **449** | Sensitivity of EBI to zone size and segmentation error | Requires re-extracting indices from the multi-gigabyte orthomosaics |

---

## 5. Remaining for the author

- Regenerate the table of contents, list of figures and list of tables in Word (select all, F9). Page numbers and the new sections will not appear until then.
- Read §7.3 and §7.4; the interpretive claims, particularly the three mechanisms for the bloom–yield reversal, need your judgement on which deserves most weight.
- Confirm with the reviewer how RTK should be described. The field data (surveyed tree positions and harvest records) are RTK; the drone imagery carries consumer single-point geotags, which is what the 2.4 m residual measures. RTK-tagged imagery would be inconsistent with that residual.
- Optionally supply the per-year alignment offsets, which are not recorded numerically anywhere in the results and would strengthen §5.5.1.
