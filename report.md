# Lab 10 Report

## Section 2.1 - Partial Dependence Variance (PDV) — Feature Importance

**Q1: Which features have higher importance in the prediction?**

Each `imp(feature) = value` is the Partial Dependence Variance importance score for that feature:

| Rank | Feature | Score | Meaning |
|------|---------|-------|---------|
| 1 | `rm` | 6.057 | Avg rooms/dwelling — strongest predictor |
| 2 | `lstat` | 4.159 | % lower status population — 2nd strongest |
| 3 | `dis` | 0.589 | Distance to employment centres |
| 4 | `ptratio` | 0.533 | Pupil-teacher ratio |
| 5 | `tax` | 0.509 | Property tax rate |
| ... | | | |
| 13 | `zn` | 0.010 | Weakest predictor |

**Answer:** `rm` and `lstat` have significantly higher importance — their scores (~6 and ~4) are an order of magnitude above the rest. The model relies heavily on the number of rooms and socioeconomic status of the neighborhood to predict housing prices.

---

## Section 2.2 - Partial Dependence (PD)

**Q2: What can you conclude with the PD plots?**

Each plot shows an **orange dashed line** (average PD curve) and **blue lines** (individual ICE curves per data point).

### High Importance Features (top row)

- **`rm` (avg rooms/dwelling)** — Strong S-shaped curve. As the number of rooms increases past ~6, predicted home value rises sharply. The ICE lines are widely spread, meaning this feature affects different homes very differently. Confirms it is the most important feature.

- **`lstat` (% lower status population)** — Sharp downward curve. As the proportion of lower-status population increases, predicted price drops steeply then flattens at high values. The effect is strong and consistent across all homes.

- **`dis` (distance to employment centres)** — Relatively flat after an initial rise. Homes very close to employment centres are predicted lower (possibly due to industrial surroundings), but the effect stabilizes quickly. Less dramatic than `rm` or `lstat`.

### Low Importance Features (bottom row)

- **`ptratio` (pupil-teacher ratio)** — Mostly flat with a slight drop at higher ratios. Weak and inconsistent effect — ICE lines do not spread much.

- **`nox` (nitric oxide concentration)** — Slight downward trend at higher pollution levels, but the effect is modest and ICE lines are tightly clustered.

- **`zn` (residential land zoning)** — Nearly flat line across all homes. Almost no effect on predictions, confirming its near-zero importance score from Q1.

### Conclusion

`rm` and `lstat` have strong, non-linear relationships with housing price — more rooms increases value sharply, while a higher proportion of lower-status population decreases it. The remaining features show weak, near-flat PD curves, meaning the model barely uses them once `rm` and `lstat` are accounted for.

---

## Section 2.2 - Two-Feature Partial Dependence (Q3)

**Q3: Discuss the results (heatmaps) — `rm` vs `zn`**

This is a **2D PD heatmap** showing the joint effect of `rm` (x-axis) and `zn` (y-axis) on predicted housing price. Color represents the predicted `medv` value — purple/dark = low, teal = medium, yellow-green = high. Numbers on contour lines are predicted prices in $1000s.

| `rm` range | Predicted price | Effect of `zn` |
|---|---|---|
| < 6.5 | ~$19.67k | None |
| 6.5 – 7.0 | $21k – $26k | None |
| > 7.5 | ~$32.5k | None |

**Key observations:**

- Color shifts entirely with **horizontal movement** (`rm`) — a small increase in `rm` past ~6.5 causes a large jump in predicted price.
- **Vertical movement** (`zn`) produces no color change at any value of `rm` — `zn` has no effect on the prediction regardless of `rm`.
- All contour lines are **perfectly vertical**, confirming there is **no interaction** between `rm` and `zn`.

**Answer:** The heatmap confirms that `rm` is the dominant driver of predicted price, following the same non-linear pattern seen in the 1D PD plot. `zn` has no meaningful interaction with `rm` — its value makes no difference to the prediction at any level of `rm`. This is fully consistent with `zn` having the lowest importance score (0.010) from Q1.

---

## Section 2.3 - Anchors (AnchorImage on InceptionV3)

### Model Predictions

Both images are of cats in paper bags. InceptionV3's top predictions were:

| | Image 1 | Image 2 |
|---|---|---|
| **Top prediction** | Persian_cat (0.502) | Persian_cat (0.574) |
| **2nd** | Siamese_cat (0.139) | Siamese_cat (0.157) |
| **3rd** | carton (0.055) | crate (0.008) |

The model correctly identifies both cats as Persian with high confidence (~50–57%). The presence of `carton` and `crate` in the top 3 shows the model partially picks up on the paper bag context, but the cat class dominates.

### Anchor Explanation Analysis

- **Original Image**: A fluffy Ragdoll cat sitting inside a paper bag on a wooden surface.
- **Segmentation Map**: SLIC divided the image into 15 superpixel regions covering the cat's fur, face, bag, and background.
- **Anchor Explanation**: The model retained the cat's **body, white fur, and the paper bag on its head** as the anchor, and **blacked out the wooden floor/table** at the bottom. These highlighted superpixels are sufficient to predict "Persian cat" with ≥95% confidence regardless of what replaces the masked regions.

### Sampling Process

The anchor algorithm ran ~31 sampling rounds (124 total model inference calls), each masking different combinations of the 15 superpixels and running them through InceptionV3 to check if the "Persian cat" prediction held. This iterative process continued until the retained superpixels satisfied the `threshold=0.95` confidence requirement — i.e., the identified anchor regions are sufficient to maintain the prediction 95% of the time regardless of what fills the rest of the image.

### Answers to Questions

**1. What part of your image did the Anchor explainer focus on?**
The cat's body, white fur, and the paper bag on its head. These superpixels form the anchor — if they are present, the model predicts "Persian cat" with high confidence no matter what the rest of the image contains.

**2. Does the highlighted region make sense compared to the model's predicted label?**
Yes. The cat's fur and body are the most visually distinctive features for classifying a Persian cat. The model anchors on the animal itself rather than the background, which is the expected and correct behavior. The fact that the paper bag is also retained aligns with `carton` appearing in the top 3 predictions — the bag contributes some signal, but the cat features dominate.

**3. What does it mean if the highlighted region seems unrelated to the object?**
It indicates the model has learned a **spurious correlation** rather than the true object features. For example, if the anchor highlighted only the paper bag and ignored the cat, the model may have associated bags with cats due to dataset bias. This would make the model fragile — it would fail on a cat without a bag, and might incorrectly classify a bag without a cat. In this case, the anchor is sensible and focused on the cat itself, suggesting the model is relying on genuine visual features.

---

## Section 3 - SHAP

### Plot 1 — Global Summary (Beeswarm)

Each dot is one test instance. The x-axis shows the SHAP value (how much that feature pushed the prediction up or down). Color represents the actual feature value (red = high, blue = low). Features are ranked top-to-bottom by overall importance.

**Key findings:**

- **`lstat`** — Most impactful globally. Red dots (high lstat) cluster far left → high lower-status population **strongly decreases** price. Blue dots (low lstat) push right → increases price. Wide spread confirms high variability of effect across homes.
- **`rm`** — Second most impactful. Red dots (many rooms) push far right → strongly **increases** price. Blue dots (few rooms) push left. Consistent with PDV and PD findings.
- **`ptratio`, `nox`, `crim`** — Moderate negative effects. High values (red) consistently push predictions down.
- **`zn`, `chas`, `rad`** — All clustered tightly near 0 → negligible impact, consistent with PDV importance scores from Q1.

---

### Plot 2 — Waterfall Plot (Single Instance, idx=5)

This explains one specific house prediction. Baseline average prediction = **$22,376**, final prediction = **$20,565**.

| Feature | Actual value (scaled) | SHAP contribution |
|---|---|---|
| Feature 5 (`rm`) | -0.674 (below avg rooms) | **-2.48** |
| Feature 12 (`lstat`) | -0.581 (low, good) | **+1.52** |
| Feature 10 (`ptratio`) | +1.158 (high ratio) | **-0.81** |
| Feature 9 (`tax`) | -0.562 | -0.29 |
| Feature 6 (`age`) | -0.446 | +0.19 |
| Feature 2 (`indus`) | -0.427 | -0.17 |

**Interpretation:** This house has **fewer rooms than average** (biggest drag, -2.48), is in a **relatively good neighbourhood** (low lstat gives +1.52 back), but is hurt by a **high pupil-teacher ratio** (-0.81). The net effect pulls the prediction $1,811 below the baseline.

---

## Section 4 - Reflection & Comparison

**Q1: Do any of the top features raise ethical or fairness concerns?**

Yes — both `lstat` (% lower status population) and `b` (racial composition proxy) are socioeconomic and demographic variables that raise significant fairness concerns. The model relies heavily on `lstat` as its second most important feature, meaning it is directly pricing homes based on the socioeconomic status of the surrounding population. `b`, while lower in importance, encodes racial composition of a neighbourhood. Using these variables embeds historical inequalities into predictions — neighbourhoods with lower-income or minority populations are systematically predicted to have lower home values, which can reinforce redlining-like patterns and perpetuate housing discrimination.

---

**Q2: Do SHAP's most important features agree with those found by PD plots or Anchor explanations?**

Yes, they are strongly consistent. All three methods identify `lstat` and `rm` as the dominant features. PDV ranked `rm` (6.057) and `lstat` (4.159) highest, PD plots showed strong non-linear curves for both, and SHAP's beeswarm plot ranks `lstat` first and `rm` second with the widest spread of impact. The waterfall plot for instance idx=5 also confirms `rm` and `lstat` as the biggest individual contributors. The methods agree because they are all measuring the same underlying model behaviour — just from different angles: PDV measures **global variance** (how much the prediction varies across the feature's range), PD plots capture **marginal effects** (how the average prediction changes as a feature changes), and SHAP computes **Shapley attribution** (how much each feature contributed to a specific prediction relative to the baseline).

---

**Q3: Summary**

The model relies primarily on `rm` (number of rooms) and `lstat` (socioeconomic status of the neighbourhood) to predict housing prices, with all other features contributing marginally by comparison. This is confirmed consistently across PDV, PD plots, and SHAP — giving high confidence in the interpretability of the model's global behaviour. However, the reliance on socioeconomic and demographic proxies raises fairness concerns that limit how responsibly this model could be deployed in practice.
