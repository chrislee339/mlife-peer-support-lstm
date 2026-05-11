# Does Peer Support Drive Weight Loss? A Deep Learning Investigation of the mLIFE Trial

**Authors:** Christopher Lee¹,², Diana Carolina Delgado-Diaz³,⁴, Gabrielle M. Turner-McGrievy³,⁴, Homayoun Valafar¹,²

¹ Department of Computer Science and Engineering, Molinaroli College of Engineering and Computing, University of South Carolina, Columbia, South Carolina, USA
² AI Institute, University of South Carolina, Columbia, South Carolina, USA
³ Prevention Research Center, Arnold School of Public Health, University of South Carolina, Columbia, South Carolina, USA
⁴ Department of Health Promotion, Education, and Behavior, Arnold School of Public Health, University of South Carolina, Columbia, South Carolina, USA

---

## Abstract

**Background.** Behavioral weight loss interventions are partially effective; identifying the active ingredients that drive successful outcomes is essential for designing scalable mHealth treatments. The mLIFE trial showed that gamifying peer support exchange more than doubled support provision and receipt and improved weight outcomes among adherent participants. A companion mediation analysis confirmed that aggregate peer support mediates weight change. However, mLIFE participants exchanged four behaviorally distinct types of peer recognition (thumbs ups for logged activity, diet, weight, and read tips), and traditional mediation cannot disentangle which type carries the active signal.

**Objective.** To use deep learning on longitudinal peer support and weight data from the mLIFE trial to identify which form(s) of peer support predict future weight change after controlling for an individual's weight trajectory, and to test the directionality and robustness of the effect.

**Methods.** We constructed a weekly panel from the mLIFE trial (N=108 complete case participants, approximately 2,200 rolling 6 week aggregation windows). An LSTM with two layers was trained to predict weight change at 20 weeks ahead from 4 week sequences of received peer support and weight trajectory, under deterministic CUDA execution and five fold cross validation grouped by participant. We performed seed paired ablations (20 seeds × 2 architectures = 40 runs per variant) to isolate the marginal contribution of each support type, with a sensitivity grid spanning window, horizon, stride, and lookback configurations and baseline ML model comparisons.

**Results.** Among the four peer support modalities, activity targeted thumbs ups uniquely predicted weight change at 20 weeks ahead (paired Δ R² = +0.070 versus the weight trajectory only baseline, p = 3×10⁻³⁴, 40/40 seeds positive). Diet targeted support contributed secondarily (Δ R² = +0.028, p = 2×10⁻¹⁵). Tip and weight log thumbs ups did not predict above noise. The activity finding remained positive across 16 of 17 alternative operationalizations of the prediction problem. Linear, tree based, and MLP baselines failed to extract this signal; only the temporal LSTM surfaced it.

**Conclusions.** Peer support drives weight change in mLIFE primarily through activity targeted recognition. This refines the implication of the prior aggregate mediation result and informs the design of future mHealth peer support interventions, because the active ingredient appears to be behavior specific reinforcement (especially around physical activity) rather than generic positive interaction.

**Keywords:** social support, weight loss, deep learning, LSTM, behavioral intervention, mHealth, mediation, ablation analysis

---

## 1. Introduction

Behavioral interventions remain the foundation of weight management for adults with overweight or obesity, with self monitoring, dietary modification, and increased physical activity established as core components (Butryn et al., 2011; Burke et al., 2011). Intensive multicomponent programs such as the Diabetes Prevention Program produce clinically meaningful weight loss in trial settings (Diabetes Prevention Program Research Group, 2002), but real world implementations show smaller and more variable effects (Ali et al., 2012). Mobile health (mHealth) platforms have emerged as a promising route to delivering behavioral weight loss treatment at scale (Cleghorn et al., 2019; Kupila et al., 2023), and randomized evidence supports gamified and socially incentivized mHealth designs in particular (Patel et al., 2021; Agarwal et al., 2021). However, mHealth interventions typically bundle multiple active components (self monitoring, education, peer support, gamification), making it difficult to identify which components drive outcomes. Active ingredient analysis is therefore essential for efficiently designing the next generation of mHealth interventions.

Peer support has long been theorized as a central active ingredient in behavioral weight loss treatment (Shumaker & Brownell, 1984; Lakey & Cohen, 2000; Kiernan et al., 2012). A recent systematic review and meta analysis confirmed that social support based weight loss interventions produce greater weight loss than non supported comparators (Jensen et al., 2024), and group based formats often outperform individual treatment (Paul-Ebhohimhen & Avenell, 2009). Within mHealth specifically, social features such as peer feeds, comment threads, and reaction based recognition have been associated with improved engagement and outcomes (Tong & Laranjo, 2018; Hales et al., 2016). The mLIFE randomized trial (Turner-McGrievy et al., 2025; DuBois et al., 2024) tested whether gamifying peer support exchange (points and leaderboards) would amplify support and improve weight outcomes. Participants used a purpose built mHealth platform supporting weight, diet, and physical activity self monitoring alongside a peer feed where users recognized each other's logged behaviors via thumbs up reactions and comments. Among adherent participants, the gamified arm achieved approximately twice the weight loss of the comparator (7.3 vs. 3.8 kg) (Turner-McGrievy et al., 2025), establishing that gamifying peer support exchange improves weight outcomes in this population.

While the trial result establishes that gamifying peer support exchange improves weight outcomes, it leaves a critical question unanswered. Which form of peer support is doing the work? mLIFE participants could give and receive recognition in four behaviorally distinct categories (logged physical activity, logged diet, logged weight, and read tips or podcasts). The trial level effect collapses these into a single bundled treatment, but mHealth features are designed component by component, and intervention scaling requires knowing which components matter. We hypothesized that the four modalities are not equivalent. Specifically, we expected that behavior specific peer recognition (e.g., for logged activity) would carry more predictive signal than generic positive interaction (e.g., for read tips). Disentangling these contributions from longitudinal panel data is a high dimensional, temporally correlated problem poorly suited to traditional regression. Recurrent neural networks, particularly long short term memory (LSTM) architectures (Hochreiter & Schmidhuber, 1997), are designed to summarize variable length temporal sequences and have shown utility in modeling clinical and behavioral panel data (Choi et al., 2016).

In this paper we apply an LSTM to weekly mLIFE panel data and use seed paired feature ablations to identify which peer support modality (or modalities) predicts weight change at 20 weeks ahead after controlling for participants' weight trajectory, assess the robustness of the resulting hierarchy across operationalizations of the prediction problem (window, stride, lookback, and horizon), and compare the temporal LSTM against linear, tree based, and shallow neural network baselines to evaluate whether temporal modeling is necessary to surface this signal.

---

## 2. Methods

### 2.1 Trial data and panel construction
Data are from the mLIFE trial (Turner-McGrievy et al., 2025; protocol in DuBois et al., 2024), a one year two arm randomized parallel behavioral intervention conducted between 2022 and 2024. The trial was based at the University of South Carolina and delivered entirely remotely. Participants used a custom mHealth platform supporting weight, diet, and physical activity self monitoring. Peer support was exchanged via thumbs up reactions on logged behavior posts, with each reaction structurally tied to one of four post categories (activity, diet, weight, or read tip). Per category thumbs up counts therefore provide a platform level decomposition of received peer recognition that is fixed by the app design rather than chosen post hoc.

We aggregated participant week data into rolling 6 week windows (stride = 1 week), yielding 4,673 windows across 122 participants who had at least 6 weeks of usable longitudinal data. Rolling window aggregation smooths short term measurement variability in both the support counts (which are zero inflated at the daily level) and weight (which fluctuates with hydration and circadian factors). After complete case filtering on demographic covariates, lookback features, and a non missing outcome at 20 weeks ahead, the modeled sample consisted of 108 participants and approximately 2,200 windows (Supplementary Figure S1, participant flow). Each window summarized:

- **Outcome:** weight change at 20 weeks ahead (`next_20wk_weight_change`, in pounds).
- **Temporal features (per week within window):** count of thumbs ups received in each of four categories (logged activity, logged diet, logged weight, read tip or podcast), and within window weight change.
- **Static features:** baseline BMI, household size.

The mLIFE trial collected additional baseline demographics including age, sex, race, ethnicity, education, employment, marital status, number of children, and household income. In a preliminary feature selection step prior to the analyses reported here, we evaluated permutation importance for the full demographic set across multiple training seeds; only baseline BMI and household size produced consistent positive importance. The remaining covariates did not improve predictive performance and were dropped from the analyzed feature set.

The 20 week horizon was selected as the reference horizon for the primary analyses because it falls within the active weight loss phase of the 52 week trial, before typical post 6 month plateaus, while retaining sufficient complete case panel windows for stable model fitting. To assess whether this choice affects our conclusions, the sensitivity grid in §3.3 re-evaluates the prediction problem at alternative horizons of 4, 8, 12, and 26 weeks.

The analyzed sample size of 108 participants is fixed by the parent trial; no a priori sample size calculation was performed for this secondary analysis. The 40 paired training runs per variant (20 seeds × 2 architectures) control training stochasticity rather than sampling variability and, combined with the deterministic CUDA training protocol described in §2.2, allow paired Δ R² (the change in held out R² between variants) comparisons to be tested at small effect sizes with high statistical power.

### 2.2 Architecture and training
We trained an LSTM with two layers and evaluated two architecture configurations in parallel, "old" (hidden=32, dropout=0.3) and "new" (hidden=64, dropout=0.2). The two configurations were chosen during an earlier pilot tuning phase; the dual configuration design lets us test whether the per type effects depend on architecture. The LSTM consumed a 4 week lookback of weekly summaries; static covariates were concatenated to the final hidden state. Training used Adam (lr = 1×10⁻³, weight decay = 1×10⁻³), MSE loss with gradient clipping at 1.0, a `ReduceLROnPlateau` learning rate scheduler (patience 10, factor 0.5), batch size 32, early stopping on the held out fold (patience 30), and a maximum of 200 epochs. Optimizer hyperparameters were set to common defaults rather than tuned via grid search, for two reasons. Hyperparameter optimization on a sample of 108 participants risks selection bias against the held-out folds, and the paired Δ R² comparisons that drive every result in this paper are far more stable to architecture and optimizer choice than to feature-set choice. The dual architecture design described above directly probes whether the per type results depend on architecture choice.

Models were implemented in PyTorch 2.1.0 (CUDA 12.1, cuDNN 8.9) on a workstation with two NVIDIA GeForce RTX 3090 Ti GPUs running Ubuntu 22.04. Training used deterministic CUDA execution to ensure bit-exact reproducibility across paired runs; the exact environment variables and PyTorch flags are documented in the code repository.

### 2.3 Cross validation
Five fold `GroupKFold` cross validation with participant ID as the grouping variable, ensuring no participant appeared in both train and test folds. Per fold R² was averaged for fold level reporting; pooled out of fold predictions yielded the overall R². Within each window, predictors are computed strictly from the lookback period and the outcome is measured strictly at 20 weeks after the window end, eliminating temporal leakage from outcome to predictors at the row level.

### 2.4 Variant definitions for ablation
We use seed paired feature set ablation as the interpretability method of this paper. Rather than relying on post hoc attribution methods such as SHAP or attention weights, we re-train the LSTM under controlled changes to its input feature set and measure the resulting change in held out R². We chose ablation over post hoc attribution for three reasons. First, ablation Δ R² has a direct intervention design interpretation. The loss in predictive performance attributable to removing a feature directly indicates which platform features should be amplified or restructured. Second, training each variant under multiple seeds and architectures yields a paired sample of held out performance that supports a standard paired t-test against the baseline, while SHAP values from a single trained model are deterministic and require re-training anyway to characterize uncertainty. Third, the four peer support types are correlated at the participant level (participants who give one type often give others), and per feature attribution methods can produce misleading individual values under collinearity, whereas an ablation that drops the entire feature stream captures interaction effects correctly.

We defined a family of feature set variants to isolate the contribution of each support type:

- `wconly`: weight trajectory only (baseline).
- `supportwc`: all four support types plus weight trajectory (full).
- `swc_only_<type>`: one support type plus weight trajectory (keep one).
- `swc_no_<type>`: full minus one support type (drop one).

Each variant was trained 40 times (20 seeds × 2 architectures). Δ R² between variants was computed across the 40 runs, matched by seed and architecture, and tested with a paired t-test plus sign count.

### 2.5 Sensitivity analyses
We re-ran `swc_only_activity` versus `wconly` across windows ∈ {2,3,5,6,8,12}, horizons ∈ {4,8,12,20,26}, strides ∈ {1,2,3,5}, and lookbacks ∈ {2,4,6,8,12}, varying one parameter at a time and holding the others at their reference values. Horizons are monthly intervals (4 week multiples plus 6 months), lookbacks span even numbered weekly counts, and strides cover small integers commonly used in rolling window panel construction. The grid was specified before any runs were executed and all cells are reported in Table 2 with no exclusions; the purpose of the sweep is to probe robustness of the reference cell finding to alternative operationalizations, not to optimize. Each cell was paired by (architecture, seed) yielding 40 paired runs per cell.

### 2.6 Baseline ML comparison
For methodological context, we trained three conventional baselines on the same reference features used by the LSTM, with the temporal sequence flattened into a vector of length `lookback × n_temporal + n_static`. The baselines were Ridge regression (α = 1.0, standardized features), Random Forest (200 trees, unbounded depth), and a small MLP (hidden layers 64 and 32, early stopping, standardized features). All three were trained on the same variant feature sets (`wconly`, `supportwc`, and `swc_only_activity`), under the same five fold GroupKFold splits and the same 20 seeds, so paired comparison against the LSTM uses identical inputs.

### 2.7 Statistical analysis
All paired comparisons used 40 runs paired by seed and architecture (20 seeds × 2 architectures). We report mean Δ R², standard deviation, paired t-statistic and p-value, and the count of seed and architecture pairs with positive sign. The reported standard deviations and p-values quantify variability across training runs rather than across participants; per prediction uncertainty (e.g., quantile or ensemble intervals) is not provided. Paired Wilcoxon signed rank p-values were computed in parallel and confirm the paired t-test results.

The primary per type ablation covers four support types and the sensitivity grid covers 17 cells. We did not formally adjust p-values for multiple comparisons. Where reported p-values are extreme (orders of magnitude below conventional thresholds, see Results), they remain significant under Bonferroni across the 21 primary and sensitivity tests, and the sensitivity grid is in any case interpreted as a robustness probe rather than a multiple hypothesis battery.

### 2.8 Code and reproducibility
All training and analysis code is available at *[repo URL to be inserted; see Code Availability]*, with the submission version archived at *[Zenodo DOI to be inserted]*. The 20 random seeds and the canary reproducibility script are included in the repository, along with the per cell sweep result CSVs that underlie Tables 1a, 1b, 2, and 3. The analysis plan was not preregistered.

---

## 3. Results

### 3.1 Sample characteristics
Of 199 randomized mLIFE participants, 108 had sufficient complete case longitudinal data for inclusion (windows with all four support categories, weight, demographics, and a valid outcome at 20 weeks ahead). The flow from the randomized sample to the analyzed sample is summarized in Supplementary Figure S1. The analyzed sample's mean baseline BMI was 34.2 (SD 6.0), mean age 49.5 (SD 11.1); 99 of 108 (92%) were women. Of these, 62 were in the gamified (mLife+Points) arms and 46 in the non-gamified (mLife) arms, pooled across the two trial cohorts. The 91 excluded participants had insufficient longitudinal weight or support data; characterization is provided in Supplementary Table S1. Included and excluded participants were balanced on baseline BMI, baseline weight, sex, arm assignment, and cohort, but included participants were on average older and had lost more total weight over the trial, consistent with engagement driven attrition.

### 3.2 Activity targeted peer support uniquely predicts weight change

Table 1a reports the paired Δ R² for each support type used alone with weight trajectory, against the weight trajectory only baseline, at the reference configuration (window = 6 weeks, stride = 1 week, lookback = 4 weeks, horizon = 20 weeks).

**Table 1a. Paired Δ R² for each support type used alone with weight trajectory, versus the weight trajectory only baseline (wconly).**

| Support type alone | Δ R² | SD | Seeds positive | t | p |
|---|---|---|---|---|---|
| **Activity** | **+0.070** | 0.011 | 40/40 | +42.4 | 3×10⁻³⁴ |
| **Diet** | **+0.028** | 0.014 | 39/40 | +12.6 | 2×10⁻¹⁵ |
| Tip | −0.000 | 0.009 | 20/40 | −0.2 | 0.83 |
| Weight | −0.001 | 0.011 | 22/40 | −0.7 | 0.46 |

We also ran the complementary drop one analyses, removing each support type from the full bundle and comparing against the full bundle (Table 1b).

**Table 1b. Paired Δ R² for drop one analyses (full bundle minus one support type, versus the full bundle); negative Δ means removal hurts predictive performance.**

| Type removed | Δ R² | SD | Seeds positive | t | p |
|---|---|---|---|---|---|
| **Activity** | **−0.045** | 0.021 | 0/40 | −13.3 | 4×10⁻¹⁶ |
| Tip | −0.019 | 0.023 | 8/40 | −5.2 | 7×10⁻⁶ |
| Diet | −0.015 | 0.015 | 8/40 | −6.2 | 3×10⁻⁷ |
| **Weight** | **+0.014** | 0.028 | 27/40 | +3.2 | 0.003 |

Activity carried the largest standalone signal (Δ R² = +0.070, Table 1a) and incurred the largest performance penalty when removed (Δ R² = −0.045, Table 1b), confirming its position as the dominant predictor. Weight targeted support carried no standalone signal (Δ R² ≈ 0, Table 1a) and, when removed from the full bundle, produced a small but reliable improvement (Δ R² = +0.014, Table 1b), indicating that weight targeted thumbs ups are uninformative and slightly degrade predictive performance when included. Diet showed a moderate but consistent effect across both analyses, with a clear standalone contribution (Δ R² = +0.028, Table 1a) and a smaller drop one penalty (Δ R² = −0.015, Table 1b), indicating that diet thumbs ups carry predictive information both in isolation and alongside the other support types. Tip showed a different pattern. Tip thumbs ups did not predict on their own (Δ R² ≈ 0, Table 1a), but their removal from the full bundle incurred a small but reliable penalty (Δ R² = −0.019, Table 1b), indicating that tip support contributes unique information when bundled with the other support types.

### 3.3 Robustness of the activity support effect

Across 16 alternative configurations (varying window, stride, lookback, and horizon one at a time) plus the reference cell, the paired Δ R² between `swc_only_activity` and `wconly` was positive at p<0.05 in 16 of 17 cells, with all 40 seed and architecture pairs positive in 13 of 17 cells. Across the 17 cells, the median Δ R² was approximately +0.056. The single non positive cell was the horizon = 8 weeks cell (Δ = −0.003, p = 0.04), consistent with the broader pattern of signal attenuation at short horizons described below.

**Table 2. Paired Δ R²(`swc_only_activity` − `wconly`) across alternative configurations, grouped by perturbed parameter.** All other parameters held at their reference values (window=6, stride=1, horizon=20, lookback=4). Each cell aggregates 40 paired runs (20 seeds × 2 architectures).

| Perturbed parameter | Value | R²(activity) | R²(wconly) | Δ R² | Seeds + | p |
|---|---|---|---|---|---|---|
| **Window (weeks)** | 2 | 0.314 | 0.256 | +0.058 | 40/40 | 9×10⁻³⁶ |
| | 3 | 0.333 | 0.284 | +0.050 | 40/40 | 3×10⁻³⁶ |
| | 5 | 0.335 | 0.265 | +0.071 | 40/40 | 2×10⁻³⁶ |
| | **6 (reference)** | **0.357** | **0.287** | **+0.070** | **40/40** | **3×10⁻³⁴** |
| | 8 | 0.353 | 0.287 | +0.066 | 40/40 | 2×10⁻³⁴ |
| | 12 | 0.350 | 0.293 | +0.056 | 40/40 | 3×10⁻³⁴ |
| **Stride (weeks)** | 2 | 0.365 | 0.314 | +0.051 | 40/40 | 9×10⁻³⁴ |
| | 3 | 0.354 | 0.260 | +0.094 | 40/40 | 2×10⁻³⁸ |
| | 5 | 0.345 | 0.306 | +0.039 | 38/40 | 4×10⁻¹⁴ |
| **Lookback (weeks)** | 2 | 0.362 | 0.268 | +0.095 | 40/40 | 9×10⁻⁴⁸ |
| | 6 | 0.372 | 0.296 | +0.076 | 40/40 | 2×10⁻³⁹ |
| | 8 | 0.381 | 0.295 | +0.086 | 40/40 | 2×10⁻⁴² |
| | 12 | 0.358 | 0.318 | +0.040 | 39/40 | 1×10⁻¹⁴ |
| **Horizon (weeks)** | 4 | 0.162 | 0.156 | +0.006 | 31/40 | 3×10⁻⁷ |
| | 8 | 0.204 | 0.207 | −0.003 | 12/40 | 0.04 |
| | 12 | 0.270 | 0.218 | +0.051 | 40/40 | 2×10⁻²⁹ |
| | 26 | 0.376 | 0.350 | +0.027 | 40/40 | 2×10⁻²⁰ |

The activity support effect is stable across window, stride, and lookback perturbations at the reference 20 week horizon. The signal weakens at short horizons (4 to 8 weeks), where less variance in weight change is available to be predicted, and is robust at the reference 20 week horizon and longer. The pattern indicates the activity support contribution is a property of multi week ahead prediction rather than an artifact of any single configuration choice.

### 3.4 LSTM versus baseline models

**Table 3. Out of fold R² across model families (reference config, mean across 20 seeds).**

| Model | wconly | supportwc | swc_only_activity |
|---|---|---|---|
| Ridge regression | 0.217 | 0.198 | 0.202 |
| Random Forest | −0.018 | 0.108 | 0.066 |
| MLP (2 layer) | 0.056 | 0.032 | 0.103 |
| **LSTM (this paper)** | **0.287** | **0.343** | **0.357** |

Ridge regression captures the weight trajectory baseline but fails to extract any peer support contribution (Δ R² = −0.015 from adding activity). Random Forest and MLP both underfit the weight trajectory baseline itself and never approach the LSTM's R² at any variant. Only the LSTM achieves both a high absolute predictive ceiling (R² ≈ 0.36 with activity) and a positive support contribution at that ceiling, indicating that the predictive signal in peer support is non-linear and temporal in nature, requiring a sequence model to extract. This justifies the deep learning approach methodologically.

---

## 4. Discussion

### 4.1 Principal findings
Across 108 mLIFE participants and approximately 2,200 longitudinal panel windows, an LSTM trained to predict weight change at 20 weeks ahead from peer support exchange revealed a clear hierarchy. Activity targeted peer recognition was the dominant unique predictor of weight change. Diet targeted support contributed a smaller but consistent effect across standalone and bundled analyses. Tip targeted thumbs ups did not predict on their own but carried unique information when bundled with the other support types, while weight log support did not predict above noise in either analysis and slightly degraded predictive performance when included in the full bundle. This refines a prior mediation result on the same trial, which showed that peer support mediates weight outcomes in aggregate, by identifying which specific support modality carries the active signal.

### 4.2 Methodological contribution
The model comparison in §3.4 carries a methodological lesson for analyzing high frequency mHealth peer support data. Ridge regression captured the weight trajectory baseline but missed the peer support contribution despite having access to the full lookback vector, while Random Forest and MLP underfit even the trajectory baseline. Only the LSTM extracted both signals at a high predictive ceiling, suggesting that high frequency peer support exchange contains non-linear and temporal structure that simpler model families cannot capture.

### 4.3 Clinical and design implications
mHealth peer support features are designed component by component, and these results suggest that activity related social reinforcement is the primary driver of observable weight outcomes in this trial. Diet related reinforcement contributes a smaller but consistent effect, tip thumbs ups contribute unique information only when bundled with the other support types, and weight log thumbs ups appear essentially uninformative.

One plausible explanation, consistent with self-determination theory (Ryan & Deci, 2000) and prior work on contingent reinforcement in mHealth (Patel et al., 2021; Agarwal et al., 2021), is that peer recognition is most effective when it targets a volitional behavior under the participant's daily control. Physical activity is highly volitional and frequently logged. Diet logging is also volitional but aggregates over many small daily decisions, which may dilute the per recognition reinforcement effect. Weight logs recognize a passive measurement rather than a daily behavior. The role of tip recognition is less clear from the data and warrants further investigation.

These findings identify activity related peer recognition as a primary structural target for intervention development within similar mHealth platforms. Although the absolute prediction error reduction from peer support features is modest (~0.4 lbs at 20 weeks ahead), the per type hierarchy is a sharper guide for intervention design than the aggregate predictive accuracy suggests. Future intervention designs may therefore benefit from prioritizing peer recognition features tied to volitional logged behaviors (especially physical activity) and from testing whether targeted activity recognition prompts amplify weight loss outcomes. Replication in independent trials is needed before broader generalization across mHealth platforms.

### 4.4 Limitations
This study has several limitations. The analyzed sample is small for deep learning (N = 108); the seed paired ablation and baseline ML comparison partially mitigate this, but replication in independent trials is needed. Included participants differed from excluded participants in age and total weight change (Supplementary Table S1), consistent with engagement driven attrition, so predictive performance applies to participants who provided sufficient longitudinal data rather than to the full randomized cohort. We performed only internal cross validation; external validation is essential before any practical deployment.

Calibration was assessed (Supplementary Figure S2) and shows modest compression of predicted magnitudes; external calibration has not been performed. Although mLIFE was randomized at the trial-arm level, our analysis predicts weight change from each participant's within-person pattern of received support, which was not itself randomized. The 20-week-ahead prediction target rules out reverse causality (support is observed strictly before the outcome window), but an unobserved confounder such as baseline motivation could still drive both higher support receipt and greater weight loss, and we cannot rule this out. The sample was 92% women, which precludes a sex based fairness assessment, and at n = 108 is too small to test subgroup performance across age or BMI. We do not measure whether activity support's predictive contribution operates through behavioral mediation (more exercise) or psychological mediation (motivation). Several of these limitations could be directly addressed by replication in independent cohorts with larger and more diverse samples.

### 4.5 Future work
Beyond replication, several specific extensions follow from these findings. Mediation analysis decomposing the pathway from activity support to weight change through measured exercise behavior would clarify whether the effect operates through actual behavior change or through psychological reinforcement. Testing whether provided (as opposed to received) activity support has independent or additive effects would distinguish the giver's experience from the receiver's.

A complementary direction is adaptive intervention design. The per type predictive contributions identified here could anchor a personalized digital twin or just in time adaptive intervention (JITAI) system (Nahum-Shani et al., 2018), in which an individualized model of each participant simulates the predicted response to different support modalities and the platform allocates prompts to the modality with the strongest expected effect. Building such a system would require extending the present model with continuous personalization, real time updating, and a validated closed loop between predicted response and intervention delivery; the per type structural foundation identified here is a first step toward that longer-term goal.

---

## Acknowledgements

## Funding

## Conflicts of Interest

## Authors' Contributions

- **Christopher Lee:** Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Writing - original draft, Visualization.
- **Diana Carolina Delgado-Diaz:** Investigation, Data curation, Writing - review and editing, Project administration.
- **Gabrielle M. Turner-McGrievy:** Conceptualization, Methodology, Resources, Writing - review and editing, Supervision, Funding acquisition, Project administration.
- **Homayoun Valafar:** Conceptualization, Methodology, Resources, Writing - review and editing, Supervision.

All authors read and approved the final manuscript.

## Data Availability

## Code Availability

## Trial Registration
The parent mLIFE randomized trial is registered at ClinicalTrials.gov as **NCT05176847** (DuBois et al., 2024).

## Abbreviations
- **BMI:** body mass index
- **DPP:** Diabetes Prevention Program
- **JITAI:** just in time adaptive intervention
- **LSTM:** long short term memory (network)
- **mHealth:** mobile health
- **MLP:** multilayer perceptron
- **MSE:** mean squared error
- **R²:** coefficient of determination

## References

1. Butryn ML, Webb V, Wadden TA. Behavioral treatment of obesity. *Psychiatr Clin North Am.* 2011;34(4):841–859.
2. Burke LE, Wang J, Sevick MA. Self monitoring in weight loss: a systematic review of the literature. *J Am Diet Assoc.* 2011;111(1):92–102.
3. Diabetes Prevention Program (DPP) Research Group. The Diabetes Prevention Program (DPP): description of lifestyle intervention. *Diabetes Care.* 2002;25(12):2165–2171.
4. Cleghorn C, Wilson N, Nair N, et al. Health benefits and cost effectiveness from promoting smartphone apps for weight loss: multistate life table modeling. *JMIR Mhealth Uhealth.* 2019;7(1):e11118.
5. Kupila SKE, Joki A, Suojanen LU, Pietiläinen KH. The effectiveness of eHealth interventions for weight loss and weight loss maintenance in adults with overweight or obesity: a systematic review of systematic reviews. *Curr Obes Rep.* 2023;12(3):371–394.
6. Patel MS, Small DS, Harrison JD, et al. Effect of behaviorally designed gamification with social incentives on lifestyle modification among adults with uncontrolled diabetes: a randomized clinical trial. *JAMA Netw Open.* 2021;4(5):e2110255.
7. Agarwal AK, Waddell KJ, Small DS, et al. Effect of gamification with and without financial incentives to increase physical activity among veterans classified as having obesity or overweight: a randomized clinical trial. *JAMA Netw Open.* 2021;4(7):e2116256.
8. Shumaker SA, Brownell A. Toward a theory of social support: closing conceptual gaps. *J Soc Issues.* 1984;40(4):11–36.
9. Lakey B, Cohen S. Social support and theory. In: *Social Support Measurement and Intervention: A Guide for Health and Social Scientists.* 2000:29.
10. Kiernan M, et al. Social support for healthy behaviors: scale psychometrics and prediction of weight loss among women in a behavioral program. *Obesity.* 2012;20(4):756–764.
11. Jensen MT, Nielsen SS, Jessen-Winge C, et al. The effectiveness of social support based weight loss interventions: a systematic review and meta analysis. *Int J Obes (Lond).* 2024;48(5):599–611.
12. Paul-Ebhohimhen V, Avenell A. A systematic review of the effectiveness of group versus individual treatments for adult obesity. *Obes Facts.* 2009;2(1):17–24.
13. Tong HL, Laranjo L. The use of social features in mobile health interventions to promote physical activity: a systematic review. *npj Digital Medicine.* 2018;1(1):1–10.
14. Hales S, Turner-McGrievy GM, Wilcox S, et al. Social networks for improving healthy weight loss behaviors for overweight and obese adults: a randomized clinical trial of the social pounds off digitally (Social POD) mobile app. *Int J Med Inform.* 2016;94:81–90.
15. Turner-McGrievy GM, Delgado-Díaz DC, DuBois KE, et al. The mLIFE randomized trial examining the impact of gamifying social support provision for weight loss. *Obesity.* 2025;33(8):1447–1456.
16. DuBois KE, Delgado-Díaz DC, McGrievy M, et al. The Mobile Lifestyle Intervention for Food and Exercise (mLife) study: protocol of a remote behavioral weight loss randomized clinical trial for type 2 diabetes prevention. *Contemp Clin Trials.* 2024;148:107735.
17. Hochreiter S, Schmidhuber J. Long short term memory. *Neural Computation.* 1997;9(8):1735–1780.
18. Ali MK, Echouffo-Tcheugui JB, Williamson DF. How effective were lifestyle interventions in real world settings that were modeled on the Diabetes Prevention Program? *Health Affairs.* 2012;31(1):67–75.
19. Choi E, Bahadori MT, Schuetz A, Stewart WF, Sun J. Doctor AI: predicting clinical events via recurrent neural networks. *Proceedings of the 1st Machine Learning for Healthcare Conference.* 2016;56:301–318.
20. Ryan RM, Deci EL. Self-determination theory and the facilitation of intrinsic motivation, social development, and well-being. *American Psychologist.* 2000;55(1):68–78.
21. Nahum-Shani I, Smith SN, Spring BJ, Collins LM, Witkiewitz K, Tewari A, Murphy SA. Just-in-time adaptive interventions (JITAIs) in mobile health: key components and design principles for ongoing health behavior support. *Annals of Behavioral Medicine.* 2018;52(6):446–462.

## Supplementary materials

**Supplementary Figure S1. Participant flow diagram.** Of 199 participants randomized in the mLIFE trial, 122 had at least six weeks of usable longitudinal weight, support, and demographic data, and 108 met the lookback requirement for the 20-week-ahead outcome under the reference six-week aggregation window. *[Figure in preparation; the final version will follow CONSORT conventions.]*

**Supplementary Figure S2. Calibration plot for the reference LSTM.** Predicted versus observed weight change at 20 weeks ahead, computed from out-of-fold predictions pooled across 5 GroupKFold folds at a single fixed seed, with a LOWESS smoother and identity reference line. Two panels show the weight trajectory only baseline (wconly) and the activity support headline variant (swc_only_activity); calibration intercepts and slopes are reported in the figure and in §4.4.

**Supplementary Table S1. Baseline characteristics and trial outcomes for participants included in the LSTM analysis versus those excluded due to insufficient longitudinal data.** Continuous variables compared with Welch's t-test (mean (SD)); binary variables compared with chi-square (count (percent)). p-values are uncorrected.

| Variable | Included (n=108) | Excluded (n=91) | p-value |
|---|---|---|---|
| Baseline BMI | 34.20 (6.03) | 35.16 (5.66) | 0.252 |
| Age (years) | 49.47 (11.09) | 44.36 (11.54) | 0.002 |
| Baseline weight (lbs) | 210.21 (45.22) | 210.16 (37.77) | 0.993 |
| Women | 99/108 (91.7%) | 81/91 (89.0%) | 0.694 |
| Points (gamified) arm | 62/108 (57.4%) | 45/91 (49.5%) | 0.328 |
| Cohort 1 | 52/108 (48.1%) | 36/91 (39.6%) | 0.284 |
| Total weight change (lbs) | −13.23 (19.66) | −5.62 (12.23) | 0.001 |
| Total weight change (%) | −5.93 (8.22) | −2.61 (5.67) | 0.001 |
