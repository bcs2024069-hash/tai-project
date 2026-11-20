## 1. Project Title
**Trustworthy Diabetes Risk-Screening Decision System: A Comparative Study of Predictive Performance, Explainability, Fairness, Calibration, and Robustness**

---

## 2. Project Overview
In chronic disease prevention, screening algorithms frequently prioritize aggregate discrimination metrics (such as ROC-AUC or raw accuracy) while overlooking essential dimensions of machine learning trustworthiness. A model that achieves high predictive accuracy may nevertheless output uncalibrated risk probabilities, perpetuate disparate impact against vulnerable demographic subgroups, operate as an opaque black box, or flip decisions under minor self-reporting noise.

This repository implements, evaluates, and compares two end-to-end predictive pipelines:
1. **Model 1: Standard Logistic Regression Baseline**  
   Standardized continuous/ordinal features and passthrough binary survey indicators fed to an $L_2$-regularized Logistic Regression classifier ($L_2$ penalty, L-BFGS solver, fixed 0.50 cutoff, raw sigmoid link).
2. **Model 2: Trust-Aware Decision Framework**  
   An identical underlying Logistic Regression pipeline wrapped with `sklearn.frozen.FrozenEstimator` and calibrated post-hoc via Platt sigmoid scaling (`CalibratedClassifierCV`) fitted strictly on held-out validation data. Audited with **SHAP** additive feature attributions, **Fairlearn** demographic parity and equalized odds metrics, and **6 controlled input perturbation stress tests**.

## 3. Problem Statement
Type 2 diabetes represents a massive global healthcare burden, affecting hundreds of millions worldwide. Early risk identification enables targeted lifestyle interventions and timely diagnostic follow-up. While machine learning offers scalable risk stratification on large-scale public health surveys like the CDC Behavioral Risk Factor Surveillance System (BRFSS), clinical deployment faces critical hurdles:
- **Overconfident or Miscalibrated Probabilities:** Clinicians rely on predicted probabilities as meaningful risk scores, but raw classification models can produce misaligned probability estimates.
- **Hidden Algorithmic Bias:** Disparities across biological sex, age brackets, and socioeconomic gradients (income) can result in disproportionate false negative or false positive rates.
- **Lack of Interpretability:** Black-box risk scores fail to reveal modifiable vs. non-modifiable risk factors for individual patients.
- **Sensitivity to Survey Noise:** Self-reported survey data naturally incurs bounded reporting noise (e.g., slight recall variance in general health ratings or BMI measurements).

---

## 4. Project Objectives
1. **Develop Reproducible Pipelines:** Build clean, leak-free scikit-learn predictive pipelines for diabetes risk screening on the CDC BRFSS 2015 cohort.
2. **Evaluate Probability Calibration:** Quantify empirical probability alignment via Platt sigmoid scaling, Expected Calibration Error (ECE across 10 decile bins), and Brier score.
3. **Audit Algorithmic Fairness:** Quantify demographic disparities across **Sex** (Female vs. Male), **Age** (13 ordinal brackets), and **Income** (8 socioeconomic brackets) using Demographic Parity Difference (DPD) and Equalized Odds Difference (EOD).
4. **Investigate Post-Processing Mitigation:** Empirically evaluate the accuracy vs. equity trade-off using Fairlearn's `ThresholdOptimizer` on sensitive demographic groups.
5. **Implement Local & Global Explainability:** Compute exact additive feature attributions using SHAP `LinearExplainer` in uncalibrated log-odds space.
6. **Stress-Test Input Robustness:** Evaluate decision stability and probability drift across 6 clinically realistic input perturbations clamped to valid codebook bounds.
7. **Deliver an Interactive Decision System:** Package all models, audits, and patient risk profiles into an intuitive Streamlit interface.

---

## 5. Key Features
- **Strict Leak-Free Partitioning:** 70% Train, 15% Validation, and 15% Held-Out Test split. Preprocessing parameters and baseline weights are fitted strictly on training data; calibration and threshold optimization are fitted strictly on validation data.
- **Survey Deduplication Analysis:** Deep empirical investigation of identical survey response vectors, uncovering conflicting target labels that confirm distinct individual respondents.
- **SHAP Feature Importance:** Global mean $|SHAP|$ importance rankings and patient-level waterfall diagrams decomposing log-odds attributions.
- **Comprehensive Fairness Audits:** Disaggregated performance tables, selection rates, True Positive Rates (TPR), and False Positive Rates (FPR) across 3 sensitive attributes.
- **Threshold Mitigation Trade-Offs:** Pareto front visualization illustrating the accuracy cost of enforcing equalized odds across income levels.
- **Robustness Sensitivity Explorer:** Interactive examination of 500 test cases under bounded perturbations of `GenHlth`, `BMI`, `PhysHlth`, and `MentHlth`.
- **Precomputed Artifact Architecture:** Zero model retraining at dashboard runtime; all 11 serialized artifacts load instantly via cached data managers.

---

## 6. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                   CDC BRFSS 2015 Survey (N = 70,692)                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
      70% Training (N=49,484)              15% Validation (N=10,604)
                  │                                   │
                  ▼                                   ▼
   ┌─────────────────────────────┐     ┌─────────────────────────────┐
   │ StandardScaler + Baseline LR │     │ Platt Sigmoid Calibration   │
   │ Pipeline (Model 1)          │     │ (FrozenEstimator, Model 2)  │
   └──────────────┬──────────────┘     └──────────────┬──────────────┘
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    │
                                    ▼
                         15% Held-Out Test Split
                              (N = 10,604)
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│   Discrimination │       │  Trustworthy AI  │       │  Stress Testing  │
│  Accuracy: 74.76%│       │  SHAP Explainer  │       │  6 Perturbations │
│  ROC-AUC: 0.8251 │       │  Fairlearn Audit │       │  500 Test Cases  │
│  Brier: 0.1698   │       │  10-Bin Calib.   │       │  Flip Rates      │
└────────┬─────────┘       └────────┬─────────┘       └────────┬─────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    │
                                    ▼
                      11 Precomputed Artifacts (.json, .csv, .joblib)
                                    │
                                    ▼
                 Interactive Multi-Page Streamlit Dashboard
```

---

## 7. Technologies Used
- **Programming Language:** Python 3.10 – 3.13
- **Machine Learning:** `scikit-learn >= 1.6.0` (`Pipeline`, `ColumnTransformer`, `LogisticRegression`, `CalibratedClassifierCV`, `FrozenEstimator`)
- **Data Engineering:** `pandas >= 2.0.0`, `numpy >= 1.24.0`, `scipy >= 1.11.0`
- **Explainability:** `shap >= 0.44.0` (`LinearExplainer`)
- **Algorithmic Fairness:** `fairlearn >= 0.9.0` (`MetricFrame`, `ThresholdOptimizer`, `demographic_parity_difference`, `equalized_odds_difference`)
- **Interactive Application:** `streamlit >= 1.28.0`
- **Visualizations:** `plotly >= 5.18.0`, `matplotlib >= 3.7.0`, `seaborn >= 0.12.0`
- **Model Serialization:** `joblib >= 1.3.0`

---

## 8. Dataset Description

The dataset is derived from the **CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015**, balanced 50/50 between diabetic/prediabetic and non-diabetic respondents:
- **Total Records:** 70,692 respondents (35,346 diabetic, 35,346 non-diabetic).
- **Features (21 survey variables):**
  - **Continuous (1):** `BMI` (Body Mass Index, 12 – 98 $kg/m^2$).
  - **Discrete Days (2):** `MentHlth` (Poor mental health days, 0 – 30), `PhysHlth` (Poor physical health days, 0 – 30).
  - **Ordinal (4):** `GenHlth` (General health, 1–5), `Age` (13 brackets: 1 = 18–24, ..., 13 = 80+), `Education` (1–6), `Income` (1–8 brackets: 1 = <\$10k, ..., 8 = $\ge$\$75k).
  - **Binary Indicators (14):** `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex`.
- **Prevalence Notice:** The 50/50 balance is curated for supervised model training. Real-world adult US diabetes prevalence is approximately 10–14%.

---

## 9. Machine Learning Methodology

### Model 1: Baseline Logistic Regression Pipeline
```python
ColumnTransformer([
    ('scaler', StandardScaler(), ['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income']),
    ('binary', 'passthrough', ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 
                              'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
                              'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex'])
]) -> LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
```

### Model 2: Trust-Aware Decision Framework
Wraps the baseline pipeline using `sklearn.frozen.FrozenEstimator` and fits a secondary Platt sigmoid calibration mapping on the validation partition:
```python
CalibratedClassifierCV(
    estimator=FrozenEstimator(baseline_pipeline),
    method='sigmoid',
    cv='prefit'
)
```

### Empirical Test Set Results (N = 10,604 Held-Out Records)

| Metric | Model 1 (Baseline LR) | Model 2 (Trust-Aware LR) | Difference (M2 - M1) | Scientific Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **ROC-AUC** | **0.8251** | **0.8251** | **0.0000** | Identical. Monotonic sigmoid scaling preserves instance rank order. |
| **Accuracy** | **74.76%** | **74.75%** | **-0.01%** | Negligible change at default 0.50 decision threshold. |
| **Recall (Sensitivity)** | **76.63%** | **76.01%** | **-0.62%** | Baseline identifies 33 additional positive cases in test cohort. |
| **Precision (PPV)** | **73.87%** | **74.14%** | **+0.27%** | Calibrated model yields slightly fewer false positives. |
| **F1-Score** | **0.7523** | **0.7506** | **-0.0016** | Harmonic mean remains virtually identical. |
| **Brier Score (MSE)** | **0.1698** | **0.1699** | **+0.0001** | Probability squared error is virtually indistinguishable. |
| **ECE (10 Bins)** | **0.0262** | **0.0280** | **+0.0018** | Baseline Logistic Regression is already inherently well-calibrated. |

---

## 10. Explainability Using SHAP
- **Algorithm:** Tree/Kernel/Linear SHAP (`shap.LinearExplainer` evaluated on the uncalibrated linear log-odds margin $z = \beta_0 + \sum \beta_i \tilde{x}_i$).
- **Top 5 Global Drivers:**
  1. `GenHlth` (Rank 1, Mean $|SHAP| = 0.5072$)
  2. `BMI` (Rank 2, Mean $|SHAP| = 0.4104$)
  3. `HighBP` (Rank 3, Mean $|SHAP| = 0.3513$)
  4. `Age` (Rank 4, Mean $|SHAP| = 0.3301$)
  5. `HighChol` (Rank 5, Mean $|SHAP| = 0.2924$)
- **Local Explanations:** The dashboard computes additive patient-level waterfalls for 15 representative clinical cases, decomposing exact risk contributions.

---

## 11. Fairness Evaluation
Disparities were audited across three sensitive attributes using **Fairlearn**:

| Sensitive Attribute | Subgroups | Demographic Parity Difference (DPD) | Equalized Odds Difference (EOD) | Key Subgroup Finding |
| :--- | :---: | :---: | :---: | :--- |
| **Sex** | Female vs. Male | **0.0572** | **0.0512** | Highly equitable; minimal disparity between sexes. |
| **Age** | 13 Brackets (18 to 80+) | **0.7387** | **0.7131** | Pronounced biological gradient: Selection rate 2.9% (18–24) vs. 76.8% (80+). |
| **Income** | 8 Brackets (<\$10k to $\ge$\$75k) | **0.5125** | **0.4307** | Strong socioeconomic disparity: Selection rate 72.0% (<\$10k) vs. 20.8% ($\ge$\$75k). |

### Fairness Mitigation Experiment (Income Attribute)
Using Fairlearn's `ThresholdOptimizer` (optimizing for Equalized Odds with validation-set fitting):
- **EOD Before:** `0.4307` $\rightarrow$ **EOD After:** `0.0944` (**78.1% disparity reduction**).
- **Accuracy Trade-off:** `74.76%` $\rightarrow$ `68.81%` (**-5.95% accuracy penalty**).
- **Operational Requirement:** Applying group-specific decision thresholds mandates that patient income must be disclosed and provided at inference time.

---

## 12. Robustness Evaluation
The system was evaluated under 6 domain-constrained input perturbations across 500 test cases (6,000 evaluations total):

| Perturbation | Perturbation Magnitude | Mean $|\Delta p|$ | Max $|\Delta p|$ | Decision Flip Rate | Flip Count / 500 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `genhlth_better` | `GenHlth` $-1$ (clamped $\ge 1$) | **0.0908** | **0.1445** | **14.0%** | 70 / 500 |
| `genhlth_worse` | `GenHlth` $+1$ (clamped $\le 5$) | **0.0920** | **0.1445** | **13.0%** | 65 / 500 |
| `bmi_minus1` | `BMI` $-1.0\text{ kg/m}^2$ | **0.0126** | **0.0187** | **1.0%** | 5 / 500 |
| `bmi_plus1` | `BMI` $+1.0\text{ kg/m}^2$ | **0.0126** | **0.0187** | **1.8%** | 9 / 500 |
| `physhlth_plus1` | `PhysHlth` $+1$ day (clamped $\le 30$) | **0.0012** | **0.0019** | **0.2%** | 1 / 500 |
| `menthlth_plus1` | `MentHlth` $+1$ day (clamped $\le 30$) | **0.0007** | **0.0011** | **0.2%** | 1 / 500 |

`GenHlth` dominates model sensitivity, triggering up to a 14% decision flip rate, whereas mental and physical illness day counts have negligible perturbation impact.


