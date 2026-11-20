# Trustworthy Diabetes Risk-Screening Decision System

An academic comparative study of standard vs. trust-aware machine learning pipelines for population-level diabetes risk screening using the **CDC BRFSS 2015 Health Survey Dataset** (70,692 respondents, 21 survey features).

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%E2%89%A51.6.0-F7931E.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-%E2%89%A50.44.0-00A4E4.svg)](https://shap.readthedocs.io/)
[![Fairlearn](https://img.shields.io/badge/Fairlearn-%E2%89%A50.9.0-008080.svg)](https://fairlearn.org/)
[![Tests](https://img.shields.io/badge/Tests-10%2F10%20Passing-brightgreen.svg)]()

> ⚠️ **CLINICAL RESEARCH & EDUCATIONAL PROTOTYPE DISCLAIMER**  
> This software is an academic research and decision-support prototype developed exclusively to investigate trustworthy artificial intelligence methodologies (probability calibration, algorithmic fairness auditing, feature explainability, and input perturbation robustness). It is **not** a certified medical diagnostic system, does not establish formal clinical diagnoses, and must **never** be used as a substitute for professional clinical judgment, laboratory testing, or medical consultations.

---

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

All experimental findings are delivered through an interactive, 7-page multi-page **Streamlit** analytical dashboard powered strictly by precomputed, serialized artifacts (zero runtime training latency).

---

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

---

## 13. Project Directory Structure

```
Trustworthy_Diabetes_Project/
│
├── app/                                 # Interactive Streamlit Multi-Page Application
│   ├── app.py                           # Main application entry point & Executive Overview
│   ├── pages/                           # 7 Modular analytical dashboard pages
│   │   ├── 1_Project_Overview.py        # Cohort summary, 21-feature codebook, data splitting
│   │   ├── 2_Model_Comparison.py        # Side-by-side metrics, ROC curves, confusion matrices
│   │   ├── 3_Case_Predictions.py        # 15 Patient clinical profiles & risk gauges
│   │   ├── 4_SHAP_Explainability.py     # Global importance & local waterfall explanations
│   │   ├── 5_Fairness_Audit.py          # Subgroup audits & ThresholdOptimizer trade-offs
│   │   ├── 6_Calibration.py             # 10-bin reliability diagrams & ECE analysis
│   │   └── 7_Robustness.py              # 6 Input perturbation stress tests
│   └── utils/                           # Application helper library
│       ├── __init__.py                  # Utility package marker
│       ├── data_loader.py               # Cached artifact loaders (@st.cache_data)
│       ├── helpers.py                   # Formatting utilities, codebook, UI helpers
│       └── visualizations.py            # Themed Plotly chart generators
│
├── artifacts/                           # 11 Serialized precomputed experiment artifacts
│   ├── baseline_model.joblib            # Model 1: Fitted baseline pipeline
│   ├── trustworthy_model.joblib         # Model 2: Fitted calibrated pipeline
│   ├── evaluation_metrics.json          # Test metrics, confusion matrices, ROC coordinates
│   ├── fairness_metrics.json            # Subgroup metrics & ThresholdOptimizer data
│   ├── model_metadata.json              # Model configuration & deduplication sensitivity
│   ├── calibration_data.csv             # 10-bin reliability curve decile coordinates
│   ├── robustness_summary.csv           # Aggregated perturbation metrics
│   ├── robustness_results.csv           # 6,000 sample-level perturbation outcomes
│   ├── sample_cases.csv                 # 15 Curated patient clinical profiles
│   ├── shap_global_importance.csv       # Global feature importance rankings
│   └── shap_individual_values.csv       # Local SHAP attributions for 15 sample cases
│
├── assets/                              # Documentation figures and visual media
│   ├── figures/                         # 11 Publication-quality analytical figures
│   └── screenshots/                     # Application UI documentation
│
├── data/                                # Primary experimental survey data
│   ├── diabetes_binary_5050split_...csv # CDC BRFSS 2015 50/50 balanced CSV (N=70,692)
│   └── README.md                        # Dataset documentation, codebook & source citations
│
├── docs/                                # Formal research publications
│   ├── Trustworthy_AI_Project_Report.pdf# Complete 8-page IEEE conference paper (PDF)
│   └── README.md                        # Documentation overview and build guide
│
├── notebooks/                           # Interactive research environments
│   └── train_models_colab.ipynb         # Self-contained Google Colab training notebook
│
├── project_report/                      # Publication sources & build engines
│   ├── Trustworthy_AI_Project_Report.pdf# Direct compiled IEEE conference paper
│   ├── Trustworthy_AI_Project_Report.docx# Official IEEE 2-column Word document
│   ├── Trustworthy_AI_Project_Report.tex# IEEEtran LaTeX source code
│   ├── report_ieee.typ                  # High-speed Typst typesetting engine source
│   ├── references.bib                   # BibTeX database with 15 verified citations
│   ├── figures/                         # 11 Publication figures (300 DPI)
│   └── README.txt                       # Build instructions for Overleaf / LaTeX / Typst
│
├── reference/                           # Architectural presentation materials
│   └── Trustworthy_Diabetes_...pptx    # Project presentation slide deck
│
├── src/                                 # Modular Python machine learning library
│   ├── __init__.py                      # Package initialization
│   ├── artifact_io.py                   # Serializers and loaders for model artifacts
│   ├── calibration.py                   # Reliability curve computation, Brier, and ECE
│   ├── data_loader.py                   # BRFSS dataset validation and stratified splitting
│   ├── explainability.py                # SHAP LinearExplainer attribution generators
│   ├── fairness.py                      # Fairlearn demographic parity & equalized odds
│   ├── mitigation.py                    # Fairlearn ThresholdOptimizer post-processing
│   ├── model_baseline.py                # Model 1 pipeline builder and evaluator
│   ├── model_trustworthy.py             # Model 2 FrozenEstimator calibration pipeline
│   ├── preprocessing.py                 # ColumnTransformer pipeline definition
│   └── robustness.py                    # Controlled survey input perturbation engine
│
├── tests/                               # Test suite
│   └── test_pipeline.py                 # 10-module end-to-end integration test suite
│
├── .gitignore                           # Git ignore rules for Python, cache, and IDEs
├── requirements.txt                     # Pinned project dependencies
├── run_full_training.py                 # Master script to reproduce all 11 artifacts
└── README.md                            # Comprehensive project documentation
```

---

## 14. Installation Instructions

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### Windows Installation (PowerShell / Command Prompt)
```powershell
# 1. Clone the repository
git clone https://github.com/surya-prathap/Trustworthy_Diabetes_Project.git
cd Trustworthy_Diabetes_Project

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip and install required dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS Installation
```bash
# 1. Clone the repository
git clone https://github.com/surya-prathap/Trustworthy_Diabetes_Project.git
cd Trustworthy_Diabetes_Project

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 15. Execution Instructions

### Launch the Streamlit Dashboard
To launch the interactive multi-page decision support dashboard:
```bash
streamlit run app/app.py
```
*The application starts immediately on `http://localhost:8501`. It reads directly from precomputed artifacts in `artifacts/` with zero runtime training latency.*

### Execute the Integration Test Suite
To verify end-to-end pipeline integrity, data loading, preprocessing, model evaluation, SHAP explainability, and artifact I/O:
```bash
python tests/test_pipeline.py
```

### Retrain Models and Reproduce All Artifacts
To execute the complete end-to-end experimental training run from raw data and re-generate all 11 serialized artifacts:
```bash
python run_full_training.py
```
*Alternatively, you can open and run `notebooks/train_models_colab.ipynb` directly in Google Colab.*

---

## 16. Expected Application Workflow

When running `streamlit run app/app.py`, the user interacts through 7 specialized pages:

```
[Main Entry: app.py] ──> Executive Summary & Model Cards
       │
       ├──> [Page 1: Project Overview]     ──> Cohort stats, 21 feature descriptions, data splits
       ├──> [Page 2: Model Comparison]     ──> Overlaid ROC curve (AUC=0.8251) & confusion matrices
       ├──> [Page 3: Case Predictions]     ──> 15 Patient cards, probability dials, risk flags
       ├──> [Page 4: SHAP Explainability]  ──> Global importance bar chart & local waterfall plots
       ├──> [Page 5: Fairness Audit]       ──> Demographic parity, equalized odds, mitigation Pareto front
       ├──> [Page 6: Calibration]          ──> 10-bin decile reliability curve, ECE, base-rate warning
       └──> [Page 7: Robustness]           ──> 6 perturbation impact charts & sample flip explorer
```

---

## 17. Limitations
- **Synthetic 50/50 Class Balance:** The BRFSS 2015 cohort was balanced to facilitate stable binary classification. Real-world adult US diabetes prevalence is ~10–14%. Uncalibrated probability estimates from balanced training cohorts reflect risks relative to a balanced sample rather than true population base rates.
- **Cross-Sectional Survey Nature:** BRFSS is an observational self-report survey. It cannot establish longitudinal etiology or direct causality.
- **Survey Discretization:** Variables such as `Age` (13 five-year brackets) and `Income` (8 brackets) are discretized, masking intra-bracket non-linearities.
- **Inference Requirement for Fairness Mitigation:** Post-processing mitigation via `ThresholdOptimizer` requires the patient's sensitive attribute (`Income`) to be collected at inference time, which may raise operational and privacy constraints.

---

## 18. Academic and Research Disclaimer
This software is an educational and academic research artifact designed to demonstrate Trustworthy Artificial Intelligence principles. It has not received clearance or approval from the U.S. Food and Drug Administration (FDA), European Medicines Agency (EMA), or any regulatory medical authority. It must not be employed for clinical diagnostics, patient triage, or direct therapy selection.

---

## 19. Team Contributions
- **Surya Prathap**: System Architecture, Machine Learning Pipeline Design, Trustworthy AI Methodology (Calibration, Fairness, Explainability, Robustness), Streamlit Dashboard Development, and Academic Paper Authorship.

---

## 20. References
1. **CDC BRFSS 2015:** Centers for Disease Control and Prevention (CDC). *Behavioral Risk Factor Surveillance System Survey Data*. Atlanta, Georgia: U.S. Department of Health and Human Services, 2015.
2. **SHAP Framework:** Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 4765–4774.
3. **Fairlearn Toolkit:** Bird, S., Dudík, M., Edgar, R., Horn, B., Lutz, R., Milan, V., Sameki, M., Wallach, H., & Walker, K. (2020). *Fairlearn: A toolkit for assessing and improving fairness in AI*. Microsoft Technical Report MSR-TR-2020-32.
4. **Platt Sigmoid Calibration:** Platt, J. (1999). *Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods*. Advances in Large Margin Classifiers, 10(3), 61–74.
5. **Expected Calibration Error:** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On calibration of modern neural networks*. International Conference on Machine Learning (ICML 2017), PMLR, 1321–1330.
6. **Equalized Odds in ML:** Hardt, M., Price, E., & Srebro, N. (2016). *Equality of opportunity in supervised learning*. Advances in Neural Information Processing Systems (NeurIPS 2016), 29, 3315–3323.
