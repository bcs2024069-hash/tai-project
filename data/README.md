# CDC BRFSS 2015 Diabetes Health Indicators Dataset

## Overview

This directory contains the dataset used for training, calibrating, and evaluating the **Trustworthy Diabetes Risk-Screening Decision System**.

- **Dataset Name:** CDC Diabetes Health Indicators (BRFSS 2015 balanced dataset)
- **Original Source:** U.S. Centers for Disease Control and Prevention (CDC) — Behavioral Risk Factor Surveillance System (BRFSS) 2015
- **Curated & Balanced Distribution:** Available on [Kaggle by Alex Teboul](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- **Primary CSV File:** `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`
- **File Size:** ~3.3 MB (3,307,815 bytes)
- **Dimensions:** 70,692 rows × 22 columns (21 feature indicators + 1 binary target)

---

## Dataset Characteristics

### Target Variable
- **`Diabetes_binary`**:
  - `0`: No diabetes diagnosis (35,346 respondents / 50.00%)
  - `1`: Prediabetes or diagnosed diabetes (35,346 respondents / 50.00%)

### Feature Codebook (21 Survey Indicators)

| Feature | Type | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `HighBP` | Binary | 0, 1 | Diagnosed with high blood pressure |
| `HighChol` | Binary | 0, 1 | Diagnosed with high blood cholesterol |
| `CholCheck` | Binary | 0, 1 | Cholesterol check within the past 5 years |
| `BMI` | Continuous | 12 – 98 | Body Mass Index ($kg/m^2$) |
| `Smoker` | Binary | 0, 1 | Smoked $\ge 100$ cigarettes in lifetime |
| `Stroke` | Binary | 0, 1 | Ever told had a stroke |
| `HeartDiseaseorAttack` | Binary | 0, 1 | Coronary heart disease or myocardial infarction |
| `PhysActivity` | Binary | 0, 1 | Physical activity in past 30 days (excluding job) |
| `Fruits` | Binary | 0, 1 | Consumes fruit $\ge 1$ time per day |
| `Veggies` | Binary | 0, 1 | Consumes vegetables $\ge 1$ time per day |
| `HvyAlcoholConsump` | Binary | 0, 1 | Heavy drinking (men $\ge 14$ drinks/week, women $\ge 7$) |
| `AnyHealthcare` | Binary | 0, 1 | Has any health insurance coverage |
| `NoDocbcCost` | Binary | 0, 1 | Needed to see doctor in past 12 mos but could not due to cost |
| `GenHlth` | Ordinal | 1 – 5 | General health self-rating (1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor) |
| `MentHlth` | Discrete | 0 – 30 | Days of poor mental health in past 30 days |
| `PhysHlth` | Discrete | 0 – 30 | Days of poor physical health/injury in past 30 days |
| `DiffWalk` | Binary | 0, 1 | Serious difficulty walking or climbing stairs |
| `Sex` | Binary | 0, 1 | Biological sex (0 = Female, 1 = Male) |
| `Age` | Ordinal | 1 – 13 | 5-year age category (1 = 18–24, ..., 13 = 80+) |
| `Education` | Ordinal | 1 – 6 | Education level (1 = None/Kindergarten, ..., 6 = College $\ge 4$ yrs) |
| `Income` | Ordinal | 1 – 8 | Income bracket (1 = <\$10k, ..., 8 = $\ge$\$75k) |

---

## Leakage Prevention & Partitioning Protocol

The dataset is partitioned into three disjoint splits using stratified sampling based on `Diabetes_binary` with random seed `42`:

- **Train Split (70%, N = 49,484):** Used exclusively for fitting `StandardScaler` parameters and model weights.
- **Validation Split (15%, N = 10,604):** Used exclusively for post-hoc probability calibration (Platt scaling) and Fairlearn threshold optimization.
- **Test Split (15%, N = 10,604):** Held-out independent evaluation set. Untouched during all training, calibration, and threshold tuning phases.

---

## Survey Deduplication Findings

In survey research involving 21 discretized variables, identical feature vectors naturally occur among distinct respondents.
- Removing exact duplicates yields 69,057 rows (1,635 removed).
- Among identical response vectors, **408 distinct feature vectors (representing 1,001 respondents)** exhibit **conflicting target labels** (one respondent diabetic, another non-diabetic).
- This confirms that duplicate rows reflect genuine population consensus rather than data pipeline errors.
- Both primary and deduplicated models achieve comparable performance (ROC-AUC 0.8251 vs. 0.8223).

---

## Reproducibility & Download Instructions

The primary balanced CSV is included directly in this directory for convenience and 100% reproducibility.

If you need to redownload the raw file:
1. Visit the Kaggle dataset page: [Kaggle: Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
2. Download `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`
3. Place the file inside the `data/` directory.
4. Run dataset validation:
   ```bash
   python tests/test_pipeline.py
   ```
