# Predicting Sepsis Risk during In-patient admissions

## Overview

This project, developed in collaboration with Royal Perth Hospital (RPH) and the Health in a Virtual Environment (HIVE), delivers machine learning–driven solutions for early detection of sepsis in hospital admissions. Sepsis, a life-threatening condition with often vague early symptoms, rapid progression, and high mortality, requires rapid diagnosis—each hour of delay in treatment markedly increases the risk of death.

## Project Goals

- Support clinicians with standardized, interpretable sepsis risk scores.
- Optimize hospital resource allocation through automated early detection.
- Reduce mortality and healthcare costs by enabling timely interventions.

## Objectives

- Develop robust predictive models using structured EHR data (demographics, comorbidities, laboratory tests).
- Benchmark and compare traditional machine learning, deep learning (LSTM), and survival analysis techniques.
- Prioritize model interpretability alongside accuracy to strengthen clinical adoption.

## Methodology

- **Data Preparation**: Leveraged MIMIC-III dataset and RPH records; excluded patients under 18, anomalous ages, or missing admission IDs. Structured early-stage data up to 8 hours after admission. Addressed class imbalance (sepsis ≈ 10.9%) with stratified sampling.
- **Feature Selection**: Guided by clinical literature and input, focused on key biomarker derangements, co-morbidities, and SOFA score. Applied Neo4j-based feature engineering to identify relationships between lab results and comorbidities.
- **Modeling Approaches**: 
  - Traditional ML: Logistic Regression, Random Forest, Gradient Boosting
  - Deep Learning: LSTM for sequential patient data (0–8 hours)
  - Survival Analysis: Cox Proportional Hazards, CoxNet, Random Survival Forests, Gradient Boosted Survival
- **Evaluation Metrics**: ML/DL—Balanced Accuracy, Precision, Recall, F1-score, AUROC; Survival Analysis—Concordance Index (C-index), Integrated Brier Score (IBS).

## Results

- **Traditional ML (Random Forest)**:  
  - Balanced Accuracy: 0.747  
  - Recall: 0.758  
  - AUROC: 0.828  
  - Top Features: Lactate, Neutrophils, Lymphocytes, WBC, Bilirubin
- **Deep Learning (LSTM)**:  
  - Balanced Accuracy: 0.766  
  - Recall: 0.747  
  - AUROC: 0.842  
  - Key Influencers: Lymphocytes, Neutrophils, Urea Nitrogen, Platelet Count (via TimeSHAP)
- **Survival Analysis (Random Survival Forest)**:  
  - C-index: 0.795  
  - IBS: 0.102  
  - Provided time-dependent risk probabilities for prioritizing care

## Key Insights

- LSTM models excelled in early detection performance.
- Random Forests achieved a valuable trade-off between accuracy and interpretability—ideal for clinical deployment.
- Survival models enhanced risk assessment by predicting the likelihood of sepsis onset over time.
- Biomarkers like Lactate, Neutrophils, Lymphocytes, and the SOFA score emerged as consistently informative.
- The project validates the use of ML and DL for early sepsis detection and demonstrates that clinician-facing risk scores enhance patient outcomes and streamline resource allocation.

## Conclusion

Integrating machine learning and survival analysis delivers accurate, interpretable, and timely sepsis identification to support hospital clinicians. This approach standardizes early warning, reduces subjective variability in clinical judgment, and sets the groundwork for scalable, life-saving interventions in acute care settings.

---
