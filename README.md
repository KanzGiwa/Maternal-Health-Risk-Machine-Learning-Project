# Maternal Health Risk Prediction

## Overview

This project analyzes maternal health indicators to predict pregnancy risk levels using machine learning techniques. The study aims to identify critical factors that contribute to pregnancy complications and develop predictive models to enhance maternal healthcare outcomes.

## Project Context

Maternal health is a critical area of focus in public health, as complications during pregnancy can have significant consequences for both mother and child. This analysis uses the *Maternal Health Risk Data* dataset containing 1,014 records with seven key features to predict pregnancy risk levels categorized as low, medium, or high risk.

## Dataset Description

The dataset includes the following features:

- **Age**: Maternal age in years during pregnancy
- **SystolicBP**: Upper blood pressure value (mmHg)
- **DiastolicBP**: Lower blood pressure value (mmHg)
- **BS (Blood Sugar)**: Blood glucose levels during pregnancy
- **BodyTemp**: Body temperature during pregnancy
- **HeartRate**: Heart rate (beats per minute)
- **RiskLevel**: Target variable (Low Risk, Medium Risk, High Risk)

**Dataset Statistics:**
- Total Records: 1,014
- Features: 6 numerical + 1 categorical target
- No missing values
- Class Distribution: Low Risk (40.0%), Medium Risk (33.1%), High Risk (26.8%)

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Statistical analysis and distribution visualization
- Correlation analysis between features
- Risk level distribution analysis
- Box plots and scatter matrix visualization
- Feature relationship exploration

### 2. Predictive Modeling
Five different machine learning models were implemented and compared:

| Model | Test Accuracy | Training Accuracy | Key Metrics |
|-------|---------------|-------------------|-------------|
| **k-NN** | **86%** | 89% | Best overall performance |
| Decision Tree | 80% | 93% | Prone to overfitting |
| MLP Neural Network | Variable | Variable | Complex patterns detection |
| Naive Bayes | 53% | - | Baseline comparison |
| SVM | 81% | - | Good boundary separation |

### 3. Model Interpretation
- **SHAP Analysis**: Feature importance and contribution analysis
- **Decision Tree Visualization**: Interpretable decision rules
- **Feature Ranking**: Blood Sugar, SystolicBP, and Age identified as top predictors

### 4. Bias Analysis
- Disparate impact assessment
- Fairness evaluation across different groups
- Recommendations for bias mitigation

## Key Findings

### Most Predictive Features (SHAP Analysis)
1. **Blood Sugar (BS)** - Most critical predictor
2. **Systolic Blood Pressure** - Strong correlation with risk levels
3. **Age** - Non-linear relationship (higher risk for <20 and >35 years)
4. **Diastolic Blood Pressure** - Complementary to systolic BP
5. **Body Temperature & Heart Rate** - Moderate importance

### Model Performance Insights
- **k-NN model** achieved the best performance with 86% accuracy and minimal overfitting
- Medium-risk pregnancies were most challenging to classify due to overlapping features
- Blood pressure and blood sugar levels showed clear thresholds for risk prediction
- Age demonstrated expected patterns with increased risk for very young and older mothers

### Clinical Implications
- Elevated blood pressure (>130/90 mmHg) strongly indicates higher risk
- Blood sugar monitoring is crucial for risk assessment
- Age-specific care protocols recommended for <20 and >35 age groups
- Multi-factor assessment needed for medium-risk classification

## Project Structure

```
maternal-health-risk-prediction/
├── README.md
├── data/
│   └── maternalRisk.csv
├── notebooks/
│   └── maternal_health_risk_ProjectCode.ipynb
├── models/
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   └── model_comparison_results.csv
├── visualizations/
│   ├── eda_plots/
│   ├── model_performance/
│   └── shap_analysis/
├── reports/
│   ├── final_analysis_report.pdf
│   └── model_performance_summary.md
└── requirements.txt
```

## Requirements

```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
xgboost>=1.5.0
shap>=0.40.0
aif360>=0.4.0
```

## Usage

1. **Data Preparation**: Load and clean the maternal health dataset
2. **Exploratory Analysis**: Run EDA cells for data understanding
3. **Model Training**: Execute model training and comparison
4. **Evaluation**: Analyze model performance and interpret results
5. **Prediction**: Use best model (k-NN) for new predictions

## Results Summary

- **Best Model**: k-NN with 86% accuracy (95% CI: 81%, 91%)
- **Key Predictors**: Blood Sugar, Systolic BP, Age
- **Clinical Insight**: Clear thresholds identified for risk stratification
- **Bias Considerations**: Disparate impact detected, requiring fairness improvements

## Future Work

1. **Feature Enhancement**: Include lifestyle, diet, and medical history
2. **Advanced Models**: Ensemble methods and deep learning approaches
3. **Bias Mitigation**: Implement fairness-aware machine learning techniques
4. **Real-time Application**: Deploy model for clinical decision support
5. **Longitudinal Analysis**: Track risk changes throughout pregnancy

## Contributing

This project supports maternal healthcare research and welcomes contributions for:
- Model improvements
- Feature engineering
- Bias reduction techniques
- Clinical validation

## Acknowledgments

- Dataset source: Maternal Health Risk Data
- Analysis frameworks: scikit-learn, SHAP, AIF360
- Visualization tools: matplotlib, seaborn, plotly

---

*This project aims to contribute to improved maternal healthcare outcomes through data-driven insights and predictive modeling.*
