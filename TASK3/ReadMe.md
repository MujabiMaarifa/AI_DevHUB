# Heart Disease Prediction Project

## 📌 Task Objective
Develop a binary classification model to predict whether a patient has heart disease based on clinical features, demonstrating:
- End-to-end ML workflow from EDA to model deployment
- Interpretation of medical features
- Comparison of model performance metrics

## 🗃️ Dataset Used
**Heart Disease UCI Dataset** (Cleveland subset)  
- **Features**: 13 clinical attributes including:
  - Demographic: `age`, `sex`
  - Physiological: `resting_blood_pressure`, `cholestoral`
  - Diagnostic: `oldpeak` (ST depression), `thalassemia`
- **Target**: `target` (0 = no disease, 1 = disease)
- **Samples**: 1025 records
- **Preprocessing**: Handled missing values, scaled numerical features

## 🤖 Models Applied
| Model | Accuracy | ROC-AUC | Key Advantage |
|-------|----------|---------|---------------|
| Logistic Regression | 73.3% | 0.7359 | Interpretability |


## 🔑 Key Findings
1. **Top Predictive Features**:
   - `thalassemia` (β=0.62)
   - `exercise_induced_angina` (β=0.51)
   - `Max_heart_rate` (β=-0.47)

2. **Performance Insights**:
   - All models achieved >70% accuracy
   - Random Forest showed best generalization (AUC 0.7370)
   - Logistic Regression provided best trade-off of performance/interpretability

3. **Clinical Relevance**:
   - Exercise-induced abnormalities strongly predictive
   - Maximum heart rate inversely correlated with risk

## 🛠️ How to Reproduce
```bash
1. python3 Heart_Disease.ipynb
 