# Advanced Time Series Forecasting with LSTM

## 📌 Project Overview
This project demonstrates an end-to-end **multivariate time series forecasting pipeline**
using a **Long Short-Term Memory (LSTM)** neural network.  
The model is compared against a classical **ARIMA baseline** and includes:

- Hyperparameter optimization (Optuna)
- Early stopping
- Explainability with SHAP
- Sensitivity analysis
- Proper inverse scaling of predictions

Synthetic data is used to simulate realistic temporal patterns.

---

## 🧠 Model Architecture
- LSTM (PyTorch)
- Fully connected output layer
- Sequence-to-one prediction
- Optimized hidden units via Optuna

---

## 📊 Dataset
Synthetic time series with:
- `feature_1`: sinusoidal pattern + noise
- `feature_2`: cosine pattern + noise
- `target`: linear + nonlinear combination of features and time

---

## ⚙️ Pipeline
1. Data generation
2. MinMax scaling
3. Sliding window sequence creation
4. Train / validation / test split
5. Hyperparameter tuning (Optuna)
6. Final training with early stopping
7. Inverse scaling and evaluation
8. ARIMA baseline comparison
9. SHAP explainability
10. Sensitivity analysis

---

## 📈 Evaluation Metrics
- RMSE
- MAE
- MAPE

Metrics are reported **on the original data scale**.

---

## 🔍 Explainability
SHAP values are computed using `shap.DeepExplainer` to:
- Identify dominant temporal features
- Understand feature impact across sequences
- Improve model transparency

---

## 🧪 Sensitivity Analysis
A simulated shock is applied to one feature to verify:
- Stability
- Smooth model response
- Robustness to input perturbations

---

## 🚀 How to Run
```bash
pip install numpy pandas torch sklearn statsmodels optuna shap matplotlib
python main.py
