# ================================
# Advanced Time Series Forecasting
# ================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

import optuna
import shap
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# 1. Configuration
# ----------------------------
SEED = 42
WINDOW = 20
EPOCHS = 100
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------
# 2. Synthetic Data Generation
# ----------------------------
time_steps = 500
t = np.arange(time_steps)

feature_1 = np.sin(0.02 * t) + np.random.normal(0, 0.1, time_steps)
feature_2 = np.cos(0.015 * t) + np.random.normal(0, 0.1, time_steps)
target = (
    0.5 * feature_1 +
    0.3 * feature_2 +
    0.05 * t +
    np.random.normal(0, 0.2, time_steps)
)

data = pd.DataFrame({
    "feature_1": feature_1,
    "feature_2": feature_2,
    "target": target
})

# ----------------------------
# 3. Scaling
# ----------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# ----------------------------
# 4. Sequence Creation
# ----------------------------
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window, -1])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, WINDOW)

# Train / Val / Test split
train_end = int(0.7 * len(X))
val_end = int(0.85 * len(X))

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32).to(DEVICE)

X_train, y_train = to_tensor(X_train), to_tensor(y_train)
X_val, y_val = to_tensor(X_val), to_tensor(y_val)
X_test, y_test = to_tensor(X_test), to_tensor(y_test)

# ----------------------------
# 5. LSTM Model
# ----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).squeeze()

# ----------------------------
# 6. Training Utilities
# ----------------------------
def train_model(model, optimizer, criterion, X_train, y_train):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_model(model, criterion, X, y):
    model.eval()
    with torch.no_grad():
        return criterion(model(X), y).item()

# ----------------------------
# 7. Optuna Hyperparameter Tuning
# ----------------------------
def objective(trial):
    hidden_size = trial.suggest_int("hidden_size", 16, 64)
    lr = trial.suggest_float("lr", 1e-4, 1e-2)

    model = LSTMModel(3, hidden_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(30):
        train_model(model, optimizer, criterion, X_train, y_train)

    return evaluate_model(model, criterion, X_val, y_val)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best_params = study.best_params
print("Best params:", best_params)

# ----------------------------
# 8. Final Training with Early Stopping
# ----------------------------
model = LSTMModel(3, best_params["hidden_size"]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
criterion = nn.MSELoss()

best_val_loss = np.inf
patience_counter = 0

for epoch in range(EPOCHS):
    train_loss = train_model(model, optimizer, criterion, X_train, y_train)
    val_loss = evaluate_model(model, criterion, X_val, y_val)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

# ----------------------------
# 9. Predictions & Inverse Scaling
# ----------------------------
model.eval()
with torch.no_grad():
    preds = model(X_test).cpu().numpy()

dummy = np.zeros((len(preds), 3))
dummy[:, -1] = preds
y_pred_inv = scaler.inverse_transform(dummy)[:, -1]

dummy[:, -1] = y_test.cpu().numpy()
y_test_inv = scaler.inverse_transform(dummy)[:, -1]

rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100

print("\nLSTM METRICS (Original Scale)")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"MAPE: {mape:.2f}%")

# ----------------------------
# 10. ARIMA Baseline
# ----------------------------
arima = ARIMA(data["target"], order=(5, 1, 0))
arima_fit = arima.fit()
arima_pred = arima_fit.forecast(len(y_test_inv))

print("\nARIMA RMSE:",
      np.sqrt(mean_squared_error(y_test_inv, arima_pred)))

# ----------------------------
# 11. SHAP Explainability
# ----------------------------
explainer = shap.DeepExplainer(model, X_train[:50])
shap_values = explainer.shap_values(X_test[:10])

shap.summary_plot(shap_values, X_test[:10], show=False)
plt.show()

# ----------------------------
# 12. Sensitivity Analysis
# ----------------------------
shock = X_test.clone()
shock[:, :, 0] += 0.2

with torch.no_grad():
    shocked_pred = model(shock).cpu().numpy()

print("\nSensitivity analysis completed successfully.")
