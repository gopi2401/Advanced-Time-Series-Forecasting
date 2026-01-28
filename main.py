# =========================================
# Advanced Time Series Forecasting
# LSTM with Custom Self-Attention
# Rolling Origin Cross-Validation
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ----------------------------
# 1. Configuration
# ----------------------------
SEED = 42
WINDOW = 20
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------
# 2. Synthetic Time Series Data
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

X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).to(DEVICE)

# ----------------------------
# 5. Self-Attention Layer
# ----------------------------
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs):
        scores = self.attn(lstm_outputs)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_outputs, dim=1)
        return context, weights

# ----------------------------
# 6. LSTM + Attention Model
# ----------------------------
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = SelfAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        output = self.fc(context).squeeze()
        return output, attn_weights

# ----------------------------
# 7. Rolling Origin Cross-Validation
# ----------------------------
def rolling_origin_cv(X, y, splits=5):
    fold_size = len(X) // splits
    rmse_scores = []

    for i in range(1, splits):
        X_train = X[:i * fold_size]
        y_train = y[:i * fold_size]
        X_test = X[i * fold_size:(i + 1) * fold_size]
        y_test = y[i * fold_size:(i + 1) * fold_size]

        model = LSTMAttentionModel(input_size=3, hidden_size=32).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for _ in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            preds, _ = model(X_train)
            loss = criterion(preds, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds, _ = model(X_test)
            rmse = torch.sqrt(criterion(preds, y_test))
            rmse_scores.append(rmse.item())

    return rmse_scores, model

rmse_scores, trained_model = rolling_origin_cv(X, y)

print("\nRolling Origin RMSE per fold:")
for i, score in enumerate(rmse_scores, 1):
    print(f"Fold {i}: {score:.4f}")

print(f"\nAverage RMSE: {np.mean(rmse_scores):.4f}")

# ----------------------------
# 8. Attention Weight Visualization
# ----------------------------
trained_model.eval()
with torch.no_grad():
    _, attention_weights = trained_model(X[-1:].to(DEVICE))

attention_weights = attention_weights.squeeze().cpu().numpy()

plt.figure(figsize=(10, 4))
plt.plot(attention_weights, marker='o')
plt.title("Learned Temporal Attention Weights")
plt.xlabel("Time Step")
plt.ylabel("Attention Importance")
plt.grid(True)
plt.show()

# ----------------------------
# 9. Final Prediction Plot
# ----------------------------
with torch.no_grad():
    predictions, _ = trained_model(X)

predictions = predictions.cpu().numpy()

dummy = np.zeros((len(predictions), 3))
dummy[:, -1] = predictions
pred_inv = scaler.inverse_transform(dummy)[:, -1]

dummy[:, -1] = y.cpu().numpy()
y_inv = scaler.inverse_transform(dummy)[:, -1]

plt.figure(figsize=(10, 4))
plt.plot(y_inv, label="Actual")
plt.plot(pred_inv, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Time Series")
plt.show()