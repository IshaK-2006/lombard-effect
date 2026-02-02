# lombard_ml.py
# Multivariate regression model for Lombard effect using PyTorch
# Predicts baseline-normalized vocal response as a function of noise level and speaker identity

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load data
# -----------------------------
data_path = r"C:\Users\Isha Kulshrestha\Desktop\IshaAcoustics\lombard_effect\data\lombard_data.csv"
df = pd.read_csv(data_path)

# -----------------------------
# 2. Compute participant baseline
#    (mean vocal level at lowest noise condition)
# -----------------------------
baseline_noise = df["noise_db"].min()

baselines = (
    df[df["noise_db"] == baseline_noise]
    .groupby("participant_id")["vocal_db"]
    .mean()
)

# Map baseline back to dataframe
df["baseline_vocal_db"] = df["participant_id"].map(baselines)

# -----------------------------
# 3. Define target variable
# -----------------------------
df["delta_vocal_db"] = df["vocal_db"] - df["baseline_vocal_db"]

# -----------------------------
# 4. Encode participant ID (one-hot)
# -----------------------------
participants = df["participant_id"].unique()
participant_to_idx = {p: i for i, p in enumerate(participants)}

participant_idx = df["participant_id"].map(participant_to_idx)
participant_onehot = np.eye(len(participants))[participant_idx]

# -----------------------------
# 5. Build input matrix X and target y
# -----------------------------
noise = df["noise_db"].values.reshape(-1, 1)
X = np.hstack([noise, participant_onehot])
y = df["delta_vocal_db"].values.reshape(-1, 1)

# Convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# -----------------------------
# 6. Define regression model
# -----------------------------
class LombardRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

model = LombardRegressor(input_dim=X.shape[1])

# -----------------------------
# 7. Training setup
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 500
loss_history = []

# -----------------------------
# 8. Training loop
# -----------------------------
for epoch in range(epochs):
    optimizer.zero_grad()

    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# -----------------------------
# 9. Evaluation & visualization
# -----------------------------
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor).numpy()

plt.figure()
plt.scatter(y, y_pred)
plt.xlabel("Observed ΔVocal dB")
plt.ylabel("Predicted ΔVocal dB")
plt.title("PyTorch Regression: Lombard Effect")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()
