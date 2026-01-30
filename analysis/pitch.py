import pandas as pd
import matplotlib.pyplot as plt

data_path = r"C:\Users\Isha Kulshrestha\Desktop\IshaAcoustics\lombard_effect\data\pitch.csv"
df = pd.read_csv(data_path)

# -----------------------------
# Prepare boxplot data
# -----------------------------
noise_levels = sorted(df["noise_db"].unique())

boxplot_data = [
    df[df["noise_db"] == n]["pitch_hz"].values
    for n in noise_levels
]

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8, 6))

# Boxplots (population-level)
plt.boxplot(
    boxplot_data,
    positions=noise_levels,
    widths=2,
    patch_artist=True,
    showfliers=True,
    boxprops=dict(alpha=0.3)
)

# Individual participant trajectories
for pid in df["participant_id"].unique():
    sub = df[df["participant_id"] == pid]
    plt.plot(
        sub["noise_db"],
        sub["pitch_hz"],
        marker="o",
        linewidth=1.5,
        alpha=0.8,
        label=pid
    )
    # -----------------------------
# Labels & styling
# -----------------------------
plt.xlabel("Background noise level (dB)")
plt.ylabel("Frequency level (Hz)")
plt.title("Lombard Effect: Vocal Output Increases with Background Noise")

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(title="Participant", fontsize=8)

plt.tight_layout()
plt.show()
