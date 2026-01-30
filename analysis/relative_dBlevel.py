import pandas as pd
import matplotlib.pyplot as plt

data_path = r"C:\Users\Isha Kulshrestha\Desktop\IshaAcoustics\lombard_effect\data\lombard_data.csv"
df = pd.read_csv(data_path)


# -----------------------------
# Compute baseline (at 40 dB) for each participant
# -----------------------------
baseline_dict = df[df["noise_db"] == 40].set_index("participant_id")["vocal_db"].to_dict()

# Add new column with baseline-corrected vocal level
df["vocal_db_baseline_corrected"] = df.apply(
    lambda row: row["vocal_db"] - baseline_dict[row["participant_id"]],
    axis=1
)

# -----------------------------
# Plotting baseline-corrected data
# -----------------------------
noise_levels = sorted(df["noise_db"].unique())

# Boxplot data (baseline-corrected)
boxplot_data = [
    df[df["noise_db"] == n]["vocal_db_baseline_corrected"].values
    for n in noise_levels
]

plt.figure(figsize=(8,6))

# Boxplots
plt.boxplot(
    boxplot_data,
    positions=noise_levels,
    widths=2,
    patch_artist=True,
    showfliers=True,
    boxprops=dict(alpha=0.3)
)

# Individual trajectories
for pid in df["participant_id"].unique():
    sub = df[df["participant_id"] == pid]
    plt.plot(
        sub["noise_db"],
        sub["vocal_db_baseline_corrected"],
        marker="o",
        linewidth=1.5,
        alpha=0.8,
        label=pid
    )

plt.xlabel("Background noise level (dB)")
plt.ylabel("Baseline-corrected vocal level (dB)")
plt.title("Lombard Effect: Vocal Output Relative to Baseline")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(title="Participant", fontsize=8)
plt.tight_layout()
plt.show()
