import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

os.makedirs("report", exist_ok=True)

results  = pd.read_csv("results/full_results.csv")
models   = ["deepgram", "whisper", "wav2vec2"]
colors   = {"deepgram": "#2563eb", "whisper": "#16a34a", "wav2vec2": "#dc2626"}

# ── Chart 1: WER by Model ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
summary = results.groupby("model")["wer"].mean()
bars    = ax.bar(summary.index, summary.values,
                 color=[colors[m] for m in summary.index], width=0.5)
ax.set_title("Average Word Error Rate by Model\n(Lower is Better)", fontsize=14, fontweight="bold")
ax.set_ylabel("Word Error Rate (WER)")
ax.set_ylim(0, 1.3)
for bar, val in zip(bars, summary.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")
ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="WER = 1.0 (complete failure)")
ax.legend()
plt.tight_layout()
plt.savefig("report/chart1_wer_by_model.png", dpi=150)
plt.close()
print("✅ Chart 1 saved")

# ── Chart 2: Locality Accuracy by Model ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
loc_acc = results.groupby("model")["locality_hit"].mean() * 100
bars    = ax.bar(loc_acc.index, loc_acc.values,
                 color=[colors[m] for m in loc_acc.index], width=0.5)
ax.set_title("Locality Name Detection Accuracy by Model\n(Higher is Better)", fontsize=14, fontweight="bold")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
for bar, val in zip(bars, loc_acc.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("report/chart2_locality_accuracy.png", dpi=150)
plt.close()
print("✅ Chart 2 saved")

# ── Chart 3: WER by Condition ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
conditions = ["quiet", "noise", "rushed", "whisper"]
x          = range(len(conditions))
width      = 0.25

for i, model in enumerate(models):
    vals = []
    for cond in conditions:
        subset = results[(results["model"] == model) & (results["condition"] == cond)]
        vals.append(subset["wer"].mean() if len(subset) > 0 else 0)
    offset = (i - 1) * width
    bars   = ax.bar([xi + offset for xi in x], vals, width,
                    label=model.capitalize(), color=colors[model])

ax.set_title("Word Error Rate by Audio Condition\n(Lower is Better)", fontsize=14, fontweight="bold")
ax.set_ylabel("Word Error Rate (WER)")
ax.set_xticks(x)
ax.set_xticklabels(["Quiet Room", "Background Noise", "Rushed Speech", "Whispered"])
ax.legend()
ax.set_ylim(0, 1.4)
plt.tight_layout()
plt.savefig("report/chart3_wer_by_condition.png", dpi=150)
plt.close()
print("✅ Chart 3 saved")

# ── Chart 4: Latency Comparison ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
latency = results.groupby("model")["latency"].mean().dropna()
bars    = ax.bar(latency.index, latency.values,
                 color=[colors[m] for m in latency.index], width=0.5)
ax.set_title("Average Latency per File by Model\n(Lower is Better)", fontsize=14, fontweight="bold")
ax.set_ylabel("Latency (seconds)")
for bar, val in zip(bars, latency.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.2f}s", ha="center", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("report/chart4_latency.png", dpi=150)
plt.close()
print("✅ Chart 4 saved")

print("\n🎉 All charts saved to report/ folder!")