import pandas as pd
import jiwer
import os

GROUND_TRUTH = "ground_truth.csv"
RESULTS_FILE = "results/raw_transcripts.csv"

gt_df      = pd.read_csv(GROUND_TRUTH)
result_df  = pd.read_csv(RESULTS_FILE)

# merge
merged = result_df.merge(gt_df[["filename","locality","ground_truth","condition"]], on="filename")

# ── WER ───────────────────────────────────────────────────────────────────
def compute_wer(ref, hyp):
    try:
        return jiwer.wer(ref, hyp) if hyp.strip() else 1.0
    except:
        return 1.0

merged["wer"] = merged.apply(
    lambda r: compute_wer(r["ground_truth"], r["transcript"]), axis=1
)

# ── Entity Accuracy (did the model get the locality name?) ────────────────
def locality_hit(row):
    locality_lower = row["locality"].lower().replace(" ", "")
    transcript     = str(row["transcript"]).lower().replace(" ", "")
    # check if any word in locality appears in transcript
    parts = row["locality"].lower().split()
    return int(any(p in transcript for p in parts if len(p) > 3))

merged["locality_hit"] = merged.apply(locality_hit, axis=1)

# ── Summary by model ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("📊 RESULTS SUMMARY")
print("="*60)

summary = merged.groupby("model").agg(
    avg_wer        = ("wer",          "mean"),
    locality_acc   = ("locality_hit", "mean"),
    avg_latency    = ("latency",      "mean")
).round(3)

print(summary.to_string())

# ── Summary by condition ──────────────────────────────────────────────────
print("\n" + "="*60)
print("📊 RESULTS BY CONDITION")
print("="*60)

by_condition = merged.groupby(["model","condition"]).agg(
    avg_wer      = ("wer",          "mean"),
    locality_acc = ("locality_hit", "mean")
).round(3)

print(by_condition.to_string())

# ── Failure cases ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("❌ LOCALITY FAILURES (where model missed the locality name)")
print("="*60)

failures = merged[merged["locality_hit"] == 0][["model","filename","locality","transcript"]]
print(failures.to_string())

# ── Save full results ─────────────────────────────────────────────────────
merged.to_csv("results/full_results.csv", index=False)
print("\n✅ Full results saved to results/full_results.csv")