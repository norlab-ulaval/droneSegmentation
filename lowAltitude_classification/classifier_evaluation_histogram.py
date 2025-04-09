import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data
path = "data/classifier_evaluation.csv"
data = pd.read_csv(path)

# Filter the data
split_overlap_data = {}
for split in ["Validation", "Test"]:
    data_filtered = data[data["Dataset"] == split]
    split_overlap_data[split] = {}
    for overlap in [0, 0.2]:
        data_overlap = data_filtered[data_filtered["Overlap"] == overlap]
        acc = data_overlap["Accuracy"]
        f1_macro = data_overlap["F1 Score - Macro"]
        f1_weighted = data_overlap["F1 Score - Weighted"]
        split_overlap_data[split][overlap] = {
            "Accuracy": acc,
            "F1 Score - Macro": f1_macro,
            "F1 Score - Weighted": f1_weighted,
        }

# Plot Validation histogram for accuracy
split = "Test"
for metric in ["Accuracy", "F1 Score - Macro", "F1 Score - Weighted"]:
    plt.title(f"Histogram for {metric} on {split}")
    plt.hist(
        split_overlap_data[split][0][metric],
        bins=10,
        alpha=0.5,
        color="blue",
        label="Overlap 0",
    )
    plt.hist(
        split_overlap_data[split][0.2][metric],
        bins=10,
        alpha=0.5,
        color="orange",
        label="Overlap 0.2",
    )
    # line for mean
    plt.axvline(
        np.mean(split_overlap_data[split][0][metric]),
        color="blue",
        linestyle="dashed",
        linewidth=1,
    )
    plt.axvline(
        np.mean(split_overlap_data[split][0.2][metric]),
        color="orange",
        linestyle="dashed",
        linewidth=1,
    )
    plt.legend()
    plt.show()
