import pandas as pd
import matplotlib as mpl

mpl.use("pdf")
import matplotlib.pyplot as plt
from gsd_utils import papermode

papermode(plt=plt, size=7, has_latex=True)

width = 4.281
height = width / 1.5

metrics_test_df = pd.read_csv(
    "lowAltitude_classification/results/phase2/center/test/phase2-test-center.csv"
)
votes_test_df = pd.read_csv(
    "lowAltitude_classification/results/avg_voters/test/Votes_test.csv"
)
metrics_val_df = pd.read_csv(
    "lowAltitude_classification/results/phase2/center/val/phase2-val-center.csv"
)
votes_val_df = pd.read_csv(
    "lowAltitude_classification/results/avg_voters/val/Votes_val.csv"
)

merged_test_df = pd.merge(
    metrics_test_df,
    votes_test_df,
    how="outer",
    on=["Central Size", "Patch Size", "Step Size", "Pad Size"],
)
merged_val_df = pd.merge(
    metrics_val_df,
    votes_val_df,
    how="outer",
    on=["Central Size", "Patch Size", "Step Size", "Pad Size"],
)

filtered_val = merged_val_df[
    (merged_val_df["Pad Size"] == 184) & (merged_val_df["Step Size"] == 27)
]
filtered_test = merged_test_df[
    (merged_test_df["Pad Size"] == 184) & (merged_test_df["Step Size"] == 27)
]

filtered_test.F1 *= 100
filtered_val.F1 *= 100

fig, (ax1, ax2) = plt.subplots(1, 2)

central_sizes = filtered_val["Central Size"].unique()
central_sizes_labels = [str(size) for size in central_sizes]

bars1 = ax1.bar(
    central_sizes_labels, filtered_val.F1, color="teal", width=0.4, edgecolor="black"
)
ax1.set_xlabel("Central Window Size")
ax1.set_ylabel("F1 score (\%)")
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax1.xaxis.set_label_coords(0.5, -0.12)
ax1.yaxis.set_label_coords(-0.20, 0.5)
ax1.set_title(r"$D_{val}^{drone}$", color="b")
# ax1.set_xticks(filtered_val['Central Size'])
ax1.set_ylim([filtered_val.F1.min() - 1, filtered_val.F1.max() + 1])
ax1.yaxis.set_major_locator(plt.MultipleLocator(0.5))

for bar in bars1:
    yval = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.07,
        f"${round(yval, 2)}$",
        fontsize="x-small",
        color="darkgreen",
        ha="center",
        va="bottom",
    )

bars2 = ax2.bar(
    central_sizes_labels, filtered_test.F1, color="teal", width=0.4, edgecolor="black"
)
ax2.set_xlabel("Central Window Size")
ax2.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
ax2.xaxis.set_label_coords(0.5, -0.12)
ax2.set_title(r"$D_{test}^{drone}$", color="b")
# ax2.set_xticks(filtered_test['Central Size'])
ax2.set_ylim([filtered_test.F1.min() - 1, filtered_test.F1.max() + 1])
ax2.yaxis.set_major_locator(plt.MultipleLocator(0.5))

for bar in bars2:
    yval = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.07,
        f"${round(yval, 2)}$",
        fontsize="x-small",
        color="darkgreen",
        ha="center",
        va="bottom",
    )

fig.subplots_adjust(top=0.90, bottom=0.15, left=0.1, right=0.95, wspace=0.25)
fig.set_size_inches(width, height)

plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True
plt.rcParams["font.serif"] = ["CMU"]

fig.savefig("lowAltitude_classification/figs/phase2/phase2-val-test-EntireVsCenter.pdf")
fig.savefig("lowAltitude_classification/figs/phase2/phase2-val-test-EntireVsCenter.png")
