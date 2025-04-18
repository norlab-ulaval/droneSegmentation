import pandas as pd
import matplotlib as mpl

mpl.use("pdf")
import matplotlib.pyplot as plt
from gsd_utils import papermode

papermode(plt=plt, size=7, has_latex=True)

width = 4.281
height = width / 1.5

df_patchsize = pd.read_csv(
    "lowAltitude_classification/results/phase2/patchsize/metrics-patch-overlap.csv"
)
df_val = df_patchsize[(df_patchsize["type"] == "val")]
df_test = df_patchsize[(df_patchsize["type"] == "test")]

df_val_filtered = df_val[(df_val["overlap"] == 0.85)]
df_test_filtered = df_test[(df_test["overlap"] == 0.85)]

df_val_filtered.F1 *= 100
df_test_filtered.F1 *= 100

fig, (ax1, ax2) = plt.subplots(1, 2)

patch_sizes = df_test_filtered["patch_size"].unique()
patch_size_labels = [str(size) for size in patch_sizes]

ax1.bar(
    patch_size_labels,
    df_val_filtered.F1,
    color="slateblue",
    width=0.3,
    edgecolor="black",
)
ax1.set_xlabel("Patch Size")
ax1.set_ylabel("F1 score (\%)")
ax1.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
ax1.xaxis.set_label_coords(0.5, -0.12)
ax1.yaxis.set_label_coords(-0.17, 0.5)
ax1.set_title(r"$D_{val}^{drone}$", color="blue")


ax2.bar(
    patch_size_labels,
    df_test_filtered.F1,
    color="slateblue",
    width=0.3,
    edgecolor="black",
)
ax2.set_xlabel("Patch Size")
ax2.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
ax2.xaxis.set_label_coords(0.5, -0.12)
ax2.set_title(r"$D_{test}^{drone}$", color="blue")

fig.subplots_adjust(top=0.90, bottom=0.15, left=0.1, right=0.95, wspace=0.25)

plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True
plt.rcParams["font.serif"] = ["CMU"]

fig.set_size_inches(width, height)

fig.savefig(
    "lowAltitude_classification/figs/phase2/phase2-val-test-DifferentPatchSize.pdf"
)
fig.savefig(
    "lowAltitude_classification/figs/phase2/phase2-val-test-DifferentPatchSize.png"
)
