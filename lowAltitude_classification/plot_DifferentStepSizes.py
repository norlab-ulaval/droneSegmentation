import pandas as pd
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from gsd_utils import papermode

papermode(plt=plt, size=7, has_latex=True)

width = 4.281
height = width / 1.5

metrics_test_df = pd.read_csv('lowAltitude_classification/results/phase2/center/test/phase2-test-center.csv')
votes_test_df = pd.read_csv('lowAltitude_classification/results/avg_voters/test/Votes_test.csv')
metrics_val_df = pd.read_csv('lowAltitude_classification/results/phase2/center/val/phase2-val-center.csv')
votes_val_df = pd.read_csv('lowAltitude_classification/results/avg_voters/val/Votes_val.csv')

merged_test_df = pd.merge(metrics_test_df, votes_test_df, how='outer', on=['Central Size', 'Patch Size', 'Step Size', 'Pad Size'])
merged_val_df = pd.merge(metrics_val_df, votes_val_df, how='outer', on=['Central Size', 'Patch Size', 'Step Size', 'Pad Size'])
filtered_df = merged_val_df[(merged_val_df['Pad Size'] == 184) & (merged_val_df['Central Size'] == 96)]
merged_test_df = merged_test_df[(merged_test_df['Pad Size'] == 184) & (merged_test_df['Central Size'] == 96)]

merged_test_df.F1 *= 100
filtered_df.F1 *= 100

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(filtered_df.Avg_Voters, filtered_df.F1, marker='o', linestyle='-', color='forestgreen')
ax1.set_xlabel('Average Number of Voters')
ax1.set_ylabel('F1 score (\%)')
ax1.grid(True)
ax1.xaxis.set_label_coords(0.5, -0.12)
ax1.yaxis.set_label_coords(-0.20, 0.5)
ax1.set_title(r'$D_{val}^{drone}$', color='blue')

ax2.plot(merged_test_df.Avg_Voters, merged_test_df.F1, marker='o', linestyle='-', color='forestgreen')
ax2.set_xlabel('Average Number of Voters')
ax2.grid(True)
ax2.xaxis.set_label_coords(0.5, -0.12)
ax2.set_title(r'$D_{test}^{drone}$', color='blue')

fig.subplots_adjust(top=0.90, bottom=0.15, left=0.1, right=0.95, wspace=0.25)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.serif'] = ['CMU']

fig.set_size_inches(width, height)
fig.savefig('lowAltitude_classification/figs/phase2/phase2-val-test-DifferentStepSize.pdf')
fig.savefig('lowAltitude_classification/figs/phase2/phase2-val-test-DifferentStepSize.png')

