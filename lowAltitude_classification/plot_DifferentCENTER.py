import pandas as pd
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from gsd_utils import papermode

papermode(plt=plt, size=8, has_latex=True)

# width as measured in inkscape
width = 3.281
height = width / 1.618

metrics_df = pd.read_csv('results/phase2/center/val/phase2-val-center.csv')
votes_df = pd.read_csv('results/avg_voters/val/Votes_val.csv')
merged_df = pd.merge(metrics_df, votes_df, how='outer', on=['Central Size', 'Patch Size', 'Step Size', 'Pad Size'])
filtered_df = merged_df[(merged_df['Pad Size'] == 184) & (merged_df['Central Size'] == 96)]
filtered_df.F1 *= 100


fig, ax = plt.subplots()
fig.subplots_adjust(top=0.98, bottom=0.17, left=0.17, right=0.95)

ax.plot(filtered_df.Avg_Voters, filtered_df.F1, marker='o', linestyle='-', color='forestgreen')
# ax.set_xscale('log')

ax.set_xlabel('Average Number of Voters ')
ax.set_ylabel('F1 score (\%)')
ax.grid(True)

ax.xaxis.set_label_coords(0.5, -0.14)

print(plt.rcParams["text.usetex"], plt.rcParams["font.serif"])

fig.set_size_inches(width, height)
# ax.set_title('F1 Score vs. Avg Number of Voters')
fig.savefig('figs/phase2/phase2-val-center.pdf')
fig.savefig('figs/phase2/phase2-val-center.png')
# plt.show()
