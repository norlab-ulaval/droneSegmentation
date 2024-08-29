import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

metrics_df = pd.read_csv('lowAltitude_classification/results/phase2/center/val/phase2-val-center.csv')
votes_df = pd.read_csv('lowAltitude_classification/results/avg_voters/val/Votes_val.csv')

merged_df = pd.merge(metrics_df, votes_df, how='outer', on=['Central Size', 'Patch Size', 'Step Size', 'Pad Size'])
# merged_df['Average Number of Voters'] = merged_df['Average Number of Voters_x'].combine_first(merged_df['Average Number of Voters_y'])

# merged_df = merged_df.drop(columns=['Average Number of Voters_x', 'Average Number of Voters_y'])
# print(merged_df)

step_marker_map = {
    16: 'o',
    24: 's',
    27: 'D',
    48: 'x',
    96: '*',
}

plt.figure(figsize=(8, 6))
palette = sns.color_palette("Set2", len(merged_df['Pad Size'].unique()))

for step_size, marker in step_marker_map.items():
    subset = merged_df[merged_df['Step Size'] == step_size]
    plt.scatter(subset['Avg_Voters'], subset['F1'], marker=marker, s=100, label=f'Step Size: {step_size}')

for (pad_size, color) in zip(merged_df['Pad Size'].unique(), palette):
    subset = merged_df[merged_df['Pad Size'] == pad_size]
    sorted_group = subset.sort_values('Avg_Voters', ascending=False)
    plt.plot(sorted_group['Avg_Voters'], sorted_group['F1'], linestyle='--', color=color, label=f'Pad Size: {pad_size}')

# Annotate points
for i in range(len(merged_df)):
    plt.annotate(
        f"Central Size: {merged_df['Central Size'][i]}",
        (merged_df['Avg_Voters'][i], merged_df['F1'][i]),
        textcoords="offset points",
        xytext=(-20,15),
        ha='center',
        fontsize=8,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
    )


# x_ticks = range(int(merged_df['Avg_Voters'].min()), int(merged_df['Avg_Voters'].max()) + 2)
# plt.xticks(x_ticks)

plt.xlabel('Avg Number of Voters')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Avg Number of Voters')
plt.grid(True)
plt.legend()
plt.show()
