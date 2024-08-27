import pandas as pd
import matplotlib.pyplot as plt

metrics_df = pd.read_csv('lowAltitude_classification/results/phase2/center/val/phase2-val-center.csv')
votes_df = pd.read_csv('lowAltitude_classification/results/avg_voters/val/Votes_val.csv')

merged_df = pd.merge(metrics_df, votes_df, how='outer', on=['Central Size', 'Patch Size', 'Step Size', 'Pad Size'])
# merged_df['Average Number of Voters'] = merged_df['Average Number of Voters_x'].combine_first(merged_df['Average Number of Voters_y'])

# merged_df = merged_df.drop(columns=['Average Number of Voters_x', 'Average Number of Voters_y'])
# print(merged_df)

plt.figure(figsize=(8, 6))
plt.scatter(merged_df['Avg_Voters'], merged_df['F1'], marker='o', linestyle='-', color='r')

for i in range(len(merged_df)):
    plt.annotate(
        f"Patch Size: {merged_df['Patch Size'][i]}\n"
        f"Step Size: {merged_df['Step Size'][i]}\n"
        f"Central Size: {merged_df['Central Size'][i]}\n"
        f"Pad Size: {merged_df['Pad Size'][i]}\n",
        (merged_df['Avg_Voters'][i], merged_df['F1'][i]),
        textcoords="offset points",
        xytext=(-20,15),
        ha='center',
        fontsize=8,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
    )


x_ticks = range(int(merged_df['Avg_Voters'].min()), int(merged_df['Avg_Voters'].max()) + 2)
plt.xticks(x_ticks)

plt.xlabel('Avg Number of Voters')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Avg Number of Voters')
plt.grid(True)
plt.show()
