import pandas as pd

df = pd.read_csv(
    "lowAltitude_classification/Phase_1_results_test/Phase_1_results_test.csv"
)
avg_df = pd.DataFrame(columns=df.columns)

for i in range(0, len(df), 5):
    chunk = df.iloc[i : i + 5]
    avg_chunk = chunk.iloc[:, 1:].mean(axis=0).round(4)
    new_row = pd.Series(
        [chunk.iloc[0, 0].split("_")[0][:-1]] + avg_chunk.tolist(), index=df.columns
    )
    avg_df = pd.concat([avg_df, pd.DataFrame([new_row])], ignore_index=True)

print(avg_df)

avg_df.to_csv(
    "lowAltitude_classification/Phase_1_results_test/Phase_1_avg_results_test.csv",
    index=False,
)
