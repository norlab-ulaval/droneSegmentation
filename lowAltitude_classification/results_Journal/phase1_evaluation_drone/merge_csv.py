import os
import pandas as pd

folder_path = "./"
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
merged_df = pd.concat([pd.read_csv(os.path.join(folder_path, file)) for file in csv_files], ignore_index=True)
output_file = os.path.join(folder_path, "phase1_evaluation_drone_merged.csv")
merged_df.to_csv(output_file, index=False)
