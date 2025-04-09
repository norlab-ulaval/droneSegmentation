import pandas as pd
from pathlib import Path

cm_dir = Path("lowAltitude_classification/confusion_matrix_Precision_Recall")

csv_files = [cm_dir / "pl/metrics_cm.csv",
             cm_dir / "pt/metrics_cm.csv",
             cm_dir / "ft/metrics_cm.csv"]



with open(cm_dir / "PrecisionRecall.dat", "w") as dat_file:
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        output_lines = []
        for i, row in df.iterrows():
            precision_value = df.iloc[i, 0]
            recall_value = df.iloc[i, 1]
            f1_value = df.iloc[i, 2]

            output_lines.append(f"{csv_file.split('/')[-2]}_precision = {precision_value * 100:.2f}")
            output_lines.append(f"{csv_file.split('/')[-2]}_recall = {recall_value * 100:.2f}")
            output_lines.append(f"{csv_file.split('/')[-2]}_f1 = {f1_value * 100:.2f}")

        dat_file.write("\n".join(output_lines) + "\n")