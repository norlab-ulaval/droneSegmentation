from pathlib import Path
import pandas as pd
import os

def p(n: float, factor: int = 100) -> float:
    return factor * n


def process_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    file_name = os.path.basename(csv_path)
    val_test = file_name.split('_')[0]

    df.loc[1, "F1diff"] = df.loc[1, "F1"] - df.loc[0, "F1"]
    df.loc[8, "F1diff"] = df.loc[8, "F1"] - df.loc[0, "F1"]
    for i in range(2, 7):
        df.loc[i, "F1diff"] = df.loc[i, "F1"] - df.loc[0, "F1"]
    df.loc[7, "F1diff"] = df.loc[7, "F1"] - df.loc[0, "F1"]
    df.loc[9, "F1diff"] = df.loc[9, "F1"] - df.loc[0, "F1"]

    outname = f"lowAltitude_classification/Cls_Different_Techniques/Classifier_techniques_results{val_test}.dat"

    counter = 0
    counter_dif = 1
    with open(outname, mode="w") as f:
        print(f"f1-base = {p(df.F1[counter]):.2f}", file=f)
        counter += 1
        print(f"f1-filt = {p(df.F1[counter]):.2f}", file=f)
        counter += 1
        for i in range(5):
            print(f"f1-aug{i} = {p(df.F1[counter]):.2f}", file=f)
            counter += 1

        print(f"f1-bal = {p(df.F1[counter]):.2f}", file=f)
        counter += 1
        print(f"f1-bg = {p(df.F1[counter]):.2f}", file=f)
        counter += 1
        print(f"f1-final = {p(df.F1[counter]):.2f}", file=f)

        print(f"f1d-filt = {p(df.F1diff[counter_dif]):.2f}", file=f)
        counter_dif += 1
        for i in range(5):
            print(f"f1d-aug{i} = {p(df.F1diff[counter_dif]):.2f}", file=f)
            counter_dif += 1
        print(f"f1d-bal = {p(df.F1diff[counter_dif]):.2f}", file=f)
        counter_dif += 1
        print(f"f1d-bg = {p(df.F1diff[counter_dif]):.2f}", file=f)
        counter_dif += 1
        print(f"f1d-final = {p(df.F1diff[counter_dif]):.2f}", file=f)


if __name__ == "__main__":
    test_df = process_csv('lowAltitude_classification/Cls_Different_Techniques/CLS_avg_csv_stages_evaluation/TEST_CLS_evaluation_drone_AVG.csv')
    val_df = process_csv('lowAltitude_classification/Cls_Different_Techniques/CLS_avg_csv_stages_evaluation/VAL_CLS_evaluation_drone_AVG.csv')
