from pathlib import Path
import pandas as pd


def p(n: float, factor: int = 100) -> float:
    return factor * n


def process_csv(csv_path: Path):
    df = pd.read_csv(csv_path)

    df.index = df.Experiment.str.lstrip("experiment ").apply(
        pd.to_numeric,
        errors="coerce",
    )
    df["F1diff"] = df.F1.diff()
    df.loc[20:, "F1diff"] = df.loc[20:].F1 - df.F1[1]

    outname = f"{csv_path.stem}.dat"

    with (outname).open(mode="w") as f:
        print(f"f1-base = {p(df.F1[0]):.2f}", file=f)
        print(f"f1-filt = {p(df.F1[1]):.2f}", file=f)
        for i in range(6):
            print(f"f1-aug{i} = {p(df.F1[20+i]):.2f}", file=f)
        for i in range(2):
            print(f"f1-bal{i} = {p(df.F1[30+i]):.2f}", file=f)
        print(f"f1-bg = {p(df.F1[4]):.2f}", file=f)
        print(f"f1-final = {p(df.F1[5]):.2f}", file=f)

        print(f"f1d-filt = {p(df.F1diff[1]):.2f}", file=f)
        for i in range(6):
            print(f"f1d-aug{i} = {p(df.F1diff[20+i]):.2f}", file=f)
        for i in range(2):
            print(f"f1d-bal{i} = {p(df.F1diff[30+i]):.2f}", file=f)
        print(f"f1d-bg = {p(df.F1diff[4]):.2f}", file=f)
        print(f"f1d-final = {p(df.F1diff[5]):.2f}", file=f)


if __name__ == "__main__":
    test_df = process_csv('lowAltitude_classification/Cls_Different_Techniques/CLS_avg_csv_stages_evaluation/TEST_CLS_evaluation_drone_AVG.csv')
    val_df = process_csv('lowAltitude_classification/Cls_Different_Techniques/CLS_avg_csv_stages_evaluation/VAL_CLS_evaluation_drone_AVG.csv')
