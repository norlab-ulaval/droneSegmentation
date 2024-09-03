from pathlib import Path
import pandas as pd


def p(n: float, factor: int = 100) -> float:
    return factor * n


def process_csv(csv_path: Path):
    df = pd.read_csv(csv_path)

    df["F1diff"] = df['F1 Score - Macro'].diff()
    df.loc[df.index >= 2, "F1diff"] = df['F1 Score - Macro'] - df['F1 Score - Macro'].iloc[1]

    outname = f"{csv_path.stem.rstrip('-avg')}.dat"

    c = 0
    with (csv_path.parent / outname).open(mode="w") as f:
        print(f"f1-base = {p(df.loc[c, 'F1 Score - Macro']):.2f}", file=f)
        c += 1
        print(f"f1-filt = {p(df.loc[c, 'F1 Score - Macro']):.2f}", file=f)
        print(f"f1d-filt = {p(df.loc[c, 'F1diff']):.2f}", file=f)
        c += 1
        for i in range(6):
            print(f"f1-aug{i} = {p(df.loc[c, 'F1 Score - Macro']):.2f}", file=f)
            print(f"f1d-aug{i} = {p(df.loc[c, 'F1diff']):.2f}", file=f)
            c += 1
        for i in range(2):
            print(f"f1-bal{i} = {p(df.loc[c, 'F1 Score - Macro']):.2f}", file=f)
            print(f"f1d-bal{i} = {p(df.loc[c, 'F1diff']):.2f}", file=f)
            c += 1
        print(f"f1-bg = {p(df.loc[c, 'F1 Score - Macro']):.2f}", file=f)
        print(f"f1d-bg = {p(df.loc[c, 'F1diff']):.2f}", file=f)
        c += 1
        print(f"f1-final = {p(df.loc[c, 'F1 Score - Macro']):.2f}", file=f)
        print(f"f1d-final = {p(df.loc[c, 'F1diff']):.2f}", file=f)
        c += 1





if __name__ == "__main__":
    test_res = Path('lowAltitude_classification/results/New_phase_1/TEST_metrics.csv')
    val_res = Path('lowAltitude_classification/results/New_phase_1/VAL_metrics.csv')

    process_csv(test_res)
    process_csv(val_res)
