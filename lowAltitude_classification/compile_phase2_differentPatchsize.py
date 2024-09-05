from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


# lac_dir = Path("lowAltitude_classification")
# results_dir = lac_dir / "results" / "phase1"
# metrics_dir = lac_dir / "metrics" / "phase1"
# metrics_dir.mkdir(parents=True, exist_ok=True)


def p(n: float, factor: int = 100) -> float:
    return factor * n


def process_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    file_result = Path(f"lowAltitude_classification/metrics/phase2_patches/{csv_path.stem}.dat")
    with file_result.open(mode="w") as f:
        for idx, row in df.iterrows():
            print(f"p{str(int(row['Patch Size']))} = {p(row['F1'])}", file=f)

    # df.index = df.Experiment.str.lstrip("experiment ").apply(
    #     pd.to_numeric,
    #     errors="coerce",
    # )
    # df["F1diff"] = df.F1.diff()
    # df.loc[20:, "F1diff"] = df.loc[20:].F1 - df.F1[1]
    #
    # outname = f"{csv_path.stem.rstrip("-avg")}.dat"
    #
    # with (metrics_dir / outname).open(mode="w") as f:
    #     print(f"f1-base = {p(df.F1[0]):.2f}", file=f)
    #     print(f"f1-filt = {p(df.F1[1]):.2f}", file=f)
    #     for i in range(6):
    #         print(f"f1-aug{i} = {p(df.F1[20+i]):.2f}", file=f)
    #     for i in range(2):
    #         print(f"f1-bal{i} = {p(df.F1[30+i]):.2f}", file=f)
    #     print(f"f1-bg = {p(df.F1[4]):.2f}", file=f)
    #     print(f"f1-final = {p(df.F1[5]):.2f}", file=f)
    #
    #     print(f"f1d-filt = {p(df.F1diff[1]):.2f}", file=f)
    #     for i in range(6):
    #         print(f"f1d-aug{i} = {p(df.F1diff[20+i]):.2f}", file=f)
    #     for i in range(2):
    #         print(f"f1d-bal{i} = {p(df.F1diff[30+i]):.2f}", file=f)
    #     print(f"f1d-bg = {p(df.F1diff[4]):.2f}", file=f)
    #     print(f"f1d-final = {p(df.F1diff[5]):.2f}", file=f)


if __name__ == "__main__":
    test_res = Path("lowAltitude_classification/results/New_phase_2/patch/test/phase2-test-patch_METRICS.csv")
    val_res = Path("lowAltitude_classification/results/New_phase_2/patch/val/phase2-val-patch_METRICS.csv")

    test_df = process_csv(test_res)
    val_df = process_csv(val_res)
