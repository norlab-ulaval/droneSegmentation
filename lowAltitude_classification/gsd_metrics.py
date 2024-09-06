import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from gsd_utils import compute_metrics

# Paths
gsddata_dir = Path("data") / "gsds"
results_dir = Path("lowAltitude_classification") / "results" / "gsd"

overlaps = [0.85]

# GSD metrics
GSD_FACTOR = 1.5
N_GSD = 4
# GSD_FACTOR=8 and N_GSD = 4
# => SCALES = [1, 1/8, 1/64, 1/512]
SCALES = np.logspace(0, -(N_GSD - 1), num=N_GSD, base=GSD_FACTOR)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="Resizing mode",
        default="resize",
        choices=["resize", "gaussian"],
    )
    parser.add_argument(
        "--psize",
        help="Base window size",
        type=int,
        default=184,
    )
    parser.add_argument(
        "--subset",
        help="Dataset subset",
        choices=["val", "test"],
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    gsddat_folder = gsddata_dir / args.mode / args.subset

    win_sizes = (args.psize * SCALES).astype(int).tolist()

    all_values = []

    for overlap in overlaps:
        patch_overlap = f"p{args.psize:04}-o{overlap * 100:.0f}"
        exp_dir = gsddat_folder / patch_overlap

        # For each window size
        for win_idx, win_size in enumerate(win_sizes):
            win_dir = exp_dir / f"WIN{win_idx}"

            # For each GSD:
            for gsd_idx, scale in enumerate(SCALES):
                gsd_dir = win_dir / f"GSD{gsd_idx}"

                gsd_plab_dir = gsd_dir / "pseudolabels"
                gsd_annot_dir = gsd_dir / "annotations"

                f1_score, pixel_acc = compute_metrics(gsd_plab_dir, gsd_annot_dir)

                gsd_values = {
                    "WIN": f"WIN{win_idx}",
                    "GSD": f"GSD{gsd_idx}",
                    "scale": scale,
                    "winsize": win_size,
                    "acc": pixel_acc,
                    "f1": f1_score,
                }

                all_values.extend(gsd_values)

        df = pd.DataFrame(all_values)
        output_dir = results_dir / args.subset / args.mode
        output_dir.mkdir(exist_ok=True, parents=True)
        df.to_csv(
            output_dir / f"multigsd-{args.subset}-p{args.psize}.csv",
            index=False,
        )

    print("[Evaluation] Processing complete.")


if __name__ == "__main__":
    main()
