import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from gsd_utils import evaluate_segmentation

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
        help="Patch size",
        type=int,
        default=128,
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    gsddat_folder = gsddata_dir / args.mode / "val"

    patch_sizes = [args.psize]

    all_values = []

    for patch_size in patch_sizes:
        for overlap in overlaps:
            patch_overlap = f"p{patch_size:04}-o{overlap * 100:.0f}"
            gsd_po_dir = gsddat_folder / patch_overlap

            # For each GSD:
            for gsd_idx, scale in enumerate(SCALES):
                gsd_dir = gsd_po_dir / f"GSD{gsd_idx}"

                gsd_plab_dir = gsd_dir / "pseudolabels"
                gsd_annot_dir = gsd_dir / "annotations"

                ious, accs, f1s, all_predictions, all_targets = evaluate_segmentation(
                    gsd_plab_dir,
                    gsd_annot_dir,
                    [1],
                    num_classes=26,
                )

                correct = all_predictions == all_targets
                print(correct.sum(), all_predictions.shape, correct.shape)

                gsd_values = [
                    {
                        "GSD": f"GSD{gsd_idx}",
                        "scale": scale,
                        "iou": iou,
                        "acc": acc,
                        "f1": f1,
                    }
                    for iou, acc, f1 in zip(ious, accs, f1s)
                ]

                all_values.extend(gsd_values)

        df = pd.DataFrame(all_values)
        output_dir = results_dir / args.mode
        output_dir.mkdir(exist_ok=True, parents=True)
        df.to_csv(
            output_dir / f"gsd-metrics-p{args.psize}.csv",
            index=False,
        )

    print("[Evaluation] Processing complete.")


if __name__ == "__main__":
    main()
