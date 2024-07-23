import numpy as np
from pathlib import Path

from gsd_utils import evaluate_segmentation, IDENTICAL_MAPPING

# Paths
data_path = Path("data") / "drone-seg"
image_folder = data_path / "test-data"
annot_folder = data_path / "test-data-annotation"
gsddat_folder = data_path / "gsds"

patch_sizes = [256]
overlaps = [0.85]

# GSD metrics
GSD_FACTOR = 2
N_GSD = 4
# GSD_FACTOR=8 and N_GSD = 4
# => SCALES = [1, 1/8, 1/64, 1/512]
SCALES = np.logspace(0, -(N_GSD - 1), num=N_GSD, base=GSD_FACTOR)


def main():
    for patch_size in patch_sizes:
        for overlap in overlaps:
            patch_overlap = f"p{patch_size:04}-o{overlap * 100:.0f}"
            gsd_po_dir = gsddat_folder / patch_overlap

            # For each GSD:
            for gsd_idx, scale in enumerate(SCALES):
                gsd_dir = gsd_po_dir / f"GSD{gsd_idx}"

                gsd_plab_dir = gsd_dir / "pseudolabels"
                gsd_annot_dir = gsd_dir / "annotations"

                avg_iou, avg_accuracy, avg_f1_score, all_predictions, all_targets = (
                    evaluate_segmentation(
                        gsd_plab_dir,
                        gsd_annot_dir,
                        IDENTICAL_MAPPING,
                        {},
                    )
                )
                print(f"Average IoU: {avg_iou:.4f}")
                print(f"Average Pixel Accuracy: {avg_accuracy:.4f}")
                print(f"Average F1 Score: {avg_f1_score:.4f}")

    print("[Evaluation] Processing complete.")


if __name__ == "__main__":
    main()
