import os
import numpy as np
from tqdm import tqdm
import time


def build_filename(xstart, ystart, row_idx, col_idx):
    """Helper function to build the filename for a subpatch, according to the naming convention: patch_xstart_ystart_rowX_colY.npz."""
    return f"patch_{xstart}_{ystart}_row{row_idx}_col{col_idx}.npz"


def main(original_wsi_size):
    """Check for missing subpatches and create them if necessary.

    Parameters
        original_wsi_size (tuple): The size of the original WSI image (width, height).
    """
    patch_folder = os.path.join('output', 'patches')
    existing_files = set(os.listdir(patch_folder))

    # Find all unique (xstart, ystart) tile origins
    # Assuming all but the last patches are 4000x4000
    expected_origins = set()
    for xstart in range(0, original_wsi_size[0], 4000):
        for ystart in range(0, original_wsi_size[1], 4000):
            expected_origins.add((xstart, ystart))
        
    # Sort the expected origins for consistent output
    expected_origins = sorted(expected_origins)
    print(f"Expected origins: {expected_origins}")

    # time.sleep(100)

    # Check missing subpatches for each unique origin
    for (xstart, ystart) in tqdm(expected_origins, desc="Checking missing subpatches"):
        for row_idx in range(16):
            for col_idx in range(16):
                expected_filename = build_filename(xstart, ystart, row_idx, col_idx)

                if expected_filename not in existing_files:
                    print(f"Missing subpatch: {expected_filename}")

                    empty_patch = np.zeros((250, 250), dtype=np.uint8)
                    save_path = os.path.join(patch_folder, expected_filename)
                    np.savez_compressed(save_path, patch=empty_patch)


if __name__ == "__main__":
    wsi_size = (127481, 88000)
    print("Checking for missing subpatches...")
    main(wsi_size)
    print("Done.")
