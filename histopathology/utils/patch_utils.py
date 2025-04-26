import numpy as np
import os
from math import ceil


def create_empty_patch(x_start, ystart, width, height, patch_size=250):
    """
    Some tiles are empty, so we skip the computation of the mask and create empty patches instead.

    Parameters
        - x_start: int: The x coordinate of the start of the tile
        - ystart: int: The y coordinate of the start of the tile
        - width: int: The width of the tile
        - height: int: The height of the tile
        - patch_size: int: The size of the patches to create
    """
    # Create empty 
    empty_patch = np.zeros((patch_size, patch_size), dtype=np.uint8)

    # Compute the number of patches in each dimension
    rows = ceil(height / patch_size)
    cols = ceil(width / patch_size)

    for row_idx in range(rows):
        for col_idx in range(cols):
            # Create patch filename
            patch_filename = f"patch_{x_start}_{ystart}_row{row_idx}_col{col_idx}.npz"
            # Save patch
            save_path = os.path.join('output', 'patches', patch_filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez_compressed(save_path, patch=empty_patch)


def split_mask_into_patches(mask, patch_size=250):
    """
    Divide a mask into subpatches of size patch_size x patch_size.It assumes that the mask is a square and its size is divisible by patch_size.

    Parameters
        - mask: np.ndarray: The mask to divide into patches
        - patch_size: int: The size of the patches

    Returns
        - list: A list of patches, each patch is a numpy array of size (patch_size, patch_size)
    """
    h, w = mask.shape

    # Check if padding is needed
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    padded_mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    patches = []

    for i in range(0, padded_mask.shape[0], patch_size):
        for j in range(0, padded_mask.shape[1], patch_size):
            patch = padded_mask[i:i + patch_size, j:j + patch_size]
            row_idx = i // patch_size
            col_idx = j // patch_size
            patches.append((patch, row_idx, col_idx))

    return patches


def save_patch(patch, patch_filename, output_dir):
    """
    Save a patch as a .npz file

    Parameters
        - patch: np.ndarray: The patch to save
        - patch_filename: str: The filename to save the patch as
        - output_dir: str: The directory to save the patch in
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, patch_filename)
    # Use compressed format - Patches are usually sparse
    np.savez_compressed(save_path, patch=patch)
