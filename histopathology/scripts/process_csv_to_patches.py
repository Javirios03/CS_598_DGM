import sys
import pandas as pd
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.file_utils import parse_filename, list_csv_files
from utils.nuclei_utils import parse_polygon_string, create_tile_mask
from utils.patch_utils import split_mask_into_patches, save_patch, create_empty_patch
import time

INPUT_DIR = 'data/csv_tiles'
OUTPUT_DIR = 'output/patches'
PATCH_SIZE = 250  # Size of the patches to create


def process_one_csv(path: str) -> None:
    """
    Process a single CSV file to create patches from the nuclei polygons
    """
    # Parse filename
    x_start, ystart, width, height, _ = parse_filename(path)

    print(f"Processing tile at ({x_start}, {ystart}) with size ({width}, {height})")

    # Read CSV file
    df = pd.read_csv(path)
    if len(df) == 0:
        print("Empty tile, creating empty patches")
        create_empty_patch(x_start, ystart, width, height)
    
    # Parse all polygons
    nuclei_polygons = []
    for _, row in df.iterrows():
        polygon_string = row['Polygon']
        points = parse_polygon_string(polygon_string)
        # Shift points to the origin of the tile
        points[:, 0] -= x_start
        points[:, 1] -= ystart
        nuclei_polygons.append(points)

    # Create mask
    mask = create_tile_mask(width, height, nuclei_polygons)

    # print(f"Mask shape: {mask.shape}")
    # print(f"Patch origin: ({x_start}, {ystart})")
    # print(f"Nucleus points (first 3): {nuclei_polygons[:3]}")
    # time.sleep(1000)  # Pause for debugging

    # Split mask into patches
    patches = split_mask_into_patches(mask, patch_size=PATCH_SIZE)

    # # We save the whole patch as a single file, using key patch_i_j for each patch
    # patch_filename = f"patch_{x_start}_{ystart}.npz"
    # save_patch(patches, patch_filename, OUTPUT_DIR)

    for patch, row_idx, col_idx in patches:
        # Create patch filename
        patch_filename = f"patch_{x_start}_{ystart}_row{row_idx}_col{col_idx}.npz"
        # Save patch
        save_patch(patch, patch_filename, OUTPUT_DIR)


def main():
    # Obtain the list of CSV files
    csv_files = list_csv_files(INPUT_DIR)
    print(f"Found {len(csv_files)} CSV files to process")

    # Process each CSV file
    for csv_file in csv_files:
        process_one_csv(csv_file)


if __name__ == "__main__":
    main()
    print("Processing completed")
