import os
import time


def parse_filename(filename: str) -> tuple:
    """
    Parse the filename to extract the values concerning the patch

    Parameters
        - filename: str: The filename to parse

    Returns
        - tuple: A tuple with the values extracted from the filename. The tuple contains:
            - xstart: int: The x coordinate of the start of the patch
            - ystart: int: The y coordinate of the start of the patch
            - width: int: The width of the patch
            - height: int: The height of the patch
            - resolution: float: The resolution of the patch
    """
    # print(filename)
    basename = os.path.basename(filename)
    # print(basename)
    name = basename.replace('_1-features.csv', '')
    parts = name.split('_')
    # print(parts)
    # time.sleep(100)
    if len(parts) != 5:
        print(filename)
        print(basename)
        print(name)
        print(parts)
        raise ValueError(f"Filename {filename} does not match expected format. Expected format: <xstart>_<ystart>_<width>_<height>_<resolution>.csv")
    
    xstart = int(parts[0])
    ystart = int(parts[1])
    width = int(parts[2])
    height = int(parts[3])
    resolution = float(parts[4])

    return xstart, ystart, width, height, resolution


def list_csv_files(path: str) -> list:
    """
    List all the csv files in a directory

    Parameters
        - path: str: The path to the directory

    Returns
        - list: A list of csv files in the directory
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
