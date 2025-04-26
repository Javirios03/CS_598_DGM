import numpy as np
import cv2


def parse_polygon_string(polygon_string: str) -> np.ndarray:
    """
    Parse a polygon string to extract the coordinates of the polygon

    Parameters
        - polygon_string: str: The polygon string to parse

    Returns
        - np.ndarray: A numpy array with the coordinates of the polygon. shape (N, 2), with N being the number of points in the polygon
    """
    polygon_string = polygon_string.strip('[]')
    coords = polygon_string.split(':')
    points = np.array(coords, dtype=float).reshape(-1, 2)

    # Ensure the polygon is closed
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack((points, points[0]))
    return points


def create_tile_mask(width, height, nuclei_polygons):
    """
    Create local 2D mask of size (width, height), filling all pixels inside the nuclei polygons with 1 and all other pixels with 0

    Parameters
        - width: int: The width of the mask
        - height: int: The height of the mask
        - nuclei_polygons: np.ndarray: A numpy array with the coordinates of the nuclei polygons. 

    Returns
        - np.ndarray: A numpy array with the mask. shape (height, width)
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for points in nuclei_polygons:
        # Draw filled polygon
        int_points = np.round(points).astype(np.int32)
        int_points = int_points.reshape((-1, 1, 2))

        cv2.fillPoly(mask, [int_points], color=1)
    return mask
