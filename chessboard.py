"""Printed 8x6 squares: 7x5 inner corners. Coordinates and size are in mm."""
import cv2
import numpy as np

PATTERN = (7, 5)


def object_points(pattern=PATTERN, square_mm=20.):
    if len(pattern) != 2 or min(pattern) < 3 or not np.isfinite(square_mm) or square_mm <= 0:
        raise ValueError('Invalid chessboard dimensions')
    points = np.zeros((pattern[0]*pattern[1], 3), dtype=float)
    points[:, :2] = np.mgrid[:pattern[0], :pattern[1]].T.reshape(-1, 2)*square_mm
    return points


def detect(image,pattern=PATTERN):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
    if not ok:
        raise ValueError(f'Need all {pattern[0]}x{pattern[1]} inner corners visible')
    corners = corners.reshape(-1, 2).astype(float)
    h, w = gray.shape
    if (corners.min() < 4 or corners[:, 0].max() > w-5 or corners[:, 1].max() > h-5):
        raise ValueError('Chessboard is too close to image border')
    grid = corners.reshape(pattern[1], pattern[0], 2)
    spacing = min(np.linalg.norm(np.diff(grid, axis=0), axis=2).min(),
                  np.linalg.norm(np.diff(grid, axis=1), axis=2).min())
    if spacing < 8:
        raise ValueError('Chessboard is too small in the image')
    return corners


def orient(corners, anchor,pattern=PATTERN):
    """Identify a fixed physical origin by click; never silently assume image top-left.

    An ordinary symmetric chessboard has no unique coded corner. The click selects
    a detected endpoint (not a manually measured pixel), retaining subpixel accuracy.
    """
    if anchor is None:
        raise ValueError('Click the same marked origin corner before saving')
    grid = np.asarray(corners, dtype=float).reshape(pattern[1], pattern[0], 2)
    ends = np.array([grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]])
    distances = np.linalg.norm(ends-np.asarray(anchor), axis=1)
    which = int(np.argmin(distances))
    spacing = min(np.linalg.norm(np.diff(grid, axis=0), axis=2).min(),
                  np.linalg.norm(np.diff(grid, axis=1), axis=2).min())
    if distances[which] > .65*spacing:
        raise ValueError('Origin moved or click missed; click the marked corner again')
    if which in (2, 3):
        grid = grid[::-1]
    if which in (1, 3):
        grid = grid[:, ::-1]
    return grid.reshape(-1, 2).copy()
