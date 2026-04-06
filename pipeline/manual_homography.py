"""Manual homography path using user-drawn calibration lines."""

from __future__ import annotations

import json
import os
from typing import Callable

import cv2
import numpy as np

from pipeline.pool_geometry import POOL_POINT_MAP, derive_calibration_points, load_calibration


def compute_manual_homography(calibration_lines: list[dict]) -> np.ndarray:
    calibration_points = derive_calibration_points(calibration_lines)
    src_pts = np.array(
        [[point["x"], point["y"]] for point in calibration_points],
        dtype=np.float64,
    )
    dst_pts = np.array(
        [POOL_POINT_MAP[point["key"]] for point in calibration_points],
        dtype=np.float64,
    )
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    if H is None:
        raise RuntimeError("Failed to compute manual homography from calibration lines")
    return H


def transform_point(H: np.ndarray, px_x: float, px_y: float) -> tuple[float, float]:
    pt = np.array([px_x, px_y, 1.0], dtype=np.float64)
    mapped = H @ pt
    mapped /= mapped[2]
    return float(mapped[0]), float(mapped[1])


def run_manual_homography(
    tracks_path: str,
    calibration_path: str,
    output_dir: str,
    progress_callback: Callable[[int], None] | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    positions_path = os.path.join(output_dir, "positions.jsonl")

    calibration = load_calibration(calibration_path)
    lines = calibration.get("lines", [])
    if not lines:
        raise RuntimeError("Manual homography requested but calibration lines are missing")

    H = compute_manual_homography(lines)

    tracks = []
    with open(tracks_path) as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))

    if not tracks:
        with open(positions_path, "w") as f:
            pass
        return positions_path

    positions = []
    total = len(tracks)
    for index, track in enumerate(tracks):
        bbox = track["bbox"]
        foot_x = (bbox[0] + bbox[2]) / 2
        foot_y = bbox[3]
        pool_x, pool_y = transform_point(H, foot_x, foot_y)
        positions.append({
            "t_seconds": track["t_seconds"],
            "frame_idx": track["frame_idx"],
            "player_id": track["player_id"],
            "team": track["team"],
            "x_metres": round(max(0.0, min(25.0, pool_x)), 2),
            "y_metres": round(max(0.0, min(13.0, pool_y)), 2),
            "h_stale": False,
        })
        if progress_callback and (index + 1) % 200 == 0:
            progress_callback(int(100 * (index + 1) / total))

    with open(positions_path, "w") as f:
        for position in positions:
            f.write(json.dumps(position) + "\n")

    if progress_callback:
        progress_callback(100)

    return positions_path
