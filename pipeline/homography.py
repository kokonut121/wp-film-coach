"""Homography module — map pixel coordinates to FINA pool coordinates.

Uses classical CV (Canny + HoughLinesP) to detect pool lines and compute
a homography matrix mapping pixel space to a 25m x 13m pool template.
"""

from __future__ import annotations

import json
import os
from typing import Callable

import cv2
import numpy as np

# FINA pool dimensions in metres
POOL_LENGTH = 25.0
POOL_WIDTH = 13.0

# Known line positions along pool length (metres from left goal line)
POOL_LINES = {
    "goal_left": 0.0,
    "2m_left": 2.0,
    "5m_left": 5.0,
    "half": 12.5,
    "5m_right": 20.0,
    "2m_right": 23.0,
    "goal_right": 25.0,
}

# Side boundaries (metres from bottom)
POOL_SIDES = {
    "bottom": 0.0,
    "top": POOL_WIDTH,
}

# HSV range for pool water (blue/teal)
WATER_HSV_LOW = np.array([85, 40, 40])
WATER_HSV_HIGH = np.array([130, 255, 255])

# Scene cut detection threshold
SCENE_CUT_THRESHOLD = 0.3

# Minimum keypoints needed for homography
MIN_KEYPOINTS = 4


def _detect_water_mask(frame: np.ndarray) -> np.ndarray:
    """Create a binary mask of the pool water area."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WATER_HSV_LOW, WATER_HSV_HIGH)
    # Clean up with morphology
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _detect_pool_lines(frame: np.ndarray, water_mask: np.ndarray) -> list[tuple]:
    """Detect pool lane lines using Canny + HoughLinesP.

    Returns list of (rho, theta, x1, y1, x2, y2) for each detected line segment.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Mask to water area only
    masked = cv2.bitwise_and(gray, gray, mask=water_mask)

    # Edge detection
    edges = cv2.Canny(masked, 50, 150, apertureSize=3)

    # Detect line segments
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=50, maxLineGap=20)

    if lines is None:
        return []

    result = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        result.append((angle, x1, y1, x2, y2))

    return result


def _cluster_lines(lines: list[tuple], frame_shape: tuple) -> dict:
    """Cluster detected lines into vertical (pool lane lines) and horizontal (side lines).

    Returns dict with 'vertical' and 'horizontal' line groups,
    each as list of (avg_x_or_y, line_segments).
    """
    if not lines:
        return {"vertical": [], "horizontal": []}

    h, w = frame_shape[:2]
    vertical = []  # roughly vertical lines (pool lane markers)
    horizontal = []  # roughly horizontal lines (side boundaries)

    for angle, x1, y1, x2, y2 in lines:
        # Vertical: angle near 90 degrees
        if 70 < angle < 110:
            mid_x = (x1 + x2) / 2
            vertical.append((mid_x, x1, y1, x2, y2))
        # Horizontal: angle near 0 or 180
        elif angle < 20 or angle > 160:
            mid_y = (y1 + y2) / 2
            horizontal.append((mid_y, x1, y1, x2, y2))

    # Cluster vertical lines by x-position (merge lines within 20px)
    vertical.sort(key=lambda l: l[0])
    v_clusters = _merge_nearby(vertical, key_idx=0, threshold=20)

    # Cluster horizontal lines by y-position
    horizontal.sort(key=lambda l: l[0])
    h_clusters = _merge_nearby(horizontal, key_idx=0, threshold=20)

    return {"vertical": v_clusters, "horizontal": h_clusters}


def _merge_nearby(lines: list[tuple], key_idx: int, threshold: float) -> list[float]:
    """Merge lines that are within threshold pixels of each other.

    Returns list of averaged positions.
    """
    if not lines:
        return []

    clusters = []
    current_cluster = [lines[0][key_idx]]

    for i in range(1, len(lines)):
        if lines[i][key_idx] - current_cluster[-1] < threshold:
            current_cluster.append(lines[i][key_idx])
        else:
            clusters.append(np.mean(current_cluster))
            current_cluster = [lines[i][key_idx]]

    clusters.append(np.mean(current_cluster))
    return clusters


def _match_lines_to_template(
    v_positions: list[float],
    h_positions: list[float],
    frame_shape: tuple,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Match detected line positions to FINA pool template.

    Returns list of (pixel_point, pool_point) correspondences.
    """
    h, w = frame_shape[:2]
    correspondences = []

    # Known vertical line positions in metres (sorted)
    template_v = sorted(POOL_LINES.values())

    # Try to match vertical lines to template
    if len(v_positions) >= 2:
        # Use the spacing pattern to find the best match
        # Try all possible alignments of detected lines to template lines
        best_match = None
        best_error = float("inf")

        for start_idx in range(len(template_v) - len(v_positions) + 1):
            template_subset = template_v[start_idx:start_idx + len(v_positions)]
            # Normalize both to [0, 1] range and compare spacing
            if len(v_positions) < 2:
                continue

            det_range = v_positions[-1] - v_positions[0]
            tmpl_range = template_subset[-1] - template_subset[0]
            if det_range < 1 or tmpl_range < 0.1:
                continue

            det_norm = [(p - v_positions[0]) / det_range for p in v_positions]
            tmpl_norm = [(p - template_subset[0]) / tmpl_range for p in template_subset]
            error = sum((d - t) ** 2 for d, t in zip(det_norm, tmpl_norm))

            if error < best_error:
                best_error = error
                best_match = list(zip(v_positions, template_subset))

        if best_match and best_error < 0.5:
            # Use horizontal positions (if any) for y-coordinate matching
            if len(h_positions) >= 2:
                y_top = min(h_positions)
                y_bot = max(h_positions)
                for px_x, pool_x in best_match:
                    correspondences.append(((px_x, y_top), (pool_x, POOL_WIDTH)))
                    correspondences.append(((px_x, y_bot), (pool_x, 0.0)))
            elif len(h_positions) == 1:
                # One horizontal line — assume it's the nearest side
                y_line = h_positions[0]
                mid_y = h / 2
                pool_y = POOL_WIDTH if y_line < mid_y else 0.0
                for px_x, pool_x in best_match:
                    correspondences.append(((px_x, y_line), (pool_x, pool_y)))
            else:
                # No horizontal lines — use frame edges as rough estimates
                for px_x, pool_x in best_match:
                    correspondences.append(((px_x, h * 0.15), (pool_x, POOL_WIDTH)))
                    correspondences.append(((px_x, h * 0.85), (pool_x, 0.0)))

    return correspondences


def compute_homography(frame: np.ndarray) -> tuple[np.ndarray | None, bool]:
    """Compute homography matrix from a single frame.

    Returns:
        (H, success) where H is the 3x3 homography matrix or None.
    """
    water_mask = _detect_water_mask(frame)
    lines = _detect_pool_lines(frame, water_mask)
    clusters = _cluster_lines(lines, frame.shape)

    correspondences = _match_lines_to_template(
        clusters["vertical"], clusters["horizontal"], frame.shape
    )

    if len(correspondences) < MIN_KEYPOINTS:
        return None, False

    src_pts = np.array([c[0] for c in correspondences], dtype=np.float64)
    dst_pts = np.array([c[1] for c in correspondences], dtype=np.float64)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        return None, False

    return H, True


def _detect_scene_cut(prev_frame: np.ndarray, curr_frame: np.ndarray) -> bool:
    """Detect scene cuts using histogram correlation."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_hist = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
    curr_hist = cv2.calcHist([curr_gray], [0], None, [64], [0, 256])
    cv2.normalize(prev_hist, prev_hist)
    cv2.normalize(curr_hist, curr_hist)
    corr = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
    return corr < SCENE_CUT_THRESHOLD


def transform_point(H: np.ndarray, px_x: float, px_y: float) -> tuple[float, float]:
    """Transform a pixel coordinate to pool coordinates using homography H."""
    pt = np.array([px_x, px_y, 1.0])
    mapped = H @ pt
    mapped /= mapped[2]
    return float(mapped[0]), float(mapped[1])


def run_homography(
    tracks_path: str,
    video_path: str,
    output_dir: str,
    fps: float,
    progress_callback: Callable[[int], None] | None = None,
) -> str:
    """Map tracked player positions from pixel to pool coordinates.

    Args:
        tracks_path: Path to tracks.jsonl.
        video_path: Path to original video.
        output_dir: Directory to write positions.jsonl.
        fps: Video FPS.
        progress_callback: Optional callback(pct: int).

    Returns:
        Path to positions.jsonl.
    """
    os.makedirs(output_dir, exist_ok=True)
    positions_path = os.path.join(output_dir, "positions.jsonl")

    # Load tracks
    tracks = []
    with open(tracks_path) as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))

    if not tracks:
        with open(positions_path, "w") as f:
            pass
        return positions_path

    # Group tracks by frame
    frames_map = {}
    for t in tracks:
        fi = t["frame_idx"]
        if fi not in frames_map:
            frames_map[fi] = []
        frames_map[fi].append(t)

    sorted_frames = sorted(frames_map.keys())

    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    current_H = None
    h_stale_count = 0

    positions = []
    total = len(sorted_frames)

    for i, frame_idx in enumerate(sorted_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Check for scene cut → recompute homography
        need_recompute = False
        if prev_frame is not None:
            if _detect_scene_cut(prev_frame, frame):
                need_recompute = True
                h_stale_count = 0

        if current_H is None or need_recompute:
            H, success = compute_homography(frame)
            if success:
                current_H = H
                h_stale_count = 0
            else:
                h_stale_count += 1

        # Transform player positions
        if current_H is not None:
            for track in frames_map[frame_idx]:
                bbox = track["bbox"]
                # Foot-point: bottom-center of bbox
                foot_x = (bbox[0] + bbox[2]) / 2
                foot_y = bbox[3]

                pool_x, pool_y = transform_point(current_H, foot_x, foot_y)

                # Clamp to pool bounds
                pool_x = max(0.0, min(POOL_LENGTH, pool_x))
                pool_y = max(0.0, min(POOL_WIDTH, pool_y))

                positions.append({
                    "t_seconds": track["t_seconds"],
                    "frame_idx": frame_idx,
                    "player_id": track["player_id"],
                    "team": track["team"],
                    "x_metres": round(pool_x, 2),
                    "y_metres": round(pool_y, 2),
                    "h_stale": h_stale_count > 0,
                })

        prev_frame = frame.copy()

        if progress_callback and (i + 1) % 200 == 0:
            progress_callback(int(100 * (i + 1) / total))

    cap.release()

    with open(positions_path, "w") as f:
        for pos in positions:
            f.write(json.dumps(pos) + "\n")

    if progress_callback:
        progress_callback(100)

    return positions_path
