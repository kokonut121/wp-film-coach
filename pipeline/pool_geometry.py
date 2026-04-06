"""Shared pool geometry helpers for filtering and manual calibration."""

from __future__ import annotations

import json
from collections.abc import Iterable

import cv2
import numpy as np

POOL_LENGTH = 25.0
POOL_WIDTH = 13.0

WATER_HSV_LOW = np.array([85, 40, 40])
WATER_HSV_HIGH = np.array([130, 255, 255])

CALIBRATION_LINE_KEYS = [
    "left_side",
    "top_side",
    "right_side",
    "bottom_side",
    "m2_left",
    "m5_left",
    "half",
    "m5_right",
    "m2_right",
]

CALIBRATION_POINT_KEYS = [
    "goal_left_top",
    "m2_left_top",
    "m5_left_top",
    "half_top",
    "m5_right_top",
    "m2_right_top",
    "goal_right_top",
    "goal_left_bottom",
    "m2_left_bottom",
    "m5_left_bottom",
    "half_bottom",
    "m5_right_bottom",
    "m2_right_bottom",
    "goal_right_bottom",
]

POOL_POINT_MAP = {
    "goal_left_top": (0.0, POOL_WIDTH),
    "m2_left_top": (2.0, POOL_WIDTH),
    "m5_left_top": (5.0, POOL_WIDTH),
    "half_top": (12.5, POOL_WIDTH),
    "m5_right_top": (20.0, POOL_WIDTH),
    "m2_right_top": (23.0, POOL_WIDTH),
    "goal_right_top": (25.0, POOL_WIDTH),
    "goal_left_bottom": (0.0, 0.0),
    "m2_left_bottom": (2.0, 0.0),
    "m5_left_bottom": (5.0, 0.0),
    "half_bottom": (12.5, 0.0),
    "m5_right_bottom": (20.0, 0.0),
    "m2_right_bottom": (23.0, 0.0),
    "goal_right_bottom": (25.0, 0.0),
}


def detect_water_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WATER_HSV_LOW, WATER_HSV_HIGH)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def infer_pool_polygon(frame: np.ndarray) -> list[list[float]] | None:
    """Best-effort auto pool boundary from the largest water contour."""
    mask = detect_water_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 5000:
        return None

    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    polygon = approx.reshape(-1, 2).astype(float).tolist()
    return polygon if len(polygon) >= 3 else None


def point_in_polygon(point: tuple[float, float], polygon: Iterable[Iterable[float]] | None) -> bool:
    if polygon is None:
        return True
    polygon_arr = np.array(list(polygon), dtype=np.float32)
    if len(polygon_arr) < 3:
        return True
    return cv2.pointPolygonTest(polygon_arr, point, False) >= 0


def apply_pool_bounds(
    players: list[dict],
    ball: dict | None,
    polygon: Iterable[Iterable[float]] | None,
) -> tuple[list[dict], dict | None]:
    if polygon is None:
        return players, ball

    kept_players = []
    for player in players:
        bbox = player["bbox"]
        foot_point = ((bbox[0] + bbox[2]) / 2, bbox[3])
        if point_in_polygon(foot_point, polygon):
            kept_players.append(player)

    kept_ball = None
    if ball:
        bbox = ball["bbox"]
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        if point_in_polygon(center, polygon):
            kept_ball = ball

    return kept_players, kept_ball


def validate_calibration_lines(lines: list[dict]) -> None:
    if len(lines) != len(CALIBRATION_LINE_KEYS):
        raise ValueError(f"Expected {len(CALIBRATION_LINE_KEYS)} calibration lines")

    keys = [line["key"] for line in lines]
    if set(keys) != set(CALIBRATION_LINE_KEYS):
        raise ValueError("Calibration lines must include the exact required keys")

    if len(keys) != len(set(keys)):
        raise ValueError("Calibration line keys must be unique")

    for line in lines:
        x1, y1, x2, y2 = float(line["x1"]), float(line["y1"]), float(line["x2"]), float(line["y2"])
        if abs(x1 - x2) < 1e-6 and abs(y1 - y2) < 1e-6:
            raise ValueError(f"Calibration line {line['key']} must have distinct endpoints")


def line_to_homogeneous(line: dict | tuple[float, float, float, float]) -> np.ndarray:
    if isinstance(line, dict):
        x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
    else:
        x1, y1, x2, y2 = line
    p1 = np.array([float(x1), float(y1), 1.0])
    p2 = np.array([float(x2), float(y2), 1.0])
    return np.cross(p1, p2)


def intersect_lines(line_a: np.ndarray, line_b: np.ndarray) -> tuple[float, float]:
    point = np.cross(line_a, line_b)
    if abs(point[2]) < 1e-6:
        raise ValueError("Calibration lines do not form a stable intersection")
    return float(point[0] / point[2]), float(point[1] / point[2])


def calibration_lines_to_dict(lines: list[dict]) -> dict[str, dict]:
    return {line["key"]: line for line in lines}


def derive_calibration_points(lines: list[dict] | dict[str, dict]) -> list[dict]:
    if isinstance(lines, dict):
        lines_by_key = lines
    else:
        lines_by_key = calibration_lines_to_dict(lines)

    top = line_to_homogeneous(lines_by_key["top_side"])
    bottom = line_to_homogeneous(lines_by_key["bottom_side"])
    left = line_to_homogeneous(lines_by_key["left_side"])
    right = line_to_homogeneous(lines_by_key["right_side"])
    m2_left = line_to_homogeneous(lines_by_key["m2_left"])
    m5_left = line_to_homogeneous(lines_by_key["m5_left"])
    half = line_to_homogeneous(lines_by_key["half"])
    m5_right = line_to_homogeneous(lines_by_key["m5_right"])
    m2_right = line_to_homogeneous(lines_by_key["m2_right"])

    intersections = {
        "goal_left_top": intersect_lines(left, top),
        "m2_left_top": intersect_lines(m2_left, top),
        "m5_left_top": intersect_lines(m5_left, top),
        "half_top": intersect_lines(half, top),
        "m5_right_top": intersect_lines(m5_right, top),
        "m2_right_top": intersect_lines(m2_right, top),
        "goal_right_top": intersect_lines(right, top),
        "goal_left_bottom": intersect_lines(left, bottom),
        "m2_left_bottom": intersect_lines(m2_left, bottom),
        "m5_left_bottom": intersect_lines(m5_left, bottom),
        "half_bottom": intersect_lines(half, bottom),
        "m5_right_bottom": intersect_lines(m5_right, bottom),
        "m2_right_bottom": intersect_lines(m2_right, bottom),
        "goal_right_bottom": intersect_lines(right, bottom),
    }

    return [{"key": key, "x": x, "y": y} for key, (x, y) in intersections.items()]


def calibration_pool_polygon(lines: list[dict] | dict[str, dict]) -> list[list[float]]:
    if isinstance(lines, dict):
        lines_by_key = lines
    else:
        lines_by_key = calibration_lines_to_dict(lines)

    left = line_to_homogeneous(lines_by_key["left_side"])
    top = line_to_homogeneous(lines_by_key["top_side"])
    right = line_to_homogeneous(lines_by_key["right_side"])
    bottom = line_to_homogeneous(lines_by_key["bottom_side"])

    return [
        list(intersect_lines(left, top)),
        list(intersect_lines(right, top)),
        list(intersect_lines(right, bottom)),
        list(intersect_lines(left, bottom)),
    ]


def mask_and_crop_to_polygon(
    frame: np.ndarray,
    polygon: list[list[float]] | None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Mask to the polygon and crop to its bounding box.

    Returns:
        cropped_frame, (offset_x, offset_y)
    """
    if polygon is None:
        return frame, (0, 0)

    polygon_arr = np.array(polygon, dtype=np.int32)
    if len(polygon_arr) < 3:
        return frame, (0, 0)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_arr], 255)
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    x, y, w, h = cv2.boundingRect(polygon_arr)
    if w <= 0 or h <= 0:
        return frame, (0, 0)

    return masked[y:y + h, x:x + w], (int(x), int(y))


def load_calibration(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
