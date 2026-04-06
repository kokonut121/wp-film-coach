import json

import numpy as np

from pipeline.manual_homography import compute_manual_homography, run_manual_homography, transform_point
from pipeline.pool_geometry import (
    CALIBRATION_LINE_KEYS,
    apply_pool_bounds,
    calibration_pool_polygon,
    derive_calibration_points,
    mask_and_crop_to_polygon,
)


def _make_calibration_lines(scale_x=10.0, scale_y=20.0):
    return [
        {"key": "left_side", "x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 13.0 * scale_y},
        {"key": "top_side", "x1": 0.0, "y1": 0.0, "x2": 25.0 * scale_x, "y2": 0.0},
        {"key": "right_side", "x1": 25.0 * scale_x, "y1": 0.0, "x2": 25.0 * scale_x, "y2": 13.0 * scale_y},
        {"key": "bottom_side", "x1": 0.0, "y1": 13.0 * scale_y, "x2": 25.0 * scale_x, "y2": 13.0 * scale_y},
        {"key": "m2_left", "x1": 2.0 * scale_x, "y1": 0.0, "x2": 2.0 * scale_x, "y2": 13.0 * scale_y},
        {"key": "m5_left", "x1": 5.0 * scale_x, "y1": 0.0, "x2": 5.0 * scale_x, "y2": 13.0 * scale_y},
        {"key": "half", "x1": 12.5 * scale_x, "y1": 0.0, "x2": 12.5 * scale_x, "y2": 13.0 * scale_y},
        {"key": "m5_right", "x1": 20.0 * scale_x, "y1": 0.0, "x2": 20.0 * scale_x, "y2": 13.0 * scale_y},
        {"key": "m2_right", "x1": 23.0 * scale_x, "y1": 0.0, "x2": 23.0 * scale_x, "y2": 13.0 * scale_y},
    ]


def test_derive_calibration_points_from_lines():
    lines = _make_calibration_lines()
    points = derive_calibration_points(lines)
    points_by_key = {point["key"]: point for point in points}

    assert points_by_key["goal_left_top"]["x"] == 0.0
    assert points_by_key["goal_right_bottom"]["x"] == 250.0
    assert points_by_key["half_top"]["x"] == 125.0


def test_compute_manual_homography_maps_expected_points():
    lines = _make_calibration_lines()
    H = compute_manual_homography(lines)

    pool_x, pool_y = transform_point(H, 125.0, 130.0)
    assert abs(pool_x - 12.5) < 0.1
    assert abs(pool_y - 6.5) < 0.1


def test_run_manual_homography_reuses_one_mapping(tmp_path):
    tracks_path = tmp_path / "tracks.jsonl"
    calibration_path = tmp_path / "calibration.json"

    tracks = [
        {"frame_idx": 0, "t_seconds": 0.0, "player_id": 1, "team": "team_a", "bbox": [120, 100, 130, 130]},
        {"frame_idx": 90, "t_seconds": 3.0, "player_id": 1, "team": "team_a", "bbox": [120, 100, 130, 130]},
    ]
    with open(tracks_path, "w") as f:
        for row in tracks:
            f.write(json.dumps(row) + "\n")
    with open(calibration_path, "w") as f:
        json.dump({"lines": _make_calibration_lines(), "line_keys": CALIBRATION_LINE_KEYS}, f)

    positions_path = run_manual_homography(str(tracks_path), str(calibration_path), str(tmp_path))
    with open(positions_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == 2
    assert rows[0]["x_metres"] == rows[1]["x_metres"]
    assert rows[0]["y_metres"] == rows[1]["y_metres"]
    assert rows[0]["h_stale"] is False


def test_calibration_pool_polygon_uses_outer_lines():
    polygon = calibration_pool_polygon(_make_calibration_lines())

    assert polygon == [[0.0, 0.0], [250.0, 0.0], [250.0, 260.0], [0.0, 260.0]]


def test_apply_pool_bounds_drops_outside_objects():
    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
    players = [
        {"bbox": [10, 10, 30, 50], "confidence": 0.9},
        {"bbox": [120, 10, 150, 40], "confidence": 0.8},
    ]
    ball = {"bbox": [130, 20, 140, 30], "confidence": 0.7}

    kept_players, kept_ball = apply_pool_bounds(players, ball, polygon)

    assert len(kept_players) == 1
    assert kept_players[0]["bbox"] == [10, 10, 30, 50]
    assert kept_ball is None


def test_apply_pool_bounds_no_polygon_is_passthrough():
    players = [{"bbox": [10, 10, 30, 50], "confidence": 0.9}]
    ball = {"bbox": [130, 20, 140, 30], "confidence": 0.7}

    kept_players, kept_ball = apply_pool_bounds(players, ball, None)

    assert kept_players == players
    assert kept_ball == ball


def test_mask_and_crop_to_polygon_returns_roi_and_offset():
    frame = (np.arange(100 * 100 * 3, dtype=np.uint8)).reshape(100, 100, 3)
    polygon = [[10, 20], [60, 20], [60, 80], [10, 80]]

    cropped, offset = mask_and_crop_to_polygon(frame, polygon)

    assert offset == (10, 20)
    assert cropped.shape[0] == 61
    assert cropped.shape[1] == 51
