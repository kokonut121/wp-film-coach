import json

import cv2
import numpy as np

from pipeline.detect import _classify_teams, run_detection


class _FakeTensor:
    def __init__(self, values):
        self._values = np.array(values, dtype=np.float32)

    def __getitem__(self, index):
        value = self._values[index]
        if np.isscalar(value):
            return value
        return _FakeTensor(value)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _FakeBox:
    def __init__(self, cls_id, conf, bbox):
        self.cls = _FakeTensor([cls_id])
        self.conf = _FakeTensor([conf])
        self.xyxy = _FakeTensor([bbox])


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self, boxes, seen_shapes):
        self._boxes = boxes
        self._seen_shapes = seen_shapes

    def __call__(self, frame, verbose=False):
        self._seen_shapes.append(frame.shape[:2])
        return [_FakeResult(self._boxes)]


def test_classify_indices_match_filtered_player_order():
    """Regression test for filtered detections causing bad second-pass indices."""
    histograms = [
        np.ones(48, dtype=np.float32),
        np.full(48, 2.0, dtype=np.float32),
    ]
    indices = [
        (0, 0),
        (0, 1),
    ]

    team_map = _classify_teams(histograms, indices)

    assert set(team_map.keys()) == {(0, 0), (0, 1)}


def test_filtered_frame_assignment_does_not_require_removed_indices():
    """Simulate the filtered-frame writeback loop with only kept players indexed."""
    frame_detections = [
        {
            "players": [
                {"bbox": [0, 0, 10, 10], "team": None},
                {"bbox": [20, 0, 30, 10], "team": None},
            ]
        }
    ]
    team_map = {(0, 0): "team_a", (0, 1): "team_b"}

    for (frame_pos, player_idx), team in team_map.items():
        frame_detections[frame_pos]["players"][player_idx]["team"] = team

    assert [p["team"] for p in frame_detections[0]["players"]] == ["team_a", "team_b"]


def test_manual_detection_runs_on_pool_roi_and_offsets_boxes(tmp_path, monkeypatch):
    video_path = str(tmp_path / "game.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (100, 60))
    assert writer.isOpened()
    writer.write(np.full((60, 100, 3), 255, dtype=np.uint8))
    writer.release()

    calibration_path = tmp_path / "calibration.json"
    calibration = {
        "lines": [
            {"key": "left_side", "x1": 20, "y1": 10, "x2": 20, "y2": 50},
            {"key": "top_side", "x1": 20, "y1": 10, "x2": 80, "y2": 10},
            {"key": "right_side", "x1": 80, "y1": 10, "x2": 80, "y2": 50},
            {"key": "bottom_side", "x1": 20, "y1": 50, "x2": 80, "y2": 50},
            {"key": "m2_left", "x1": 28, "y1": 10, "x2": 28, "y2": 50},
            {"key": "m5_left", "x1": 35, "y1": 10, "x2": 35, "y2": 50},
            {"key": "half", "x1": 50, "y1": 10, "x2": 50, "y2": 50},
            {"key": "m5_right", "x1": 65, "y1": 10, "x2": 65, "y2": 50},
            {"key": "m2_right", "x1": 72, "y1": 10, "x2": 72, "y2": 50},
        ]
    }
    calibration_path.write_text(json.dumps(calibration))

    seen_shapes = {"player": [], "ball": []}

    def fake_yolo(model_name):
        if model_name == "yolov8m.pt":
            return _FakeModel([_FakeBox(0, 0.9, [5, 5, 15, 20])], seen_shapes["player"])
        return _FakeModel([], seen_shapes["ball"])

    monkeypatch.setattr("pipeline.detect.YOLO", fake_yolo)

    output_dir = tmp_path / "out"
    det_path = run_detection(
        video_path=video_path,
        output_dir=str(output_dir),
        homography_mode="manual",
        calibration_path=str(calibration_path),
    )

    with open(det_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == 1
    assert seen_shapes["player"] == [(41, 61)]
    assert rows[0]["players"][0]["bbox"] == [25, 15, 35, 30]
    assert rows[0]["players"][0]["team"] == "team_a"
