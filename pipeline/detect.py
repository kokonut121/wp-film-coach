"""Player and ball detection module using YOLOv8 + HSV ball fallback."""

from __future__ import annotations

import json
import os
from typing import Callable

import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO

from pipeline.pool_geometry import (
    apply_pool_bounds,
    calibration_pool_polygon,
    load_calibration,
    mask_and_crop_to_polygon,
)

# COCO class IDs
PERSON_CLASS = 0
SPORTS_BALL_CLASS = 32

# HSV thresholds for orange water polo ball
BALL_HSV_LOW = np.array([5, 100, 100])
BALL_HSV_HIGH = np.array([25, 255, 255])
BALL_MIN_AREA = 50
BALL_MAX_AREA = 5000
BALL_MIN_CIRCULARITY = 0.4

# Frame processing stride (every 3rd frame → ~10fps from 30fps source)
FRAME_STRIDE = 3

# Progress reporting interval
PROGRESS_INTERVAL = 500


def _detect_ball_hsv(frame: np.ndarray) -> dict | None:
    """Detect orange ball using HSV color thresholding."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BALL_HSV_LOW, BALL_HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < BALL_MIN_AREA or area > BALL_MAX_AREA:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < BALL_MIN_CIRCULARITY:
            continue
        score = circularity * area
        if score > best_score:
            best_score = score
            x, y, w, h = cv2.boundingRect(cnt)
            best = {
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "confidence": float(min(circularity, 0.95)),
                "source": "hsv",
            }

    return best


def _extract_cap_histogram(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
    """Extract HSV histogram from the top 30% of a player bounding box (cap region)."""
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    cap_y2 = y1 + max(int(h * 0.3), 1)
    cap_crop = frame[y1:cap_y2, x1:x2]

    if cap_crop.size == 0:
        return np.zeros(48, dtype=np.float32)

    hsv_crop = cv2.cvtColor(cap_crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv_crop], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv_crop], [1], None, [16], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv_crop], [2], None, [16], [0, 256]).flatten()
    hist = np.concatenate([hist_h, hist_s, hist_v])

    total = hist.sum()
    if total > 0:
        hist /= total
    return hist.astype(np.float32)


def _offset_bbox(bbox: list[int], offset_x: int, offset_y: int) -> list[int]:
    return [
        int(bbox[0] + offset_x),
        int(bbox[1] + offset_y),
        int(bbox[2] + offset_x),
        int(bbox[3] + offset_y),
    ]


def _classify_teams(all_histograms: list[np.ndarray], all_indices: list[tuple[int, int]]) -> dict:
    """Classify detections into team_a, team_b, goalie using K-means on cap histograms.

    Args:
        all_histograms: List of HSV histograms for each detection.
        all_indices: List of (frame_position_in_detections, player_index) tuples.

    Returns:
        dict mapping (frame_pos, player_idx) -> team label ("team_a", "team_b", "goalie")
    """
    if len(all_histograms) < 3:
        return {idx: "team_a" for idx in all_indices}

    hists = np.array(all_histograms)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(hists)

    # Count cluster sizes — two largest are field player teams, smallest is goalies
    cluster_counts = np.bincount(labels, minlength=3)
    sorted_clusters = np.argsort(cluster_counts)
    goalie_cluster = sorted_clusters[0]
    team_a_cluster = sorted_clusters[2]  # largest
    team_b_cluster = sorted_clusters[1]

    label_map = {
        team_a_cluster: "team_a",
        team_b_cluster: "team_b",
        goalie_cluster: "goalie",
    }

    return {all_indices[i]: label_map[labels[i]] for i in range(len(labels))}


def run_detection(
    video_path: str,
    output_dir: str,
    progress_callback: Callable[[int], None] | None = None,
    model_dir: str = "/models",
    homography_mode: str = "auto",
    calibration_path: str | None = None,
) -> str:
    """Run player and ball detection on a video.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to write detections.jsonl.
        progress_callback: Optional callback(pct: int) for progress updates.
        model_dir: Directory for caching model weights.

    Returns:
        Path to the detections.jsonl file.
    """
    os.makedirs(output_dir, exist_ok=True)
    det_path = os.path.join(output_dir, "detections.jsonl")

    # Load models
    player_model = YOLO("yolov8m.pt")
    ball_model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # First pass: detect players and balls, collect cap histograms
    frame_detections = []
    all_histograms = []
    all_hist_indices = []  # (frame_pos, player_idx)

    frame_idx = 0
    processed = 0
    manual_polygon = None
    if homography_mode == "manual" and calibration_path and os.path.exists(calibration_path):
        manual_polygon = calibration_pool_polygon(load_calibration(calibration_path)["lines"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue

        t_seconds = frame_idx / fps
        detection_frame = frame
        offset_x, offset_y = 0, 0
        if homography_mode == "manual" and manual_polygon is not None:
            detection_frame, (offset_x, offset_y) = mask_and_crop_to_polygon(frame, manual_polygon)
            if detection_frame.size == 0:
                frame_detections.append({
                    "frame_idx": frame_idx,
                    "t_seconds": round(t_seconds, 3),
                    "players": [],
                    "ball": None,
                })
                processed += 1
                frame_idx += 1
                continue

        # Player detection
        player_results = player_model(detection_frame, verbose=False)[0]
        candidate_players = []
        for i, box in enumerate(player_results.boxes):
            cls = int(box.cls[0])
            if cls != PERSON_CLASS:
                continue
            conf = float(box.conf[0])
            if conf < 0.3:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            bbox = _offset_bbox([x1, y1, x2, y2], offset_x, offset_y)

            hist = _extract_cap_histogram(frame, bbox)
            player_entry = {
                "bbox": bbox,
                "confidence": round(conf, 3),
                "team": None,  # filled in second pass
            }
            candidate_players.append((player_entry, hist))

        # Ball detection — YOLO first
        ball_entry = None
        ball_results = ball_model(detection_frame, verbose=False)[0]
        for box in ball_results.boxes:
            cls = int(box.cls[0])
            if cls != SPORTS_BALL_CLASS:
                continue
            conf = float(box.conf[0])
            if conf < 0.2:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            bbox = _offset_bbox([x1, y1, x2, y2], offset_x, offset_y)
            ball_entry = {
                "bbox": bbox,
                "confidence": round(conf, 3),
                "source": "yolo",
            }
            break  # take highest confidence ball

        # HSV fallback if no YOLO ball
        if ball_entry is None:
            ball_entry = _detect_ball_hsv(detection_frame)
            if ball_entry is not None:
                ball_entry["bbox"] = _offset_bbox(ball_entry["bbox"], offset_x, offset_y)

        frame_pos = len(frame_detections)
        if homography_mode == "manual":
            players, ball_entry = apply_pool_bounds(
                [player for player, _ in candidate_players],
                ball_entry,
                manual_polygon,
            )
            kept_bboxes = {tuple(player["bbox"]) for player in players}
            kept_players = [
                (player_entry, hist)
                for player_entry, hist in candidate_players
                if tuple(player_entry["bbox"]) in kept_bboxes
            ]
        else:
            players = [player for player, _ in candidate_players]
            kept_players = candidate_players

        for player_idx, (_, hist) in enumerate(kept_players):
            all_histograms.append(hist)
            all_hist_indices.append((frame_pos, player_idx))

        frame_detections.append({
            "frame_idx": frame_idx,
            "t_seconds": round(t_seconds, 3),
            "players": players,
            "ball": ball_entry,
        })

        processed += 1
        if progress_callback and processed % PROGRESS_INTERVAL == 0:
            pct = int(100 * frame_idx / max(total_frames, 1))
            progress_callback(pct)

        frame_idx += 1

    cap.release()

    # Second pass: classify teams using collected cap histograms
    team_map = _classify_teams(all_histograms, all_hist_indices)

    for (frame_pos, player_idx), team in team_map.items():
        frame_detections[frame_pos]["players"][player_idx]["team"] = team

    # Write detections
    with open(det_path, "w") as f:
        for det in frame_detections:
            f.write(json.dumps(det) + "\n")

    if progress_callback:
        progress_callback(100)

    return det_path
