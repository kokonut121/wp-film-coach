"""Player and ball detection module using YOLOv8 + HSV ball fallback.

In manual homography mode, player detection uses HSV-based cap detection
instead of YOLO, since players are mostly submerged and only caps are visible.
"""

from __future__ import annotations

import json
import os
from typing import Callable

import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO

from pipeline.pool_geometry import (
    WATER_HSV_LOW,
    WATER_HSV_HIGH,
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

# HSV cap detection parameters (manual mode)
CAP_MIN_AREA = 30
CAP_MAX_AREA = 3000

# Frame processing stride (every 3rd frame → ~10fps from 30fps source)
FRAME_STRIDE = 3

# Progress reporting interval
PROGRESS_INTERVAL = 500

# Number of early frames used to train team classifier
TEAM_TRAIN_FRAMES = 10


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


def _detect_caps_hsv(frame: np.ndarray) -> list[dict]:
    """Detect player caps as non-water blobs within the frame.

    Works by creating a water mask, inverting it to find everything that
    is NOT water, then filtering contours by cap-appropriate size.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    water_mask = cv2.inRange(hsv, WATER_HSV_LOW, WATER_HSV_HIGH)

    # Also exclude the ball color range so ball blobs aren't counted as caps
    ball_mask = cv2.inRange(hsv, BALL_HSV_LOW, BALL_HSV_HIGH)

    # Non-water mask: anything in the frame that isn't water or ball
    non_water = cv2.bitwise_not(water_mask)
    non_water = cv2.bitwise_and(non_water, cv2.bitwise_not(ball_mask))

    # Morphological cleanup
    non_water = cv2.morphologyEx(non_water, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    non_water = cv2.morphologyEx(non_water, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(non_water, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    caps = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < CAP_MIN_AREA or area > CAP_MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Confidence based on how well the blob area fills its bounding box
        fill_ratio = area / max(w * h, 1)
        caps.append({
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "confidence": round(float(min(fill_ratio + 0.3, 0.95)), 3),
        })

    return caps


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
    """Classify detections into team_a, team_b using K-means on cap histograms.

    Args:
        all_histograms: List of HSV histograms for each detection.
        all_indices: List of (frame_position_in_detections, player_index) tuples.

    Returns:
        dict mapping (frame_pos, player_idx) -> team label ("team_a", "team_b")
    """
    if len(all_histograms) < 2:
        return {idx: "team_a" for idx in all_indices}

    hists = np.array(all_histograms)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(hists)

    cluster_counts = np.bincount(labels, minlength=2)
    team_a_cluster = int(np.argmax(cluster_counts))
    team_b_cluster = 1 - team_a_cluster

    label_map = {
        team_a_cluster: "team_a",
        team_b_cluster: "team_b",
    }

    return {all_indices[i]: label_map[labels[i]] for i in range(len(labels))}


def _train_team_classifier(
    histograms: list[np.ndarray],
) -> KMeans | None:
    """Train a K-means(k=2) classifier on early-frame cap histograms.

    Returns the fitted KMeans model, or None if too few samples.
    """
    if len(histograms) < 2:
        return None
    hists = np.array(histograms)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(hists)
    return kmeans


def _assign_team(kmeans: KMeans, histogram: np.ndarray) -> str:
    """Assign a single cap histogram to the nearest team cluster."""
    cluster_counts = np.bincount(kmeans.labels_, minlength=2)
    team_a_cluster = int(np.argmax(cluster_counts))
    label = kmeans.predict(histogram.reshape(1, -1))[0]
    return "team_a" if label == team_a_cluster else "team_b"


def run_detection(
    video_path: str,
    output_dir: str,
    progress_callback: Callable[[int], None] | None = None,
    model_dir: str = "/models",
    homography_mode: str = "auto",
    calibration_path: str | None = None,
) -> str:
    """Run player and ball detection on a video.

    In manual mode, players are detected via HSV cap detection (non-water blobs).
    In auto mode, players are detected via YOLO person class.
    Ball detection (YOLO + HSV fallback) is the same in both modes.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to write detections.jsonl.
        progress_callback: Optional callback(pct: int) for progress updates.
        model_dir: Directory for caching model weights.
        homography_mode: "auto" or "manual".
        calibration_path: Path to calibration.json (manual mode).

    Returns:
        Path to the detections.jsonl file.
    """
    os.makedirs(output_dir, exist_ok=True)
    det_path = os.path.join(output_dir, "detections.jsonl")

    use_cap_detection = homography_mode == "manual"

    # Load YOLO models — player model only needed in auto mode
    if not use_cap_detection:
        player_model = YOLO("yolov8m.pt")
    ball_model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    frame_detections = []
    all_histograms = []
    all_hist_indices = []  # (frame_pos, player_idx)

    # Manual mode: early team classifier trained on first N frames
    team_classifier: KMeans | None = None
    training_histograms: list[np.ndarray] = []
    # Maps (frame_pos, player_idx) -> histogram for retroactive assignment
    pending_team_assignments: dict[tuple[int, int], np.ndarray] = {}

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

        # ── Player detection ──
        candidate_players = []

        if use_cap_detection:
            # Manual mode: HSV cap detection
            cap_detections = _detect_caps_hsv(detection_frame)
            for det in cap_detections:
                bbox = _offset_bbox(det["bbox"], offset_x, offset_y)
                hist = _extract_cap_histogram(frame, bbox)

                # Assign team immediately if classifier is trained
                team = None
                if team_classifier is not None:
                    team = _assign_team(team_classifier, hist)

                player_entry = {
                    "bbox": bbox,
                    "confidence": det["confidence"],
                    "team": team,
                }
                candidate_players.append((player_entry, hist))
        else:
            # Auto mode: YOLO person detection
            player_results = player_model(detection_frame, verbose=False)[0]
            for box in player_results.boxes:
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

        # ── Ball detection — YOLO first, HSV fallback ──
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

        if ball_entry is None:
            ball_entry = _detect_ball_hsv(detection_frame)
            if ball_entry is not None:
                ball_entry["bbox"] = _offset_bbox(ball_entry["bbox"], offset_x, offset_y)

        # ── Pool bounds filtering ──
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

        # ── Histogram collection & early team training (manual mode) ──
        if use_cap_detection:
            if team_classifier is None:
                # Still in training phase: collect histograms
                for player_idx, (_, hist) in enumerate(kept_players):
                    training_histograms.append(hist)
                    pending_team_assignments[(frame_pos, player_idx)] = hist

                # Train classifier after TEAM_TRAIN_FRAMES processed frames
                if processed + 1 >= TEAM_TRAIN_FRAMES:
                    team_classifier = _train_team_classifier(training_histograms)
                    if team_classifier is not None:
                        # Retroactively assign teams to already-appended frames
                        for (fp, pi), hist in pending_team_assignments.items():
                            if fp < len(frame_detections):
                                frame_detections[fp]["players"][pi]["team"] = _assign_team(team_classifier, hist)
                        # Assign current frame's players (not yet appended)
                        for player_idx, (_, hist) in enumerate(kept_players):
                            players[player_idx]["team"] = _assign_team(team_classifier, hist)
                        pending_team_assignments.clear()
        else:
            # Auto mode: collect for second-pass classification
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

    # If video ended before TEAM_TRAIN_FRAMES, train with what we have
    if use_cap_detection and team_classifier is None and pending_team_assignments:
        team_classifier = _train_team_classifier(training_histograms)
        if team_classifier is not None:
            for (fp, pi), hist in pending_team_assignments.items():
                frame_detections[fp]["players"][pi]["team"] = _assign_team(team_classifier, hist)
        else:
            # Too few samples for clustering — assign all to team_a
            for (fp, pi) in pending_team_assignments:
                frame_detections[fp]["players"][pi]["team"] = "team_a"

    # Auto mode: second pass team classification
    if not use_cap_detection:
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
