"""Player tracking module using ByteTrack + ResNet18 re-ID."""

from __future__ import annotations

import json
import os
from typing import Callable

import cv2
import numpy as np
import supervision as sv
import torch
import torchvision.models as models
import torchvision.transforms as T

# Scene cut detection threshold (histogram correlation)
SCENE_CUT_THRESHOLD = 0.3

# Re-ID cosine similarity threshold for cross-cut matching
REID_THRESHOLD = 0.7

# Max frames a track can be lost before being dropped (2 seconds at ~10fps)
MAX_LOST_FRAMES = 20

# ResNet18 embedding transform
_reid_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ReIDExtractor:
    """Extract appearance embeddings using a pretrained ResNet18."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the final FC layer to get 512-dim embeddings
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def extract(self, frame: np.ndarray, bboxes: list[list[int]]) -> np.ndarray:
        """Extract embeddings for a list of bounding boxes.

        Returns:
            np.ndarray of shape (N, 512) with L2-normalized embeddings.
        """
        if not bboxes:
            return np.zeros((0, 512), dtype=np.float32)

        crops = []
        for x1, y1, x2, y2 in bboxes:
            crop = frame[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
            if crop.size == 0:
                crop = np.zeros((64, 32, 3), dtype=np.uint8)
            crops.append(_reid_transform(crop))

        batch = torch.stack(crops).to(self.device)
        embeddings = self.model(batch).cpu().numpy()

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        return (embeddings / norms).astype(np.float32)


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


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two sets of embeddings."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return a @ b.T


def run_tracking(
    detections_path: str,
    video_path: str,
    output_dir: str,
    progress_callback: Callable[[int], None] | None = None,
) -> str:
    """Run tracking on detections to produce stable player IDs.

    Args:
        detections_path: Path to detections.jsonl from detect module.
        video_path: Path to original video (for re-ID crop extraction).
        output_dir: Directory to write tracks.jsonl.
        progress_callback: Optional callback(pct: int).

    Returns:
        Path to tracks.jsonl.
    """
    os.makedirs(output_dir, exist_ok=True)
    tracks_path = os.path.join(output_dir, "tracks.jsonl")

    # Load detections
    detections_list = []
    with open(detections_path) as f:
        for line in f:
            detections_list.append(json.loads(line))

    if not detections_list:
        with open(tracks_path, "w") as f:
            pass
        return tracks_path

    # Open video for re-ID crops
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    reid = ReIDExtractor()
    tracker = sv.ByteTrack(
        track_activation_threshold=0.3,
        lost_track_buffer=MAX_LOST_FRAMES,
        minimum_matching_threshold=0.8,
        frame_rate=int(fps / 3),  # effective fps after stride
    )

    # State for cross-cut re-ID
    prev_frame = None
    prev_shot_embeddings = {}  # track_id -> embedding
    global_id_offset = 0
    local_to_global = {}  # maps ByteTrack local IDs to global IDs
    next_global_id = 0

    all_tracks = []
    total = len(detections_list)

    for i, det in enumerate(detections_list):
        frame_idx = det["frame_idx"]

        # Seek to the correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        players = det["players"]
        if not players:
            prev_frame = frame
            continue

        # Prepare supervision Detections
        bboxes = np.array([p["bbox"] for p in players], dtype=np.float32)
        # Convert from [x1,y1,x2,y2] — already in that format
        confidences = np.array([p["confidence"] for p in players], dtype=np.float32)

        sv_dets = sv.Detections(
            xyxy=bboxes,
            confidence=confidences,
        )

        # Check for scene cut
        is_cut = False
        if prev_frame is not None:
            is_cut = _detect_scene_cut(prev_frame, frame)

        if is_cut:
            # Save embeddings from previous shot
            active_bboxes = []
            active_global_ids = []
            for local_id, global_id in local_to_global.items():
                # Find last known bbox for this track in recent frames
                for recent in reversed(all_tracks[-MAX_LOST_FRAMES:]):
                    if recent["player_id"] == global_id:
                        active_bboxes.append(recent["bbox"])
                        active_global_ids.append(global_id)
                        break

            if active_bboxes and prev_frame is not None:
                prev_embeddings = reid.extract(prev_frame, active_bboxes)
                prev_shot_embeddings = {
                    gid: emb for gid, emb in zip(active_global_ids, prev_embeddings)
                }

            # Reset tracker for new shot
            tracker = sv.ByteTrack(
                track_activation_threshold=0.3,
                lost_track_buffer=MAX_LOST_FRAMES,
                minimum_matching_threshold=0.8,
                frame_rate=int(fps / 3),
            )
            local_to_global = {}

        # Run ByteTrack
        tracked = tracker.update_with_detections(sv_dets)

        if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
            # Handle cross-cut re-ID for new local IDs
            new_local_ids = [
                tid for tid in tracked.tracker_id if tid not in local_to_global
            ]

            if new_local_ids and prev_shot_embeddings and is_cut:
                # Extract embeddings for new detections
                new_bboxes = []
                new_local_list = []
                for j, tid in enumerate(tracked.tracker_id):
                    if tid in new_local_ids:
                        new_bboxes.append(tracked.xyxy[j].astype(int).tolist())
                        new_local_list.append(tid)

                if new_bboxes:
                    new_embeddings = reid.extract(frame, new_bboxes)
                    prev_gids = list(prev_shot_embeddings.keys())
                    prev_embs = np.array([prev_shot_embeddings[g] for g in prev_gids])
                    sim_matrix = _cosine_similarity_matrix(new_embeddings, prev_embs)

                    matched_prev = set()
                    for j in range(len(new_local_list)):
                        if sim_matrix.shape[1] > 0:
                            best_idx = np.argmax(sim_matrix[j])
                            best_sim = sim_matrix[j, best_idx]
                            if best_sim >= REID_THRESHOLD and prev_gids[best_idx] not in matched_prev:
                                local_to_global[new_local_list[j]] = prev_gids[best_idx]
                                matched_prev.add(prev_gids[best_idx])
                                continue
                        # No match — assign new global ID
                        local_to_global[new_local_list[j]] = next_global_id
                        next_global_id += 1
            else:
                for tid in new_local_ids:
                    local_to_global[tid] = next_global_id
                    next_global_id += 1

            # Build track entries
            for j, tid in enumerate(tracked.tracker_id):
                global_id = local_to_global.get(tid, tid)
                bbox = tracked.xyxy[j].astype(int).tolist()
                conf = float(tracked.confidence[j]) if tracked.confidence is not None else 0.5

                # Find team label from original detection
                team = "unknown"
                for p in players:
                    if p["bbox"] == bbox or _bbox_iou(p["bbox"], bbox) > 0.5:
                        team = p.get("team") or "unknown"
                        break

                all_tracks.append({
                    "frame_idx": frame_idx,
                    "t_seconds": round(frame_idx / fps, 3),
                    "player_id": int(global_id),
                    "team": team,
                    "bbox": bbox,
                    "confidence": round(conf, 3),
                })

        prev_frame = frame.copy()

        if progress_callback and (i + 1) % 200 == 0:
            progress_callback(int(100 * (i + 1) / total))

    cap.release()

    # Write tracks
    with open(tracks_path, "w") as f:
        for track in all_tracks:
            f.write(json.dumps(track) + "\n")

    if progress_callback:
        progress_callback(100)

    return tracks_path


def _bbox_iou(box1: list[int], box2: list[int]) -> float:
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / max(union, 1e-6)
