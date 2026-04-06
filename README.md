# wp-film-coach

A serverless computer-vision pipeline for water polo tactical analysis. Upload a match video and receive a structured breakdown of player movements, possession phases, formation transitions, and set-piece events — delivered through a React dashboard with an interactive pool map, event timeline, and Claude-powered tactical report.

## What it does

The pipeline processes raw broadcast or handheld footage end-to-end:

1. **Detection** — YOLOv8 identifies players and the ball on every third frame. A secondary HSV-based detector fills in ball detections that YOLO misses. Cap-region HSV histograms are clustered via K-means into `team_a`, `team_b`, and `goalie` labels without any manual annotation.

2. **Tracking** — [ByteTrack](https://github.com/ifzhang/ByteTrack) maintains within-shot player identities. A pretrained ResNet18 produces appearance embeddings that are matched across scene cuts via cosine similarity, giving each player a stable ID across the full match regardless of camera switches or replays.

3. **Homography** — Two modes are supported:
   - *Auto*: Classical CV detects pool-water regions and lane-line geometry, then computes a perspective transform into a canonical 25 m × 13 m pool coordinate frame. Transforms are propagated between keyframes and flagged `h_stale` when fresh feature matches are unavailable.
   - *Manual*: The user draws nine reference lines (side walls, half, 2 m and 5 m lines) directly on a calibration frame. Intersection points are extracted and used to compute a single high-accuracy homography that covers the entire clip.

4. **Event classification** — Rule-based logic over the pool-coordinate track history detects turnovers (possession switches), man-up phases, player exclusions, counter-attacks, and press-like defensive contractions. Formation snapshots are generated at regular intervals by projecting player centroids onto the pool template.

5. **Metrics** — Per-team possession fractions, convex hull area over time, centroid spread, and spatial heatmaps are derived from the position stream and packed into the final `events.json`.

6. **LLM report** — The structured output is compressed into a concise text summary and passed to Claude Sonnet 4.6, which generates a tactical markdown report. A follow-up chat interface streams Claude responses grounded on the same summarized context.

## Architecture

```
Browser (React + Vite)
  └─ FastAPI (Modal @asgi_app)
       └─ run_pipeline() — Modal GPU function (T4)
            ├─ pipeline/detect.py     YOLOv8 + HSV + HSV cap clustering
            ├─ pipeline/track.py      ByteTrack + ResNet18 re-ID
            ├─ pipeline/homography.py Auto classical-CV homography
            ├─ pipeline/manual_homography.py  User-calibrated homography
            ├─ pipeline/pool_geometry.py      Shared geometry / ROI helpers
            ├─ pipeline/events.py     Event detection + metrics
            └─ pipeline/agent.py      Claude summary / report / chat
```

The web API is a Modal-hosted FastAPI app. All compute-heavy work is dispatched via `run_pipeline.spawn(...)`, keeping API responses fast while the GPU worker runs asynchronously on Modal infrastructure. Results are persisted to a shared Modal volume and polled by the frontend.

## Repository layout

```
app.py                 Modal app, FastAPI endpoints, pipeline orchestration
local_proxy.py         Local helper that downloads YouTube videos and forwards them to Modal
pipeline/
  download.py          YouTube download + local video probing (yt-dlp)
  detect.py            YOLOv8 player/ball detection + HSV fallback + cap-color clustering
  track.py             ByteTrack tracking + ResNet18 cross-cut re-identification
  homography.py        Classical CV pixel-to-pool mapping
  manual_homography.py User-calibrated pixel-to-pool mapping
  pool_geometry.py     Shared pool geometry, calibration, and pool-bound helpers
  events.py            Event detection, formations, metrics, final events.json
  agent.py             Claude summary/report/chat layer
frontend/              Vite + React 18 + D3 frontend
tests/                 Unit and integration tests
```

## Data contracts

### `detections.jsonl`
```json
{"frame_idx": 90, "t_seconds": 3.0, "players": [{"bbox": [1, 2, 3, 4], "confidence": 0.9, "team": "team_a"}], "ball": {"bbox": [1, 2, 3, 4], "confidence": 0.6, "source": "yolo"}}
```

### `tracks.jsonl`
```json
{"frame_idx": 90, "t_seconds": 3.0, "player_id": 4, "team": "team_a", "bbox": [1, 2, 3, 4], "confidence": 0.91}
```

### `positions.jsonl`
```json
{"frame_idx": 90, "t_seconds": 3.0, "player_id": 4, "team": "team_a", "x_metres": 12.5, "y_metres": 6.5, "h_stale": false}
```

### `events.json`
Top-level keys: `meta`, `positions`, `events`, `formations`, `metrics`, `report`

### `calibration.json`
Nine line segments keyed by:
`left_side`, `top_side`, `right_side`, `bottom_side`, `m2_left`, `m5_left`, `half`, `m5_right`, `m2_right`

## API surface

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/uploads/init` | Begin a chunked upload session |
| `PUT` | `/uploads/{job_id}/chunk` | Append an upload chunk |
| `POST` | `/uploads/{job_id}/complete` | Finalize upload and spawn the pipeline |
| `GET` | `/uploads/{job_id}/calibration-frame` | Fetch the extracted frame for manual calibration |
| `POST` | `/uploads/{job_id}/calibration` | Submit nine calibration lines and start the pipeline |
| `POST` | `/process-upload` | One-shot multipart upload (used by `local_proxy.py`) |
| `POST` | `/process` | Queue a YouTube-based job |
| `GET` | `/status/{job_id}` | Read `progress.json` |
| `GET` | `/results/{job_id}` | Fetch final `events.json` |
| `POST` | `/chat` | Stream Claude chat as SSE |
| `DELETE` | `/jobs/{job_id}` | Delete a job directory from the results volume |

## Pipeline flow

```
Uploaded video — auto homography
  → /uploads/init → /uploads/{job_id}/chunk → /uploads/{job_id}/complete
  → detect.py → track.py → homography.py → events.py → agent.py

Uploaded video — manual homography
  → /uploads/init → /uploads/{job_id}/chunk → /uploads/{job_id}/complete
  → awaiting_calibration
  → /uploads/{job_id}/calibration-frame  (user draws 9 reference lines)
  → /uploads/{job_id}/calibration
  → detect.py (pool ROI constrained by calibration polygon)
  → track.py → manual_homography.py → events.py → agent.py

YouTube URL (via local proxy)
  → local_proxy.py → /process-upload
  → detect.py → track.py → homography.py → events.py → agent.py
```

## Artifacts per job

All files are written under `/results/<job_id>/` in the Modal results volume:

| File | Description |
|------|-------------|
| `game.mp4` | Source video |
| `progress.json` | Live stage + percentage |
| `upload.json` | Chunked upload state |
| `calibration_frame.jpg` | Frame shown to user for manual calibration |
| `calibration.json` | Submitted calibration lines |
| `detections.jsonl` | Per-frame player and ball detections |
| `tracks.jsonl` | Per-frame tracked player identities |
| `positions.jsonl` | Per-frame pool-coordinate positions |
| `events.json` | Final structured analysis output |

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Powers report generation and chat |
| `YOUTUBE_COOKIES` | No | Cookie string for age-restricted YouTube downloads |
| `WP_API_URL` | No | Override the Modal backend URL used by `local_proxy.py` |
| `VITE_API_URL` | No | Override the API base URL in the frontend build |

In Modal, `ANTHROPIC_API_KEY` and YouTube cookies are injected as Modal secrets.

## Testing

```bash
# Core local suite
pytest tests/test_detect.py tests/test_events.py tests/test_agent.py \
       tests/test_homography.py tests/test_manual_homography.py tests/test_uploads.py -v

# Single test
pytest tests/test_events.py::TestTurnoverDetection::test_clear_possession_switch -v

# Integration tests (requires network)
pytest -m integration -v
```

## Modal notes

- `results_vol.commit()` is required after any write inside a Modal function
- `results_vol.reload()` is required before any read in the web app
- Volume mount points: `RESULTS_DIR = "/results"`, `MODELS_DIR = "/models"`
- `run_pipeline.spawn(...)` is the async boundary between the web app and the GPU worker
