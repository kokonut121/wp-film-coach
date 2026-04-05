# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all unit tests (no GPU/network required)
/Users/kokon/Library/Python/3.9/bin/pytest tests/test_events.py tests/test_agent.py tests/test_homography.py -v

# Run a single test
/Users/kokon/Library/Python/3.9/bin/pytest tests/test_events.py::TestTurnoverDetection::test_clear_possession_switch -v

# Run integration tests (requires network)
/Users/kokon/Library/Python/3.9/bin/pytest -m integration -v

# Deploy to Modal
modal deploy app.py

# Run locally (serves FastAPI on localhost)
modal serve app.py
```

## Architecture

The system is a serverless CV pipeline that processes YouTube water polo footage into tactical analysis. Data flows one-way through sequential modules, each writing a file to a Modal Volume:

```
YouTube URL → download.py → detect.py → track.py → homography.py → events.py → agent.py
               game.mp4   detections.jsonl  tracks.jsonl  positions.jsonl  events.json  report (in events.json)
```

**`app.py`** is the Modal entrypoint. It defines two things:
1. A `@modal.asgi_app()` wrapping FastAPI — this is the HTTP server clients talk to.
2. A `@app.function(gpu="T4")` called `run_pipeline` — this is the GPU worker that runs the full CV chain. The FastAPI `/process` endpoint calls `run_pipeline.spawn()` to kick it off asynchronously.

Both share two Modal Volumes: `wp-results` (job outputs) and `wp-model-cache` (YOLO weights).

**Progress tracking** uses a `progress.json` file written to the job directory after each stage. The frontend polls `GET /status/{job_id}` which reads this file.

**`pipeline/detect.py`** does two passes over the video: first pass collects all player cap-region HSV histograms, second pass (after K-means clustering) labels each player with `team_a`/`team_b`/`goalie`. This means team labels are only available after the full video is processed.

**`pipeline/track.py`** handles two levels of ID consistency:
- *Within a shot*: ByteTrack via the `supervision` library
- *Across camera cuts*: ResNet18 embeddings + cosine similarity matching. Scene cuts are detected by histogram correlation drop below 0.3.

**`pipeline/homography.py`** maps pixel coordinates → pool metres using Canny + HoughLinesP to find pool lane lines, then matches detected line intersections to known FINA template coordinates (25m × 13m). When fewer than 4 keypoints are found, it reuses the last valid H matrix with a `h_stale` flag.

**`pipeline/events.py`** produces the canonical `events.json` consumed by the frontend and agent. All rule-based detection (turnover, man-up, exclusion, counter-attack, press trigger) happens here, along with formation labelling (K-means vs 5 archetype templates) and all tactical metrics.

**`pipeline/agent.py`** never receives raw `positions` arrays — it receives a compact summary string from `summarize_events()` that represents events as one-line records and metrics as tables, targeting <8k tokens.

## Key data contracts

`detections.jsonl` — one JSON line per processed frame (every 3rd frame):
```json
{"frame_idx": 90, "t_seconds": 3.0, "players": [{"bbox": [x1,y1,x2,y2], "confidence": 0.9, "team": "team_a"}], "ball": {"bbox": [...], "source": "yolo|hsv|null"}}
```

`positions.jsonl` — one entry per tracked player per frame:
```json
{"t_seconds": 3.0, "player_id": 4, "team": "team_a", "x_metres": 12.5, "y_metres": 6.5, "h_stale": false}
```

`events.json` — final output with top-level keys: `meta`, `positions`, `events`, `formations`, `metrics`, `report`.

## Modal-specific notes

- `results_vol.commit()` must be called after writing files inside the GPU function, otherwise changes aren't visible to the web function.
- `results_vol.reload()` must be called in the web function before reading job files.
- The `RESULTS_DIR = "/results"` and `MODELS_DIR = "/models"` constants define volume mount points used throughout all pipeline modules.
