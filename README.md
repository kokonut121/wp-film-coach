# wp-film-coach

Computer-vision-driven water polo analysis. Submit a YouTube link or upload a video, run a Modal-hosted pipeline, and review the result in a React dashboard with a tactical map, event timeline, metrics, and Claude chat/reporting.

## Current stack

- Backend: FastAPI served from Modal
- Async pipeline worker: Modal GPU function on `T4`
- CV pipeline: YOLOv8 detection, ByteTrack + ResNet18 re-ID, classical CV homography, rule-based event classification
- LLM layer: Anthropic Claude Sonnet 4.6
- Frontend: React 18 + Vite + D3 + react-markdown
- Local YouTube bridge: FastAPI proxy in `local_proxy.py`

## Repository layout

```text
app.py                 Modal app, FastAPI API, upload endpoints, pipeline orchestration
local_proxy.py         Local helper that downloads YouTube videos then uploads them to Modal
pipeline/
  download.py          YouTube download + local video probing
  detect.py            YOLO player/ball detection + HSV fallback + cap-color clustering
  track.py             ByteTrack tracking + ResNet18 cross-cut re-identification
  homography.py        Pixel-to-pool mapping
  events.py            Event detection, formations, metrics, final events.json
  agent.py             Claude summary/report/chat layer
frontend/              Vite app for submission, processing, results, and chat
tests/                 Backend and pipeline tests
```

## Setup

Backend:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Frontend:

```bash
cd frontend
npm install
```

## Environment

- `ANTHROPIC_API_KEY`: required for report generation and chat in Modal
- `YOUTUBE_COOKIES`: optional cookie text used by `pipeline/download.py` inside Modal
- `WP_API_URL`: optional override for `local_proxy.py`; defaults to the deployed Modal URL
- `VITE_API_URL`: optional frontend override for the FastAPI base URL; defaults to `http://localhost:8000`

For deployed Modal runs, `ANTHROPIC_API_KEY` and optional YouTube cookies are expected as Modal secrets.

## Run locally

Backend API via Modal:

```bash
source .venv/bin/activate
modal serve app.py
```

Frontend:

```bash
cd frontend
npm run dev
```

Optional local YouTube proxy:

```bash
source .venv/bin/activate
python local_proxy.py
```

Notes:

- File uploads go straight to the FastAPI backend.
- YouTube URL submissions from the frontend go through `local_proxy.py` on `127.0.0.1:8001` to avoid remote download/IP issues, then get forwarded to Modal via `/process-upload`.

## Deploy

```bash
source .venv/bin/activate
modal deploy app.py
```

The frontend is a standalone Vite app in `frontend/` and can be deployed separately, for example with Vercel.

## API surface

- `POST /process`: queue a YouTube-based job
- `POST /uploads/init`: begin chunked upload session
- `PUT /uploads/{job_id}/chunk`: append an upload chunk
- `POST /uploads/{job_id}/complete`: finalize upload and spawn the pipeline
- `POST /process-upload`: one-shot multipart upload path used by `local_proxy.py`
- `GET /status/{job_id}`: read `progress.json`
- `GET /results/{job_id}`: fetch final `events.json`
- `POST /chat`: stream Claude chat as SSE
- `DELETE /jobs/{job_id}`: delete a job directory from the results volume

## Pipeline flow

```text
YouTube URL or uploaded video
  -> download.py / probe_video()
  -> detect.py
  -> track.py
  -> homography.py
  -> events.py
  -> agent.py
```

Artifacts written under `/results/<job_id>/`:

- `game.mp4`
- `progress.json`
- `upload.json` for chunked uploads
- `detections.jsonl`
- `tracks.jsonl`
- `positions.jsonl`
- `events.json`

## Data contracts

`detections.jsonl`

```json
{"frame_idx": 90, "t_seconds": 3.0, "players": [{"bbox": [1, 2, 3, 4], "confidence": 0.9, "team": "team_a"}], "ball": {"bbox": [1, 2, 3, 4], "confidence": 0.6, "source": "yolo"}}
```

`tracks.jsonl`

```json
{"frame_idx": 90, "t_seconds": 3.0, "player_id": 4, "team": "team_a", "bbox": [1, 2, 3, 4], "confidence": 0.91}
```

`positions.jsonl`

```json
{"frame_idx": 90, "t_seconds": 3.0, "player_id": 4, "team": "team_a", "x_metres": 12.5, "y_metres": 6.5, "h_stale": false}
```

`events.json`

- Top-level keys: `meta`, `positions`, `events`, `formations`, `metrics`, `report`

## Frontend behavior

- Home screen supports either file upload or YouTube URL mode
- Processing screen polls `/status/{job_id}` and can show debug counters
- Results screen renders a pool map, event timeline, timeline scrubber, metrics, and AI chat
- Local game history is stored client-side and used to reopen prior jobs

## Testing

Current fast local suite:

```bash
source .venv/bin/activate
pytest tests/test_events.py tests/test_agent.py tests/test_homography.py tests/test_uploads.py -v
```

Single test example:

```bash
source .venv/bin/activate
pytest tests/test_events.py::TestTurnoverDetection::test_clear_possession_switch -v
```

Download integration tests:

```bash
source .venv/bin/activate
pytest -m integration -v
```

## Modal notes

- `results_vol.commit()` is required after writes inside Modal functions
- `results_vol.reload()` is required before reads from the web app
- Shared volume mount points are `RESULTS_DIR = "/results"` and `MODELS_DIR = "/models"`
- `run_pipeline.spawn(...)` is the async boundary between the web app and the GPU worker
