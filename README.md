# wp-film-coach

Automated tactical analysis for water polo. Paste a YouTube URL, get back a bird's-eye pool map, event timeline, formation detection, and a Claude-powered chat interface to interrogate the game.

## Stack

- **Backend**: FastAPI on [Modal](https://modal.com) serverless GPU (T4)
- **CV pipeline**: YOLOv8 (ultralytics) → ByteTrack (supervision) → classical CV homography → rule-based event classifier
- **LLM**: Claude Sonnet 4.6 via Anthropic API
- **Frontend**: React + Vite + D3 (forthcoming), hosted on Vercel

## Quickstart

```bash
pip install -r requirements.txt
modal deploy app.py
```

Set `ANTHROPIC_API_KEY` in your environment or as a Modal secret.

## How it works

1. User submits a YouTube URL → `POST /process` returns a `job_id`
2. A Modal GPU function downloads the video and runs the CV pipeline
3. Frontend polls `GET /status/{job_id}` for progress
4. Results available at `GET /results/{job_id}` once `stage == "done"`
5. Chat via `POST /chat` streams Claude responses over SSE
