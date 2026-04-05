from __future__ import annotations

import json
import os
import shutil
import uuid

import modal
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------

app = modal.App("wp-film-coach")

results_vol = modal.Volume.from_name("wp-results", create_if_missing=True)
model_vol = modal.Volume.from_name("wp-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands(
        "apt-get update -qq && apt-get install -y --no-install-recommends "
        "libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg nodejs && "
        "rm -rf /var/lib/apt/lists/*"
    )
    .pip_install(
        "fastapi>=0.115.0",
        "python-multipart>=0.0.9",
        "uvicorn>=0.30.0",
        "ultralytics>=8.2.0",
        "supervision>=0.22.0",
        "opencv-python-headless>=4.10.0",
        "yt-dlp>=2025.3.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "scikit-learn>=1.4.0",
        "anthropic>=0.39.0",
    )
    .add_local_python_source("pipeline")
)

RESULTS_DIR = "/results"
MODELS_DIR = "/models"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _job_dir(job_id: str) -> str:
    return os.path.join(RESULTS_DIR, job_id)


def update_progress(job_dir: str, stage: str, pct: int, error_message: str | None = None):
    os.makedirs(job_dir, exist_ok=True)
    data = {"stage": stage, "pct": pct}
    if error_message:
        data["error_message"] = error_message
    with open(os.path.join(job_dir, "progress.json"), "w") as f:
        json.dump(data, f)
    results_vol.commit()


# ---------------------------------------------------------------------------
# GPU pipeline function
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    volumes={RESULTS_DIR: results_vol, MODELS_DIR: model_vol},
    secrets=[
        modal.Secret.from_name("youtube-cookies"),
        modal.Secret.from_name("anthropic-api-key"),
    ],
)
def run_pipeline(job_id: str, youtube_url: str | None = None, label: str | None = None):
    from pipeline.download import download_video, probe_video
    from pipeline.detect import run_detection
    from pipeline.track import run_tracking
    from pipeline.homography import run_homography
    from pipeline.events import run_event_classification
    from pipeline.agent import summarize_events, generate_report

    job_dir = _job_dir(job_id)

    try:
        video_path = os.path.join(job_dir, "game.mp4")
        if youtube_url:
            # --- Download ---
            update_progress(job_dir, "downloading", 0)
            meta = download_video(youtube_url, job_dir)
            update_progress(job_dir, "downloading", 100)
        else:
            # Video was pre-uploaded — just probe metadata
            update_progress(job_dir, "downloading", 50)
            meta = probe_video(video_path)
            update_progress(job_dir, "downloading", 100)

        # --- Detection ---
        def det_progress(pct):
            update_progress(job_dir, "detecting", pct)

        update_progress(job_dir, "detecting", 0)
        det_path = run_detection(meta["path"], job_dir, progress_callback=det_progress)

        # --- Tracking ---
        def track_progress(pct):
            update_progress(job_dir, "tracking", pct)

        update_progress(job_dir, "tracking", 0)
        tracks_path = run_tracking(det_path, meta["path"], job_dir, progress_callback=track_progress)

        # --- Homography ---
        def homo_progress(pct):
            update_progress(job_dir, "homography", pct)

        update_progress(job_dir, "homography", 0)
        pos_path = run_homography(
            tracks_path, meta["path"], job_dir, meta["fps"], progress_callback=homo_progress
        )

        # --- Event classification ---
        update_progress(job_dir, "classifying", 0)
        events_path = run_event_classification(pos_path, tracks_path, job_dir, meta)
        update_progress(job_dir, "classifying", 100)

        # --- Report generation ---
        update_progress(job_dir, "generating_report", 0)
        with open(events_path) as f:
            events_data = json.load(f)
        summary = summarize_events(events_data)
        try:
            report = generate_report(summary)
        except Exception as e:
            # Keep the pipeline usable even if the LLM step is misconfigured.
            report = (
                "## Report unavailable\n"
                f"Automatic report generation failed: {e}\n\n"
                "The CV analysis completed successfully, but the Anthropic client "
                "could not generate the written report."
            )
        events_data["report"] = report
        if label:
            events_data["meta"]["label"] = label
        with open(events_path, "w") as f:
            json.dump(events_data, f)
        results_vol.commit()
        update_progress(job_dir, "done", 100)

    except Exception as e:
        update_progress(job_dir, "error", 0, error_message=str(e))
        raise


# ---------------------------------------------------------------------------
# FastAPI web app
# ---------------------------------------------------------------------------

web_app = FastAPI(title="WP Film Coach")
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    youtube_url: str
    label: str | None = None


class ChatRequest(BaseModel):
    job_id: str
    messages: list[dict]


class UploadInitRequest(BaseModel):
    filename: str
    label: str | None = None
    content_type: str | None = None
    total_size: int | None = None


@web_app.post("/process")
async def process_video(req: ProcessRequest):
    job_id = uuid.uuid4().hex
    run_pipeline.spawn(job_id, req.youtube_url, req.label)
    return {"job_id": job_id}


@web_app.post("/uploads/init")
async def init_upload(req: UploadInitRequest):
    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    os.makedirs(job_dir, exist_ok=True)

    manifest = {
        "filename": req.filename,
        "label": req.label,
        "content_type": req.content_type,
        "total_size": req.total_size,
        "received_bytes": 0,
        "status": "uploading",
    }
    with open(os.path.join(job_dir, "upload.json"), "w") as f:
        json.dump(manifest, f)
    with open(os.path.join(job_dir, "game.mp4"), "wb"):
        pass
    results_vol.commit()
    return {"job_id": job_id}


@web_app.put("/uploads/{job_id}/chunk")
async def upload_chunk(
    job_id: str,
    request: Request,
    index: int,
    total_chunks: int,
    start_byte: int,
):
    job_dir = _job_dir(job_id)
    upload_path = os.path.join(job_dir, "game.mp4")
    meta_path = os.path.join(job_dir, "upload.json")

    results_vol.reload()
    if not os.path.exists(upload_path) or not os.path.exists(meta_path):
        raise HTTPException(404, "Upload session not found")

    body = await request.body()
    if not body:
        raise HTTPException(400, "Chunk body is empty")

    with open(meta_path) as f:
        manifest = json.load(f)

    expected_start = manifest.get("received_bytes", 0)
    if start_byte != expected_start:
        raise HTTPException(
            409,
            f"Unexpected chunk offset {start_byte}; expected {expected_start}",
        )

    with open(upload_path, "ab") as f:
        f.write(body)

    manifest["received_bytes"] = expected_start + len(body)
    manifest["last_chunk_index"] = index
    manifest["total_chunks"] = total_chunks

    with open(meta_path, "w") as f:
        json.dump(manifest, f)

    results_vol.commit()
    return {
        "ok": True,
        "received_bytes": manifest["received_bytes"],
    }


@web_app.post("/uploads/{job_id}/complete")
async def complete_upload(job_id: str):
    job_dir = _job_dir(job_id)
    upload_path = os.path.join(job_dir, "game.mp4")
    meta_path = os.path.join(job_dir, "upload.json")

    results_vol.reload()
    if not os.path.exists(upload_path) or not os.path.exists(meta_path):
        raise HTTPException(404, "Upload session not found")

    with open(meta_path) as f:
        manifest = json.load(f)

    total_size = manifest.get("total_size")
    received_bytes = manifest.get("received_bytes", 0)
    if total_size is not None and received_bytes != total_size:
        raise HTTPException(
            400,
            f"Upload incomplete: received {received_bytes} of {total_size} bytes",
        )

    manifest["status"] = "uploaded"
    with open(meta_path, "w") as f:
        json.dump(manifest, f)

    results_vol.commit()
    run_pipeline.spawn(job_id, None, manifest.get("label"))
    return {"job_id": job_id}


@web_app.post("/process-upload")
async def process_upload(file: UploadFile = File(...), label: str | None = Form(None)):
    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    os.makedirs(job_dir, exist_ok=True)
    video_path = os.path.join(job_dir, "game.mp4")
    with open(video_path, "wb") as f:
        while chunk := await file.read(8 * 1024 * 1024):
            f.write(chunk)
    results_vol.commit()
    run_pipeline.spawn(job_id, None, label)
    return {"job_id": job_id}


@web_app.get("/status/{job_id}")
async def get_status(job_id: str):
    progress_path = os.path.join(_job_dir(job_id), "progress.json")
    results_vol.reload()
    if not os.path.exists(progress_path):
        return {"stage": "queued", "pct": 0}
    with open(progress_path) as f:
        return json.load(f)


@web_app.get("/results/{job_id}")
async def get_results(job_id: str):
    events_path = os.path.join(_job_dir(job_id), "events.json")
    results_vol.reload()
    if not os.path.exists(events_path):
        raise HTTPException(404, "Results not ready or job not found")
    with open(events_path) as f:
        return json.load(f)


@web_app.post("/chat")
async def chat(req: ChatRequest):
    from pipeline.agent import summarize_events, stream_chat

    events_path = os.path.join(_job_dir(req.job_id), "events.json")
    results_vol.reload()
    if not os.path.exists(events_path):
        raise HTTPException(404, "Results not ready or job not found")
    with open(events_path) as f:
        events_data = json.load(f)
    summary = summarize_events(events_data)

    def event_stream():
        for chunk in stream_chat(summary, req.messages):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@web_app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    job_dir = _job_dir(job_id)
    results_vol.reload()
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
        results_vol.commit()
    return {"deleted": True}


@app.function(image=image, volumes={RESULTS_DIR: results_vol})
@modal.asgi_app()
def fastapi_app():
    return web_app
