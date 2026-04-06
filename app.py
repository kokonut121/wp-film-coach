from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Mapping
from typing import Literal

import modal
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from pipeline.pool_geometry import CALIBRATION_LINE_KEYS, validate_calibration_lines

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
HomographyMode = Literal["auto", "manual"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _job_dir(job_id: str) -> str:
    return os.path.join(RESULTS_DIR, job_id)


def _upload_manifest_path(job_dir: str) -> str:
    return os.path.join(job_dir, "upload.json")


def _calibration_path(job_dir: str) -> str:
    return os.path.join(job_dir, "calibration.json")


def _calibration_frame_path(job_dir: str) -> str:
    return os.path.join(job_dir, "calibration_frame.jpg")


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


def _extract_calibration_frame(video_path: str, output_path: str) -> None:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video for calibration frame extraction")

    total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    target_frame = max(total_frames // 2, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Could not extract calibration frame from uploaded video")

    if not cv2.imwrite(output_path, frame):
        raise RuntimeError("Failed to write calibration frame")


def _deep_merge(base: dict, updates: Mapping) -> dict:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def update_progress(
    job_dir: str,
    stage: str,
    pct: int,
    error_message: str | None = None,
    debug_data: dict | None = None,
):
    os.makedirs(job_dir, exist_ok=True)
    progress_path = os.path.join(job_dir, "progress.json")
    data = {}
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            data = json.load(f)
    data.update({"stage": stage, "pct": pct})
    if error_message:
        data["error_message"] = error_message
    if debug_data:
        _deep_merge(data, {"debug": debug_data})
    with open(progress_path, "w") as f:
        json.dump(data, f)
    results_vol.commit()


def _summarize_detection(det_path: str) -> dict:
    frames = 0
    frames_with_players = 0
    total_players = 0
    frames_with_ball = 0
    for line in open(det_path):
        row = json.loads(line)
        frames += 1
        players = row.get("players", [])
        total_players += len(players)
        if players:
            frames_with_players += 1
        if row.get("ball"):
            frames_with_ball += 1
    return {
        "frames_processed": frames,
        "frames_with_players": frames_with_players,
        "total_player_detections": total_players,
        "frames_with_ball": frames_with_ball,
    }


def _summarize_tracks(tracks_path: str) -> dict:
    rows = 0
    player_ids = set()
    teams = set()
    for line in open(tracks_path):
        if not line.strip():
            continue
        row = json.loads(line)
        rows += 1
        player_ids.add(row["player_id"])
        if row.get("team"):
            teams.add(row["team"])
    return {
        "track_rows": rows,
        "unique_players": len(player_ids),
        "teams_seen": sorted(teams),
    }


def _summarize_positions(positions_path: str) -> dict:
    rows = 0
    stale_rows = 0
    frames = set()
    player_ids = set()
    for line in open(positions_path):
        if not line.strip():
            continue
        row = json.loads(line)
        rows += 1
        frames.add(row["frame_idx"])
        player_ids.add(row["player_id"])
        if row.get("h_stale"):
            stale_rows += 1
    return {
        "position_rows": rows,
        "mapped_frames": len(frames),
        "unique_players": len(player_ids),
        "stale_rows": stale_rows,
    }


def _summarize_events(events_path: str) -> dict:
    with open(events_path) as f:
        data = json.load(f)
    return {
        "events": len(data.get("events", [])),
        "formations": len(data.get("formations", [])),
        "positions": len(data.get("positions", [])),
        "heatmap_players": len(data.get("metrics", {}).get("heatmaps", {})),
        "possession_periods": len(data.get("metrics", {}).get("possession", {})),
    }


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
def run_pipeline(
    job_id: str,
    youtube_url: str | None = None,
    label: str | None = None,
    debug: bool = False,
    homography_mode: HomographyMode = "auto",
):
    from pipeline.download import download_video, probe_video
    from pipeline.detect import run_detection
    from pipeline.track import run_tracking
    from pipeline.homography import run_homography
    from pipeline.manual_homography import run_manual_homography
    from pipeline.events import run_event_classification
    from pipeline.agent import summarize_events, generate_report

    job_dir = _job_dir(job_id)
    calibration_path = _calibration_path(job_dir)

    try:
        video_path = os.path.join(job_dir, "game.mp4")
        if youtube_url:
            # --- Download ---
            update_progress(
                job_dir,
                "downloading",
                0,
                debug_data={
                    "enabled": debug,
                    "current_stage": "downloading",
                    "stages": {
                        "download": {
                            "status": "running",
                            "source": "youtube",
                        }
                    },
                } if debug else None,
            )
            meta = download_video(youtube_url, job_dir)
            update_progress(
                job_dir,
                "downloading",
                100,
                debug_data={
                    "enabled": True,
                    "current_stage": "downloading",
                    "stages": {
                        "download": {
                            "status": "done",
                            "source": "youtube",
                            "fps": round(meta.get("fps", 0), 2),
                            "duration_s": round(meta.get("duration_s", 0), 2),
                            "video_path": os.path.basename(meta.get("path", "")),
                            "homography_mode": homography_mode,
                        }
                    },
                } if debug else None,
            )
        else:
            # Video was pre-uploaded — just probe metadata
            update_progress(
                job_dir,
                "downloading",
                50,
                debug_data={
                    "enabled": debug,
                    "current_stage": "downloading",
                    "stages": {"download": {"status": "probing", "source": "upload"}},
                } if debug else None,
            )
            meta = probe_video(video_path)
            update_progress(
                job_dir,
                "downloading",
                100,
                debug_data={
                    "enabled": True,
                    "current_stage": "downloading",
                    "stages": {
                        "download": {
                            "status": "done",
                            "source": "upload",
                            "fps": round(meta.get("fps", 0), 2),
                            "duration_s": round(meta.get("duration_s", 0), 2),
                            "video_path": os.path.basename(meta.get("path", "")),
                            "homography_mode": homography_mode,
                        }
                    },
                } if debug else None,
            )

        # --- Detection ---
        def det_progress(pct):
            update_progress(
                job_dir,
                "detecting",
                pct,
                debug_data={
                    "enabled": debug,
                    "current_stage": "detecting",
                    "stages": {"detection": {"status": "running"}},
                } if debug else None,
            )

        update_progress(job_dir, "detecting", 0, debug_data={
            "enabled": debug,
            "current_stage": "detecting",
            "stages": {"detection": {"status": "running"}},
        } if debug else None)
        det_path = run_detection(
            meta["path"],
            job_dir,
            progress_callback=det_progress,
            homography_mode=homography_mode,
            calibration_path=calibration_path if homography_mode == "manual" else None,
        )
        if debug:
            update_progress(
                job_dir,
                "detecting",
                100,
                debug_data={
                    "enabled": True,
                    "current_stage": "detecting",
                    "stages": {
                        "detection": {
                            "status": "done",
                            **_summarize_detection(det_path),
                        }
                    },
                },
            )

        # --- Tracking ---
        def track_progress(pct):
            update_progress(
                job_dir,
                "tracking",
                pct,
                debug_data={
                    "enabled": debug,
                    "current_stage": "tracking",
                    "stages": {"tracking": {"status": "running"}},
                } if debug else None,
            )

        update_progress(job_dir, "tracking", 0, debug_data={
            "enabled": debug,
            "current_stage": "tracking",
            "stages": {"tracking": {"status": "running"}},
        } if debug else None)
        tracks_path = run_tracking(det_path, meta["path"], job_dir, progress_callback=track_progress)
        if debug:
            update_progress(
                job_dir,
                "tracking",
                100,
                debug_data={
                    "enabled": True,
                    "current_stage": "tracking",
                    "stages": {
                        "tracking": {
                            "status": "done",
                            **_summarize_tracks(tracks_path),
                        }
                    },
                },
            )

        # --- Homography ---
        def homo_progress(pct):
            update_progress(
                job_dir,
                "homography",
                pct,
                debug_data={
                    "enabled": debug,
                    "current_stage": "homography",
                    "stages": {"homography": {"status": "running"}},
                } if debug else None,
            )

        update_progress(job_dir, "homography", 0, debug_data={
            "enabled": debug,
            "current_stage": "homography",
            "stages": {"homography": {"status": "running"}},
        } if debug else None)
        if homography_mode == "manual":
            pos_path = run_manual_homography(
                tracks_path,
                calibration_path,
                job_dir,
                progress_callback=homo_progress,
            )
        else:
            pos_path = run_homography(
                tracks_path, meta["path"], job_dir, meta["fps"], progress_callback=homo_progress
            )
        if debug:
            update_progress(
                job_dir,
                "homography",
                100,
                debug_data={
                    "enabled": True,
                    "current_stage": "homography",
                    "stages": {
                        "homography": {
                            "status": "done",
                            **_summarize_positions(pos_path),
                        }
                    },
                },
            )

        # --- Event classification ---
        update_progress(job_dir, "classifying", 0, debug_data={
            "enabled": debug,
            "current_stage": "classifying",
            "stages": {"classification": {"status": "running"}},
        } if debug else None)
        events_path = run_event_classification(pos_path, tracks_path, job_dir, meta)
        update_progress(
            job_dir,
            "classifying",
            100,
            debug_data={
                "enabled": True,
                "current_stage": "classifying",
                "stages": {
                    "classification": {
                        "status": "done",
                        **_summarize_events(events_path),
                    }
                },
            } if debug else None,
        )

        # --- Report generation ---
        update_progress(job_dir, "generating_report", 0, debug_data={
            "enabled": debug,
            "current_stage": "generating_report",
            "stages": {"report": {"status": "running"}},
        } if debug else None)
        with open(events_path) as f:
            events_data = json.load(f)
        summary = summarize_events(events_data)
        try:
            report = generate_report(summary)
            report_debug = {"status": "done", "mode": "llm"}
        except Exception as e:
            # Keep the pipeline usable even if the LLM step is misconfigured.
            report = (
                "## Report unavailable\n"
                f"Automatic report generation failed: {e}\n\n"
                "The CV analysis completed successfully, but the Anthropic client "
                "could not generate the written report."
            )
            report_debug = {"status": "fallback", "error": str(e), "mode": "static"}
        events_data["report"] = report
        if label:
            events_data["meta"]["label"] = label
        with open(events_path, "w") as f:
            json.dump(events_data, f)
        results_vol.commit()
        update_progress(
            job_dir,
            "done",
            100,
            debug_data={
                "enabled": True,
                "current_stage": "done",
                "stages": {"report": report_debug},
            } if debug else None,
        )

    except Exception as e:
        update_progress(
            job_dir,
            "error",
            0,
            error_message=str(e),
            debug_data={
                "enabled": debug,
                "current_stage": "error",
                "stages": {
                    "failure": {
                        "status": "error",
                        "message": str(e),
                    }
                },
            } if debug else None,
        )
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
    debug: bool = False
    homography_mode: HomographyMode = "auto"


class ChatRequest(BaseModel):
    job_id: str
    messages: list[dict]


class UploadInitRequest(BaseModel):
    filename: str
    label: str | None = None
    content_type: str | None = None
    total_size: int | None = None
    debug: bool = False
    homography_mode: HomographyMode = "auto"


class CalibrationLineRequest(BaseModel):
    key: str
    x1: float
    y1: float
    x2: float
    y2: float


class CalibrationSubmitRequest(BaseModel):
    lines: list[CalibrationLineRequest]


@web_app.post("/process")
async def process_video(req: ProcessRequest):
    if req.homography_mode == "manual":
        raise HTTPException(400, "Manual homography is only supported for uploaded videos")
    job_id = uuid.uuid4().hex
    update_progress(
        _job_dir(job_id),
        "queued",
        0,
        debug_data={
            "enabled": req.debug,
            "current_stage": "queued",
            "input": {"homography_mode": req.homography_mode},
        },
    )
    run_pipeline.spawn(job_id, req.youtube_url, req.label, req.debug, req.homography_mode)
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
        "debug": req.debug,
        "homography_mode": req.homography_mode,
    }
    _write_json(_upload_manifest_path(job_dir), manifest)
    with open(os.path.join(job_dir, "game.mp4"), "wb"):
        pass
    results_vol.commit()
    update_progress(
        job_dir,
        "uploading",
        0,
        debug_data={
            "enabled": req.debug,
            "current_stage": "uploading",
            "input": {
                "filename": req.filename,
                "homography_mode": req.homography_mode,
                "total_size_mb": round((req.total_size or 0) / (1024 * 1024), 2),
            },
        },
    )
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
    meta_path = _upload_manifest_path(job_dir)

    results_vol.reload()
    if not os.path.exists(upload_path) or not os.path.exists(meta_path):
        raise HTTPException(404, "Upload session not found")

    body = await request.body()
    if not body:
        raise HTTPException(400, "Chunk body is empty")

    manifest = _read_json(meta_path)

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

    _write_json(meta_path, manifest)

    if manifest.get("debug"):
        update_progress(
            job_dir,
            "uploading",
            int(100 * manifest["received_bytes"] / max(manifest.get("total_size") or 1, 1)),
            debug_data={
                "enabled": True,
                "current_stage": "uploading",
                "input": {
                    "filename": manifest.get("filename"),
                    "homography_mode": manifest.get("homography_mode", "auto"),
                    "received_mb": round(manifest["received_bytes"] / (1024 * 1024), 2),
                    "total_size_mb": round((manifest.get("total_size") or 0) / (1024 * 1024), 2),
                    "chunks_uploaded": index + 1,
                    "total_chunks": total_chunks,
                },
            },
        )
    results_vol.commit()
    return {
        "ok": True,
        "received_bytes": manifest["received_bytes"],
    }


@web_app.post("/uploads/{job_id}/complete")
async def complete_upload(job_id: str):
    job_dir = _job_dir(job_id)
    upload_path = os.path.join(job_dir, "game.mp4")
    meta_path = _upload_manifest_path(job_dir)

    results_vol.reload()
    if not os.path.exists(upload_path) or not os.path.exists(meta_path):
        raise HTTPException(404, "Upload session not found")

    manifest = _read_json(meta_path)

    total_size = manifest.get("total_size")
    received_bytes = manifest.get("received_bytes", 0)
    if total_size is not None and received_bytes != total_size:
        raise HTTPException(
            400,
            f"Upload incomplete: received {received_bytes} of {total_size} bytes",
        )

    manifest["status"] = "uploaded"
    _write_json(meta_path, manifest)

    results_vol.commit()
    homography_mode = manifest.get("homography_mode", "auto")
    if homography_mode == "manual":
        frame_path = _calibration_frame_path(job_dir)
        _extract_calibration_frame(upload_path, frame_path)
        update_progress(
            job_dir,
            "awaiting_calibration",
            0,
            debug_data={
                "enabled": bool(manifest.get("debug")),
                "current_stage": "awaiting_calibration",
                "input": {
                    "filename": manifest.get("filename"),
                    "homography_mode": homography_mode,
                    "total_size_mb": round((manifest.get("total_size") or 0) / (1024 * 1024), 2),
                },
                "calibration": {
                    "status": "required",
                    "lines_required": len(CALIBRATION_LINE_KEYS),
                    "lines_completed": 0,
                    "frame_ready": True,
                },
            },
        )
        return {"job_id": job_id, "needs_calibration": True}

    update_progress(
        job_dir,
        "queued",
        0,
        debug_data={
            "enabled": bool(manifest.get("debug")),
            "current_stage": "queued",
            "input": {
                "filename": manifest.get("filename"),
                "homography_mode": homography_mode,
                "total_size_mb": round((manifest.get("total_size") or 0) / (1024 * 1024), 2),
            },
        },
    )
    run_pipeline.spawn(
        job_id,
        None,
        manifest.get("label"),
        bool(manifest.get("debug")),
        homography_mode,
    )
    return {"job_id": job_id}


@web_app.get("/uploads/{job_id}/calibration-frame")
async def get_calibration_frame(job_id: str):
    frame_path = _calibration_frame_path(_job_dir(job_id))
    results_vol.reload()
    if not os.path.exists(frame_path):
        raise HTTPException(404, "Calibration frame not found")
    return FileResponse(frame_path, media_type="image/jpeg")


@web_app.post("/uploads/{job_id}/calibration")
async def submit_calibration(job_id: str, req: CalibrationSubmitRequest):
    job_dir = _job_dir(job_id)
    meta_path = _upload_manifest_path(job_dir)
    frame_path = _calibration_frame_path(job_dir)

    results_vol.reload()
    if not os.path.exists(meta_path) or not os.path.exists(frame_path):
        raise HTTPException(404, "Calibration session not found")

    manifest = _read_json(meta_path)
    if manifest.get("homography_mode") != "manual":
        raise HTTPException(400, "Calibration is only valid for manual homography jobs")

    lines = [line.model_dump() for line in req.lines]
    try:
        validate_calibration_lines(lines)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    calibration = {
        "lines": lines,
        "line_keys": CALIBRATION_LINE_KEYS,
    }
    _write_json(_calibration_path(job_dir), calibration)
    results_vol.commit()

    update_progress(
        job_dir,
        "queued",
        0,
        debug_data={
            "enabled": bool(manifest.get("debug")),
            "current_stage": "queued",
            "input": {
                "filename": manifest.get("filename"),
                "homography_mode": "manual",
                "total_size_mb": round((manifest.get("total_size") or 0) / (1024 * 1024), 2),
            },
            "calibration": {
                "status": "complete",
                "lines_required": len(CALIBRATION_LINE_KEYS),
                "lines_completed": len(lines),
                "frame_ready": True,
            },
        },
    )
    run_pipeline.spawn(job_id, None, manifest.get("label"), bool(manifest.get("debug")), "manual")
    return {"job_id": job_id}


@web_app.post("/process-upload")
async def process_upload(
    file: UploadFile = File(...),
    label: str | None = Form(None),
    debug: bool = Form(False),
    homography_mode: HomographyMode = Form("auto"),
):
    if homography_mode != "auto":
        raise HTTPException(400, "Manual homography is only supported through the chunked upload flow")
    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    os.makedirs(job_dir, exist_ok=True)
    update_progress(
        job_dir,
        "uploading",
        0,
        debug_data={"enabled": debug, "current_stage": "uploading"},
    )
    video_path = os.path.join(job_dir, "game.mp4")
    with open(video_path, "wb") as f:
        while chunk := await file.read(8 * 1024 * 1024):
            f.write(chunk)
    results_vol.commit()
    run_pipeline.spawn(job_id, None, label, debug, homography_mode)
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
