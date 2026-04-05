"""Local proxy that downloads YouTube videos and uploads them to Modal.

Run:  python local_proxy.py
Keep it running while using the frontend. It listens on port 8001.
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

MODAL_API_URL = os.environ.get(
    "WP_API_URL",
    "https://patrickfengsr--wp-film-coach-fastapi-app.modal.run",
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    youtube_url: str
    label: str | None = None


@app.post("/process")
def process_video(req: ProcessRequest):
    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, "game.mp4")

    # Download locally with yt-dlp (use the one from our venv)
    yt_dlp_bin = shutil.which("yt-dlp") or os.path.join(os.path.dirname(sys.executable), "yt-dlp")
    result = subprocess.run(
        [
            yt_dlp_bin,
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
                  "bestvideo[height<=720]+bestaudio/"
                  "best[height<=720]/best",
            "--merge-output-format", "mp4",
            "-o", video_path,
            req.youtube_url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"error": f"yt-dlp failed: {result.stderr.strip()}"}

    # yt-dlp may write the file with a slightly different name
    if not os.path.exists(video_path):
        candidates = glob.glob(os.path.join(tmpdir, "game.*"))
        if candidates:
            video_path = candidates[0]
        else:
            return {"error": f"Download completed but no file found. yt-dlp output: {result.stdout[-500:]}"}

    # Upload to Modal
    with open(video_path, "rb") as f:
        files = {"file": ("game.mp4", f, "video/mp4")}
        data = {"label": req.label} if req.label else {}
        resp = requests.post(f"{MODAL_API_URL}/process-upload", files=files, data=data)

    os.remove(video_path)
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    if not resp.ok:
        return {"error": f"Upload failed: {resp.text}"}

    return resp.json()


if __name__ == "__main__":
    print(f"Local proxy running — downloads YouTube videos and uploads to Modal")
    print(f"Modal API: {MODAL_API_URL}")
    uvicorn.run(app, host="127.0.0.1", port=8001)
