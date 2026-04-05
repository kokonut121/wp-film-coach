"""Video download module using yt-dlp."""

import os
import tempfile

import cv2
import yt_dlp


def probe_video(video_path: str) -> dict:
    """Extract metadata from an already-downloaded video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoUnavailableError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = frame_count / fps if fps else 0
    cap.release()
    return {
        "path": video_path,
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": duration_s,
    }


class VideoUnavailableError(Exception):
    """Raised when a YouTube video cannot be downloaded."""
    pass


def download_video(youtube_url: str, output_dir: str) -> dict:
    """Download a YouTube video and return metadata.

    Args:
        youtube_url: Full YouTube URL.
        output_dir: Directory to save the video file.

    Returns:
        dict with keys: path, fps, frame_count, duration_s

    Raises:
        VideoUnavailableError: If the video is private, deleted, age-restricted, etc.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "game.mp4")

    format_preferences = [
        # Preferred: 720p mp4+m4a
        (
            "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]"
            "/bestvideo[height<=720]+bestaudio"
            "/best[height<=720]"
        ),
        # Fallback: any resolution mp4
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio",
        # Last resort: any single pre-merged format
        "best",
    ]

    base_opts = {
        "outtmpl": output_path,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"player_js_runtime": ["nodejs"]}},
    }

    # If YouTube cookies are available (injected as Modal Secret), write them to
    # a temp file so yt-dlp can pass them to YouTube and bypass bot-detection.
    cookie_content = os.environ.get("YOUTUBE_COOKIES")
    cookie_file = None
    if cookie_content:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        tmp.write(cookie_content)
        tmp.flush()
        cookie_file = tmp.name

    def _try_download(use_cookies: bool) -> dict | None:
        """Attempt download with format fallbacks. Returns info dict or None."""
        for fmt in format_preferences:
            opts = {**base_opts, "format": fmt}
            if use_cookies and cookie_file:
                opts["cookiefile"] = cookie_file
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    return ydl.extract_info(youtube_url, download=True)
            except yt_dlp.utils.DownloadError:
                if os.path.exists(output_path):
                    os.remove(output_path)
                continue
        return None

    try:
        # Try with cookies first (helps with bot-detection), then without
        # (stale cookies can cause YouTube to return no video formats).
        info = _try_download(use_cookies=True) if cookie_file else None
        if info is None:
            info = _try_download(use_cookies=False)
        if info is None:
            raise VideoUnavailableError(
                "All format/cookie combinations failed. Video may be unavailable."
            )
    finally:
        if cookie_file and os.path.exists(cookie_file):
            os.unlink(cookie_file)

    if not os.path.exists(output_path):
        raise VideoUnavailableError("Download completed but output file not found")

    fps = info.get("fps") or 30
    duration_s = info.get("duration") or 0
    frame_count = int(fps * duration_s)

    return {
        "path": output_path,
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": duration_s,
    }
