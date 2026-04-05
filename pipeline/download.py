"""Video download module using yt-dlp."""

import os

import yt_dlp


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

    ydl_opts = {
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "outtmpl": output_path,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
    except yt_dlp.utils.DownloadError as e:
        raise VideoUnavailableError(str(e)) from e

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
