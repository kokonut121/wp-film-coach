"""Tests for pipeline/download.py."""

import os
import tempfile

import pytest

from pipeline.download import VideoUnavailableError, download_video


class TestDownloadVideoInvalid:
    """Test error handling for invalid URLs."""

    def test_private_video_raises_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(VideoUnavailableError):
                download_video("https://www.youtube.com/watch?v=invalid_id_12345", tmpdir)

    def test_nonsense_url_raises_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(VideoUnavailableError):
                download_video("https://www.youtube.com/watch?v=ZZZZZZZZZZZZ", tmpdir)


@pytest.mark.integration
class TestDownloadVideoValid:
    """Integration tests that require network access."""

    def test_valid_short_video(self):
        """Download a short public domain video and verify metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a well-known short Creative Commons video
            result = download_video(
                "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # "Me at the zoo" (first YT video)
                tmpdir,
            )

            assert os.path.exists(result["path"])
            assert result["fps"] >= 24
            assert result["fps"] <= 60
            assert result["duration_s"] > 10
            assert result["frame_count"] > 0
