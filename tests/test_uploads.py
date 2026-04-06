import json
import os

from fastapi.testclient import TestClient

import app as modal_app
from pipeline.pool_geometry import CALIBRATION_LINE_KEYS


def test_chunked_upload_flow(monkeypatch, tmp_path):
    spawned = []

    monkeypatch.setattr(modal_app, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(modal_app.results_vol, "commit", lambda: None)
    monkeypatch.setattr(modal_app.results_vol, "reload", lambda: None)
    monkeypatch.setattr(modal_app.run_pipeline, "spawn", lambda *args: spawned.append(args))

    client = TestClient(modal_app.web_app)
    payload = b"abcdefghij"

    init_res = client.post(
        "/uploads/init",
        json={
            "filename": "test.mp4",
            "label": "Scrimmage",
            "content_type": "video/mp4",
            "total_size": len(payload),
        },
    )
    assert init_res.status_code == 200
    job_id = init_res.json()["job_id"]

    first = client.put(
        f"/uploads/{job_id}/chunk?index=0&total_chunks=2&start_byte=0",
        content=payload[:4],
        headers={"content-type": "application/octet-stream"},
    )
    assert first.status_code == 200
    assert first.json()["received_bytes"] == 4

    second = client.put(
        f"/uploads/{job_id}/chunk?index=1&total_chunks=2&start_byte=4",
        content=payload[4:],
        headers={"content-type": "application/octet-stream"},
    )
    assert second.status_code == 200
    assert second.json()["received_bytes"] == len(payload)

    complete = client.post(f"/uploads/{job_id}/complete")
    assert complete.status_code == 200
    assert complete.json() == {"job_id": job_id}
    assert spawned == [(job_id, None, "Scrimmage", False, "auto")]

    upload_path = tmp_path / job_id / "game.mp4"
    meta_path = tmp_path / job_id / "upload.json"
    assert upload_path.read_bytes() == payload

    with open(meta_path) as f:
        manifest = json.load(f)

    assert manifest["status"] == "uploaded"
    assert manifest["received_bytes"] == len(payload)
    assert manifest["filename"] == "test.mp4"


def test_chunk_rejects_offset_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(modal_app, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(modal_app.results_vol, "commit", lambda: None)
    monkeypatch.setattr(modal_app.results_vol, "reload", lambda: None)

    client = TestClient(modal_app.web_app)
    init_res = client.post(
        "/uploads/init",
        json={"filename": "test.mp4", "total_size": 6},
    )
    job_id = init_res.json()["job_id"]

    res = client.put(
        f"/uploads/{job_id}/chunk?index=0&total_chunks=1&start_byte=3",
        content=b"abcdef",
        headers={"content-type": "application/octet-stream"},
    )

    assert res.status_code == 409
    assert "expected 0" in res.json()["detail"]


def test_manual_upload_waits_for_calibration(monkeypatch, tmp_path):
    spawned = []

    monkeypatch.setattr(modal_app, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(modal_app.results_vol, "commit", lambda: None)
    monkeypatch.setattr(modal_app.results_vol, "reload", lambda: None)
    monkeypatch.setattr(modal_app.run_pipeline, "spawn", lambda *args: spawned.append(args))
    monkeypatch.setattr(
        modal_app,
        "_extract_calibration_frame",
        lambda video_path, frame_path: open(frame_path, "wb").write(b"jpg"),
    )

    client = TestClient(modal_app.web_app)
    payload = b"abcdefghij"

    init_res = client.post(
        "/uploads/init",
        json={
            "filename": "manual.mp4",
            "total_size": len(payload),
            "homography_mode": "manual",
        },
    )
    job_id = init_res.json()["job_id"]

    client.put(
        f"/uploads/{job_id}/chunk?index=0&total_chunks=1&start_byte=0",
        content=payload,
        headers={"content-type": "application/octet-stream"},
    )

    complete = client.post(f"/uploads/{job_id}/complete")
    assert complete.status_code == 200
    assert complete.json() == {"job_id": job_id, "needs_calibration": True}
    assert spawned == []

    status = client.get(f"/status/{job_id}")
    assert status.status_code == 200
    assert status.json()["stage"] == "awaiting_calibration"


def test_submit_calibration_starts_manual_pipeline(monkeypatch, tmp_path):
    spawned = []

    monkeypatch.setattr(modal_app, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(modal_app.results_vol, "commit", lambda: None)
    monkeypatch.setattr(modal_app.results_vol, "reload", lambda: None)
    monkeypatch.setattr(modal_app.run_pipeline, "spawn", lambda *args: spawned.append(args))
    monkeypatch.setattr(
        modal_app,
        "_extract_calibration_frame",
        lambda video_path, frame_path: open(frame_path, "wb").write(b"jpg"),
    )

    client = TestClient(modal_app.web_app)
    payload = b"abcdefghij"

    init_res = client.post(
        "/uploads/init",
        json={
            "filename": "manual.mp4",
            "label": "Manual Match",
            "total_size": len(payload),
            "homography_mode": "manual",
        },
    )
    job_id = init_res.json()["job_id"]

    client.put(
        f"/uploads/{job_id}/chunk?index=0&total_chunks=1&start_byte=0",
        content=payload,
        headers={"content-type": "application/octet-stream"},
    )
    client.post(f"/uploads/{job_id}/complete")

    lines = [
        {"key": key, "x1": float(index * 10), "y1": 0.0, "x2": float(index * 10 + 5), "y2": 10.0}
        for index, key in enumerate(CALIBRATION_LINE_KEYS)
    ]
    res = client.post(f"/uploads/{job_id}/calibration", json={"lines": lines})

    assert res.status_code == 200
    assert res.json() == {"job_id": job_id}
    assert spawned == [(job_id, None, "Manual Match", False, "manual")]


def test_submit_calibration_rejects_incomplete_points(monkeypatch, tmp_path):
    monkeypatch.setattr(modal_app, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(modal_app.results_vol, "commit", lambda: None)
    monkeypatch.setattr(modal_app.results_vol, "reload", lambda: None)
    monkeypatch.setattr(
        modal_app,
        "_extract_calibration_frame",
        lambda video_path, frame_path: open(frame_path, "wb").write(b"jpg"),
    )

    client = TestClient(modal_app.web_app)
    payload = b"abcdefghij"

    init_res = client.post(
        "/uploads/init",
        json={
            "filename": "manual.mp4",
            "total_size": len(payload),
            "homography_mode": "manual",
        },
    )
    job_id = init_res.json()["job_id"]

    client.put(
        f"/uploads/{job_id}/chunk?index=0&total_chunks=1&start_byte=0",
        content=payload,
        headers={"content-type": "application/octet-stream"},
    )
    client.post(f"/uploads/{job_id}/complete")

    res = client.post(
        f"/uploads/{job_id}/calibration",
        json={"lines": [{"key": CALIBRATION_LINE_KEYS[0], "x1": 1, "y1": 2, "x2": 1, "y2": 2}]},
    )

    assert res.status_code == 400
    assert "Expected" in res.json()["detail"] or "distinct endpoints" in res.json()["detail"]
