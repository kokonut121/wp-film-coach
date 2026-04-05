"""Tests for pipeline/events.py."""

import json
import os
import tempfile

from pipeline.events import run_event_classification


def _write_jsonl(path: str, records: list[dict]):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_position(t: float, player_id: int, team: str, x: float, y: float) -> dict:
    return {
        "t_seconds": t,
        "frame_idx": int(t * 10),
        "player_id": player_id,
        "team": team,
        "x_metres": x,
        "y_metres": y,
        "h_stale": False,
    }


def _make_track(t: float, player_id: int, team: str) -> dict:
    return {
        "frame_idx": int(t * 10),
        "t_seconds": t,
        "player_id": player_id,
        "team": team,
        "bbox": [100, 100, 200, 200],
        "confidence": 0.9,
    }


class TestTurnoverDetection:
    def test_clear_possession_switch(self):
        """Positions where team_a controls for 5s then team_b for 5s → one turnover."""
        positions = []
        tracks = []

        # Team A near opponent goal (attacking) for 0-5s
        for t in [float(i) for i in range(6)]:
            for pid in range(6):
                positions.append(_make_position(t, pid, "team_a", 20.0 + pid * 0.5, 6.5))
                tracks.append(_make_track(t, pid, "team_a"))
            for pid in range(6, 12):
                positions.append(_make_position(t, pid, "team_b", 5.0 + (pid - 6) * 0.5, 6.5))
                tracks.append(_make_track(t, pid, "team_b"))

        # Team B near opponent goal (attacking) for 6-11s
        for t in [float(i) for i in range(6, 12)]:
            for pid in range(6):
                positions.append(_make_position(t, pid, "team_a", 5.0 + pid * 0.5, 6.5))
                tracks.append(_make_track(t, pid, "team_a"))
            for pid in range(6, 12):
                positions.append(_make_position(t, pid, "team_b", 20.0 + (pid - 6) * 0.5, 6.5))
                tracks.append(_make_track(t, pid, "team_b"))

        with tempfile.TemporaryDirectory() as tmpdir:
            pos_path = os.path.join(tmpdir, "positions.jsonl")
            trk_path = os.path.join(tmpdir, "tracks.jsonl")
            _write_jsonl(pos_path, positions)
            _write_jsonl(trk_path, tracks)

            events_path = run_event_classification(
                pos_path, trk_path, tmpdir, {"fps": 30, "duration_s": 12}
            )

            with open(events_path) as f:
                result = json.load(f)

            turnovers = [e for e in result["events"] if e["type"] == "turnover"]
            assert len(turnovers) >= 1, f"Expected at least 1 turnover, got {len(turnovers)}"


class TestManUpDetection:
    def test_six_vs_five(self):
        """One team with 6 players, other with 5 → man_up event."""
        positions = []
        tracks = []

        for t in [0.0, 1.0, 2.0, 3.0]:
            # Team A: 6 players
            for pid in range(6):
                positions.append(_make_position(t, pid, "team_a", 10.0 + pid, 6.5))
                tracks.append(_make_track(t, pid, "team_a"))
            # Team B: 5 players (one excluded)
            for pid in range(6, 11):
                positions.append(_make_position(t, pid, "team_b", 15.0 + (pid - 6), 6.5))
                tracks.append(_make_track(t, pid, "team_b"))

        with tempfile.TemporaryDirectory() as tmpdir:
            pos_path = os.path.join(tmpdir, "positions.jsonl")
            trk_path = os.path.join(tmpdir, "tracks.jsonl")
            _write_jsonl(pos_path, positions)
            _write_jsonl(trk_path, tracks)

            events_path = run_event_classification(
                pos_path, trk_path, tmpdir, {"fps": 30, "duration_s": 4}
            )

            with open(events_path) as f:
                result = json.load(f)

            man_ups = [e for e in result["events"] if e["type"] == "man_up"]
            assert len(man_ups) >= 1, f"Expected at least 1 man_up event, got {len(man_ups)}"
            assert "team_a" in man_ups[0]["detail"]


class TestFormationDetection:
    def test_three_three_formation(self):
        """Positions in a 3-3 formation should be labelled as such."""
        positions = []
        tracks = []

        # Create a clear 3-3: three at ~2m line, three at ~5m line
        # Attacking left goal (x near 0)
        front_three = [(2.0, 3.0), (2.0, 6.5), (2.0, 10.0)]
        back_three = [(5.0, 3.0), (5.0, 6.5), (5.0, 10.0)]
        all_pos = front_three + back_three

        for t in [float(i) for i in range(11)]:  # need >= FORMATION_INTERVAL (10s)
            for pid, (x, y) in enumerate(all_pos):
                positions.append(_make_position(t, pid, "team_a", x, y))
                tracks.append(_make_track(t, pid, "team_a"))
            # Put opposing team on other side
            for pid in range(6, 12):
                positions.append(_make_position(t, pid, "team_b", 20.0 + (pid - 6) * 0.5, 6.5))
                tracks.append(_make_track(t, pid, "team_b"))

        with tempfile.TemporaryDirectory() as tmpdir:
            pos_path = os.path.join(tmpdir, "positions.jsonl")
            trk_path = os.path.join(tmpdir, "tracks.jsonl")
            _write_jsonl(pos_path, positions)
            _write_jsonl(trk_path, tracks)

            events_path = run_event_classification(
                pos_path, trk_path, tmpdir, {"fps": 30, "duration_s": 11}
            )

            with open(events_path) as f:
                result = json.load(f)

            formations = result["formations"]
            team_a_formations = [f for f in formations if f["team"] == "team_a"]
            assert len(team_a_formations) >= 1, "Expected at least one formation detection for team_a"
            # The formation should ideally be 3-3 given the positions
            labels = [f["formation"] for f in team_a_formations]
            assert "3-3" in labels, f"Expected '3-3' in formation labels, got {labels}"


class TestOutputStructure:
    def test_events_json_has_all_sections(self):
        """Verify the output events.json has all required sections."""
        positions = []
        tracks = []
        for t in [0.0, 1.0]:
            for pid in range(6):
                positions.append(_make_position(t, pid, "team_a", 10.0, 6.5))
                tracks.append(_make_track(t, pid, "team_a"))

        with tempfile.TemporaryDirectory() as tmpdir:
            pos_path = os.path.join(tmpdir, "positions.jsonl")
            trk_path = os.path.join(tmpdir, "tracks.jsonl")
            _write_jsonl(pos_path, positions)
            _write_jsonl(trk_path, tracks)

            events_path = run_event_classification(
                pos_path, trk_path, tmpdir, {"fps": 30, "duration_s": 2}
            )

            with open(events_path) as f:
                result = json.load(f)

            assert "meta" in result
            assert "positions" in result
            assert "events" in result
            assert "formations" in result
            assert "metrics" in result
            assert "heatmaps" in result["metrics"]
            assert "possession" in result["metrics"]
            assert "hull_area" in result["metrics"]
            assert "centroid_spread" in result["metrics"]
