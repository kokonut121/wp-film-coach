"""Event classifier, formation detector, and tactical metrics."""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
from scipy.spatial import ConvexHull

# Pool dimensions
POOL_LENGTH = 25.0
POOL_WIDTH = 13.0

# Goal zones (x ranges for each goal)
GOAL_LEFT_X = 0.5
GOAL_RIGHT_X = 24.5
GOAL_Y_MIN = 4.0  # approximate goal post positions
GOAL_Y_MAX = 9.0

# Event detection parameters
POSSESSION_WINDOW = 1.0  # seconds to confirm possession change
BALL_PROXIMITY_RADIUS = 3.0  # metres
EXCLUSION_ABSENCE_THRESHOLD = 20.0  # seconds
COUNTER_ATTACK_WINDOW = 5.0  # seconds
COUNTER_ATTACK_MIN_PLAYERS = 3
PRESS_HULL_CONTRACTION = 0.3  # 30% contraction
PRESS_WINDOW = 3.0  # seconds
SHOT_VELOCITY_THRESHOLD = 3.0  # metres per second toward goal

# Formation templates (normalized positions for 6 field players on attack)
# Each template is a list of (x_norm, y_norm) where x is distance from goal (0-1), y is lateral (0-1)
FORMATION_TEMPLATES = {
    "3-3": [
        # Front three (near 2m line)
        (0.15, 0.25), (0.15, 0.50), (0.15, 0.75),
        # Back three (near 5m line)
        (0.35, 0.25), (0.35, 0.50), (0.35, 0.75),
    ],
    "4-2": [
        # Front four
        (0.15, 0.15), (0.15, 0.38), (0.15, 0.62), (0.15, 0.85),
        # Back two
        (0.35, 0.35), (0.35, 0.65),
    ],
    "arc": [
        # Semicircle around goal
        (0.10, 0.50), (0.15, 0.25), (0.15, 0.75),
        (0.25, 0.15), (0.25, 0.85), (0.35, 0.50),
    ],
    "umbrella": [
        # One player high, rest spread in an arc
        (0.45, 0.50),  # point/high player
        (0.15, 0.20), (0.15, 0.50), (0.15, 0.80),
        (0.25, 0.35), (0.25, 0.65),
    ],
    "spread": [
        # Even distribution
        (0.15, 0.25), (0.15, 0.75),
        (0.30, 0.15), (0.30, 0.50), (0.30, 0.85),
        (0.45, 0.50),
    ],
}

# Heatmap grid
HEATMAP_BINS_X = 25  # 1m resolution along length
HEATMAP_BINS_Y = 13  # 1m resolution along width

# Formation detection interval
FORMATION_INTERVAL = 10.0  # seconds


def _load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def _group_by_time(positions: list[dict], window: float = 0.5) -> list[list[dict]]:
    """Group positions into time windows."""
    if not positions:
        return []
    groups = []
    current_group = [positions[0]]
    for pos in positions[1:]:
        if pos["t_seconds"] - current_group[0]["t_seconds"] <= window:
            current_group.append(pos)
        else:
            groups.append(current_group)
            current_group = [pos]
    groups.append(current_group)
    return groups


def _estimate_possession(positions: list[dict], ball_positions: list[dict] | None = None) -> list[dict]:
    """Estimate which team has possession at each time window.

    Uses player clustering as a proxy when ball position is unavailable.
    Returns list of {t_seconds, team} entries.
    """
    time_groups = _group_by_time(positions, window=1.0)
    possession_log = []

    for group in time_groups:
        t = group[0]["t_seconds"]
        teams = defaultdict(list)
        for p in group:
            teams[p["team"]].append((p["x_metres"], p["y_metres"]))

        # Heuristic: team with centroid closer to opponent's goal is attacking (has possession)
        best_team = None
        best_attack_score = -1

        for team, positions_list in teams.items():
            if team in ("unknown", "goalie"):
                continue
            centroid_x = np.mean([p[0] for p in positions_list])
            # Score: how far into opponent territory
            attack_score = abs(centroid_x - POOL_LENGTH / 2)
            if attack_score > best_attack_score:
                best_attack_score = attack_score
                best_team = team

        if best_team:
            possession_log.append({"t_seconds": t, "team": best_team})

    return possession_log


def _detect_turnovers(possession_log: list[dict]) -> list[dict]:
    """Detect possession changes (turnovers)."""
    events = []
    if len(possession_log) < 2:
        return events

    prev_team = possession_log[0]["team"]
    change_start = None

    for entry in possession_log[1:]:
        if entry["team"] != prev_team:
            if change_start is None:
                change_start = entry["t_seconds"]
            if entry["t_seconds"] - change_start >= POSSESSION_WINDOW:
                events.append({
                    "t_seconds": round(change_start, 2),
                    "type": "turnover",
                    "detail": f"{prev_team}→{entry['team']}",
                    "location": None,
                })
                prev_team = entry["team"]
                change_start = None
        else:
            change_start = None

    return events


def _detect_player_counts(positions: list[dict]) -> list[dict]:
    """Detect man-up/man-down situations based on player counts per team."""
    events = []
    time_groups = _group_by_time(positions, window=2.0)

    for group in time_groups:
        team_counts = defaultdict(set)
        for p in group:
            if p["team"] not in ("unknown", "goalie"):
                team_counts[p["team"]].add(p["player_id"])

        teams = [t for t in team_counts if t not in ("unknown", "goalie")]
        if len(teams) == 2:
            count_a = len(team_counts[teams[0]])
            count_b = len(team_counts[teams[1]])
            if count_a == 6 and count_b == 5:
                events.append({
                    "t_seconds": round(group[0]["t_seconds"], 2),
                    "type": "man_up",
                    "detail": f"{teams[0]} man-up (6v5)",
                    "location": None,
                })
            elif count_b == 6 and count_a == 5:
                events.append({
                    "t_seconds": round(group[0]["t_seconds"], 2),
                    "type": "man_up",
                    "detail": f"{teams[1]} man-up (6v5)",
                    "location": None,
                })

    # Deduplicate: only keep first event in each continuous man-up sequence
    if not events:
        return events

    deduped = [events[0]]
    for e in events[1:]:
        if e["t_seconds"] - deduped[-1]["t_seconds"] > 5.0 or e["detail"] != deduped[-1]["detail"]:
            deduped.append(e)

    return deduped


def _detect_exclusions(tracks: list[dict]) -> list[dict]:
    """Detect probable exclusions (player absent for 20+ seconds)."""
    events = []

    # Build per-player timeline
    player_times = defaultdict(list)
    player_teams = {}
    for t in tracks:
        pid = t["player_id"]
        player_times[pid].append(t["t_seconds"])
        player_teams[pid] = t["team"]

    for pid, times in player_times.items():
        times.sort()
        for i in range(1, len(times)):
            gap = times[i] - times[i - 1]
            if gap >= EXCLUSION_ABSENCE_THRESHOLD:
                events.append({
                    "t_seconds": round(times[i - 1], 2),
                    "type": "exclusion",
                    "detail": f"player_{pid} ({player_teams.get(pid, 'unknown')}) absent {gap:.0f}s",
                    "location": None,
                })

    return events


def _detect_counter_attacks(positions: list[dict], turnovers: list[dict]) -> list[dict]:
    """Detect counter-attacks: turnover followed by rapid advance."""
    events = []

    for turnover in turnovers:
        t_start = turnover["t_seconds"]
        t_end = t_start + COUNTER_ATTACK_WINDOW

        # Find which team gained possession
        parts = turnover["detail"].split("→")
        if len(parts) != 2:
            continue
        attacking_team = parts[1]

        # Count players from attacking team past half line within window
        advanced = set()
        for p in positions:
            if p["t_seconds"] < t_start or p["t_seconds"] > t_end:
                continue
            if p["team"] != attacking_team:
                continue
            # "Past half line" means in opponent's half
            if p["x_metres"] > POOL_LENGTH / 2 or p["x_metres"] < POOL_LENGTH / 2:
                # Determine attacking direction from turnover context
                advanced.add(p["player_id"])

        if len(advanced) >= COUNTER_ATTACK_MIN_PLAYERS:
            events.append({
                "t_seconds": round(t_start, 2),
                "type": "counter_attack",
                "detail": f"{attacking_team} counter-attack ({len(advanced)} players advanced)",
                "location": None,
            })

    return events


def _compute_convex_hull_area(points: list[tuple[float, float]]) -> float:
    """Compute convex hull area of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(np.array(points))
        return float(hull.volume)  # In 2D, 'volume' is area
    except Exception:
        return 0.0


def _detect_press_triggers(positions: list[dict]) -> list[dict]:
    """Detect pressing triggers: defending team's hull contracts >30% within 3s."""
    events = []
    time_groups = _group_by_time(positions, window=1.0)

    team_hull_history = defaultdict(list)

    for group in time_groups:
        t = group[0]["t_seconds"]
        team_points = defaultdict(list)
        for p in group:
            if p["team"] not in ("unknown", "goalie"):
                team_points[p["team"]].append((p["x_metres"], p["y_metres"]))

        for team, pts in team_points.items():
            area = _compute_convex_hull_area(pts)
            team_hull_history[team].append((t, area))

    # Check for rapid contraction
    for team, history in team_hull_history.items():
        for i in range(len(history)):
            t_i, area_i = history[i]
            if area_i < 1.0:
                continue
            for j in range(i + 1, len(history)):
                t_j, area_j = history[j]
                if t_j - t_i > PRESS_WINDOW:
                    break
                if area_j < area_i * (1 - PRESS_HULL_CONTRACTION):
                    events.append({
                        "t_seconds": round(t_i, 2),
                        "type": "press_trigger",
                        "detail": f"{team} hull contracted {((area_i - area_j) / area_i * 100):.0f}%",
                        "location": None,
                    })
                    break

    return events


def _detect_formations(positions: list[dict], duration_s: float) -> list[dict]:
    """Detect offensive formations every FORMATION_INTERVAL seconds."""
    formations = []

    for t in np.arange(0, duration_s, FORMATION_INTERVAL):
        t_start = t
        t_end = t + FORMATION_INTERVAL

        # Collect positions in this window per team
        team_positions = defaultdict(list)
        for p in positions:
            if p["t_seconds"] < t_start or p["t_seconds"] >= t_end:
                continue
            if p["team"] in ("unknown", "goalie"):
                continue
            team_positions[p["team"]].append((p["x_metres"], p["y_metres"]))

        for team, pts in team_positions.items():
            if len(pts) < 6:
                continue

            # Average positions to get representative snapshot
            pts_array = np.array(pts)
            # Cluster into ~6 player positions using simple binning
            from sklearn.cluster import KMeans as KM
            n_clusters = min(6, len(pts))
            km = KM(n_clusters=n_clusters, random_state=42, n_init=5)
            km.fit(pts_array)
            centroids = km.cluster_centers_

            # Normalize to [0, 1] relative to attacking zone
            # Determine attacking direction: team closer to goal on left attacks right, vice versa
            mean_x = centroids[:, 0].mean()
            if mean_x < POOL_LENGTH / 2:
                # Attacking left goal
                norm_x = centroids[:, 0] / (POOL_LENGTH / 2)
                norm_y = centroids[:, 1] / POOL_WIDTH
            else:
                # Attacking right goal
                norm_x = (POOL_LENGTH - centroids[:, 0]) / (POOL_LENGTH / 2)
                norm_y = centroids[:, 1] / POOL_WIDTH

            player_positions = list(zip(norm_x, norm_y))

            # Match against formation templates using cosine similarity
            best_formation = "unknown"
            best_similarity = -1

            for name, template in FORMATION_TEMPLATES.items():
                if len(template) != n_clusters:
                    continue
                sim = _formation_similarity(player_positions, template)
                if sim > best_similarity:
                    best_similarity = sim
                    best_formation = name

            if best_similarity > 0.5:
                formations.append({
                    "t_seconds": round(t_start, 2),
                    "team": team,
                    "formation": best_formation,
                    "confidence": round(best_similarity, 3),
                })

    return formations


def _formation_similarity(
    positions: list[tuple[float, float]],
    template: list[tuple[float, float]],
) -> float:
    """Compute similarity between detected positions and a formation template.

    Uses minimum-cost assignment via greedy matching on Euclidean distance.
    Returns a similarity score in [0, 1].
    """
    if len(positions) != len(template):
        return 0.0

    pos = np.array(positions)
    tmpl = np.array(template)

    # Compute pairwise distances
    dists = np.linalg.norm(pos[:, None] - tmpl[None, :], axis=2)

    # Greedy assignment
    total_dist = 0.0
    used_tmpl = set()
    for _ in range(len(pos)):
        min_val = float("inf")
        min_i, min_j = 0, 0
        for i in range(len(pos)):
            for j in range(len(tmpl)):
                if j in used_tmpl:
                    continue
                if dists[i, j] < min_val:
                    min_val = dists[i, j]
                    min_i, min_j = i, j
        total_dist += min_val
        used_tmpl.add(min_j)
        dists[min_i, :] = float("inf")

    avg_dist = total_dist / len(pos)
    # Convert distance to similarity: 0 distance → 1.0 similarity
    similarity = max(0.0, 1.0 - avg_dist)
    return similarity


def _compute_heatmaps(positions: list[dict]) -> dict:
    """Compute per-player 2D position heatmaps."""
    player_positions = defaultdict(list)
    for p in positions:
        player_positions[p["player_id"]].append((p["x_metres"], p["y_metres"]))

    heatmaps = {}
    for pid, pts in player_positions.items():
        pts_array = np.array(pts)
        hist, _, _ = np.histogram2d(
            pts_array[:, 0], pts_array[:, 1],
            bins=[HEATMAP_BINS_X, HEATMAP_BINS_Y],
            range=[[0, POOL_LENGTH], [0, POOL_WIDTH]],
        )
        heatmaps[str(pid)] = hist.tolist()

    return heatmaps


def _compute_hull_area_timeline(positions: list[dict]) -> list[dict]:
    """Compute team convex hull area over time."""
    time_groups = _group_by_time(positions, window=2.0)
    timeline = []

    for group in time_groups:
        t = group[0]["t_seconds"]
        team_points = defaultdict(list)
        for p in group:
            if p["team"] not in ("unknown", "goalie"):
                team_points[p["team"]].append((p["x_metres"], p["y_metres"]))

        entry = {"t_seconds": round(t, 2)}
        for team, pts in team_points.items():
            entry[f"{team}_area"] = round(_compute_convex_hull_area(pts), 2)
        timeline.append(entry)

    return timeline


def _compute_possession_by_period(possession_log: list[dict], duration_s: float) -> dict:
    """Compute possession percentage per period (quarter)."""
    if not possession_log:
        return {}

    period_length = duration_s / 4
    periods = {}

    for period in range(4):
        t_start = period * period_length
        t_end = (period + 1) * period_length

        team_counts = defaultdict(int)
        total = 0
        for entry in possession_log:
            if t_start <= entry["t_seconds"] < t_end:
                team_counts[entry["team"]] += 1
                total += 1

        if total > 0:
            periods[f"period_{period + 1}"] = {
                team: round(count / total, 3) for team, count in team_counts.items()
            }

    return periods


def _compute_centroid_spread(positions: list[dict]) -> list[dict]:
    """Compute average inter-player distance within each team over time."""
    time_groups = _group_by_time(positions, window=2.0)
    timeline = []

    for group in time_groups:
        t = group[0]["t_seconds"]
        team_points = defaultdict(list)
        for p in group:
            if p["team"] not in ("unknown", "goalie"):
                team_points[p["team"]].append((p["x_metres"], p["y_metres"]))

        entry = {"t_seconds": round(t, 2)}
        for team, pts in team_points.items():
            if len(pts) < 2:
                entry[f"{team}_spread"] = 0.0
                continue
            pts_arr = np.array(pts)
            dists = np.linalg.norm(pts_arr[:, None] - pts_arr[None, :], axis=2)
            n = len(pts)
            avg_dist = dists.sum() / (n * (n - 1)) if n > 1 else 0.0
            entry[f"{team}_spread"] = round(float(avg_dist), 2)

        timeline.append(entry)

    return timeline


def run_event_classification(
    positions_path: str,
    tracks_path: str,
    output_dir: str,
    meta: dict,
) -> str:
    """Run event detection, formation analysis, and tactical metrics.

    Args:
        positions_path: Path to positions.jsonl.
        tracks_path: Path to tracks.jsonl.
        output_dir: Directory to write events.json.
        meta: Video metadata dict with fps, duration_s, etc.

    Returns:
        Path to events.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    events_path = os.path.join(output_dir, "events.json")

    positions = _load_jsonl(positions_path)
    tracks = _load_jsonl(tracks_path)
    duration_s = meta.get("duration_s", 0)
    fps = meta.get("fps", 30)

    # Detect team colours from tracks
    team_colours = {}
    for t in tracks:
        if t["team"] not in ("unknown", "goalie") and t["team"] not in team_colours:
            team_colours[t["team"]] = t["team"]

    # Estimate possession
    possession_log = _estimate_possession(positions)

    # Detect events
    turnovers = _detect_turnovers(possession_log)
    man_ups = _detect_player_counts(positions)
    exclusions = _detect_exclusions(tracks)
    counter_attacks = _detect_counter_attacks(positions, turnovers)
    press_triggers = _detect_press_triggers(positions)

    all_events = turnovers + man_ups + exclusions + counter_attacks + press_triggers
    all_events.sort(key=lambda e: e["t_seconds"])

    # Detect formations
    formations = _detect_formations(positions, duration_s)

    # Compute metrics
    heatmaps = _compute_heatmaps(positions)
    possession_pct = _compute_possession_by_period(possession_log, duration_s)
    hull_area = _compute_hull_area_timeline(positions)
    centroid_spread = _compute_centroid_spread(positions)

    # Build output
    output = {
        "meta": {
            "duration_s": duration_s,
            "fps": fps,
            "team_a_colour": team_colours.get("team_a", "unknown"),
            "team_b_colour": team_colours.get("team_b", "unknown"),
        },
        "positions": positions,
        "events": all_events,
        "formations": formations,
        "metrics": {
            "heatmaps": heatmaps,
            "possession": possession_pct,
            "hull_area": hull_area,
            "centroid_spread": centroid_spread,
        },
    }

    with open(events_path, "w") as f:
        json.dump(output, f)

    return events_path
