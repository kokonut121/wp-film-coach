"""Claude-powered tactical analysis agent for water polo."""

from __future__ import annotations

import os
from typing import Generator

import anthropic

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an expert water polo tactical analyst. You have deep knowledge of \
water polo strategy, formations, positioning, and game flow.

You are analysing a specific water polo game based on structured event data from a computer \
vision pipeline. The data includes detected events (turnovers, shots, goals, exclusions, \
counter-attacks, man-up/man-down situations), formation detections, and tactical metrics \
(possession percentages, player heatmaps, team spread, convex hull area).

Guidelines:
- Always cite specific timestamps (MM:SS format) when referencing events.
- Clearly distinguish between the two teams (team_a and team_b) throughout.
- Flag when your inference is uncertain or based on limited data.
- Ground strategic recommendations in the observed data, not generic advice.
- When discussing formations, reference the specific formation labels detected (3-3, 4-2, arc, umbrella, spread).
- When discussing positioning, reference pool coordinates (e.g., "near the 2m line", "in the 5m zone").
- Be concise but thorough. A coach should be able to read your analysis and immediately identify actionable takeaways."""

REPORT_PROMPT = """Based on the following game event data, generate a comprehensive tactical analysis report.

{events_summary}

Structure your report with these exact sections:

## Summary
A 3-5 sentence overview of the game flow, key trends, and overall tactical observations.

## Key Moments
A timestamped list of the most impactful moments in the game (turnovers that led to goals, \
critical exclusions, momentum shifts). Include 5-10 moments.

## Tactical Patterns
Analysis of recurring patterns: formation tendencies, pressing intensity, possession flow, \
counter-attack frequency and effectiveness. Reference the metrics data.

## Individual Notes
Notable individual player performances or positioning patterns based on heatmap data and \
event involvement. Reference specific player IDs.

## Recommendations
3-5 specific, actionable tactical recommendations based on the observed patterns. Each \
recommendation should cite the data that supports it."""


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def summarize_events(events_json: dict) -> str:
    """Compress events.json into a compact context string for Claude.

    Target: under 8,000 tokens.
    """
    meta = events_json.get("meta", {})
    events = events_json.get("events", [])
    formations = events_json.get("formations", [])
    metrics = events_json.get("metrics", {})

    lines = []

    # Meta
    duration = meta.get("duration_s", 0)
    lines.append(f"GAME DURATION: {_format_timestamp(duration)}")
    lines.append(f"TEAMS: team_a ({meta.get('team_a_colour', '?')}) vs team_b ({meta.get('team_b_colour', '?')})")
    lines.append("")

    # Events log
    lines.append("=== EVENT LOG ===")
    for e in events:
        ts = _format_timestamp(e["t_seconds"])
        loc = ""
        if e.get("location"):
            loc = f" | loc=({e['location']['x']:.1f}m, {e['location']['y']:.1f}m)"
        lines.append(f"T={ts} | {e['type'].upper()} | {e.get('detail', '')}{loc}")
    lines.append("")

    # Event summary counts
    event_counts = {}
    for e in events:
        event_counts[e["type"]] = event_counts.get(e["type"], 0) + 1
    lines.append("=== EVENT COUNTS ===")
    for etype, count in sorted(event_counts.items()):
        lines.append(f"  {etype}: {count}")
    lines.append("")

    # Formations
    lines.append("=== FORMATION TIMELINE ===")
    for f_entry in formations:
        ts = _format_timestamp(f_entry["t_seconds"])
        lines.append(
            f"T={ts} | {f_entry['team']} | {f_entry['formation']} (conf={f_entry['confidence']:.2f})"
        )
    lines.append("")

    # Possession
    possession = metrics.get("possession", {})
    if possession:
        lines.append("=== POSSESSION BY PERIOD ===")
        for period, pcts in possession.items():
            pct_str = ", ".join(f"{team}: {pct:.1%}" for team, pct in pcts.items())
            lines.append(f"  {period}: {pct_str}")
        lines.append("")

    # Hull area summary
    hull_data = metrics.get("hull_area", [])
    if hull_data:
        lines.append("=== TEAM SPREAD (CONVEX HULL AREA) ===")
        for team_key in [k for k in hull_data[0].keys() if k != "t_seconds"]:
            values = [h.get(team_key, 0) for h in hull_data if team_key in h]
            if values:
                lines.append(
                    f"  {team_key}: avg={np.mean(values):.1f}m², min={min(values):.1f}m², max={max(values):.1f}m²"
                )
        lines.append("")

    # Centroid spread summary
    spread_data = metrics.get("centroid_spread", [])
    if spread_data:
        lines.append("=== INTER-PLAYER DISTANCE ===")
        for team_key in [k for k in spread_data[0].keys() if k != "t_seconds"]:
            values = [s.get(team_key, 0) for s in spread_data if team_key in s]
            if values:
                lines.append(f"  {team_key}: avg={np.mean(values):.1f}m")
        lines.append("")

    # Heatmap summary (top positions per player)
    heatmaps = metrics.get("heatmaps", {})
    if heatmaps:
        lines.append("=== PLAYER POSITIONING (TOP ZONES) ===")
        for pid, hist in heatmaps.items():
            hist_arr = np.array(hist)
            if hist_arr.sum() == 0:
                continue
            # Find top 3 zones
            flat = hist_arr.flatten()
            top_indices = np.argsort(flat)[-3:][::-1]
            zones = []
            for idx in top_indices:
                if flat[idx] == 0:
                    break
                x_bin = idx // hist_arr.shape[1]
                y_bin = idx % hist_arr.shape[1]
                zones.append(f"({x_bin}m, {y_bin}m)")
            if zones:
                lines.append(f"  player_{pid}: primary zones = {', '.join(zones)}")
        lines.append("")

    return "\n".join(lines)


# Need numpy for the summarize function
import numpy as np


def generate_report(events_summary: str) -> str:
    """Generate a tactical analysis report using Claude.

    Args:
        events_summary: Compact event summary string from summarize_events().

    Returns:
        Markdown-formatted report string.
    """
    client = anthropic.Anthropic()

    message = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": REPORT_PROMPT.format(events_summary=events_summary)}
        ],
    )

    return message.content[0].text


def stream_chat(
    events_summary: str,
    messages: list[dict],
) -> Generator[str, None, None]:
    """Stream a chat response about the game.

    Args:
        events_summary: Compact event summary string.
        messages: Conversation history as list of {role, content} dicts.

    Yields:
        Text chunks of the Claude response.
    """
    client = anthropic.Anthropic()

    system = f"{SYSTEM_PROMPT}\n\nHere is the game data you are analysing:\n\n{events_summary}"

    with client.messages.stream(
        model=MODEL,
        max_tokens=2048,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text
