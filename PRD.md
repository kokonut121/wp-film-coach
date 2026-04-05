# PRD: Water Polo Tactical Analysis System

## Problem Statement

Competitive water polo players and coaches lack access to affordable, automated tactical analysis tools. Professional-grade sports analytics (used in soccer, basketball, etc.) require dedicated camera crews, expensive licensed software, and data science teams. A Caltech varsity water polo player wants a tool that can take any video of a water polo game — pasted as a YouTube URL — and automatically produce a timestamped tactical breakdown, flag mistakes, detect formations, and allow a coach or player to ask natural-language questions about what happened in the game. The system should be consumer-facing (accessible via a shared web URL), require no local GPU, and cost essentially nothing to run for personal use.

---

## Solution

A full-stack web application where a user pastes a YouTube link to a water polo game, kicks off an async CV pipeline that runs on Modal serverless GPU infrastructure, and receives back a rich tactical analysis including a real-time bird's-eye pool map, player tracking, event timeline, formation detection, and a Claude-powered chat interface they can use to interrogate the game. All model weights are sourced from HuggingFace Hub (free). The LLM analysis layer uses the Anthropic API. The frontend is hosted on Vercel.

### High-Level Architecture

```
[Vercel Frontend (React)]
        ↕ HTTPS / SSE
[Modal Backend (FastAPI + GPU functions)]
        ↓
[CV Pipeline: YOLO → ByteTrack → Homography → Event Classifier]
        ↓
[Structured Event JSON (Modal Volume)]
        ↓
[Claude API — Tactical Agent + Chat Q&A]
```

---

## User Stories

### Video Submission

1. As a player, I want to paste a YouTube URL into a web app, so that I don't have to download or upload large video files myself.
2. As a player, I want the app to accept any publicly available YouTube game footage, so that I can analyse both my own games and opponents' games.
3. As a player, I want the submission form to validate that the URL is a real YouTube link before submitting, so that I don't wait 30 minutes to find out the input was wrong.
4. As a player, I want to optionally provide a label for the game (e.g. "vs UCLA, Oct 12"), so that I can identify it later in my history.
5. As a player, I want the app to tell me approximately how long processing will take after I submit, so that I know when to come back.

### Processing & Progress

6. As a player, I want to see a live progress indicator showing which pipeline stage is running (downloading, detecting, tracking, etc.), so that I know the job is alive and not stuck.
7. As a player, I want the progress indicator to show a percentage within each stage, so that I can estimate time remaining.
8. As a player, I want to be able to close the browser and return later to find my results ready, so that I don't have to keep the tab open for 30+ minutes.
9. As a player, I want a unique shareable URL for each processed game, so that I can send the analysis to a coach.
10. As a player, I want to receive a clear error message if the video is private, age-restricted, or otherwise unavailable, so that I know immediately what went wrong.
11. As a player, I want the system to handle long games (up to 90 minutes) without timing out, so that full match footage can be analysed.

### Bird's-Eye Pool Map

12. As a player, I want to see an animated top-down pool map showing all player positions over time, so that I can understand spatial patterns that aren't visible from broadcast angles.
13. As a player, I want the two teams colour-coded differently on the map (matching cap colour where possible), so that I can instantly distinguish team movements.
14. As a player, I want the ball to be shown on the map as a distinct marker, so that I can track possession flow.
15. As a player, I want to be able to scrub through the game timeline and see the pool map update to that moment, so that I can review specific sequences.
16. As a player, I want player trails (last 3–5 seconds of movement) shown on the map, so that I can see directional momentum.
17. As a player, I want the map to use real FINA pool dimensions (25m × 13m) with 2m/5m/half-distance lines marked, so that distances on the map are meaningful.

### Detection & Tracking

18. As a player, I want each player to maintain a consistent ID throughout the video even when they submerge or the camera cuts, so that individual player analysis is accurate.
19. As a player, I want the goalie to be identified and tracked separately from field players, so that goalkeeper-specific analysis is possible.
20. As a player, I want ball tracking to be as reliable as possible even through splashing and partial occlusion, so that possession events are accurately detected.
21. As a player, I want the system to handle multiple camera angles and cuts in broadcast footage without losing tracking continuity, so that commercial game footage can be used.

### Event Detection & Timeline

22. As a player, I want a timestamped event timeline showing detected events (shot, goal, turnover, exclusion, counter-attack, man-up, man-down), so that I can jump to key moments.
23. As a player, I want to click any event in the timeline and have it highlighted on the pool map and in the analysis, so that I can see the spatial context of each event.
24. As a player, I want exclusion events (ejections) detected and logged with which player was excluded and for how long, so that man-up/man-down sequences can be reviewed.
25. As a player, I want counter-attack transitions detected — moments where possession changes and a team rapidly advances — so that transition defence/offence can be analysed.
26. As a player, I want shot attempts distinguished from goals, so that conversion rate can be computed.

### Formation Detection

27. As a player, I want the current offensive and defensive formation detected and labelled (e.g. 3-3, 4-2, arc, umbrella) at each moment, so that formation tendencies can be studied.
28. As a player, I want to see a formation timeline chart showing how formations shifted over the course of the game, so that patterns relative to score, fatigue, or opponent adjustments are visible.
29. As a player, I want formation transitions flagged as events in the timeline, so that I can see exactly when and why a team shifted shape.

### Tactical Metrics

30. As a player, I want a per-player heatmap showing where on the pool each player spent the most time, so that positioning tendencies are clear.
31. As a player, I want team convex hull area computed over time (effective playing space), so that pressing intensity can be measured.
32. As a player, I want possession percentage broken down by period, so that dominance patterns are visible.
33. As a player, I want average shot clock pressure index computed per possession, so that rushed versus composed attack patterns can be identified.
34. As a player, I want centroid spread (average inter-player distance within a team) tracked over time, so that compactness and stretching tendencies are visible.

### LLM Tactical Agent — Post-Game Analysis

35. As a player, I want the system to automatically generate a written tactical summary after processing, so that I have an immediate overview without needing to interact with the data manually.
36. As a player, I want the summary to include timestamped mistake flags (e.g. "at 14:32 your team lost possession in the 2m zone due to over-dribbling"), so that specific correctable moments are highlighted.
37. As a player, I want the summary to identify recurring patterns in turnovers, so that systemic weaknesses are surfaced.
38. As a player, I want the agent to identify man-up sequences and evaluate whether the team's positioning was effective, so that set-piece efficiency can be improved.
39. As a player, I want the agent to compare offensive formation effectiveness (e.g. goals/shots from 3-3 vs arc), so that tactical adjustments can be prioritised.
40. As a player, I want the auto-generated report to be structured with clear sections (Summary, Key Moments, Tactical Patterns, Individual Notes, Recommendations), so that a coach can read it efficiently.

### LLM Chat Interface

41. As a player, I want to ask natural-language questions about the game (e.g. "when did we lose possession the most?"), so that I can explore the data without needing to know what metrics exist.
42. As a player, I want the chat to remember context from earlier in the conversation, so that follow-up questions work naturally.
43. As a player, I want the chat responses to cite specific timestamps when referring to events, so that I can verify claims by scrubbing to that moment.
44. As a player, I want to ask strategic questions (e.g. "what formation should we run against a high-press team?") and get answers grounded in both the game data and general water polo knowledge, so that the tool is useful for planning not just reviewing.
45. As a player, I want chat responses to stream token-by-token rather than appear all at once, so that the interface feels responsive.

### Game History

46. As a player, I want a list of all previously processed games accessible from the home page, so that I can return to old analyses.
47. As a player, I want to be able to delete a processed game from my history, so that storage doesn't accumulate indefinitely.

---

## Implementation Decisions

### Module Breakdown

#### Module 1: Frontend (Vite + React + D3 + Vercel)

- Single-page React app (plain JavaScript, no TypeScript) built with Vite, deployed to Vercel on the free tier.
- **Charts and visualisations use D3.js** for full control over custom pool map rendering, heatmaps, hull area timelines, and formation charts.
- Three main views: (1) Home / URL submission, (2) Processing / progress screen, (3) Results / analysis view.
- Results view has four panels: bird's-eye pool map (D3 SVG/canvas), event timeline (scrollable list), tactical metrics (D3 charts), and chat pane.
- The pool map is rendered using D3 with position data from the event JSON. It is not a video player — it is a data visualisation that mirrors game time.
- A shared timeline scrubber (a range slider) controls the current time displayed across all panels simultaneously.
- Chat pane uses Server-Sent Events (SSE) to stream Claude responses.
- Game history is stored in `localStorage` as a list of `{ job_id, label, youtube_url, timestamp }` objects. No user auth required.
- The frontend communicates with the Modal backend via a `VITE_API_URL` environment variable pointing to the Modal HTTPS endpoint.

#### Module 2: Modal Backend (FastAPI + Modal)

- A single Modal app containing a FastAPI ASGI app (for the HTTP endpoints) and one GPU-backed function (for the CV pipeline).
- HTTP endpoints:
  - `POST /process` — accepts `{ youtube_url, label }`, spawns the pipeline as a background Modal function call, returns `{ job_id }` immediately.
  - `GET /status/{job_id}` — reads a `progress.json` file from Modal Volume, returns `{ stage, pct }`.
  - `GET /results/{job_id}` — returns the full `events.json` from Modal Volume once processing is complete.
  - `POST /chat` — accepts `{ job_id, messages }`, loads event JSON from volume as context, streams Claude API response via SSE.
  - `DELETE /jobs/{job_id}` — removes job files from Modal Volume.
- CORS is open (`allow_origins=["*"]`) since this is a single-user personal tool.
- The Modal Volume named `wp-results` persists all job data between invocations.
- The GPU function uses a `T4` GPU. `timeout=3600` (1 hour) to handle long games.
- All model weights are downloaded on first run inside the Modal image and cached in a separate Modal Volume named `wp-model-cache` to avoid re-downloading on every cold start.

#### Module 3: Video Download (`pipeline/download.py`)

- Uses `yt-dlp` inside the Modal container.
- Downloads the best available video quality up to 720p (balances quality vs processing time).
- Output saved to `/results/{job_id}/game.mp4` on the Modal Volume.
- Returns the video path, fps, total frame count, and duration in seconds.
- Raises a typed exception (`VideoUnavailableError`) for private/age-restricted/deleted videos so the API can return a clean 400 error.

#### Module 4: Detection (`pipeline/detect.py`)

- Loads `YOLOv8m` weights (player detection) and `YOLOv8n` weights (ball detection) from the `ultralytics` auto-download mechanism, cached in the model volume.
- Runs inference every 3rd frame (effectively 10fps), which is sufficient for tactical analysis and keeps processing time under 40 minutes for a 45-minute game on a T4.
- Two detection models run per frame:
  1. **Player detector** (`YOLOv8m`) — COCO-pretrained, filters for the `person` class. Returns bounding boxes + confidence.
  2. **Ball detector** (`YOLOv8n`) — COCO-pretrained, filters for the `sports ball` class. If no detection is returned, a secondary HSV-based orange ball detector runs on the frame (H: 5–25, S > 100, V > 100; contours filtered by circularity and size).
- No separate keypoint detection model — pool line keypoints are extracted by the homography module using classical CV (see Module 6).
- Team classification (cap colour) runs as a post-detection step using K-means (k=3: team A, team B, goalie) on the HSV histogram of the **top 30% of each player bounding box** (the cap region), not the full crop. This reduces noise from water splashing and body exposure. Runs on CPU after the full video is processed.
- Writes per-frame detection results to a `detections.jsonl` file (one JSON line per processed frame) so progress is not lost if the job is interrupted.
- Reports progress to the `progress.json` file after every 500 frames.

#### Module 5: Tracking (`pipeline/track.py`)

- Uses `ByteTrack` (via the `supervision` library) for within-shot player ID assignment. Runs on CPU.
- Uses **ResNet18** (pretrained via `torchvision`, final FC layer removed to produce 512-dim embeddings) as the re-ID embedding model for cross-cut continuity. Embeddings are L2-normalised; cross-cut matching uses cosine similarity with a threshold of 0.7. This replaces OSNet/torchreid for a lighter dependency footprint with a clear upgrade path.
- A `Kalman filter` per player predicts position through occlusion windows of up to 2 seconds (20 frames at 10fps). Players unmatched longer than 2 seconds are considered lost and re-matched via re-ID on reappearance.
- Scene cuts are detected by frame-to-frame histogram correlation drop below 0.3, triggering a tracker reset and cross-cut re-ID pass.
- Outputs a `tracks.jsonl` with `{ frame_idx, t_seconds, player_id, team, bbox, confidence }` per row.

#### Module 6: Homography (`pipeline/homography.py`)

- Detects scene cuts using frame-to-frame histogram difference (correlation drop below 0.3). Recomputes homography on each new shot.
- Uses **classical CV** (no learned model) to find pool line keypoints:
  1. Convert frame to HSV and create a water mask (H: 85–130) to restrict edge detection to the pool area.
  2. Run Canny edge detection on the masked grayscale frame.
  3. Run `HoughLinesP` to extract line segments; cluster into vertical (pool lane markers) and horizontal (side boundaries) groups.
  4. Compute line intersections and match to known FINA template coordinates (goal lines at 0m/25m, 2m lines, 5m lines, half at 12.5m, side boundaries at 0m/13m).
- Requires ≥ 4 matched keypoints to compute `cv2.findHomography` with RANSAC. If fewer are found, the last valid `H` matrix is reused with a `h_stale` flag set in the output.
- Applies `H` to the foot-point (bottom-centre of each player bounding box) to produce `(x, y)` in metres, clamped to pool bounds.
- Outputs a `positions.jsonl` with `{ t_seconds, frame_idx, player_id, team, x_metres, y_metres, h_stale }` per entry.

#### Module 7: Event Classifier (`pipeline/events.py`)

- Consumes `positions.jsonl` and `tracks.jsonl`.
- Rule-based + lightweight heuristics for the following event types:
  - **Shot attempt** — ball detected moving at high velocity toward goal zone.
  - **Goal** — ball enters goal bounding box.
  - **Turnover** — possession (ball proximity majority) switches teams.
  - **Exclusion** — a player disappears from their team's detected set for 20+ seconds (flagged as probable ejection).
  - **Counter-attack** — turnover followed by 3+ attacking players advancing past half line within 5 seconds.
  - **Man-up** — one team has 6 players detected, the other has 5.
  - **Press trigger** — convex hull of defending team contracts by >30% within 3 seconds.
- Formation detection: runs K-means (k=2 per half of pool) on attacking team positions every 10 seconds to label formation as one of: `3-3`, `4-2`, `arc`, `umbrella`, `spread`. Uses cosine similarity against archetype position templates.
- Tactical metrics computed globally: per-player heatmap (2D histogram), team convex hull area over time, possession % per period, centroid spread.
- Outputs a single `events.json` file structured as:
  ```json
  {
    "meta": { "duration_s", "fps", "team_a_colour", "team_b_colour" },
    "positions": [...],
    "events": [...],
    "formations": [...],
    "metrics": { "heatmaps", "possession", "hull_area", "centroid_spread" }
  }
  ```

#### Module 8: Tactical Agent (`pipeline/agent.py`)

- Summarises `events.json` into a compact context string (since the full positions array would exceed context limits). The summary includes: event log with timestamps, formation transitions, turnover locations, man-up sequences, and aggregated metrics. Target: under 8,000 tokens.
- `generate_report(events_summary) -> str` — calls `claude-sonnet-4-6` with a water-polo-expert system prompt and the summarised context. Returns a structured markdown report with sections: Summary, Key Moments, Tactical Patterns, Individual Notes, Recommendations.
- `stream_chat(events_summary, messages) -> Generator` — takes the full conversation history plus the events summary as system context, calls Claude with `stream=True`, yields SSE chunks.
- The system prompt instructs Claude to: cite timestamps when referencing events, distinguish between the two teams consistently, flag uncertain inferences, and ground strategic recommendations in observed data rather than generic advice.

### Data Flow Summary

```
YouTube URL
  → yt-dlp → game.mp4
  → detect.py (GPU) → detections.jsonl
  → track.py (CPU) → tracks.jsonl
  → homography.py (CPU) → positions.jsonl
  → events.py (CPU) → events.json
  → agent.py → report (markdown string, stored in events.json)
  → frontend reads events.json → renders all UI panels
  → chat calls agent.py → streams response
```

### Key Technical Constraints

- All model weights must be sourced from HuggingFace Hub or the `ultralytics` auto-download mechanism. No proprietary APIs for inference.
- The Modal T4 GPU function has a 1-hour timeout. A 90-minute game at 10fps effective = ~54,000 frames through YOLO. At ~30fps throughput on T4 this takes ~30 minutes. Within budget.
- The Modal free tier provides $30/month in credits. At ~$0.60/hr for T4, this is 50 hours of GPU — sufficient for ~100 full-game analyses per month at personal use scale.
- The Anthropic API is called only twice per game (report generation + on-demand chat). At `claude-sonnet-4-6` pricing, the report generation costs ~$0.002–0.005. Chat responses are ~$0.001 each.
- No user authentication is implemented. The app is single-user by design. Job IDs are UUIDs; knowledge of a job ID grants access to its results.
- The frontend stores game history in `localStorage`. No database is needed.

### API Contract (Backend → Frontend)

**POST /process**
```
Request:  { youtube_url: string, label?: string }
Response: { job_id: string }
```

**GET /status/{job_id}**
```
Response: { stage: "queued"|"downloading"|"detecting"|"tracking"|"homography"|"classifying"|"generating_report"|"done"|"error", pct: number, error_message?: string }
```

**GET /results/{job_id}**
```
Response: events.json (full object as described in Module 7)
```

**POST /chat**
```
Request:  { job_id: string, messages: [{role: "user"|"assistant", content: string}] }
Response: text/event-stream — SSE chunks of Claude response text
```

**DELETE /jobs/{job_id}**
```
Response: { deleted: true }
```

---

## Testing Decisions

### What makes a good test

Tests should verify the external behaviour of each module given a known input — not the internal implementation. For example, the homography module should be tested by giving it a synthetic frame with known pixel coordinates of pool line intersections and asserting that the output position in metres is within an acceptable tolerance of the ground truth. Tests should not assert which OpenCV function was called internally.

### Modules to test

**`pipeline/download.py`**
- Given a valid public YouTube URL, assert that a video file is downloaded and the returned metadata (fps, duration) is plausible (fps between 24–60, duration > 60 seconds).
- Given a known-invalid URL (404, private video), assert that `VideoUnavailableError` is raised.

**`pipeline/homography.py`**
- Given a synthetic image with pool line keypoints at known pixel positions, assert that the computed `(x, y)` metres output is within ±0.5m of the ground truth for at least 4 test points covering different regions of the pool.
- Given two sequential frames with a simulated camera cut (large histogram difference), assert that homography is recomputed and the previous `H` is discarded.

**`pipeline/events.py`**
- Given a hand-crafted `positions.jsonl` representing a clear possession switch (ball near team A for 5s then ball near team B for 5s), assert that exactly one `turnover` event is emitted.
- Given positions where one team has 6 players and the other has 5, assert that a `man_up` event is emitted.
- Given positions representing a 3-3 formation (three attackers spread across the 2m line, three at 5m), assert that the formation label is `3-3`.

**`pipeline/agent.py`**
- Given a minimal synthetic `events.json`, assert that `generate_report()` returns a string containing all expected section headers (Summary, Key Moments, etc.) and at least one timestamp reference.
- Given a multi-turn `messages` list, assert that `stream_chat()` is a generator that yields at least one non-empty string chunk.

**Frontend (integration-level)**
- Given a mock `/status` endpoint that cycles through all stages, assert that the progress UI transitions through each stage label correctly.
- Given a mock `/results` endpoint returning a minimal events JSON, assert that the pool map canvas renders without errors and the event timeline shows the correct number of items.

---

## Out of Scope

- **User authentication and multi-user accounts.** The app is intentionally single-user. Adding auth, user accounts, or per-user storage is a future concern.
- **Real-time / live game analysis.** The pipeline is batch-only. A live streaming pipeline would require a fundamentally different architecture (frame buffer, sub-second latency model serving) and is out of scope.
- **Fine-tuning the ball detector.** Ball detection uses a pretrained YOLOv8n (COCO `sports ball` class) with an HSV orange-ball fallback detector. Fine-tuning on water polo-specific crops is a follow-on task once labelled data is collected.
- **Individual player identification by number.** Jersey number OCR is not included. Players are identified by persistent tracking IDs and team colour only.
- **Audio analysis** (referee whistle detection, crowd noise). Only visual data is used.
- **Mobile app.** The frontend is a responsive web app, not a native iOS/Android app.
- **Video download from sources other than YouTube.** Only YouTube URLs are supported. Direct video file upload is out of scope for v1.
- **Export to PDF or video overlay.** The analysis is web-only. Exporting the annotated video or a PDF report is a future feature.
- **Opponent scouting mode.** While the tool can technically analyse any game, the UX and agent prompts are designed around analysing your own team. A dedicated opponent-scouting workflow is out of scope.

---

## Further Notes

### Water Polo-Specific CV Challenges

The hardest single sub-problem in this entire system is ball detection. The water polo ball is small (21cm diameter), frequently partially submerged, subject to heavy motion blur during passes, and occluded by splashing white water. The initial approach — running a high-resolution crop detector on the upper pool area — will have meaningful miss rates. The event classifier should be designed to be robust to missing ball detections, using velocity extrapolation and possession-zone heuristics when the ball is not detected.

Player detection from head/shoulders only (rather than full body) means standard COCO-pretrained YOLO will underperform. The pretrained model should be evaluated on a sample of water polo footage early in development. If precision/recall is below ~0.7, fine-tuning on a small labelled dataset (500–1000 annotated frames) should be prioritised before building downstream modules.

### Homography Reliability

Pool markings (lane ropes, goal posts, line markings) are much more reliably visible than grass field markings in soccer. However, broadcast cameras frequently zoom in heavily on action, reducing visible keypoints. The homography module must gracefully handle frames where fewer than 4 keypoints are visible — in these cases it should reuse the last valid `H` matrix with a staleness flag rather than failing.

### Agent Context Design

The Claude agent receives a summarised event log, not raw positions. The summarisation step (`pipeline/events.py` → context string) is critical: it must be compact enough to fit in context while retaining enough spatial and temporal detail for the agent to make meaningful observations. A good target: represent each event as a one-line structured string (`T=14:32 | TURNOVER | team_a→team_b | location=(18.2m, 6.1m)`) and include aggregated metrics as a short table. The full positions array (potentially millions of rows) is never sent to the API.

### Development Order

The recommended build sequence to get to a working end-to-end system as fast as possible:

1. Detection + homography on a 5-minute clip → verify bird's-eye map is geometrically correct.
2. Add ByteTrack → verify player IDs are stable across a 2-minute clip.
3. Add event classifier (turnovers + shots only) → verify timeline events are plausible.
4. Wire up Modal endpoints + frontend URL input + progress polling.
5. Add Claude report generation + chat interface.
6. Add formation detection, heatmaps, and remaining metrics.
7. Polish frontend (scrubber, pool map trails, chart panels).
