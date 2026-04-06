# PRD: HSV-Based Cap Detection for Water Polo Players

## Problem Statement

The current player detection pipeline relies on YOLOv8's generic COCO "person" class to detect water polo players. This fails almost entirely because players are submerged in water with only their heads and caps visible — YOLO expects to see torsos, limbs, and full or partial human bodies. The result is near-zero player detections, which breaks the entire downstream pipeline (tracking, homography mapping, event detection, and tactical analysis).

Water polo caps are the single most visually distinctive feature of players in the water. They are brightly colored (typically white vs. dark/navy), highly saturated, and contrast sharply against the teal/cyan pool water. A detection approach based on cap color will be far more reliable than generic object detection for this domain.

## Solution

Replace YOLO-based player detection with HSV color-based cap detection when operating in manual homography mode. The approach leverages the fact that within the calibrated pool polygon, the water surface is a near-uniform teal/cyan color, and anything that is NOT water-colored is either a player's cap, the ball, or a rare artifact.

The detection algorithm:

1. Mask the frame to the calibration polygon (already implemented)
2. Create a water mask using existing HSV water thresholds
3. Invert the water mask to find all non-water blobs within the pool region
4. Filter blobs by a fixed pixel size range appropriate for caps
5. Generate tight bounding boxes around each detected cap
6. Classify caps into two teams using K-means clustering on cap color histograms, trained once on the first 10 processed frames and then applied to all subsequent frames via nearest-cluster-center assignment

YOLO-based player detection is preserved for auto homography mode and future use.

## User Stories

1. As a coach uploading game footage, I want players to be detected reliably even though only their caps are visible above water, so that the tactical analysis pipeline produces meaningful results.
2. As a coach using manual calibration mode, I want the system to detect all players within the calibrated pool area, so that I get accurate position tracking and event detection.
3. As a coach, I want the system to automatically distinguish between the two teams based on cap color, so that team-specific tactical metrics and formations are correctly attributed.
4. As a coach, I want team classification to be determined early in the video and remain consistent throughout, so that player identities do not flip between teams mid-game.
5. As a coach, I want the detection step to complete in a reasonable time without requiring a GPU for the cap detection portion, so that processing is fast and cost-effective.
6. As a coach, I want the ball to continue being detected separately from players, so that possession tracking and event detection still work correctly.
7. As a developer, I want the cap detection logic contained within the existing detection module, so that tracking, homography, events, and agent modules require no changes.
8. As a developer, I want YOLO-based player detection preserved for auto mode, so that the auto homography path remains functional and I can revisit YOLO-based approaches in the future.
9. As a developer, I want the cap detection output format (bbox, confidence, team) to match the existing player detection format, so that downstream consumers are unaffected.
10. As a developer, I want the team classification to run once on a small sample of early frames rather than on every frame, so that classification is stable and efficient.
11. As a coach, I want the system to handle varying cap colors across different games (white vs. dark, red vs. blue, etc.) without manual configuration, so that I do not need to adjust settings per video.
12. As a developer, I want the detection thresholds (size ranges, HSV bounds) kept as internal implementation details rather than exposed configuration, so that the module interface stays simple.
13. As a coach, I want the system to not miss players who are in the pool but partially occluded by water splash or wave patterns, so that detection coverage is high throughout the game.
14. As a developer, I want the cap detection to reuse the existing calibration polygon and water HSV thresholds from pool_geometry, so that there is a single source of truth for pool boundary and water color definitions.

## Implementation Decisions

### Detection Path Branching

- When `homography_mode="manual"`, the detection function skips YOLO player detection and uses the new HSV cap detection path instead.
- When `homography_mode="auto"`, the existing YOLO person detection path runs unchanged.
- Ball detection (YOLO + HSV orange fallback) remains identical in both modes.

### Cap Detection Algorithm

- The frame is masked and cropped to the calibration polygon using the existing `mask_and_crop_to_polygon()` function.
- A water mask is created using the existing HSV water thresholds (`WATER_HSV_LOW`, `WATER_HSV_HIGH` from pool_geometry).
- The water mask is inverted to produce a "non-water" mask — all pixels within the pool region that are not water-colored.
- Morphological operations (open to remove noise, close to fill small gaps) are applied to the non-water mask.
- Contours are extracted from the non-water mask.
- Contours are filtered by area using fixed pixel thresholds appropriate for cap-sized blobs at typical camera distances.
- Each surviving contour produces a tight bounding box representing the detected cap.
- The bounding box is offset back to full-frame coordinates using the existing offset mechanism.

### Team Classification Strategy

- Cap histograms are collected from the first 10 processed frames only.
- K-means clustering with k=2 is run once on these histograms to establish two team cluster centers.
- The two clusters are labeled `team_a` (larger cluster) and `team_b` (smaller cluster).
- For all subsequent frames (frame 11+), each detected cap's histogram is compared to the two cluster centers, and the cap is assigned to the nearest cluster.
- The existing `_extract_cap_histogram()` function is reused for histogram extraction.
- There is no explicit goalie cluster — goalies are assigned to one of the two team clusters based on cap color proximity.

### Output Format Compatibility

- The output format in detections.jsonl remains identical: each player entry has `bbox`, `confidence`, and `team` fields.
- Confidence for HSV-detected caps can be derived from blob quality metrics (area, circularity, color saturation).
- A `source` field may optionally be added to player entries (similar to ball entries) to distinguish `"hsv_cap"` detections from `"yolo"` detections, but this is not required by downstream consumers.

### What Stays Unchanged

- Ball detection (YOLO + HSV orange fallback)
- YOLO player detection in auto mode
- The tracking module (track.py)
- The homography modules
- The events module
- The agent module
- The detections.jsonl schema
- The progress reporting mechanism

## Testing Decisions

### What Makes a Good Test

Tests should verify external behavior — given a frame or set of frames with known properties, does the detection module produce the expected detections? Tests should not depend on internal implementation details like specific HSV threshold values or morphological kernel sizes.

### Modules to Test

- **Cap detection function**: Given a synthetic or real frame with colored blobs on a teal background within a polygon, verify that the correct number of cap detections are returned with reasonable bounding boxes.
- **Team classification**: Given a set of histograms with two distinct color distributions, verify that K-means produces two clusters and subsequent assignments match the nearest cluster.
- **Detection path branching**: Verify that manual mode invokes cap detection (not YOLO player detection) and auto mode invokes YOLO player detection (not cap detection).
- **Output format**: Verify that detections.jsonl entries from cap detection have the same schema as entries from YOLO detection.

### Prior Art

- `tests/test_detect.py` — existing detection tests provide the pattern for test structure and assertions.
- `tests/test_manual_homography.py` — tests for manual-mode-specific logic.
- The existing `_detect_ball_hsv()` function and its behavior provide a reference for how HSV detection should be tested.

## Out of Scope

- Fine-tuning a custom YOLO model on water polo training data (insufficient compute and labeled data available).
- Cap detection in auto homography mode (only manual mode is addressed).
- Explicit goalie detection or three-cluster team classification.
- Adaptive or configurable size thresholds (fixed pixel ranges are used).
- Lane line, goal post, or other non-player object filtering (the calibrated pool area is assumed to contain only players and the ball).
- Changes to any module outside of detect.py.
- Changes to the detections.jsonl schema (output format is preserved).

## Further Notes

- Water polo caps follow standardized color conventions (white vs. dark for field players, red for goalies), but the negative-space approach (detecting non-water blobs) is color-agnostic and will work regardless of specific cap colors used in a given game.
- The first-10-frames classification window assumes that players are in the water and roughly grouped by team at the start of each period. If the video starts during active play with teams intermixed, the classification may be less reliable, but K-means should still separate the two dominant cap colors.
- If future games reveal that 10 frames is too few for reliable clustering (e.g., not enough players visible yet), this window can be increased, but it should remain a small early sample rather than a full-video pass.
- The cap-only bounding box is intentionally tight. Downstream modules (tracking, homography) use centroids or foot-points derived from the bbox, so a tight cap bbox provides a more accurate position signal than an expanded body-sized bbox for submerged players.
- This approach may also improve the existing cap histogram extraction for team classification, since the bbox now tightly frames the cap rather than including water and body below it.
