# 20-second review improvements

The baseline reviewed was `outputs/match_20s_review/review.mp4`. Sampled original
frames at 0, 10 and 19.96 s show small text, overlapping labels and many detected
people without track IDs. Some distant people are not detected at all.

## Tracking experiment

Reused exactly the same 7,972 person detections and pitch projections. Re-ran only
the existing ByteTrack algorithm on all 500 contiguous frames. This avoids
confounding a tracking comparison with new detector or calibration predictions.

| Configuration | Tracked detections | Coverage | Track IDs | Tracks of <=5 detections |
| --- | ---: | ---: | ---: | ---: |
| Original threshold 0.6, tracker FPS 30 | 5,055 | 63.41% | 44 | 2 |
| Threshold 0.45, tracker FPS 25 | 7,145 | 89.63% | 68 | 11 |
| **Selected threshold 0.5, tracker FPS 25** | **6,680** | **83.79%** | **60** | **6** |

The detector accepts confidence >=0.4. ByteTrack starts tracks at
`track_thresh + 0.1`, so the old configuration required confidence >=0.7 to
initialize a track. Lowering the threshold retains more observations. The selected
0.5 setting trades some coverage for fewer short tracks than the 0.45 candidate.
The MVP configuration now uses 0.5 and the decoded video's actual FPS.

There is **no labeled identity evaluation** here: increased coverage and increased
track count do not establish fewer identity switches. The additional short tracks
are a tradeoff, not hidden by renumbering or merging. Track IDs are not player or
jersey counts. No tracks are joined across gaps, no new detections are invented,
and no pitch coordinates are changed. Distant missed detections still require
detector improvements, and calibration accuracy remains unverified.

## Video presentation

`outputs/match_20s_improved_final/review.mp4` uses a 1920x900 canvas with a larger
1280x720 match view, readable labels, stable colors per track, and a pitch with
penalty areas. Label placement avoids nearby labels and markers. Colors are not
team predictions. The frame counter uses the 500 processed frames, not the 750
source frames.

The main minimap shows tracked finite positions inside the canonical field.
Outside-field detections remain visible in gray on the match image and stay in
the JSON. This is a display filter, not role classification. A border-touching
box is marked `*` because its foot position can be unreliable.

Trails contain up to 0.4 seconds of observed points and reset after a missing
frame or >3 m consecutive displacement. They do not interpolate, smooth, modify
or certify the underlying coordinates. The original diagnostic view, including
the cyan calibration overlay, is still available without `--clean`.

## Reproduce

Use fresh output directories:

```powershell
.venv/Scripts/python.exe tools/retune_tracking.py --run outputs/match_20s --output outputs/match_20s_tuned_050 --threshold 0.5
.venv/Scripts/python.exe tools/review_coordinates.py --run outputs/match_20s_tuned_050 --output outputs/match_20s_improved_final --clean
.venv/Scripts/python.exe -m pytest tests -q
```

The replay tool accepts trusted local MVP state only. Full-pipeline role/team
votes must not be reused with new track IDs. Regression tests cover retaining
input geometry and detection indices, aging tracks through empty frames,
rejecting discontinuous frame metadata, and clearing trails at gaps/jumps.
