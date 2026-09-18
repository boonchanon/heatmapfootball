# Real-video coordinate review (2026-09-16)

## 20-second test deliverable (2026-09-17)

The requested longer MVP run is complete: `outputs/match_20s` processes frames
0 through 499 at 25 FPS. `outputs/match_20s_review/review.mp4` is the completed
20-second annotated video; `tracking.json` and `metadata.json` in the same folder
contain the matching data. These files supersede the 4-second example for testing.

Validation decoded all 500 output frames at 1500x720, confirmed 20.0 s duration,
and checked every exported timestamp against frame / FPS. There are 5,055 tracked
positions across 44 track IDs, including 313 outside-field positions. Track IDs
are not a player count. `validation.json` records these checks. This run uses MVP,
so role, team and jersey attributes remain null; quality is still unverified.

The source-video metadata still correctly describes the 30-second source clip;
the `processing` object describes the 20-second analyzed segment.

## Original 4-second review

Input: `test_videos/match_test.mp4`, 1920x1080, 25 FPS, 750 frames.
Reviewed state: `outputs/match_test/tracker_state.pklz`, first 100 frames (4 s).
This review reuses the original MVP predictions; it is not a new inference run.

## Findings

- 1,956 person detections, including 1,435 tracked detections across 17 IDs.
- 204 detections project outside the canonical 105x68 m field. Of these,
  100 have a track ID, and **all 100 belong to track 7**. The other 104 detections
  have no assigned track ID and were absent from the trajectory JSON.
- Visual inspection of frames 0, 50 and 99 shows track 7 on a dark-clothed person
  standing outside the near touchline. Its outside-field coordinates are consistent
  with the image; this is not evidence of an axis swap or unit conversion error.
  The person's precise role cannot be established from these images alone.
- Track 7 has x between -5.42 and -4.81 m, y between 34.45 and 37.34 m.
  The nominal near touchline is y=34 m.
- Its bounding box touches the image border in 68 of 100 frames. The box's lower
  midpoint is then an unreliable substitute for the person's true ground contact.
  This limits the accuracy of the exact outside-field distance.
- The projected halfway line and center circle broadly align with the actual
  markings in the sampled frames. Visible touchline offsets remain. There is no
  independent ground truth or quantitative calibration accuracy certification.
- MVP detects people, including people beside the field; it has no role classifier.
  Track IDs are not a verified player roster. Do not interpret every track as a player.

## Artifacts and reproduction

```powershell
.venv/Scripts/python.exe tools/review_coordinates.py --run outputs/match_test --output outputs/match_coordinate_audit
```

Use a new output directory for a second run. The script reads trusted local pickle
state and writes `review.mp4`, three annotated sample images,
`coordinate_audit.json`, and updated `tracking.json` / `metadata.json`.
The original inference output is preserved.

The cyan overlay reconstructs the saved pixel-to-field mapping from the detection
input/output pairs. Its agreement with the actual painted lines is a visual check;
the reconstructed mapping itself is **not** independent evidence of accuracy.

## Export behavior

Each position now includes `pitch_valid` (finite coordinates only), `inside_pitch`
(null when projection is missing), and `bbox_truncated` when a usable image bbox
is available. Original coordinates remain unclipped, including outside-field points.
`role`, `team` and `jersey_number` preserve upstream predictions when present and
are null when unavailable, such as in MVP. `quality_verified` remains false.

Consumers can filter `inside_pitch == true` for field occupancy, but should not
use that flag alone as a player classification: a player can leave the field and
a referee can stand inside it. Role predictions also require visual validation.

## Full-stack installation

The existing Visual Studio Build Tools installation at `D:/DVS2026BuildTools`
was missing the C++ compiler. Added `Microsoft.VisualStudio.Component.VC.Tools.x86.x64`
and `Microsoft.VisualStudio.Component.Windows11SDK.26100` without restarting.

Installed into the project `.venv`, preserving torch 1.13.1+cu117 and torchvision
0.14.1+cu117:

| Package | Version / source |
| --- | --- |
| prtreid | 1.3.1, source commit 30617a75967e84d5d516959c4b84cbeea6f56493 |
| torchreid | 1.2.4, BPBreID source commit 02570d5c893977685de7a547a58c0aa87fa4788c |
| mmcv | 2.0.1, official Windows CPython 3.9 CUDA 11.7 / torch 1.13 wheel |
| mmengine | 0.10.7 |
| mmocr | 1.0.1 |
| mmdet | 3.1.0 |
| albumentations | 1.3.1 |
| monai | 1.3.2 |
| tb-nightly | 2.20.0a20250501 |

The two Re-ID extensions were compiled from the cached source revisions, using
isolated build dependencies numpy 1.26.4, Cython 3.0.12 and setuptools 75.8.2.
Runtime support dependencies were also installed. This is a tested Windows
environment, not a claim that an unmodified `uv sync --locked` reproduces it.

Checks passed: full dependency metadata check, `uv pip check`, imports of PRTReId,
StrongSORT, MMOCR and team clustering, and actual `mmcv.ops.nms` execution on CUDA.
Official references: [MMOCR installation](https://mmocr.readthedocs.io/en/v1.0.1/get_started/install.html)
and [MMCV wheel index](https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html).

## Full-pipeline proof and remaining accuracy problems

The full pipeline completed successfully on CUDA with exit code 0:

```powershell
.venv/Scripts/python.exe tools/analyze_video.py --video test_videos/match_test.mp4 --output outputs/match_full_probe --stage full --device cuda --max-frames 10
```

`outputs/match_full_probe/inference_evidence.json` records 196 detections, all 196
tracked and projected, with 22 track IDs across 10 frames (0.4 s). This validates
execution of detection, Re-ID, StrongSORT, calibration, OCR, voting and team
clustering. It does not validate identity, team or jersey accuracy.

- Roles after voting: 175 player, 10 goalkeeper, 10 referee, and 1 ball detection.
  In particular, the dark-clothed person outside the near touchline is incorrectly
  labeled goalkeeper. The detector is a person detector; the single ball role is
  also not proof of ball tracking.
- Teams after voting: 135 left, 50 right, 11 missing. These are model labels, not
  verified home/away affiliations. The exporter now normalizes the upstream string
  `"nan"` to JSON null; original inference files are preserved as run evidence.
- All 196 jersey predictions are missing. OCR model initialization and inference
  succeed, but no jersey number was recognized in this short segment. A separate
  real crop OCR smoke test also completed with no recognized text.
- Both Re-ID checkpoint MD5 hashes passed. The native Cython ranking extensions
  for both Re-ID packages import successfully. OCR checkpoints are also downloaded.
- Removed the unused duplicate combined OCR inferencer, which previously loaded
  an additional detector and recognizer. Future runner executions keep Torch Hub
  checkpoints under the selected `model_dir/torch_hub`.

The 4-second MVP coordinate audit and the 0.4-second full-pipeline probe are
different runs with different trackers; track IDs are not interchangeable.
Further quality work requires labeled examples and longer clips, particularly
for role mistakes, team assignment, jersey visibility and ID continuity.
