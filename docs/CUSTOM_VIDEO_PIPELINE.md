# Custom MP4 pipeline: implementation and proof status

See [the real-video review](REAL_VIDEO_REVIEW.md) for the latest coordinate audit,
completed full-stack dependency installation and validation evidence. Historical
blockers below describe the initial setup and are superseded by that review.

**MVP status (2026-09-16): MVP inference has produced real-video output;
quality remains unverified.** The saved `outputs/match_test/inference_evidence.json`
records 100 processed frames from a 750-frame clip, 17 tracks and 1,435 tracked
finite pitch positions on CUDA. MVP uses YOLO, ByteTrack and NBJW calibration;
it does not run Re-ID, jersey recognition or team classification. Full inference
now has the Re-ID and OpenMMLab dependencies installed. The installation narrative and
acceptance table below describe the earlier audit, not the latest MVP results.

Use the MVP-specific environment check and a fresh output folder:

```powershell
.venv/Scripts/python.exe tools/analyze_video.py --stage mvp --check-environment
.venv/Scripts/python.exe tools/analyze_video.py --video test_videos/match_test.mp4 --output outputs/match_test_fixed --stage mvp --device cuda --max-frames 100
```

The CLI now checks MVP's completion gate. New JSON exports accept NumPy bounding
boxes and include processed frame coverage separately from source video duration.
Existing output files are not automatically rewritten. Finite coordinates alone
do not establish correct calibration or stable player identities.

See [CUSTOM_VIDEO_AUDIT.md](CUSTOM_VIDEO_AUDIT.md) for the source-based architecture,
versions, weights, coordinate convention, and tracking limitations.

## Architecture

`tools/analyze_video.py` calls `sn_gamestate.custom_video.runner.run_video`.
It decodes local MP4 metadata, composes the existing SoccerNet Hydra config with
`custom_video.yaml`, constructs TrackLab's `ExternalVideo`, and calls the existing
`Pipeline`, `TrackerState`, and `OfflineTrackingEngine`. The user does not create
any SoccerNet dataset directories or annotations.

The TrackLab 1.3.24 direct video reader has Windows drive-colon and frame-offset
issues. The runner reuses TrackLab's `write_video_images_to_disk` and points image
metadata at the generated JPEGs. It validates file existence and frame counts.
This takes additional disk space and multiple video reads, and introduces the
existing helper's JPEG compression. Intended initial use: a short single-shot clip.

Changes to model behavior are limited to actual video dimensions, batch sizes,
and disabling previous-homography reuse. Two upstream correctness fixes preserve
nonzero frame indexing and detection indices during team clustering. No model,
tracker, or projection algorithm has been replaced.

## Installation and current blocker

Supported project Python is **3.9**, with torch **1.13.1**, torchvision **0.14.1**,
and TrackLab **1.3.24**. The existing machine-wide Python 3.11 / torch 2.7 environment
does not match. An isolated Python 3.9.25 and `.venv` were created in the workspace;
system Python was not modified.

From a shell with `uv` installed, the intended locked installation is:

```powershell
uv venv --python 3.9
uv sync --locked
uv run --no-sync mim install mmcv==2.0.1
.venv/Scripts/python.exe tools/analyze_video.py --check-environment
```

On Linux, use `.venv/bin/python` instead. The lock includes the local calibration
plugin and pins the Re-ID Git revisions. MMCV is a separate step in upstream's
README and is not present in the current lock. Inspect its install/import result;
the command above is not a claim that a compatible compiled wheel is available
for every platform. `--check-environment` verifies package metadata only, not DLLs,
CUDA kernels or model initialization. Its success is only readiness for import checks.

**Actual full-install failure on this Windows host:** building
`torchreid @ ...bpbreid@02570d5c893977685de7a547a58c0aa87fa4788c` ended with:

```text
Microsoft Visual C++ 14.0 or greater is required.
```

No usable compiler was found via `cl`/Visual Studio installer discovery. WSL did
not expose an installed Linux distribution. A compatible C++ build environment
or an available Linux environment is required before retrying the locked install.
Installing just a similarly named PyPI Re-ID package is not a valid substitute.

The current `.venv` contains **test dependencies and TrackLab/config package
metadata installed without full dependencies**. It is not an inference environment.
Do not run `uv sync` and mistake successful resolution for successful installation.

## Device and weights

`--device auto` selects CUDA if torch reports it available, otherwise CPU, and
prints the selected device. `--device cuda` fails if unavailable; it never silently
changes to CPU. CPU operation of the entire dependency stack remains unverified.
PRTReID internally checks CUDA availability in addition to accepting a device;
for a strict CPU-only process, hide CUDA before process startup if necessary.

The RTX 3050 here has 4 GB VRAM. Custom config uses small batches, but full-model
memory consumption is unmeasured. The CPU torch wheel resolved for Windows may
not enable this GPU. Choose a matching torch/torchvision CUDA build if needed;
do not infer runtime compatibility merely from the driver's displayed CUDA version.

Existing modules automatically fetch YOLO11m, PRTReID SoccerNet, HRNet32, NBJW
`SV_kp` and `SV_lines`, and (in full stage) MMOCR DBNet/SAR weights. These downloads
have **not** been performed by the custom runner. `--model-dir` selects their
storage directory; default is the repository's `pretrained_models` directory.

## Commands and ordered proof gates

After installation, supply an actual short football MP4. Use a fresh output
directory for each run; existing nonempty directories are rejected so old tracker
state cannot masquerade as new output.

```powershell
.venv/Scripts/python.exe tools/analyze_video.py --video path/to/match.mp4 --output outputs/match_detect --stage detect
.venv/Scripts/python.exe tools/analyze_video.py --video path/to/match.mp4 --output outputs/match_track --stage track
.venv/Scripts/python.exe tools/analyze_video.py --video path/to/match.mp4 --output outputs/match_calibrate --stage calibrate
.venv/Scripts/python.exe tools/analyze_video.py --video path/to/match.mp4 --output outputs/match_full --stage full
```

Stages reuse prefixes of the audited pipeline:

* `detect`: YOLO person detection.
* `track`: detection, PRTReID, BPBreID StrongSORT.
* `calibrate` (default): those stages, NBJW landmarks and calibration/projection.
* `full`: all of the above plus jersey OCR, role/jersey voting, team clustering
  and left/right side labels. This step is not needed to establish the initial
  geometric proof, and costs more memory/time.

Exit code 1 indicates an exception, 2 indicates invalid arguments or failed
environment check, 3 indicates completion without the stage's required detections
or tracked finite pitch points. Exit 0 means those numerical counts are nonzero;
it does **not** prove identity continuity or metric calibration accuracy. Review
the actual short clip against the predicted tracks and field landmarks before
accepting real pitch positions. There is no claim of cross-cut tracking.

## Current output contract

On a successful run, the implemented runner saves:

```text
outputs/<run>/
  resolved_config.yaml
  tracker_state.pklz
  inference_evidence.json
  tracking.json
  metadata.json
  tmp/<video_stem>/<video_stem>_000000.jpg
  ...
```

`tracker_state.pklz` is TrackLab's native persisted detection/image DataFrames.
It contains IDs, bboxes, pitch positions, frame metadata, and diagnostics produced
by the selected stage. Preserve it to avoid re-running AI for subsequent export.
`inference_evidence.json` contains actual numerical counts, video metadata, stage,
device and dependency versions. `quality_verified` is always false: automated
counts do not validate accuracy. No fake positions or success records are emitted.

The runner now writes `tracking.json` and `metadata.json` from the actual
TrackLab DataFrames after inference. The exporter preserves missing calibration
as JSON `null`, sorts positions by zero-based frame, and never substitutes image
coordinates for pitch meters. A successful real run is still required before
these files can be treated as proof of valid pitch coordinates. `preview.mp4`
is not implemented; existing visualization was audited, but its fixed radar
layout is not enabled for arbitrary-sized custom clips.

## Frontend schema and coordinates (design for the next gated step)

The runner implements the following mapping:

| Frontend field | Audited source / meaning |
| --- | --- |
| `video.fps`, width, height, duration | Decoded metadata; duration = decoded count / nominal FPS |
| `players[].track_id` | TrackLab `track_id`, video-local identity only |
| `team`, `role`, `jersey_number` | Actual voted upstream attributes, null when absent; team is left/right, never inferred home/away |
| `positions[].frame` | Join detection `image_id` to image metadata's zero-based `frame` |
| `time` | `frame / fps`; nominal timing, not VFR presentation timestamp or match clock |
| `image_x`, `image_y` | Lower middle of `bbox_ltwh`: left + width/2, top + height |
| `pitch_x`, `pitch_y` | `bbox_pitch.x_bottom_middle`, `y_bottom_middle`; finite pair only, else both null |
| `detection_confidence` | `bbox_conf`; do not label this tracking confidence |
| tracking diagnostics | Existing `matched_with`, `costs`, hits, age, state, time_since_update |
| validity | Separate finite projection, inside canonical field, and any calibration quality assessment |

Retain the native center-origin meter convention: x [-52.5,52.5], y [-34,34]
on the canonical NBJW 105x68 field. No rescaling to measured venue dimensions is
justified by this implementation. No coordinate transformation is currently added.
An optional corner-origin conversion must explicitly add (52.5,34) and have tests.
Do not clamp off-field projections or substitute image pixels for pitch meters.

The export sorts positions by frame/time, serializes missing values as JSON null
(never NaN/Infinity), skips untracked detections, and retains missing attributes
honestly. It does not perform a coordinate transformation.

## Heatmap and HTTP integration (pending)

Once the exporter passes the real-video gate, a frontend can select the persisted
player ID, filter by video time, select finite/valid pitch points, and bin them in
native field coordinates. Use duration weighting when interpreting occupancy;
gaps, out-of-view players, replay frames and ID switches bias heatmaps. Do not
interpolate across cuts or calculate distance/speed across missing spans without
an explicit quality policy. Detection trajectories are not a whole-match roster.

API work is intentionally gated on real custom inference success. Future upload,
analyze/status/tracking/players/positions endpoints should run this CLI in an
isolated worker process, since the adapter changes its process working directory.
No HTTP handler should directly call it concurrently. The lock already includes
FastAPI and python-multipart transitively via Lightning, but that does not mean
a backend exists or that direct application dependencies can be omitted later.

## Tests and completed validation

```powershell
.venv/Scripts/python.exe -m pytest tests -q
git diff --check
```

Latest result: **29 passed, 0 failed**. Tests cover real MP4 encoding/decoding of
clearly synthetic color-frame fixtures, metadata JSON serialization, invalid FPS,
invalid/empty files, zero-based frame file mapping, empty tracking diagnostics,
missing/NaN/Infinity projections, all four real Hydra configurations, restoration
of working directory, and actual NBJW homography fitting against known geometric
correspondences. Regression tests execute the edited methods extracted from their
source AST to avoid importing neural networks; they do not validate heavy imports.
No test downloads model weights. Exported player ordering and missing calibration
serialization are covered by focused exporter tests.

## Commands actually executed in this workspace

Read-only inspection used `rg --files`, `rg -n`, `Get-Content`,
`Get-ChildItem`, `git status --short`, `git rev-parse HEAD`, `py -0p`,
`python -m pip show tracklab torch opencv-python pandas pytest`, `nvidia-smi`,
compiler discovery and `wsl --list --quiet`. File-by-file component references
are in the audit. The network-restricted first TrackLab download failed with
WinError 10013; the reviewed network-enabled retry succeeded.

```powershell
python -m pip download tracklab==1.3.24 --no-deps --dest .audit/vendor
python -m zipfile -e .audit/vendor/tracklab-1.3.24-py3-none-any.whl .audit/tracklab
python -m pip install uv --target .audit/tooling --no-deps --no-warn-script-location
$env:UV_PYTHON_INSTALL_DIR = Join-Path (Get-Location) '.audit/python'
.audit/tooling/bin/uv.exe python install 3.9 --cache-dir .audit/cache --no-bin
.audit/tooling/bin/uv.exe venv .venv --python .audit/python/cpython-3.9.25-windows-x86_64-none/python.exe --cache-dir .audit/cache
.audit/tooling/bin/uv.exe sync --locked --dry-run --python .audit/python/cpython-3.9.25-windows-x86_64-none/python.exe --cache-dir .audit/cache
.audit/tooling/bin/uv.exe sync --locked --python .audit/python/cpython-3.9.25-windows-x86_64-none/python.exe --cache-dir .audit/cache
```

The last command failed building torchreid. To validate independent work, installed
only the test stack and package/config metadata (not a replacement inference setup):

```powershell
.audit/tooling/bin/uv.exe pip install --python .venv/Scripts/python.exe --cache-dir .audit/cache hydra-core==1.3.2 pytest==8.3.5 numpy==1.26.4 pandas==2.2.3 opencv-python==4.11.0.86 scipy==1.13.1
.audit/tooling/bin/uv.exe pip install --python .venv/Scripts/python.exe --cache-dir .audit/cache --no-deps .audit/vendor/tracklab-1.3.24-py3-none-any.whl -e . -e plugins/calibration
python tools/analyze_video.py --help
python tools/analyze_video.py --check-environment
.venv/Scripts/python.exe tools/analyze_video.py --check-environment
.venv/Scripts/python.exe -m pytest tests -q
git diff --check
```

Both environment checks reported missing/incompatible inference dependencies.
Tests were first run with 25 passing cases, then rerun with the three added
regression cases: 28 passed. No real inference command was run without a clip.

## Acceptance status

| Item | Result |
| --- | --- |
| Repository audit | Complete |
| Custom MP4 end-to-end inference | **FAIL / blocked before inference** |
| Player Detection on real clip | NOT TESTED |
| Player Tracking on real clip | NOT TESTED |
| Team Classification on real clip | NOT TESTED |
| Camera Calibration on real clip | NOT TESTED |
| Pitch Projection on real clip | NOT TESTED |
| Real 2D Pitch Coordinates | NOT TESTED |
| Frontend JSON Export | NOT TESTED / pending proof gate |
| API | Not started, as required by the gate |

There are no real-video `tracking.json`, `metadata.json`, or `preview.mp4` output
paths to report. Supply the football clip path and resolve the native build
toolchain before continuing the remaining ordered steps.
