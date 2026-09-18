# Custom video audit

Historical source audit. For subsequent real-video results and resolved dependency
blockers, see [REAL_VIDEO_REVIEW.md](REAL_VIDEO_REVIEW.md).

Audit date: 2026-09-16. Repository commit: `1c958345067218297d221e45e1a6405f975f83e0`.
Completed before custom pipeline implementation. This document distinguishes source
inspection from actual inference: **no real custom football clip has yet been run**.

## Sources and dependencies

Read `README.md`, `FAQ.md`, `pyproject.toml`, `uv.lock`,
`plugins/calibration/pyproject.toml`, `sn_gamestate/config_finder.py`,
`sn_gamestate/configs/soccernet.yaml`, the dataset, visualization and module configs,
and the implementations listed below. Generic components are not in this repository.
For those, inspected the **published TrackLab 1.3.24 wheel**, downloaded without
installing it into the existing Python environment. Its extracted sources are in
the ignored `.audit/tracklab/` directory. Paths prefixed `tracklab/` and
`bpbreid_strong_sort/` below refer to that exact dependency, not guessed local files.

* Python: project and calibration plugin require `>=3.9,<3.10`.
* PyTorch: project pins `1.13.1`; `uv.lock` resolves torchvision `0.14.1`.
* TrackLab: `1.3.24` pinned in both project and UV lock.
* Other direct requirements include NumPy `1.26.4`, SoccerNet `0.1.55`,
  MMOCR `1.0.1`, MMDetection `~=3.1.0`, Lightning `2.0.9`, EasyOCR `1.7.1`,
  and PRTReID / BPBreID Git dependencies. Calibration uses Kornia `0.6.3`.
* README install: `uv venv --python 3.9`, `uv pip install -e .`,
  `uv run mim install mmcv==2.0.1`. The local calibration source is defined under
  `[tool.uv.sources]`; ordinary pip does not interpret that table. When installing
  with pip, supply `plugins/calibration` explicitly. Do not assume a plain editable
  install reproduces the checked-in lock.
* README's conda recipe specifies PyTorch CUDA 11.7. This is an example environment,
  not a requirement that the installed driver report exactly CUDA 11.7.
* TrackLab selects MPS/CUDA/CPU in `tracklab/main.py:init_environment`.
  NBJW loads with `map_location=device` and `.to(device)`; the BPBreID tracker
  computes its part-based feature distances with `use_gpu=False`. No mandatory
  CUDA operation was found in this selected path. Full CPU execution is **unverified**,
  especially transitive PRTReID/MMCV dependencies. PRTReID sets `cfg.use_gpu` from
  CUDA availability independently of the supplied device.

## Actual execution chain

`tracklab/main.py:main` uses Hydra `instantiate` to construct the dataset and
modules, builds `tracklab.pipeline.Pipeline` (implemented in
`tracklab/pipeline/module.py`), creates `TrackerState`, and calls
`OfflineTrackingEngine.track_dataset`. The offline engine processes one module
over the video before the next module; it does not process the entire chain once
per frame. `sn_gamestate.config_finder.ConfigFinder` exposes the local config package.

| Stage | Actual implementation | Main inputs / outputs |
| --- | --- | --- |
| Detection | `tracklab/wrappers/bbox_detector/yolo_ultralytics_api.py:YOLOUltralytics.__init__/preprocess/process` | YOLO person class 0, confidence >=0.4; `image_id`, `video_id`, `bbox_ltwh`, `bbox_conf`, category 1 |
| Re-ID and role | `sn_gamestate/reid/prtreid_api.py:PRTReId.__init__/download_models/preprocess/process` | Crops -> embeddings, visibility scores, body masks, `role_detection`, `role_confidence` |
| Tracking | `tracklab/wrappers/track/bpbreid_strong_sort_api.py:BPBReIDStrongSORT.reset/preprocess/process`; `bpbreid_strong_sort/strong_sort.py:StrongSORT.update` | Bboxes and embeddings -> track IDs, Kalman bboxes, matching costs/state/age/hits |
| Pitch landmarks | `sn_gamestate/calibration/nbjw_calib.py:NBJW_Calib_Keypoints.__init__/preprocess/process`, `kp_to_line` | Resize to 960x540, HRNet keypoint and line heatmaps -> normalized keypoints and lines |
| Calibration and projection | Same file: `NBJW_Calib.process`, `get_bbox_pitch`; `plugins/calibration/nbjw_calib/utils/utils_calib.py:FramebyFrameCalib.get_homography_from_ground_plane` | RANSAC ground-plane homography -> `bbox_pitch` and camera `parameters` |
| Jersey recognition | `sn_gamestate/jersey/mmocr_api.py:MMOCR.preprocess/process/run_mmocr_inference` | DBNet + SAR -> `jersey_number_detection`, `jersey_number_confidence` |
| Attribute voting | `tracklab/wrappers/tracklet_agg/majority_vote_api.py:MajorityVoteTracklet.process` | Confidence-weighted tracklet jersey/role voting |
| Teams | `sn_gamestate/team/tracklet_team_clustering_api.py:TrackletTeamClustering.process` | Mean player embeddings per track -> two KMeans clusters |
| Team sides | `sn_gamestate/team/tracklet_team_side_labeling_api.py:TrackletTeamSideLabeling.process` | Mean pitch x of clusters -> `left`/`right`; goalkeeper side from x sign |

This is the order in `soccernet.yaml`. Roles are player, goalkeeper, referee,
other, ball, or unknown in the Re-ID mapping. Detector still detects people;
this is not a ball-tracking pipeline. Team labels are **not home/away**. If pitch
coordinates are missing, team-side code can still assign sides via its NaN
comparison branch; those labels must not be presented as reliable team identities.

Other inspected calibration paths: `calibration/pitch.py:BaselinePitch`,
`calibration/baseline.py:BaselineCalibration`, `calibration/tvcalib.py:TVCalib`,
`calibration/pnlcalib.py:PnLCalib`, and
`calibration/bbox2pitch.py:Bbox2Pitch/get_bbox_pitch/get_bbox_pitch_homography`.
These are alternatives, not additional stages of the current NBJW default.
The calibration plugin package list does not include `pnlcalib`, another reason
not to switch to that alternative without installation testing.

## Weights

| Component | Configured checkpoint / source |
| --- | --- |
| Person detector | `pretrained_models/yolo/yolo11m.pt`, Ultralytics automatic download |
| Re-ID | `pretrained_models/reid/prtreid-soccernet-baseline.pth.tar`, Zenodo record 10653453, MD5 checked in `PRTReId.download_models` |
| Re-ID backbone | `pretrained_models/reid/hrnetv2_w32_imagenet_pretrained.pth`, Zenodo record 10604211 |
| NBJW keypoints | `pretrained_models/calibration/SV_kp`, Zenodo record 12626395 |
| NBJW lines | `pretrained_models/calibration/SV_lines`, same record |
| Jersey OCR | MMOCR identifiers `dbnet_resnet18_fpnc_1200e_icdar2015` and `SAR`; downloaded by MMOCR |

At initial inspection the repository had only calibration mean/std arrays and a
radar image, not these learned weights. No new detector or tracker is warranted.

## Inputs, persistence, and existing visualization

The default `soccernet_gs` dataset uses SoccerNet splits and annotations, but
**SoccerNet dataset structure is not necessary for custom inference**.
`sn_gamestate/configs/dataset/youtube.yaml` points to
`tracklab/wrappers/dataset/external_video.py:ExternalVideo`. Despite the config
name, it accepts a local MP4. It creates a `val` TrackingSet with zero-based
`frame`, image IDs, video IDs and `vid://...:<frame>` file paths, with no detections GT.
Its current implementation reads from the video; its docstring about always
extracting images is stale. The same file also provides
`write_video_images_to_disk`, an existing zero-based JPEG extraction helper.

`tracklab/datastruct/tracker_state.py:TrackerState.save/load` persists a ZIP-based
`.pklz` containing per-video detection and image DataFrames and `summary.json`.
Detection rows already contain the position data needed downstream:
`bbox_pitch.x_bottom_middle`, `bbox_pitch.y_bottom_middle`, `track_id`, and
`image_id`. Join image IDs to image metadata to obtain frame numbers; never assume
an arbitrary detection index is a frame. Existing benchmark/MOT exports are not
the requested frontend trajectory schema.

`sn_gamestate/visualization/pitch.py:Pitch/Radar/draw_pitch/draw_radar_view`
and `visualization/players.py:CompletePlayerEllipse` already draw the pitch,
minimap and player labels. `tracklab/visualization/visualization_engine.py` writes
`visualization/videos/<name>.mp4`. Its default FPS is 25, and radar placement uses
hard-coded 1920x1080 coordinates. Preview at arbitrary sizes needs additional
work and should not gate position inference.

## Coordinates and validity

`plugins/calibration/nbjw_calib/utils/utils_calib.py` defines landmarks on a
105x68 canonical field and explicitly subtracts `(52.5, 34)` from them.
`get_homography_from_ground_plane(inverse=True)` returns image -> pitch.
`nbjw_calib.get_bbox_pitch` projects the lower left, lower right and lower middle
of each detection bbox onto ground plane Z=0. The lower middle is the desired
player ground-contact proxy, not bbox center and not necessarily the actual feet.

Native coordinates are already canonical meters: x nominally [-52.5,52.5],
y [-34,34], center origin. x increases toward the right canonical goal and
y toward the bottom canonical touchline. There is no attacking-direction
normalization. If corner-origin coordinates are later needed, explicitly use
`x_corner = x_native + 52.5`, `y_corner = y_native + 34`, with tests. These are
model field dimensions, **not measured dimensions of the uploaded match venue**.

Missing homography produces `bbox_pitch=None` unless `use_prev_homography=True`.
Upstream enables reuse and carries `last_h/last_params` without a shot boundary
check. Custom inference must disable this reuse so failed frames remain missing.
NaN checks in upstream projection do not catch infinity; downstream validity must
check both coordinates are finite. Off-field coordinates should not be clipped
into plausible positions. Numerical finiteness is not proof of calibration accuracy.

## Tracking quality and compatibility findings

* StrongSORT matches appearance with Kalman gating, then unmatched detections
  using IoU. EMA appearance features help partial occlusion but cannot guarantee ID.
* Config uses `max_age=300`, `n_init=0`, appearance threshold 0.5, IoU threshold
  0.8, and prediction cutoff 7. These are lifecycle settings, not a guarantee of
  re-identification after 300 physical video frames. Empty-detection processing
  returns early in the wrapper and must be considered when evaluating long gaps.
* Similar shirts and partial occlusion can cause ID switches. Deleted tracks are
  not a persistent roster; return after leaving the image may receive a new ID.
* Camera compensation code exists (`prepare_next_frame`, ECC), but config disables
  ECC and the inspected offline path does not call `prepare_next_frame`.
* No shot detection, replay exclusion, cross-cut identity reconciliation, or
  full-match identity guarantee was found. Pan/zoom affect matching, and cuts and
  replays require segmentation/validation before heatmap or distance use.
* `matched_with`, `costs`, `hits`, `age`, `time_since_update`, `state` exist.
  They are diagnostics, not a calibrated tracking confidence probability.
* TrackLab `utils/cv2.py:cv2_load_image` splits video URIs on every `:` (breaks
  Windows drive letters); `VideoReader.__getitem__` seeks `idx-1` although
  ExternalVideo frames are zero-based. Reuse the existing JPEG extraction helper
  and replace dataset file paths, retaining zero-based frame metadata.
* `NBJW_Calib.process` reads `metadatas['keypoints'][0]`, label-based lookup in a
  pandas Series with integer index. For later batches use `.iloc[0]` and enforce
  batch size 1 in custom config; otherwise later frame IDs may raise KeyError.
* Default NBJW dimensions 1920x1080 must be replaced with decoded video dimensions.
* `TrackletTeamClustering.process` merges with a fresh RangeIndex. This can break
  TrackLab row identity if detection IDs have gaps; a focused fix must preserve IDs.
* TrackLab main calls Unix-style multiprocessing sharing setup unconditionally.
  A small programmatic runner can reuse Pipeline/TrackerState/OfflineTrackingEngine
  without invoking that environment helper or the unused GT evaluator.

## Initial environment and implementation decision

Initial environment: Python 3.11, torch 2.7.1+cu118, pandas 2.3.3, OpenCV 4.13,
pytest 7.2, no installed TrackLab, no Python 3.9 discovered by the launcher.
NVIDIA RTX 3050 laptop GPU, 4 GB VRAM; driver reports CUDA capability 13.1.
No MP4 was found in the repository; requested the user's football clip path.

Proceed with an isolated environment and a thin custom-video runner that reuses
TrackLab and existing SoccerNet modules, uses decoded dimensions, avoids previous
homography reuse, saves raw tracker state, and reports evidence honestly. Complete
real detection/tracking/calibration proof before claiming success or starting the
HTTP API. No frontend is needed. The ordered proof gate also precedes the final
frontend tracking JSON export; unit fixtures are not real inference evidence.
