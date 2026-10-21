"""Thin programmatic adapter around TrackLab 1.3.24's existing offline engine."""

import json
import math
import os
import platform
import gc
import shutil
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

from .video import (extracted_frame_paths, inspect_video, resolve_frame_step, stream_resized_frames,
                    validate_max_frames)


STAGES = {
    'mvp': ['bbox_detector', 'reid', 'track', 'pitch', 'calibration'],
    'detect': ['bbox_detector'],
    'track': ['bbox_detector', 'reid', 'track'],
    'calibrate': ['bbox_detector', 'reid', 'track', 'pitch', 'calibration'],
    'full': ['bbox_detector', 'reid', 'track', 'pitch', 'calibration',
             'jersey_number_detect', 'tracklet_agg', 'team', 'team_side'],
}


def memory_checkpoint(phase):
    """Fail safely before the container OOM killer restarts the web service."""
    try:
        rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        limit_text = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        limit = 0 if limit_text == 'max' else int(limit_text)
    except (OSError, ValueError, IndexError):
        print(f'[RAM] phase={phase} rss=unavailable', flush=True)
        return
    ratio = float(os.environ.get('RAM_LIMIT_RATIO', '0.85'))
    print(f'[RAM] phase={phase} rss_mb={rss / 1048576:.1f} limit_mb={limit / 1048576:.1f}', flush=True)
    if limit and rss >= limit * ratio:
        raise MemoryError(f'RAM guard at {phase}: {rss / 1048576:.0f} MB exceeds {ratio:.0%} of limit')


def check_environment(require_optional=True):
    """Report required pinned versions without importing optional model packages."""
    versions = {'python': platform.python_version()}
    errors = []
    if platform.python_version_tuple()[:2] != ('3', '9'):
        errors.append('Inference requires Python 3.9 (see pyproject.toml)')
    for package, required in [('torch', '1.13.1'), ('tracklab', '1.3.24'),
                              ('torchvision', '0.14.1')]:
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed = None
        versions[package] = installed
        if installed is None or installed.split('+')[0] != required:
            errors.append(f'{package} requires {required}; found {installed}')
    packages = ['sn-gamestate', 'tracklab-calibration']
    if require_optional:
        packages += ['prtreid', 'torchreid', 'mmcv', 'mmengine', 'mmocr', 'mmdet']
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
            errors.append(f'Missing package: {package}')
    return {'versions': versions, 'errors': errors,
            'ready_for_import_check': not errors}


@contextmanager
def working_directory(path):
    """Only used by the standalone CLI; not suitable for concurrent HTTP workers."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def build_config(video, output, model_dir, video_metadata, stage):
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf, open_dict

    with initialize_config_module(config_module='sn_gamestate.configs', version_base=None):
        config_name = 'custom_video_mvp' if stage == 'mvp' else 'custom_video'
        cfg = compose(config_name=config_name, overrides=[
            'hydra.searchpath=[pkg://tracklab.configs]',
        ])
    # Set paths as values, never interpolate user paths into Hydra override syntax.
    with open_dict(cfg):
        cfg.project_dir = str(output)
        cfg.home_dir = str(Path.home())
        cfg.model_dir = str(model_dir)
        cfg.data_dir = str(output / 'data')
        cfg.dataset.video_path = str(video)
        cfg.dataset.dataset_path = str(output)
        cfg.pipeline = STAGES[stage]
        cfg.state.save_file = str(output / 'tracker_state.pklz')
        cfg.state.load_file = None
        if stage == 'mvp' and 'hyperparams' in cfg.modules.track.cfg:
            # The tracker sees sampled frames, so its motion model needs the analysed rate.
            cfg.modules.track.cfg.hyperparams.frame_rate = video_metadata.get('analysis_fps', video_metadata['fps'])
        for name in ['pitch', 'calibration']:
            cfg.modules[name].image_width = video_metadata['width']
            cfg.modules[name].image_height = video_metadata['height']
    OmegaConf.resolve(cfg)
    return cfg


def summarize_state(detections, images):
    """Numerical evidence only; does not claim identity or projection accuracy."""
    total = len(detections)
    tracked = 0
    tracks = 0
    projected = 0
    tracked_projected = 0
    if 'track_id' in detections:
        tracked = int(detections.track_id.notna().sum())
        tracks = int(detections.track_id.nunique())
    if 'bbox_pitch' in detections:
        for _, row in detections.iterrows():
            point = row.bbox_pitch
            try:
                valid = isinstance(point, dict) and all(math.isfinite(float(point[k]))
                    for k in ['x_bottom_middle', 'y_bottom_middle'])
            except (KeyError, TypeError, ValueError):
                valid = False
            if valid:
                projected += 1
                track_id = row.get('track_id')
                if track_id is not None and math.isfinite(float(track_id)):
                    tracked_projected += 1
    return {
        'frames': len(images), 'detections': total, 'tracked_detections': tracked,
        'track_count': tracks, 'finite_pitch_positions': projected,
        'tracked_finite_pitch_positions': tracked_projected,
        'quality_verified': False,
    }


def resolve_start_frame(video_metadata, start_seconds=0):
    """Validate start_seconds and convert it to an original-video frame index."""
    try:
        start_seconds = float(start_seconds)
    except (TypeError, ValueError):
        raise ValueError('start_seconds must be a finite number greater than or equal to zero')
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError('start_seconds must be a finite number greater than or equal to zero')
    fps = float(video_metadata['fps'])
    frame_count = int(video_metadata['frame_count'])
    start_frame = round(start_seconds * fps)
    if start_frame < 0 or start_frame >= frame_count:
        raise ValueError('start_seconds is outside the video duration')
    return start_frame


def select_frame_segment(image_metadatas, video_metadata, start_seconds=0, max_frames=None, frame_step=1):
    """Select original video frame rows without changing their frame identifiers.

    ``max_frames`` counts selected rows; with ``frame_step`` > 1 every Nth original
    frame is selected.  ``end_frame`` is one past the last selected original frame.
    """
    if max_frames is not None and max_frames <= 0:
        raise ValueError('max_frames must be positive')
    if frame_step < 1:
        raise ValueError('frame_step must be at least 1')
    frame_count = int(video_metadata['frame_count'])
    start_frame = resolve_start_frame(video_metadata, start_seconds)
    span = (max_frames * frame_step) if max_frames is not None else frame_count - start_frame
    stop = min(frame_count, start_frame + span)
    selected = image_metadatas.iloc[start_frame:stop:frame_step].copy()
    if selected.empty:
        raise ValueError('Selected frame range is empty')
    end_frame = start_frame + (len(selected) - 1) * frame_step + 1
    return selected, start_frame, end_frame


def configure_tracking_set_segment(tracking_set, video_metadata, image_folder, video_stem,
                                   start_seconds=0, max_frames=None, frame_step=1):
    """Synchronize one TrackLab set with an original-frame video segment.

    TrackLab's ReID setup also reads ``dataset.training_sets``.  Therefore the
    caller must apply this operation to both the active and training set before
    constructing modules; keeping only the active set in sync loses image_gt's
    required video_id column in the ReID path.
    """
    selected, start_frame, end_frame = select_frame_segment(
        tracking_set.image_metadatas, video_metadata, start_seconds, max_frames, frame_step)
    selected['file_path'] = extracted_frame_paths(image_folder, video_stem, selected.frame)
    tracking_set.image_metadatas = selected
    # image_gt is consumed by TrackLab/ReID even for external video input.
    # Copy the complete metadata schema, including video_id and original ids.
    tracking_set.image_gt = selected.copy()
    return start_frame, end_frame


def run_video(video, output, model_dir, stage='calibrate', device='auto', max_frames=None, start_seconds=0,
              sample_fps=None):
    video = Path(video).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    model_dir = Path(model_dir).expanduser().resolve()
    if stage not in STAGES:
        raise ValueError(f'Unknown stage: {stage}')
    if device not in {'auto', 'cpu', 'cuda'}:
        raise ValueError(f'Unknown device: {device}')
    max_frames = validate_max_frames(max_frames)
    video_metadata = inspect_video(video)
    frame_step = resolve_frame_step(video_metadata['fps'], sample_fps)
    analysis_fps = video_metadata['fps'] / frame_step
    environment = check_environment(require_optional=stage != 'mvp')
    if environment['errors']:
        raise RuntimeError('\n'.join(environment['errors']))
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty; use a new run directory')

    # Imports are delayed: --help, metadata tests and diagnostics never load models.
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from tracklab.datastruct import TrackerState
    from tracklab.engine import OfflineTrackingEngine
    from tracklab.pipeline import Pipeline
    from tracklab.wrappers.dataset.external_video import ExternalVideo

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing silent CPU fallback')
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(model_dir / 'torch_hub'))
    frame_cache = output / '_frames_tmp'
    modules = pipeline = state = engine = dataset = None
    try:
        start_frame = resolve_start_frame(video_metadata, start_seconds)
        memory_checkpoint('before_decode')
        # Decode sequentially and cap/resize before any model sees a frame.
        streamed = stream_resized_frames(video, frame_cache, max_frames=max_frames,
                                         start_frame=start_frame, frame_step=frame_step)
        memory_checkpoint('after_decode')
        # Pitch/calibration need the resized dims; frame_count/fps stay the source
        # video's, so start_seconds keeps meaning the original timeline below.
        resized_metadata = dict(video_metadata, width=streamed['width'], height=streamed['height'],
                                analysis_fps=analysis_fps)
        cfg = build_config(video, output, model_dir, resized_metadata, stage)
        with working_directory(output):
            OmegaConf.save(cfg, output / 'resolved_config.yaml')
            print(f'Running stage={stage}, device={device}, frame_step={frame_step} '
                  f'({analysis_fps:.2f} analysed frames/s), output={output}', flush=True)
            dataset = ExternalVideo(**OmegaConf.to_container(cfg.dataset, resolve=True))
            image_folder = frame_cache
            tracking_set = dataset.sets['val']
            training_set = dataset.training_sets['val']
            if len(tracking_set.image_metadatas) != video_metadata['frame_count']:
                raise RuntimeError('Frame count changed between metadata scan and TrackLab')
            segment = configure_tracking_set_segment(
                tracking_set, video_metadata, image_folder, video.stem, start_seconds,
                streamed['frame_count'], frame_step)
            training_segment = configure_tracking_set_segment(
                training_set, video_metadata, image_folder, video.stem, start_seconds,
                streamed['frame_count'], frame_step)
            if segment != (streamed['start_frame'], streamed['end_frame']):
                raise RuntimeError('Streamed frame range does not match the TrackLab segment')
            if training_segment != segment:
                raise RuntimeError('TrackLab training set selected a different frame range')
            start_frame, end_frame = segment
            modules = [instantiate(cfg.modules[name], device=device, tracking_dataset=dataset)
                       for name in cfg.pipeline]
            pipeline = Pipeline(models=modules)
            state = TrackerState(tracking_set, pipeline=pipeline,
                                 save_file=output / 'tracker_state.pklz')
            engine = OfflineTrackingEngine(modules=pipeline, tracker_state=state,
                                           num_workers=0, callbacks={})
            memory_checkpoint('before_inference')
            engine.track_dataset()
            memory_checkpoint('after_inference')
            evidence = summarize_state(state.detections_pred, state.image_pred)
            from .export import export_tracking

            tracking, quality_metadata = export_tracking(
                state.detections_pred, state.image_pred, video_metadata, output,
                start_seconds=start_seconds, start_frame=start_frame, end_frame=end_frame)
            # This is run evidence, not frontend tracking export or a fabricated result.
            evidence.update(video=video_metadata, stage=stage, device=device,
                    start_seconds=start_seconds, start_frame=start_frame, end_frame=end_frame,
                    frames_processed=len(state.image_pred), frame_step=frame_step,
                    analysis_fps=analysis_fps,
                    # Detector boxes are in these (resized) pixel coordinates.
                    analysis_width=streamed['width'], analysis_height=streamed['height'],
                            environment=environment['versions'],
                            quality_summary=tracking['quality_summary'],
                            debug_artifacts=['tracking.json', 'metadata.json',
                                             'rejected_records.json'])
            (output / 'inference_evidence.json').write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False),
                encoding='utf-8')
            print(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False))
            return evidence
    finally:
        # Model tensors and decoded frame cache must not survive a job.
        del engine, state, pipeline, modules, dataset
        shutil.rmtree(frame_cache, ignore_errors=True)
        try:
            if 'torch' in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            gc.collect()
