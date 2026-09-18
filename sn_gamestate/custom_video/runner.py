"""Thin programmatic adapter around TrackLab 1.3.24's existing offline engine."""

import json
import math
import os
import platform
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

from .video import extracted_frame_paths, inspect_video


STAGES = {
    'mvp': ['bbox_detector', 'reid', 'track', 'pitch', 'calibration'],
    'detect': ['bbox_detector'],
    'track': ['bbox_detector', 'reid', 'track'],
    'calibrate': ['bbox_detector', 'reid', 'track', 'pitch', 'calibration'],
    'full': ['bbox_detector', 'reid', 'track', 'pitch', 'calibration',
             'jersey_number_detect', 'tracklet_agg', 'team', 'team_side'],
}


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
            cfg.modules.track.cfg.hyperparams.frame_rate = video_metadata['fps']
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


def run_video(video, output, model_dir, stage='calibrate', device='auto', max_frames=None):
    video = Path(video).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    model_dir = Path(model_dir).expanduser().resolve()
    if stage not in STAGES:
        raise ValueError(f'Unknown stage: {stage}')
    if device not in {'auto', 'cpu', 'cuda'}:
        raise ValueError(f'Unknown device: {device}')
    video_metadata = inspect_video(video)
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
    from tracklab.wrappers.dataset.external_video import (
        ExternalVideo, write_video_images_to_disk,
    )

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing silent CPU fallback')
    cfg = build_config(video, output, model_dir, video_metadata, stage)
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(model_dir / 'torch_hub'))
    with working_directory(output):
        OmegaConf.save(cfg, output / 'resolved_config.yaml')
        print(f'Running stage={stage}, device={device}, output={output}', flush=True)
        dataset = ExternalVideo(**OmegaConf.to_container(cfg.dataset, resolve=True))
        image_folder = write_video_images_to_disk(video).resolve()
        tracking_set = dataset.sets['val']
        if max_frames is not None:
            if max_frames <= 0:
                raise ValueError('max_frames must be positive')
            tracking_set.image_metadatas = tracking_set.image_metadatas.iloc[:max_frames].copy()
            tracking_set.image_gt = tracking_set.image_metadatas.copy()
        if len(tracking_set.image_metadatas) != video_metadata['frame_count']:
            if max_frames is None:
                raise RuntimeError('Frame count changed between metadata scan and TrackLab')
        tracking_set.image_metadatas['file_path'] = extracted_frame_paths(
            image_folder, video.stem, tracking_set.image_metadatas.frame)
        tracking_set.image_gt = tracking_set.image_metadatas.copy()
        modules = [instantiate(cfg.modules[name], device=device, tracking_dataset=dataset)
                   for name in cfg.pipeline]
        pipeline = Pipeline(models=modules)
        state = TrackerState(tracking_set, pipeline=pipeline,
                             save_file=output / 'tracker_state.pklz')
        engine = OfflineTrackingEngine(modules=pipeline, tracker_state=state,
                                       num_workers=0, callbacks={})
        engine.track_dataset()
        evidence = summarize_state(state.detections_pred, state.image_pred)
        from .export import export_tracking

        export_tracking(state.detections_pred, state.image_pred, video_metadata, output)
        # This is run evidence, not frontend tracking export or a fabricated result.
        evidence.update(video=video_metadata, stage=stage, device=device,
                frames_processed=len(state.image_pred),
                        environment=environment['versions'])
        (output / 'inference_evidence.json').write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False),
            encoding='utf-8')
        print(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False))
        return evidence
