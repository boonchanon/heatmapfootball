"""Isolated tests of edited upstream methods, without importing neural networks.

Extract the actual method AST, not a copy of its implementation. This deliberately
does not test module imports/model construction; full inference remains a separate
gate. Only the torch.no_grad decorator is omitted in the CPU-only test environment.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def method_from_source(relative_path, class_name, method_name, **extra_globals):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == method_name)
    method.decorator_list = []
    namespace = dict(np=np, pd=pd, Any=Any, **extra_globals)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[method_name]


def test_calibration_reads_nonzero_frame_index():
    observed = []
    camera = SimpleNamespace(update=observed.append,
                             get_homography_from_ground_plane=lambda **_: None)
    instance = SimpleNamespace(cam=camera, use_prev_homography=False)
    process = method_from_source('sn_gamestate/calibration/nbjw_calib.py',
                                 'NBJW_Calib', 'process')
    detections = pd.DataFrame({'bbox_ltwh': [[0, 0, 10, 20]]}, index=[17])
    images = pd.DataFrame({'keypoints': [{1: {'x': 0.2, 'y': 0.3}}]}, index=[43])
    output, metadata = process(instance, None, detections, images)
    assert observed == [images.iloc[0].keypoints]
    assert list(output.index) == [17]
    assert output.bbox_pitch.iloc[0] is None
    assert list(metadata.index) == [43]
    assert metadata.parameters.iloc[0] == {}


def test_team_assignment_preserves_detection_ids_and_unknown_tracks():
    process = method_from_source('sn_gamestate/team/tracklet_team_clustering_api.py',
                                 'TrackletTeamClustering', 'process')
    rows = pd.DataFrame({
        'track_id': [7., 7., np.nan],
        'role': ['player', 'player', 'other'],
        'embeddings': [np.array([1., 2.])] * 3,
    }, index=pd.Index([31, 90, 105], name='detection_id'))
    result = process(None, rows, pd.DataFrame())
    pd.testing.assert_index_equal(result.index, rows.index)
    assert list(result.team_cluster.iloc[:2]) == [0, 0]
    assert pd.isna(result.team_cluster.iloc[2])
    assert 'team_cluster' not in rows


def test_team_assignment_empty_input():
    process = method_from_source('sn_gamestate/team/tracklet_team_clustering_api.py',
                                 'TrackletTeamClustering', 'process')
    rows = pd.DataFrame(columns=['track_id', 'role', 'embeddings'])
    result = process(None, rows, pd.DataFrame())
    assert result.empty
    assert 'team_cluster' in result
