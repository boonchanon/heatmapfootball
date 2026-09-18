"""Exercise actual NBJW geometric fitting without loading either neural network."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def geometry():
    path = Path(__file__).resolve().parents[1] / 'plugins/calibration/nbjw_calib/utils/utils_calib.py'
    spec = importlib.util.spec_from_file_location('audited_nbjw_geometry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_pitch_units_and_origin(geometry):
    points = geometry.keypoint_world_coords_2D
    assert points[0] == [-52.5, -34]
    assert points[2] == [52.5, -34]
    assert points[27] == [-52.5, 34]
    assert points[29] == [52.5, 34]


def test_missing_keypoints_have_no_homography(geometry):
    camera = geometry.FramebyFrameCalib(640, 480, denormalize=True)
    camera.update({})
    assert camera.get_homography_from_ground_plane(inverse=True) is None


def test_ground_plane_projection_against_known_correspondences(geometry):
    # Controlled correspondences are a mathematical unit test, not detected players.
    camera = geometry.FramebyFrameCalib(640, 480, denormalize=True)
    expected = np.array([[4, 0, 320], [0, 4, 240], [0, 0, 1]], dtype=float)
    keypoints = {}
    for index in [1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30]:
        x, y = geometry.keypoint_world_coords_2D[index - 1]
        pixel = expected @ np.array([x, y, 1])
        keypoints[index] = {'x': pixel[0] / 640, 'y': pixel[1] / 480, 'p': 1.0}
    camera.update(keypoints)
    inverse = camera.get_homography_from_ground_plane(inverse=True)
    assert inverse is not None
    result = inverse @ np.array([360, 260, 1])
    np.testing.assert_allclose(result[:2] / result[2], [10, 5], atol=1e-4)
