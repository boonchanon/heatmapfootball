"""Regression checks for cached tracking replay and honest visualization trails."""

from collections import deque

import numpy as np
import pandas as pd
import pytest

from tools.retune_tracking import retrack
from sn_gamestate.custom_video.review import trail_points


def detections_for(frames, confidence=.65):
    return pd.DataFrame([{'image_id':f, 'bbox_ltwh':np.array([100.,100.,30.,70.]),
                          'bbox_conf':confidence, 'category_id':1, 'track_id':np.nan,
                          'bbox_pitch':{'x_bottom_middle':1.,'y_bottom_middle':2.}}
                         for f in frames],index=np.arange(len(frames))*3+17)


def test_replay_retains_geometry_and_detection_identity():
    detections = detections_for([0,1,2])
    images = pd.DataFrame({'frame':[0,1,2]})
    before = retrack(detections,images,25,.6)
    after = retrack(detections,images,25,.5)
    assert before.track_id.notna().sum()==0
    assert after.track_id.notna().sum()==3
    assert after.track_id.nunique()==1
    pd.testing.assert_frame_equal(after[detections.columns.difference(['track_id'])],
                                  detections[detections.columns.difference(['track_id'])])
    assert after.index.equals(detections.index)


def test_empty_frames_age_out_tracks():
    result = retrack(detections_for([0,1,40,41]),pd.DataFrame({'frame':range(42)}),25,.5)
    assert result.iloc[0].track_id != result.iloc[-1].track_id
    assert pd.notna(result.iloc[-1].track_id)


def test_replay_rejects_discontinuous_metadata():
    with pytest.raises(ValueError,match='contiguous'):
        retrack(detections_for([0,2]),pd.DataFrame({'frame':[0,2]},index=[0,2]),25,.5)


def test_trails_reset_at_gaps_jumps_and_limit_length():
    history=deque()
    assert trail_points(history,0,(0,0),2)==[(0,0)]
    assert trail_points(history,1,(.1,0),2)==[(0,0),(.1,0)]
    assert trail_points(history,2,(.2,0),2)==[(.1,0),(.2,0)]
    assert trail_points(history,4,(.3,0),2)==[(.3,0)]
    assert trail_points(history,5,(10,0),2)==[(10,0)]
