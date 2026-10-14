import pytest

from tools.build_heatmap import prepare_data


def fixture(positions):
    return {'video':{'fps':25,'name':'clip.mp4'},
            'processing':{'processed_duration_seconds':20},
            'players':[{'track_id':1,'positions':positions}]}


def point(x=0, y=0, frame=0):
    return {'pitch_x':x,'pitch_y':y,'frame':frame,'time':frame/25,'bbox_truncated':False}


def test_heatmap_excludes_missing_infinite_and_outside_points():
    data=prepare_data(fixture([point(),point(None,0,1),point(53,0,2),point(float('inf'),0,3)]))
    assert len(data['samples'])==1
    assert data['excluded_invalid_or_outside']==3
    assert data['samples'][0][-2] is False


def test_duplicate_observation_cannot_double_occupancy():
    with pytest.raises(ValueError,match='Duplicate'):
        prepare_data(fixture([point(),point()]))


def test_unknown_box_quality_is_excluded_by_default():
    p=point()
    del p['bbox_truncated']
    assert prepare_data(fixture([p]))['samples'][0][-2] is True


def test_heatmap_requires_matching_nominal_timestamps():
    p=point()
    p['time']=10
    with pytest.raises(ValueError,match='timestamp'):
        prepare_data(fixture([p]))


def test_heatmap_accepts_nonzero_original_frame_segment():
    tracking = fixture([{'pitch_x': 1, 'pitch_y': 2, 'frame': 500, 'time': 10.0, 'bbox_truncated': False}])
    tracking['video']['fps'] = 50
    tracking['processing'] = {'processed_duration_seconds': 2, 'start_frame': 500, 'end_frame': 600}
    data = prepare_data(tracking)
    assert data['start_time'] == pytest.approx(10.0)
    assert data['duration'] == pytest.approx(12.0)
    assert data['samples'][0][3] == 500


def test_track_option_id_matches_samples_for_integral_float_track_id():
    tracking = fixture([point()])
    tracking['players'][0]['track_id'] = 96.0
    data = prepare_data(tracking)
    assert data['track_options'][0]['id'] == '96'
    assert data['samples'][0][4] == '96'
