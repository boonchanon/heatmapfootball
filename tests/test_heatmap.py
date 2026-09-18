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
    assert data['samples'][0][-1] is False


def test_duplicate_observation_cannot_double_occupancy():
    with pytest.raises(ValueError,match='Duplicate'):
        prepare_data(fixture([point(),point()]))


def test_unknown_box_quality_is_excluded_by_default():
    p=point()
    del p['bbox_truncated']
    assert prepare_data(fixture([p]))['samples'][0][-1] is True


def test_heatmap_requires_matching_nominal_timestamps():
    p=point()
    p['time']=10
    with pytest.raises(ValueError,match='timestamp'):
        prepare_data(fixture([p]))
