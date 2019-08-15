import os
import pytest
import musdb
import simplejson as json
import museval
import numpy as np


test_track = 'Music Delta - 80s Rock'

json_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'data/%s.json' % test_track,
)


@pytest.fixture()
def mus():
    return musdb.DB(download=True)


def test_estimate_and_evaluate(mus):
    # return any number of targets
    with open(json_path) as json_file:
        ref = json.loads(json_file.read())

    track = [track for track in mus.tracks if track.name == test_track][0]

    np.random.seed(0)
    random_voc = np.random.random(track.audio.shape)
    random_acc = np.random.random(track.audio.shape)

    # create a silly regression test
    estimates = {
        'vocals': random_voc,
        'accompaniment': random_acc
    }

    scores = museval.eval_mus_track(
        track, estimates
    )

    assert scores.validate() is None

    with open(
        os.path.join('.', track.name) + '.json', 'w+'
    ) as f:
        f.write(scores.json)

    scores = json.loads(scores.json)

    for target in ref['targets']:
        for metric in ['SDR', 'SIR', 'SAR', 'ISR']:

            ref = np.array([d['metrics'][metric] for d in target['frames']])
            idx = [t['name'] for t in scores['targets']].index(target['name'])
            est = np.array(
                [
                    d['metrics'][metric]
                    for d in scores['targets'][idx]['frames']
                ]
            )

            assert np.allclose(ref, est, atol=1e-05)


def test_one_estimate(mus):
    track = [track for track in mus.tracks if track.name == test_track][0]

    np.random.seed(0)
    random_voc = np.random.random(track.audio.shape)

    estimates = {
        'vocals': random_voc
    }

    with pytest.warns(UserWarning):
        scores = museval.eval_mus_track(
            track, estimates
        )

    scores = json.loads(scores.json)

    assert len(scores['targets']) == 0
