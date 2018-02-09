import os
import pytest
import musdb
import simplejson as json
import museval


json_path = 'tests/data/PR - Oh No.json'


@pytest.fixture()
def mus():
    return musdb.DB(root_dir='data/MUS-STEMS-SAMPLE')


def test_estimate_and_evaluate(mus):
    # return any number of targets
    with open(json_path) as json_file:
        reference_scores = json.loads(json_file.read())

    print(os.path.basename(json_path))
    track = mus.load_mus_tracks(
        tracknames=[os.path.splitext(os.path.basename(json_path))[0]]
    )[0]

    fake = (
        track.sources['vocals'].audio +
        track.sources['other'].audio +
        track.sources['drums'].audio +
        track.sources['bass'].audio
    )

    estimates = {
        'vocals': fake,
        'accompaniment': fake
    }

    scores = museval.eval_mus_track(
        track, estimates
    )

    scores = json.loads(json.dumps(scores))
    assert reference_scores == scores
