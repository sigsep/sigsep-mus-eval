import os
import pytest
import musdb
import simplejson as json
import museval


json_path = 'tests/data/PR - Oh No.json'


@pytest.fixture()
def mus():
    return musdb.DB(root_dir='data/MUS-STEMS-SAMPLE', is_wav=True)


def test_estimate_and_evaluate(mus):
    # return any number of targets
    with open(json_path) as json_file:
        reference_scores = json.loads(json_file.read())

    print(os.path.basename(json_path))
    track = mus.load_mus_tracks(
        tracknames=[os.path.splitext(os.path.basename(json_path))[0]]
    )[0]

    # create a silly regression test
    estimates = {
        'vocals': track.targets['accompaniment'].audio,
        'accompaniment': track.targets['vocals'].audio
    }

    scores = museval.eval_mus_track(
        track, estimates
    )

    with open(
        os.path.join('.', track.name) + '.json', 'w+'
    ) as f:
        f.write(scores.json)

    scores = json.loads(scores.json)
    assert reference_scores == scores
