import os
import pytest
import musdb
import simplejson as json
import museval
import numpy as np


@pytest.fixture()
def mus():
    musdb_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data/MUSDB18-7-SAMPLE",
    )
    return musdb.DB(root=musdb_path)


@pytest.fixture(params=["Music Delta - 80s Rock"])
def track_name(request):
    return request.param


@pytest.fixture()
def reference(mus, track_name):
    track = [track for track in mus.tracks if track.name == track_name][0]

    json_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data/%s.json" % track_name,
    )

    with open(json_path) as json_file:
        ref_json = json.loads(json_file.read())

    return track, ref_json


def test_aggregate(reference):
    track, _ = reference

    np.random.seed(0)
    random_voc = np.random.random(track.audio.shape)
    random_acc = np.random.random(track.audio.shape)

    # create a silly regression test
    estimates = {"vocals": random_voc, "accompaniment": random_acc}

    scores = museval.eval_mus_track(track, estimates)

    print(scores.df)

    results = museval.EvalStore()
    results.add_track(scores)
    agg = results.agg_frames_scores()
    print(results)


def test_track_scores(reference):
    track, ref_scores = reference

    np.random.seed(0)
    random_voc = np.random.random(track.audio.shape)
    random_acc = np.random.random(track.audio.shape)

    # create a silly regression test
    estimates = {"vocals": random_voc, "accompaniment": random_acc}

    scores = museval.eval_mus_track(track, estimates)

    est_scores = json.loads(scores.json)
    for target in ref_scores["targets"]:
        for metric in ["SDR", "SIR", "SAR", "ISR"]:
            ref = np.array([d["metrics"][metric] for d in target["frames"]])
            idx = [t["name"] for t in est_scores["targets"]].index(target["name"])
            est = np.array(
                [d["metrics"][metric] for d in est_scores["targets"][idx]["frames"]]
            )

            assert np.allclose(ref, est, atol=1e-01, equal_nan=True)


def test_random_estimate(reference):
    track, _ = reference
    np.random.seed(0)
    random_voc = np.random.random(track.audio.shape)
    random_acc = np.random.random(track.audio.shape)

    # create a silly regression test
    estimates = {"vocals": random_voc, "accompaniment": random_acc}

    scores = museval.eval_mus_track(track, estimates)

    # save json
    with open(os.path.join(".", track.name) + ".json", "w+") as f:
        f.write(scores.json)

    # validate json
    assert scores.validate() is None


def test_one_estimate(reference):
    track, _ = reference

    np.random.seed(0)
    random_voc = np.random.random(track.audio.shape)

    estimates = {"vocals": random_voc}

    with pytest.warns(UserWarning):
        est_scores = museval.eval_mus_track(track, estimates)

    est_json = json.loads(est_scores.json)

    assert len(est_json["targets"]) == 0
