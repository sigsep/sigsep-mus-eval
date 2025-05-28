import os
import pytest
import musdb
import simplejson as json
import museval.cli
import numpy as np


json_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data/Music Delta - Rock.json",
)


def test_evaluate_mus_dir():
    museval.cli.museval(
        ["data/EST", "-o", "data/EST_scores2", "--musdb", "data/MUS-STEMS-SAMPLE"]
    )
