import numpy as np
import pytest
import museval.metrics as metrics
import museval


@pytest.fixture(params=[2])
def nb_sources(request):
    return request.param


@pytest.fixture(params=[2])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[1000])
def nb_samples(request):
    return request.param


@pytest.fixture(params=[-10, 10])
def nb_samples_diff(request):
    return request.param


def test_pad_or_truncate(nb_sources, nb_channels, nb_samples, nb_samples_diff):
    references = np.random.random((nb_sources, nb_samples, nb_channels))
    estimates = np.random.random(
        (nb_sources, nb_samples + nb_samples_diff, nb_channels)
    )

    references, estimates = museval.pad_or_truncate(references, estimates)
    assert references.shape[1] == estimates.shape[1]
