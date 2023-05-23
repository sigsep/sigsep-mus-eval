import numpy as np
import pytest
import museval.metrics as metrics


@pytest.fixture(params=[2])
def nb_sources(request):
    return request.param


@pytest.fixture(params=[2000])
def nb_win(request):
    return request.param


@pytest.fixture(params=[2000])
def nb_hop(request):
    return request.param


@pytest.fixture(params=[2])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[1])
def nb_samples(request, rate):
    return int(request.param * rate)


@pytest.fixture(params=[2000])
def rate(request):
    return request.param


@pytest.fixture(params=[True, False])
def is_sources(request):
    return request.param


@pytest.fixture
def references(request, nb_sources, nb_samples, nb_channels, is_sources):
    if is_sources:
        return np.random.random((nb_sources, nb_samples))
    else:
        return np.random.random((nb_sources, nb_samples, nb_channels))


@pytest.fixture
def estimates(request, nb_sources, nb_samples, nb_channels, is_sources):
    if is_sources:
        return np.random.random((nb_sources, nb_samples))
    else:
        return np.random.random((nb_sources, nb_samples, nb_channels))


@pytest.fixture(params=[True, False])
def is_framewise(request):
    return request.param


def test_empty_input(is_framewise, is_sources, nb_win, nb_hop):
    inputs = [np.array([]), np.array([])]

    with pytest.warns(UserWarning):
        output = metrics.bss_eval(
            *inputs,
            framewise_filters=is_framewise,
            bsseval_sources_version=is_sources,
            window=nb_win,
            hop=nb_hop
        )

        assert np.allclose(output, np.array([]))


def test_silent_input(references, estimates, is_framewise, is_sources, nb_win, nb_hop):
    estimates = np.zeros(references.shape)

    with pytest.raises(ValueError):
        metrics.bss_eval(
            references,
            estimates,
            framewise_filters=is_framewise,
            bsseval_sources_version=is_sources,
            window=nb_win,
            hop=nb_hop,
        )


def test_metric(references, estimates, is_framewise, is_sources, nb_win, nb_hop):
    metrics.bss_eval(
        references,
        estimates,
        framewise_filters=is_framewise,
        bsseval_sources_version=is_sources,
        window=nb_win,
        hop=nb_hop,
    )


def test_wrappers(references, estimates, is_framewise, is_sources, nb_win, nb_hop):
    functions = [
        metrics.bss_eval_sources,
        metrics.bss_eval_images,
        metrics.bss_eval_sources_framewise,
        metrics.bss_eval_images_framewise,
    ]

    for function in functions:
        function(
            references,
            estimates,
        )
