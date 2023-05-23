import os.path as op
import numpy as np
import os
import glob
import soundfile as sf
import functools
import musdb
import warnings
import pandas as pd
from .aggregate import TrackStore, MethodStore, EvalStore, json2df
from . import metrics


def _load_track_estimates(track, estimates_dir, output_dir, ext="wav"):
    """load estimates from disk instead of processing"""
    user_results = {}
    track_estimate_dir = os.path.join(estimates_dir, track.subset, track.name)
    for target in glob.glob(track_estimate_dir + "/*." + ext):
        target_name = op.splitext(os.path.basename(target))[0]
        try:
            target_audio, _ = sf.read(target, always_2d=True)
            user_results[target_name] = target_audio
        except RuntimeError:
            pass

    if user_results:
        eval_mus_track(track, user_results, output_dir=output_dir)

    return None


def eval_dir(
    reference_dir,
    estimates_dir,
    output_dir=None,
    mode="v4",
    win=1.0,
    hop=1.0,
):
    """Compute bss_eval metrics for two given directories assuming file
    names are identical for both, reference source and estimates.

    Parameters
    ----------
    reference_dir : str
        path to reference sources directory.
    estimates_dir : str
        path to estimates directory.
    output_dir : str
        path to output directory used to save evaluation results. Defaults to
        `None`, meaning no evaluation files will be saved.
    mode : str
        bsseval version number. Defaults to 'v4'.
    win : int
        window size in

    Returns
    -------
    scores : TrackStore
        scores object that holds the framewise and global evaluation scores.
    """

    reference = []
    estimates = []

    data = TrackStore(win=win, hop=hop, track_name=os.path.basename(reference_dir))

    global_rate = None
    reference_glob = os.path.join(reference_dir, "*.wav")
    # Load in each reference file in the supplied dir
    for reference_file in glob.glob(reference_glob):
        ref_audio, rate = sf.read(reference_file, always_2d=True)
        # Make sure fs is the same for all files
        assert global_rate is None or rate == global_rate
        global_rate = rate
        reference.append(ref_audio)

    if not reference:
        raise ValueError("`reference_dir` contains no wav files")

    estimated_glob = os.path.join(estimates_dir, "*.wav")
    targets = []
    for estimated_file in glob.glob(estimated_glob):
        targets.append(os.path.basename(estimated_file))
        ref_audio, rate = sf.read(estimated_file, always_2d=True)
        assert global_rate is None or rate == global_rate
        global_rate = rate
        estimates.append(ref_audio)

    SDR, ISR, SIR, SAR = evaluate(
        reference,
        estimates,
        win=int(win * global_rate),
        hop=int(hop * global_rate),
        mode=mode,
    )
    for i, target in enumerate(targets):
        values = {
            "SDR": SDR[i].tolist(),
            "SIR": SIR[i].tolist(),
            "ISR": ISR[i].tolist(),
            "SAR": SAR[i].tolist(),
        }

        data.add_target(target_name=target, values=values)

    return data


def eval_mus_dir(dataset, estimates_dir, output_dir=None, ext="wav"):
    """Run evaluation of musdb estimate dir

    Parameters
    ----------
    dataset : DB(object)
        MUSDB18 Database object.
    estimates_dir : str
        Path to estimates folder.
    output_dir : str
        Output folder where evaluation json files are stored.
    ext : str
        estimate file extension, defaults to `wav`
    """
    # create a new musdb instance for estimates with the same file structure
    est = musdb.DB(root=estimates_dir, is_wav=True)
    # get a list of track names
    tracks_to_be_estimated = [t.name for t in est.tracks]

    for track in dataset:
        if track.name not in tracks_to_be_estimated:
            continue
        _load_track_estimates(
            track=track, estimates_dir=estimates_dir, output_dir=output_dir
        )


def eval_mus_track(track, user_estimates, output_dir=None, mode="v4", win=1.0, hop=1.0):
    """Compute all bss_eval metrics for the musdb track and estimated signals,
    given by a `user_estimates` dict.

    Parameters
    ----------
    track : Track
        musdb track object loaded using musdb
    estimated_sources : Dict
        dictionary, containing the user estimates as np.arrays.
    output_dir : str
        path to output directory used to save evaluation results. Defaults to
        `None`, meaning no evaluation files will be saved.
    mode : str
        bsseval version number. Defaults to 'v4'.
    win : int
        window size in

    Returns
    -------
    scores : TrackStore
        scores object that holds the framewise and global evaluation scores.
    """

    audio_estimates = []
    audio_reference = []

    # make sure to always build the list in the same order
    # therefore track.targets is an OrderedDict
    eval_targets = []  # save the list of target names to be evaluated
    for key, target in list(track.targets.items()):
        try:
            # try to fetch the audio from the user_results of a given key
            user_estimates[key]
        except KeyError:
            # ignore wrong key and continue
            continue

        # append this target name to the list of target to evaluate
        eval_targets.append(key)

    data = TrackStore(win=win, hop=hop, track_name=track.name)

    # check if vocals and accompaniment is among the targets
    has_acc = all(x in eval_targets for x in ["vocals", "accompaniment"])
    if has_acc:
        # remove accompaniment from list of targets, because
        # the voc/acc scenario will be evaluated separately
        eval_targets.remove("accompaniment")

    if len(eval_targets) >= 2:
        # compute evaluation of remaining targets
        for target in eval_targets:
            audio_estimates.append(user_estimates[target])
            audio_reference.append(track.targets[target].audio)

        SDR, ISR, SIR, SAR = evaluate(
            audio_reference,
            audio_estimates,
            win=int(win * track.rate),
            hop=int(hop * track.rate),
            mode=mode,
        )

        # iterate over all evaluation results except for vocals
        for i, target in enumerate(eval_targets):
            if target == "vocals" and has_acc:
                continue

            values = {
                "SDR": SDR[i].tolist(),
                "SIR": SIR[i].tolist(),
                "ISR": ISR[i].tolist(),
                "SAR": SAR[i].tolist(),
            }

            data.add_target(target_name=target, values=values)
    elif not has_acc:
        warnings.warn(
            UserWarning(
                "Incorrect usage of BSSeval : at least two estimates must be provided. Target score will be empty."
            )
        )

    # add vocal accompaniment targets later
    if has_acc:
        # add vocals and accompaniments as a separate scenario
        eval_targets = ["vocals", "accompaniment"]

        audio_estimates = []
        audio_reference = []

        for target in eval_targets:
            audio_estimates.append(user_estimates[target])
            audio_reference.append(track.targets[target].audio)

        SDR, ISR, SIR, SAR = evaluate(
            audio_reference,
            audio_estimates,
            win=int(win * track.rate),
            hop=int(hop * track.rate),
            mode=mode,
        )

        # iterate over all targets
        for i, target in enumerate(eval_targets):
            values = {
                "SDR": SDR[i].tolist(),
                "SIR": SIR[i].tolist(),
                "ISR": ISR[i].tolist(),
                "SAR": SAR[i].tolist(),
            }

            data.add_target(target_name=target, values=values)

    if output_dir:
        # validate against the schema
        data.validate()

        try:
            subset_path = op.join(output_dir, track.subset)

            if not op.exists(subset_path):
                os.makedirs(subset_path)

            with open(op.join(subset_path, track.name) + ".json", "w+") as f:
                f.write(data.json)

        except IOError:
            pass

    return data


def pad_or_truncate(audio_reference, audio_estimates):
    """Pad or truncate estimates by duration of references:
    - If reference > estimates: add zeros at the and of the estimated signal
    - If estimates > references: truncate estimates to duration of references

    Parameters
    ----------
    references : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing true reference sources
    estimates : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing estimated sources
    Returns
    -------
    references : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing true reference sources
    estimates : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing estimated sources
    """
    est_shape = audio_estimates.shape
    ref_shape = audio_reference.shape
    if est_shape[1] != ref_shape[1]:
        if est_shape[1] >= ref_shape[1]:
            audio_estimates = audio_estimates[:, : ref_shape[1], :]
        else:
            # pad end with zeros
            audio_estimates = np.pad(
                audio_estimates,
                [(0, 0), (0, ref_shape[1] - est_shape[1]), (0, 0)],
                mode="constant",
            )

    return audio_reference, audio_estimates


def evaluate(
    references, estimates, win=1 * 44100, hop=1 * 44100, mode="v4", padding=True
):
    """BSS_EVAL images evaluation using metrics module

    Parameters
    ----------
    references : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing true reference sources
    estimates : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing estimated sources
    window : int, defaults to 44100
        window size in samples
    hop : int
        hop size in samples, defaults to 44100 (no overlap)
    mode : str
        BSSEval version, default to `v4`
    Returns
    -------
    SDR : np.ndarray, shape=(nsrc,)
        vector of Signal to Distortion Ratios (SDR)
    ISR : np.ndarray, shape=(nsrc,)
        vector of Source to Spatial Distortion Image (ISR)
    SIR : np.ndarray, shape=(nsrc,)
        vector of Source to Interference Ratios (SIR)
    SAR : np.ndarray, shape=(nsrc,)
        vector of Sources to Artifacts Ratios (SAR)
    """

    estimates = np.array(estimates)
    references = np.array(references)

    if padding:
        references, estimates = pad_or_truncate(references, estimates)

    SDR, ISR, SIR, SAR, _ = metrics.bss_eval(
        references,
        estimates,
        compute_permutation=False,
        window=win,
        hop=hop,
        framewise_filters=(mode == "v3"),
        bsseval_sources_version=False,
    )

    return SDR, ISR, SIR, SAR
