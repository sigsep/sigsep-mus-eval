from __future__ import division
from . import metrics
import simplejson as json
import os.path as op
import numpy as np
import os
from decimal import Decimal as D
import glob
import soundfile as sf
from jsonschema import validate
import functools
import musdb
import museval


class EvalStore(object):
    """
    Evaluation Track Data Storage Object

    Attributes
    ----------
    win : float
        evaluation window duration in seconds, default to 1s
    hop : float
        hop length in seconds, defaults to 1s
    rate : int
        Track sample rate
    scores : Dict
        Nested Dictionary of all scores
    """
    def __init__(self, win=1, hop=1):
        super(EvalStore, self).__init__()
        self.win = win
        self.hop = hop

        schema_path = os.path.join(
            museval.__path__[0], 'musdb.schema.json'
        )
        with open(schema_path) as json_file:
            self.schema = json.load(json_file)
        self.scores = {
            'targets': []
        }

    def add_target(self, target_name, values):
        """add target to scores Dictionary

        Parameters
        ----------
        target_name : str
            name of target to be added to list of targets
        values : List(Dict)
            List of framewise data entries, see `musdb.schema.json`
        """
        target_data = {
            'name': target_name,
            'frames': []
        }
        for i, v in enumerate(values['SDR']):
            frame_data = {
                'time': i * self.hop,
                'duration': self.win,
                'metrics': {
                    "SDR": self._q(values['SDR'][i]),
                    "SIR": self._q(values['SIR'][i]),
                    "SAR": self._q(values['SAR'][i]),
                    "ISR": self._q(values['ISR'][i])
                }
            }
            target_data['frames'].append(frame_data)

        self.scores['targets'].append(target_data)

    @property
    def json(self):
        """add target to scores Dictionary

        Returns
        ----------
        json_string : str
            json dump of the scores dictionary
        """
        json_string = json.dumps(
            self.scores,
            indent=2,
            allow_nan=True
        )
        return json_string

    def __repr__(self):
        """Print the mean values instead of all frames

        Returns
        ----------
        str
            mean values of all target metrics
        """
        out = ""
        for t in self.scores['targets']:
            out += t['name'].ljust(20) + "=> "
            for metric in ['SDR', 'SIR', 'ISR', 'SAR']:
                out += metric + ":" + \
                    "%0.3f" % np.nanmean(
                        [np.float(f['metrics'][metric]) for f in t['frames']]
                    ) + "dB, "
            out += "\n"
        return out

    def validate(self):
        """Validate scores against `musdb.schema`"""
        return validate(self.scores, self.schema)

    def _q(self, number, precision='.00001'):
        """quantiztion of BSSEval values"""
        if np.isinf(number):
            return np.nan
        else:
            return D(D(number).quantize(D(precision)))


def _load_track_estimates(track, estimates_dir, output_dir):
    """load estimates from disk instead of processing"""
    user_results = {}
    track_estimate_dir = os.path.join(
        estimates_dir,
        track.subset,
        track.name
    )
    for target in glob.glob(
        track_estimate_dir + '/*.wav'
    ):

        target_name = op.splitext(
            os.path.basename(target)
        )[0]
        try:
            target_audio, rate = sf.read(
                target,
                always_2d=True
            )
            user_results[target_name] = target_audio
        except RuntimeError:
            pass

    if user_results:
        eval_mus_track(
            track,
            user_results,
            output_dir=output_dir
        )

    return None


def eval_dir(
    reference_dir,
    estimates_dir,
    output_dir=None,
    mode='v4',
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
    scores : EvalStore
        scores object that holds the framewise and global evaluation scores.
    """

    reference = []
    estimates = []

    data = EvalStore(win=win, hop=hop)

    global_rate = None
    reference_glob = os.path.join(reference_dir, '*.wav')
    # Load in each reference file in the supplied dir
    for reference_file in glob.glob(reference_glob):
        ref_audio, rate = sf.read(
            reference_file,
            always_2d=True
        )
        # Make sure fs is the same for all files
        assert (global_rate is None or rate == global_rate)
        global_rate = rate
        reference.append(ref_audio)

    if not reference:
        raise ValueError('`reference_dir` contains no wav files')

    estimated_glob = os.path.join(estimates_dir, '*.wav')
    targets = []
    for estimated_file in glob.glob(estimated_glob):
        targets.append(os.path.basename(estimated_file))
        ref_audio, rate = sf.read(
            estimated_file,
            always_2d=True
        )
        assert (global_rate is None or rate == global_rate)
        global_rate = rate
        estimates.append(ref_audio)

    SDR, ISR, SIR, SAR = evaluate(
        reference,
        estimates,
        win=int(win*global_rate),
        hop=int(hop*global_rate),
        mode=mode
    )
    for i, target in enumerate(targets):
        values = {
            "SDR": SDR[i].tolist(),
            "SIR": SIR[i].tolist(),
            "ISR": ISR[i].tolist(),
            "SAR": SAR[i].tolist()
        }

        data.add_target(
            target_name=target,
            values=values
        )

    return data


def eval_mus_dir(
    dataset,
    estimates_dir,
    output_dir=None,
    *args, **kwargs
):
    """Run musdb.run for the purpose of evaluation of musdb estimate dir

    Parameters
    ----------
    dataset : DB(object)
        Musdb Database object.
    estimates_dir : str
        Path to estimates folder.
    output_dir : str
        Output folder where evaluation json files are stored.
    *args
        Variable length argument list for `musdb.run()`.
    **kwargs
        Arbitrary keyword arguments for `musdb.run()`.
    """
    # create a new musdb instance for estimates with the same file structure
    est = musdb.DB(root_dir=estimates_dir, is_wav=True)
    # load all estimates track_names
    est_tracks = est.load_mus_tracks()
    # get a list of track names
    tracknames = [t.name for t in est_tracks]
    # load only musdb tracks where we have estimate tracks
    tracks = dataset.load_mus_tracks(tracknames=tracknames)
    # wrap the estimate loader
    run_fun = functools.partial(
        _load_track_estimates,
        estimates_dir=estimates_dir,
        output_dir=output_dir
    )
    # evaluate tracks
    dataset.run(run_fun, estimates_dir=None, tracks=tracks, *args, **kwargs)


def eval_mus_track(
    track,
    user_estimates,
    output_dir=None,
    mode='v4',
    win=1.0,
    hop=1.0
):
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
    scores : EvalStore
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

    data = EvalStore(win=win, hop=hop)

    # check if vocals and accompaniment is among the targets
    has_acc = all(x in eval_targets for x in ['vocals', 'accompaniment'])
    if has_acc:
        # remove accompaniment from list of targets, because
        # the voc/acc scenario will be evaluated separately
        eval_targets.remove('accompaniment')

    if len(eval_targets) >= 2:
        # compute evaluation of remaining targets
        for target in eval_targets:
            audio_estimates.append(user_estimates[target])
            audio_reference.append(track.targets[target].audio)

        SDR, ISR, SIR, SAR = evaluate(
            audio_reference,
            audio_estimates,
            win=int(win*track.rate),
            hop=int(hop*track.rate),
            mode=mode
        )

        # iterate over all evaluation results except for vocals
        for i, target in enumerate(eval_targets):
            if target == 'vocals' and has_acc:
                continue

            values = {
                "SDR": SDR[i].tolist(),
                "SIR": SIR[i].tolist(),
                "ISR": ISR[i].tolist(),
                "SAR": SAR[i].tolist()
            }

            data.add_target(
                target_name=target,
                values=values
            )

    # add vocal accompaniment targets later
    if has_acc:
        # add vocals and accompaniments as a separate scenario
        eval_targets = ['vocals', 'accompaniment']

        audio_estimates = []
        audio_reference = []

        for target in eval_targets:
            audio_estimates.append(user_estimates[target])
            audio_reference.append(track.targets[target].audio)

        SDR, ISR, SIR, SAR = evaluate(
            audio_reference,
            audio_estimates,
            win=int(win*track.rate),
            hop=int(hop*track.rate),
            mode=mode
        )

        # iterate over all targets
        for i, target in enumerate(eval_targets):
            values = {
                "SDR": SDR[i].tolist(),
                "SIR": SIR[i].tolist(),
                "ISR": ISR[i].tolist(),
                "SAR": SAR[i].tolist()
            }

            data.add_target(
                target_name=target,
                values=values
            )

    if output_dir:
        # validate against the schema
        data.validate()

        try:
            subset_path = op.join(
                output_dir,
                track.subset
            )

            if not op.exists(subset_path):
                os.makedirs(subset_path)

            with open(
                op.join(subset_path, track.name) + '.json', 'w+'
            ) as f:
                f.write(data.json)

        except (IOError):
            pass

    return data


def pad_or_truncate(
    audio_reference,
    audio_estimates
):
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
            audio_estimates = audio_estimates[:, :ref_shape[1], :]
        else:
            # pad end with zeros
            audio_estimates = np.pad(
                audio_estimates,
                [
                    (0, 0),
                    (0, ref_shape[1] - est_shape[1]),
                    (0, 0)
                ],
                mode='constant'
            )

    return audio_reference, audio_estimates


def evaluate(
    references,
    estimates,
    win=1*44100,
    hop=1*44100,
    mode='v4',
    padding=True
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
        bsseval_sources_version=False
    )

    return SDR, ISR, SIR, SAR
