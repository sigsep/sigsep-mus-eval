from __future__ import division
from . import metrics
import simplejson as json
import os.path as op
import numpy as np
import os
from decimal import Decimal as D
import glob
import soundfile as sf


class TrackData(object):
    """Creates track dict with scores
    """
    def __init__(self, win, hop, rate):
        super(TrackData, self).__init__()
        self.win = win
        self.hop = hop
        self.rate = rate
        self.scores = {
            'targets': []
        }

    def add_target(self, target_name, values):
        target_data = {
            'name': target_name,
            'frames': []
        }
        for i, v in enumerate(values['SDR']):
            frame_data = {
                'time': i * self.hop / self.rate,
                'duration': self.win / self.rate,
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
        json_string = json.dumps(
            self.scores,
            indent=2,
            allow_nan=True
        )
        return json_string

    def _q(self, number, precision='.00001'):
        """quantiztion of values"""
        if np.isinf(number):
            return np.nan
        else:
            return D(D(number).quantize(D(precision)))


def eval_estimates_dir(
    dataset,
    estimates_dir,
    output_path=None
):
    def load_estimates(track):
        # load estimates from disk instead of processing
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

        eval_mus_track(
            track,
            user_results,
            output_path=output_path
        )

        return None

    dataset.run(load_estimates, estimates_dir=None, subsets="Test")


def eval_mus_track(
    track,
    user_estimates,
    output_path=None,
    mode='v4',
):
    """Compute all bss_eval metrics for the musdb track and estimated signals,
    given by a `user_estimates` dict.

    Parameters
    ----------
    track : Track
        musdb track object loaded using musdb
    estimated_sources : Dict
        dictionary, containing the user estimates.
    output_path : str
        path to output directory used to save evaluation results. Defaults to
        `None`, meaning no evaluation files will be saved.
    mode : str
        bsseval version number. Defaults to 'v4'."""

    audio_estimates = []
    audio_reference = []

    window = track.rate * 1
    hop = track.rate * 1

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

    data = TrackData(
        win=window,
        hop=hop,
        rate=track.rate
    )

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

        SDR, ISR, SIR, SAR = safe_eval(
            audio_reference,
            audio_estimates,
            win=window, hop=hop, mode=mode
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

        SDR, ISR, SIR, SAR = safe_eval(
            audio_reference,
            audio_estimates,
            win=window, hop=hop, mode=mode
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

    if output_path:
        try:
            # schema.validate(data.scores)

            subset_path = op.join(
                output_path,
                track.subset
            )

            if not op.exists(subset_path):
                os.makedirs(subset_path)

            with open(
                op.join(subset_path, track.name) + '.json', 'w+'
            ) as f:
                f.write(data.json)

        except (ValueError, IOError):
            pass


def safe_eval(
    audio_reference,
    audio_estimates,
    win, hop, mode
):

    audio_estimates = np.array(audio_estimates)
    audio_reference = np.array(audio_reference)

    if audio_estimates.shape[1] != audio_reference.shape[1]:
        if audio_estimates.shape[1] > audio_reference.shape[1]:
            audio_estimates = audio_estimates[:, :audio_reference.shape[1], :]
        else:
            # pad end with zeros
            np.pad(
                audio_estimates,
                [
                    (0, 0, 0),
                    (0, 0, audio_reference.shape[1] - audio_estimates[1]),
                    (0, 0, 0)
                ],
                mode='constant'
            )

    SDR, ISR, SIR, SAR = _evaluate(
        audio_estimates, audio_reference, win, hop, mode
    )

    return SDR, ISR, SIR, SAR


def _evaluate(
    estimates,
    references,
    window=1*44100,
    hop=1*44100,
    mode='v4'
):
    """BSS_EVAL images evaluation using mir_eval.separation module
    Parameters
    ----------
    references : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing true reference sources
    estimates : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing estimated sources
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

    sdr, isr, sir, sar, _ = metrics.bss_eval(
        references,
        estimates,
        compute_permutation=False,
        window=window,
        hop=hop,
        framewise_filters=(mode == "v3"),
        bsseval_sources_version=False
    )

    return sdr, isr, sir, sar
