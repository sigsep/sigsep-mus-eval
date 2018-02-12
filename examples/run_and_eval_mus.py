import musdb
import museval

"""
Hook in the the user_function and do evaluation in the `run` loop
"""

output_dir = ...
estimates_dir = ...


def estimate_and_evaluate(track):
    # return any number of targets
    estimates = {
        'vocals': track.audio,
        'accompaniment': track.audio
    }

    museval.eval_mus_track(
        track, estimates, output_dir=output_dir
    )

    return estimates


# initiate musdb
mus = musdb.DB()

mus.run(
    estimate_and_evaluate,
    estimates_dir=estimates_dir,
    subsets="test"
)
