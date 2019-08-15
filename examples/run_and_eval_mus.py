import musdb
import museval

"""
Hook in the the user_function and do evaluation in the `run` loop
"""

output_dir = ...
estimates_dir = ...

# initiate musdb
mus = musdb.DB(subsets="test")

for track in mus:
    # return any number of targets
    estimates = {
        'vocals': track.audio,
        'accompaniment': track.audio
    }

    mus.save_estimates(estimates, track, estimates_dir)

    museval.eval_mus_track(
        track, estimates, output_dir=output_dir
    )