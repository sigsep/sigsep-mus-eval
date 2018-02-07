import musdb
import museval

"""
Hook in the the user_function and do evaluation in the `run` loop
"""


def my_function(track):
    # return any number of targets
    estimates = {
        'vocals': track.audio,
        'accompaniment': track.audio
    }

    museval.eval_user_estimates(
        track, estimates, output_dir="ORL1_2018_1", mode='v4'
    )

    return estimates


# initiate musdb
mus = musdb.DB("data/MUS-STEMS-SAMPLE")

# this might take 3 days to finish
mus.run(my_function, estimates_dir="ORL1", subsets="test")
