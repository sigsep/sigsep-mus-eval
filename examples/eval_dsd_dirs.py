import museval
import dsdtools
import glob
import os.path as op
import os
import soundfile as sf

# initiate dsdtools
dsd = dsdtools.DB()


user_estimates_dir = ...
output_dir = ...


def load_and_eval_estimates(track):
    # load estimates from disk instead of processing
    user_results = {}
    track.name = track.filename
    track_estimate_dir = os.path.join(
        user_estimates_dir,
        track.subset,
        track.filename
    )
    for target in glob.glob(track_estimate_dir + '/*.wav'):

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

    museval.eval_mus_track(
        track,
        user_results,
        output_dir=output_dir,
        mode='v3'  # use bss_eval v3 to reproduce sisec 2016 results
    )

    return None


dsd.run(
    load_and_eval_estimates,
    estimates_dir=None,
    subsets="Test"
)
