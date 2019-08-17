import musdb
import museval

# initiate musdb
mus = musdb.DB(download=True)

# set up museval store
results = museval.EvalStore()

for track in mus:
    print(track)
    # return any number of targets
    estimates = {
        'vocals': track.audio,
        'accompaniment': track.audio
    }

    track_scores = museval.eval_mus_track(track, estimates)
    print(track_scores)
    results.add_track(track_scores.df)

print(results)
results.save('AAA.pandas')
methods = museval.MethodStore()
methods.add_evalstore(results, name="AAA")
methods.add_sisec18()
print(methods.agg_frames_tracks_scores())


