import musdb
import museval

# initiate musdb
mus = musdb.DB(download=True)

# set up museval store
results = museval.EvalStore()

for track in mus[:3]:
    print(track)
    # return any number of targets
    estimates = {
        'vocals': track.audio,
        'accompaniment': track.audio
    }

    track_scores = museval.eval_mus_track(track, estimates)
    print(track_scores)
    results.add_track(track_scores)

print(results)
results.save('SUP.pandas')
comparison = museval.MethodsStore()
comparison.add_method(results, method_name="SUP")
comparison.add_sisec18()
print(comparison.aggregate_score())


