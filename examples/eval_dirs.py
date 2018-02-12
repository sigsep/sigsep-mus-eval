import musdb
import museval


# initiate musdb
mus = musdb.DB()
estimate_dir = ...
output_dir = ...

# evaluate an existing estimate folder with wav files
museval.eval_estimates_dir(
    dataset=mus,  # instance of musdb
    estimates_dir='./DUR/',  # path to estiamte folder
    output_dir='DURv4',  # set a folder to write eval json files
)
