import musdb
import museval


# initiate musdb
mus = musdb.DB()

# evaluate an existing estimate folder with wav files
museval.eval_mus_dir(
    dataset=mus,  # instance of musdb
    estimates_dir='eval_test/MWF',  # path to estiamte folder
    output_dir='./EST2',  # set a folder to write eval json files
    subsets="Test",
    parallel=True
)
