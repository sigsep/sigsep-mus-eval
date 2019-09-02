# museval

[![Build Status](https://travis-ci.org/sigsep/sigsep-mus-eval.svg?branch=master)](https://travis-ci.org/sigsep/sigsep-mus-eval)
[![Latest Version](https://img.shields.io/pypi/v/museval.svg)](https://pypi.python.org/pypi/museval)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/museval.svg)](https://pypi.python.org/pypi/museval)

A python package to evaluate source separation results using the [MUSDB18](https://sigsep.github.io/musdb) dataset. This package was part of the [MUS task](https://sisec.inria.fr/home/2018-professionally-produced-music-recordings/) of the [Signal Separation Evaluation Campaign (SISEC)](https://sisec.inria.fr/).

### BSSEval v4

The BSSEval metrics, as implemented in the [MATLAB toolboxes](http://bass-db.gforge.inria.fr/bss_eval/) and their re-implementation in [mir_eval](http://craffel.github.io/mir_eval/#module-mir_eval.separation) are widely used in the audio separation literature. One particularity of BSSEval is to compute the metrics after optimally matching the estimates to the true sources through linear distortion filters. This allows the criteria to be robust to some linear mismatches. Apart from the optional evaluation for all possible permutations of the sources, this matching is the reason for most of the computation cost of BSSEval, especially considering it is done for each evaluation window when the metrics are computed on a framewise basis.

For this package, we enabled the option of having _time invariant_ distortion filters, instead of necessarily taking them as varying over time as done in the previous versions of BSS eval. First, enabling this option _significantly reduces_ the computational cost for evaluation because matching needs to be done only once for the whole signal. Second, it introduces much more dynamics in the evaluation, because time-varying matching filters turn out to over-estimate performance. Third, this makes matching more robust, because true sources are not silent throughout the whole recording, while they often were for short windows.

## Installation

### Package installation

You can install the `museval` parsing package using pip:

```bash
pip install museval
```

## Usage

The purpose of this package is to evaluate source separation results and write out validated `json` files. We want to encourage users to use this evaluation output format as the standardized way to share source separation results. `museval` is designed to work in conjuction with the [musdb](https://github.com/sigsep/sigsep-mus-db) tools and the MUSDB18 dataset (however, `museval` can also be used without `musdb`).

### Separate MUSDB18 tracks and Evaluate on-the-fly

- If you want to perform evaluation while processing your source separation results, you can make use `musdb` track objects.
Here is an example for such a function separating the mixture into a __vocals__ and __accompaniment__ track:

```python
import musdb
import museval

def estimate_and_evaluate(track):
    # assume mix as estimates
    estimates = {
        'vocals': track.audio,
        'accompaniment': track.audio
    }

    # Evaluate using museval
    scores = museval.eval_mus_track(
        track, estimates, output_dir="path/to/json"
    )

    # print nicely formatted and aggregated scores
    print(scores)

mus = musdb.DB()
for track in mus:
    estimate_and_evaluate(track)

```

Make sure `output_dir` is set. `museval` will recreate the `musdb` file structure in that folder and write the evaluation results to this folder.

### Evaluate MUSDB18 tracks later

If you have already computed your estimates, we provide you with an easy-to-use function to process evaluation results afterwards.

Simply use the `museval.eval_mus_dir` to evaluate your `estimates_dir` and write the results into the `output_dir`. For convenience, the `eval_mus_dir` function accepts all parameters of the `musdb.run()`. That way e.g. multiprocessing can easily be enabled by setting `parallel=True`:

```python
import musdb
import museval

# initiate musdb
mus = musdb.DB()

# evaluate an existing estimate folder with wav files
museval.eval_mus_dir(
    dataset=mus,  # instance of musdb
    estimates_dir=...,  # path to estimate folder
    output_dir=...,  # set a folder to write eval json files
    subsets="test",
    is_wav=False
)
```

### Aggregate and Analyze Scores

Scores for each track can also be aggregated in a pandas DataFrame for easier analysis or the creation of boxplots.
To aggregate multiple tracks in a DataFrame, create `museval.EvalStore()` object and add the track scores successively.

```python
results = museval.EvalStore(frames_agg='median', tracks_agg='median')
for track in tracks:
    # ...
    results.add_track(museval.eval_mus_track(track, estimates))
```

When all tracks have been added, the aggregated scores can be shown using `print(results)` and results may be saved as a pandas DataFrame `results.save('my_method.pandas')`.

To compare multiple methods, create a `museval.MethodStore()` object add the results

```python
methods = museval.MethodStore()
methods.add_evalstore(results, name="XZY")
```

To compare against participants from [SiSEC MUS 2018](https://github.com/sigsep/sigsep-mus-2018), we provide a convenient method to load the existing scores on demand using `methods.add_sisec18()`. For the creation of plots and statistical significance tests we refer to our [list of examples](/examples).

#### Commandline tool

We provide a command line wrapper of `eval_mus_dir` by calling the `museval` command line tool. The following example is equivalent to the code example above:

```
museval -p --musdb path/to/musdb -o path/to/output_dir path/to/estimate_dir
```

:bulb: you use the `--iswav` flag to use the decoded wav _musdb_ dataset.

### Using Docker for Evaluation

If you don't want to set up a Python environment to run the evaluation, we would recommend to use [Docker](http://docker.com). Assuming you have already computed your estimates and installed docker in your machine, you just need to run the following two lines in your terminal:

#### 1. Pull Docker Container

Pull our precompiled `sigsep-mus-eval` image from [dockerhub](https://hub.docker.com/r/faroit/sigsep-mus-eval/):

```
docker pull faroit/sigsep-mus-eval
```

#### 2. Run evaluation

To run the evaluation inside of the docker, three absolute paths are required:

* `estimatesdir` will stand here for the absolute path to the estimates directory. (For instance `/home/faroit/dev/mymethod/musdboutput`)
* `musdbdir` will stand here for the absolute path to the root folder of musdb. (For instance `/home/faroit/dev/data/musdb18`)
* `outputdir` will stand here for the absolute path to the output directory. (For instance `/home/faroit/dev/mymethod/scores`)

We just mount these directories into the docker container using the `-v` flags and start the docker instance:

```
docker run --rm -v estimatesdir:/est -v musdbdir:/mus -v outputdir:/out faroit/sigsep-mus-eval --musdb /mus -o /out /est
```

In the line above, replace `estimatesdir`, `musdbdir` and `outputdir` by the absolute paths for your setting.  Please note that docker requires absolute paths so you have to rely on your command line environment to convert relative paths to absolute paths (e.g. by using `$HOME/` on Unix).

:warning: `museval` requires a significant amount of memory for the evaluation. Evaluating all five targets for _MUSDB18_ may require more than 4GB of RAM. If you use multiprocessing by using the `-p` switch in `museval`, this results in 16GB of RAM. It is recommended to adjust your Docker preferences, because the docker container might just quit if its out of memory.

## How to contribute

_museval_ is a community focused project, we therefore encourage the community to submit bug-fixes and requests for technical support through [github issues](https://github.com/sigsep/sigsep-mus-eval/issues/new). For more details of how to contribute, please follow our [`CONTRIBUTING.md`](CONTRIBUTING.md). 

## References

A. If you use the `museval` in the context of source separation evaluation comparing a method it to other methods of [SiSEC 2018](http://sisec18.unmix.app/), please cite

```
@InProceedings{SiSEC18,
  author="St{\"o}ter, Fabian-Robert and Liutkus, Antoine and Ito, Nobutaka",
  title="The 2018 Signal Separation Evaluation Campaign",
  booktitle="Latent Variable Analysis and Signal Separation:
  14th International Conference, LVA/ICA 2018, Surrey, UK",
  year="2018",
  pages="293--305"
}
```

B. if you use the software for any other purpose, you can cite the software release

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3376621.svg)](https://doi.org/10.5281/zenodo.3376621)
