# museval

[![Build Status](https://travis-ci.org/sigsep/sigsep-mus-eval.svg?branch=master)](https://travis-ci.org/sigsep/sigsep-mus-eval)
[![Coverage Status](https://coveralls.io/repos/github/sigsep/sigsep-mus-eval/badge.svg?branch=master)](https://coveralls.io/github/sigsep/sigsep-mus-eval?branch=master)
[![Latest Version](https://img.shields.io/pypi/v/museval.svg)](https://pypi.python.org/pypi/museval)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/museval.svg)](https://pypi.python.org/pypi/museval)
[![Docs Status](https://readthedocs.org/projects/museval/badge/?version=latest)](https://museval.readthedocs.org/en/latest/)


A python package to evaluate source separation results using the [MUSDB18](https://sigsep.github.io/musdb) dataset. This package is part of the [MUS task](https://sisec.inria.fr/home/2018-professionally-produced-music-recordings/) of the [Signal Separation Evaluation Campaign (SISEC)](https://sisec.inria.fr/). Using this package mandatory for submitting results for SiSEC as it includes the reference implementation of the new BSSEval version 4.

### BSSEval v4

The BSSEval metrics, as implemented in the [MATLAB toolboxes](http://bass-db.gforge.inria.fr/bss_eval/) and its re-implementation in [mir_eval](http://craffel.github.io/mir_eval/#module-mir_eval.separation) are widely used in the audio separation literature. One particularity of BSSEval is to compute the metrics after optimally matching the estimates to the true sources through linear distortion filters. This allows the criteria to be robust to some linear mismatches. Apart from the optional computation of all possible permutations of the sources, this matching is the reason for most of the computation cost of BSSEval, especially considering it is done for each evaluation window when the metrics are computed on a framewise basis.

For this package, we dropped the assumption that distortion filters could be varying over time, but considered instead they are fixed for the whole length of the track. First, this _significantly reduces_ the computational cost for evaluation because matching needs to be done only once for the whole signal. Second, this introduces much more dynamics in the evaluation, because time-varying matching filters turn out to over-estimate performance. Third, this makes matching more robust, because true sources are not silent throughout the whole recording, while they often were for short windows.

## Installation

### Package installation

You can install the `museval` parsing package using pip:

```bash
pip install museval
```

## Usage

The purpose of this package is to evaluate source separation results and write out standardized `json` files that can easiliy parsed by the SiSEC submission system. Furthermore we want to encourage users to use this evaluation output format as the standardized way to share source separation results for processed tracks. We provide two different ways to use `museval` in conjunction with your source separation results.

### Run and Evaluate

- If you want to evaluate while processing your source separation results, you can hook `museval` into your `musdb` user_function:

Here is an example for such a function separating the mixture into a __vocals__ and __accompaniment__ track:

```python
import musdb
import museval

output_dir = ...
estimates_dir = ...

def estimate_and_evaluate(track):
    # generate your estimates
    estimates = {
        'vocals': track.audio,
        'accompaniment': track.audio
    }

    # Evaluate using museval
    scores = museval.eval_mus_track(
        track, estimates, output_dir=output_dir
    )

    # print nicely formatted mean scores
    print(scores)

    # return estimates as usual
    return estimates

# you usual way to run musdb
musdb.DB().run()
```

- Make sure `output_dir` is set. `museval` will recreate the `musdb` file structure in that folder and writes the evaluation results to this folder. __This folder should be submitted for your SiSEC contribution__.

### Evaluate later

If you have already computed your estimates (maybe through the use of MATLAB), we provide you with an easy-to-use function to process evaluation results afterwards.

Simply use the `museval.eval_mus_dir` to evaluate your `estimates_dir` and write the results into the `output_dir`. For convenience, the `eval_mus_dir` function accepts all parameters of the `musdb.run()`. That way e.g. multiprocessing can easily be enabled by setting `parallel=True`:

```python
import musdb
import museval

# initiate musdb
mus = musdb.DB()

# evaluate an existing estimate folder with wav files
museval.eval_mus_dir(
    dataset=mus,  # instance of musdb
    estimates_dir=...,  # path to estiamte folder
    output_dir=...,  # set a folder to write eval json files
    subsets="Test",
    parallel=True
)
```

#### Commandline tool

We provide a commandline wrapper of `eval_mus_dir` by calling the `museval` commandline tool. The following example is equivalent to code example above:

```
museval your_estimate_dir -o your_output_dir -p
```


### Docker



## Evaluation and Submission

Please refer to our [Submission site](https://github.com/sigsep/sigsep-mus-2018).


## References

LVA/ICA 2018 publication t.b.a
