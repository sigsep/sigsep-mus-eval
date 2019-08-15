import sys
import argparse
from . import eval_mus_dir, eval_dir
from .version import _version
import musdb


def bsseval(inargs=None):
    """
    Generic cli app for bsseval results. Expects two folder with
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'reference_dir',
        type=str
    )

    parser.add_argument(
        'estimates_dir',
        type=str
    )

    parser.add_argument('-o', help='output_dir')

    parser.add_argument(
        '--win', type=float, help='Window size in seconds', default=1.0
    )

    parser.add_argument(
        '--hop', type=float, help='Hop size in seconds', default=1.0
    )

    parser.add_argument(
        '-m', type=str, help='bss_eval version [`v3`, `v4`]', default='v4'
    )

    parser.add_argument(
        '--version', '-v',
        action='version',
        version='%%(prog)s %s' % _version
    )

    args = parser.parse_args(inargs)

    if not args.o:
        output_dir = args.estimates_dir
    else:
        output_dir = args.o

    # evaluate an existing estimate folder with wav files
    data = eval_dir(
        args.reference_dir,
        args.estimates_dir,
        output_dir=output_dir,
        mode=args.m,
        win=args.win,
        hop=args.hop
    )

    print(data)


def museval(inargs=None):
    """
    Commandline interface for museval evaluation tools
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'estimates_dir',
        type=str
    )

    parser.add_argument('-o', help='output_dir')

    parser.add_argument(
        '--musdb',
        help='path to musdb',
        type=str
    )

    parser.add_argument(
        '--is-wav', help='Read musdb wav instead of stems',
        action='store_true',
    )

    args = parser.parse_args(inargs)
    mus = musdb.DB(root=args.musdb, is_wav=args.is_wav)

    if not args.o:
        output_dir = args.estimates_dir
    else:
        output_dir = args.o

    # evaluate an existing estimate folder with wav files
    eval_mus_dir(
        dataset=mus,  # instance of musdb
        estimates_dir=args.estimates_dir,  # path to estiamte folder
        output_dir=output_dir,  # set a folder to write eval json files
    )


if __name__ == '__main__':
    museval(sys.argv[1:])
