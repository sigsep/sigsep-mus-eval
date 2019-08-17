import museval
import argparse

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(
        description='MUSDB18 Evaluation',
        add_help=False
    )

    parser.add_argument(
        '--eval-path',
        type=str,
        help='Path evaluation root folder'
    )


    args = parser.parse_args()
    methods = museval.EvalStore(frames_agg="median", tracks_agg="median")
    methods.add_eval_dir(args.eval_path)
    print(methods)
