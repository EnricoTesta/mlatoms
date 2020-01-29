from postprocessor import Aggregator
import argparse


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Aggregator')
    parser.add_argument(
        '--method',
        default='average',
        metavar='method',
        help='Aggregation method')
    parser.add_argument(
        '--model-dir',
        default='/mlatoms/test/outputs',
        metavar='model_dir',
        help='The directory to store the object')
    parser.add_argument(
        '--train-files',
        default='/mlatoms/test/aggregatedir/',
        metavar='train_files',
        help='The directory to fetch train data')

    args = parser.parse_args()
    return args


class ScoreAggregator:

    @staticmethod
    def transform(data):
        return data.groupby(level=0).mean()


def main():
    # Training settings
    args = get_args()
    args_dict = vars(args)

    # Make param dict
    param_dict = {'algo': {}, 'fit': {}}
    for item in args_dict:
        if item not in ('model_dir', 'train_files'):
            param_dict['algo'][item] = args_dict[item]

    t = Aggregator(data_path=args.train_files, model_path=args.model_dir, algo=ScoreAggregator, params=param_dict)
    t.run()


if __name__ == '__main__':
    main()
