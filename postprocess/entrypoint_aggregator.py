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
        '--score-dir',
        default='/mlatoms/test/aggregatedir/',
        metavar='score_dir',
        help='The directory to fetch score data')
    parser.add_argument(
        '--output-dir',
        default='/mlatoms/test/outputs/',
        metavar='output_dir',
        help='The directory to store output data')

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
    param_dict = {}
    for item in args_dict:
        param_dict[item] = args_dict[item]

    t = Aggregator(data_path=param_dict['score_dir'], output_dir=param_dict['output_dir'],
                   algo=ScoreAggregator, params=param_dict)
    t.run()


if __name__ == '__main__':
    main()
