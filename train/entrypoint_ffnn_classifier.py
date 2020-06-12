from tensorflow import keras
from trainer import Trainer
from numpy import linspace
import argparse
import os


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Light Gradient Boosting Classifier')
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        metavar='optimizer',
        help='Can be adam, sgd')
    parser.add_argument(
        '--n_hidden_layers',
        type=int,
        default=1,
        metavar='n_hidden_layers',
        help='Number of hidden layers')
    parser.add_argument(
        '--architecture',
        type=str,
        default='funnel',
        metavar='architecture',
        help='Network architecture regarding number of neurons per layer')
    parser.add_argument(
        '--activation',
        type=str,
        default='relu',
        metavar='activation',
        help="Neuron's activation function.")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        metavar='batch_size',
        help='Number of samples for stochastic gradient steps')
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        metavar='epochs',
        help='Number of epochs')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        metavar='learning_rate',
        help='Learning rate.')
    parser.add_argument(
        '--loss',
        type=str,
        default='categorical_crossentropy',
        metavar='loss',
        help='Training loss function')
    parser.add_argument(
        '--model-dir',
        type=str,
        default='/mlatoms/test/modeldir',
        metavar='model_dir',
        help='The directory to store the model')
    parser.add_argument(
        '--train-files',
        type=str,
        default='/mlatoms/data/classification/multi/numer/train/',
        metavar='train_files',
        help='The directory to fetch train data')
    parser.add_argument(
        '--hypertune-loss',
        default=None,
        metavar='hypertune_loss',
        help='Hypertune loss name')

    args = parser.parse_args()
    return args


class FFNNTrainer(Trainer):

    def fit(self, x, y, algo_params, fit_params):

        ffnn_classifier = self.algo()

        # Architecture parsing
        if algo_params['architecture'] == 'funnel' and algo_params['n_hidden_layers'] > 0:
            n_neurons = linspace(x.shape[1], y.shape[1],
                                 num=algo_params['n_hidden_layers'] + 1,
                                 endpoint=False,
                                 dtype=int)[1:]  # remove first element as it's the input layer
        else:
            raise NotImplementedError

        # Add hidden layers
        for i in range(algo_params['n_hidden_layers']):
            if i == 0:
                ffnn_classifier.add(keras.layers.Dense(n_neurons[i],
                                    activation=algo_params['activation'],
                                    input_shape=(x.shape[1],)))
            else:
                ffnn_classifier.add(keras.layers.Dense(n_neurons[i], activation=algo_params['activation']))

        # Add output layer
        if algo_params['n_hidden_layers'] > 0:
            ffnn_classifier.add(keras.layers.Dense(y.shape[1]))
        else:
            ffnn_classifier.add(keras.layers.Dense(y.shape[1], input_shape=(x.shape[1],)))

        # Define optimizer and loss function
        if algo_params['optimizer'] == 'adam':
            opt = keras.optimizers.Adam(learning_rate=algo_params['learning_rate'])
        else:
            raise NotImplementedError

        if algo_params['loss'] == 'categorical_crossentropy':
            loss_function = keras.losses.CategoricalCrossentropy(from_logits=True)
        else:
            raise NotImplementedError

        ffnn_classifier.compile(optimizer=opt, loss=loss_function)

        # fit_params: batch_size, epochs
        ffnn_classifier.fit(x, y, **fit_params)

        # attach softmax layer (from log-odds to probabilities)
        return keras.Sequential([ffnn_classifier, keras.layers.Softmax()])

    def export_model_file(self, obj, folder_name):
        tmp_folder_name = os.path.join(self.local_path, folder_name)
        obj.save(tmp_folder_name)
        if not self.model_path.startswith('gs://') and not os.path.isdir(os.path.join(self.model_path, folder_name)):
            os.makedirs(os.path.join(self.model_path, folder_name))
        os.system(' '.join(['gsutil', 'rsync -r', tmp_folder_name, os.path.join(self.model_path, folder_name)]))  # subprocess.check_call fails


def main():
    # Training settings
    args = get_args()
    args_dict = vars(args)

    # TODO: normalize input data
    # Make param dict
    param_dict = {'algo': {}, 'fit': {}, 'read': {'encode_features_to_int': False, 'encode_features_to_one_hot': True,
                                                  'encode_target_to_int': False, 'encode_target_to_one_hot': True}}
    for item in args_dict:
        if item not in ('model_dir', 'train_files', 'hypertune_loss', 'epochs', 'batch_size'):
            param_dict['algo'][item] = args_dict[item]
        elif item in ('epochs', 'batch_size'):
            param_dict['fit'][item] = args_dict[item]

    t = FFNNTrainer(data_path=args.train_files, model_path=args.model_dir, algo=keras.models.Sequential,
                    params=param_dict, hypertune_loss=args.hypertune_loss)
    t.run()


if __name__ == '__main__':
    main()
