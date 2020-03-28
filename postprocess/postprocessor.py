from shutil import rmtree
from subprocess import check_call
from atoms import Atom
from pandas import read_csv, concat, DataFrame
import os


class Aggregator(Atom):

    def __init__(self, data_path=None, output_dir=None, algo=None, params=None):
        super().__init__(data_path, None, algo, params)
        self._output_dir = output_dir
        self._original_index = None

    def read_data(self, encode_features_to_int=False, encode_features_to_one_hot=False,
                  encode_target_to_int=False, encode_target_to_one_hot=False):
        try:
            file_list = list()
            for (dirpath, dirnames, filenames) in os.walk(self.local_path):
                file_list += [os.path.join(dirpath, file) for file in filenames if file.split(".")[-1] == 'csv']

            if len(file_list) == 1:
                self.data = read_csv(os.path.join(self.local_path, file_list[0]))
                self._original_index = DataFrame(self.data.iloc[:, 0])  # Assume first column is index column
            else:
                dfs = []
                for file in file_list:
                    dfs.append(read_csv(os.path.join(self.local_path, file)))
                self._original_index = DataFrame(dfs[0].iloc[:, 0])  # Assume first column is index & all files share same index
                self.data = concat(dfs, axis=0)
            self.data.set_index(self.data.columns[0], inplace=True, drop=True)  # Assume first column is index column
        except:
            raise Exception("Unable to load data file.")

    def run(self):

        if self.params['method'] != 'average':
            raise NotImplementedError("Only supported aggregation method is: 'average'")

        # Fetch data
        self.retrieve_data()
        self.read_data()

        # Fit model
        self.trained_model = self.algo(self._original_index)
        self.predictions = self.transform(self.trained_model, self.data)

        # Export
        local_results_uri = os.path.join(self.local_path, 'results.csv')
        self.predictions.to_csv(path_or_buf=local_results_uri, index=False)
        check_call(['gsutil', 'cp', local_results_uri, self._output_dir])

        # Clean-up
        rmtree(self.local_path)
