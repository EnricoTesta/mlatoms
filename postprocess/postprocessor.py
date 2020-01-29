from shutil import rmtree
from subprocess import check_call
from atoms import Atom
from pandas import read_csv, concat
import os


class Aggregator(Atom):

    def __init__(self, data_path=None, output_dir=None, algo=None, params=None):
        super().__init__(data_path, None, algo, params)
        self._output_dir = output_dir

    def read_data(self):
        try:
            file_list = list()
            for (dirpath, dirnames, filenames) in os.walk(self.local_path):
                file_list += [os.path.join(dirpath, file) for file in filenames if file.split(".")[-1] == 'csv']

            if len(file_list) == 1:
                self.data = read_csv(os.path.join(self.local_path, file_list[0]))
            else:
                dfs = []
                for file in file_list:
                    dfs.append(read_csv(os.path.join(self.local_path, file)))
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
        self.trained_model = self.algo()
        self.predictions = self.transform(self.trained_model, self.data).reset_index()  # TODO: verify index

        # Export
        local_results_uri = os.path.join(self.local_path, 'results.csv')
        self.predictions.to_csv(path_or_buf=local_results_uri, index=False)
        check_call(['gsutil', 'cp', local_results_uri, self._output_dir])

        # Clean-up
        rmtree(self.local_path)
