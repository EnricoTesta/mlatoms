from shutil import rmtree
from atoms import Atom


class Encoder(Atom):

    def __init__(self, data_path=None, model_path=None, algo=None, params=None):
        super().__init__(data_path, model_path, algo, params)

    def run(self):

        # Fetch data
        self.retrieve_data()
        self.read_info()
        self.read_data()

        # Fit model
        self.trained_model = self.fit(self.data, None)
        self.predictions = self.transform(self.trained_model, self.data).reset_index()  # TODO: verify index

        # Export
        unique_id = self.generate_unique_id()
        self.export_trained_model(unique_id)
        self.export_predictions(unique_id)

        # Clean-up
        rmtree(self.local_path)
