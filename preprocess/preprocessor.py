from shutil import rmtree
from google.cloud import storage
from atoms import Atom
import os


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


class DataEvaluator(Atom):

    def __init__(self, data_path=None, model_path=None, algo=None, params=None):
        super().__init__(data_path, model_path, algo, params)
        self.data_size_in_gb = self.get_data_size()
        self.metadata = {}

    def get_data_size(self):
        try:
            gcs_client = storage.Client()
            blob_list = [blob for blob
                         in list(gcs_client.list_blobs(bucket_or_name=self.data_path.split("/")[2],
                                                       prefix="/".join(self.data_path.split("/")[3:])))
                         if blob.name[-1] != "/"]
        except:
            # look locally
            blob_list = [item for item in os.listdir(self.data_path)
                         if os.path.isfile(os.path.join(self.data_path, item)) and item.split(".")[-1] == 'csv']

        # Get data size in bytes
        data_size = 0
        for blob in blob_list:
            try:
                if blob.name[-3:] == 'csv':
                    data_size += blob.size
            except AttributeError:
                if blob[-3:] == 'csv':
                    data_size += os.path.getsize(os.path.join(self.data_path, blob))
        return data_size/pow(10, 9)  # in GB

    def run(self):

        # Fetch data
        if self.data_size_in_gb < 20:
            self.retrieve_data()
        else:
            raise NotImplementedError("Data size above 20 GB.")  # TODO: implement chunk retrieve
        self.read_info()
        self.read_data()

        # Compute metadata
        self.metadata['size'] = self.data_size_in_gb
        self.metadata['n_rows'] = self.data.shape[0]
        self.metadata['n_cols'] = self.data.shape[1]
        self.metadata['rows_cols_ratio'] = self.data.shape[0] / self.data.shape[1]
        self.metadata['column_names'] = list(self.data.columns)
        self.metadata['missing_data_rate'] = self.data.isna().mean(axis=0).to_dict()
        self.metadata['unique_value_count'] = self.data.nunique(dropna=True).to_dict()

        # Export metadata
        self.export_json(self.metadata, 'metadata.json')

        # CLean-up
        rmtree(self.local_path)
