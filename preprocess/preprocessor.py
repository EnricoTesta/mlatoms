from shutil import rmtree
from logging import getLogger
from google.cloud import storage
from atoms import Atom
import os

logger = getLogger("preprocess_logger")

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
        self.read_info(fill=True)
        self.read_data()

        # Compute metadata
        self.metadata['size'] = self.data_size_in_gb
        self.metadata['n_rows'] = self.data.shape[0]
        self.metadata['n_cols'] = self.data.shape[1]
        self.metadata['rows_cols_ratio'] = self.data.shape[0] / self.data.shape[1]
        self.metadata['column_names'] = list(self.data.columns)

        self.metadata['missing_data_rate'] = self.data.isna().mean(axis=0).to_dict()

        self.metadata['unique_value_count'] = {}
        self.metadata['column_types'] = {}
        self.metadata['column_data_types'] = {}
        self.metadata['category_encodings'] = {}
        self.metadata['category_value_distribution'] = {}
        for idx, col in enumerate(self.data.columns):

            logger.info("Processing column %s of %s: %s".format(idx+1, self.metadata['n_cols'], col))

            # Very high memory operation. Do it column-wise.
            self.metadata['unique_value_count'][col] = self.data[col].nunique(dropna=True)

            if self.data.dtypes[col].name == 'object' or col in self.info["CATEGORICAL_COLUMN"] or \
                    (col == self.info["TARGET_COLUMN"] and self.info["PROBLEM_TYPE"] == 'classification'):
                self.metadata['column_types'][col] = 'categorical'
                self.metadata['column_data_types'][col] = 'category'
                self.metadata['category_encodings'][col] = list(self.data[col].astype('category').dtype.categories)
                self.metadata['category_value_distribution'][col] = self.data[col].value_counts(normalize=True, dropna=False).to_dict()
            elif col in self.info["ORDINAL_COLUMN"]:
                self.metadata['column_types'][col] = 'ordinal'
                self.metadata['column_data_types'][col] = 'float64'
            else:
                self.metadata['column_types'][col] = 'scale'
                self.metadata['column_data_types'][col] = 'float64'

        # Export metadata
        self.export_file(self.metadata, 'metadata.json')

        # Clean-up
        rmtree(self.local_path)
