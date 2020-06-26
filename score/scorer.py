import os
import pickle
import subprocess
import numpy as np
from scipy import stats
from shutil import rmtree
from pandas import DataFrame
from logging import getLogger
from sklearn.preprocessing import MinMaxScaler
from atoms import Atom, squeeze_proba

logger = getLogger("sklearn_predictor")


class BatchPredictor(Atom):
    def __init__(self, data_path, model_path, preprocess_path=None,
                 algo=None, params=None, output_dir=None, use_probabilities=False):
        super().__init__(data_path, model_path, algo, params)
        self.preprocess_path = preprocess_path  # this is either a string or a list of strings
        self.stratification_df = None
        self._model = None
        self._preprocessor = None
        self._output_dir = output_dir
        self._use_probabilities = use_probabilities

    def default_preprocess(self, info, inputs):
        logger.info("Dropping useless columns. Fetching ids...")
        column_to_drop_list = [col for col in self.data.columns if self.info["TARGET_COLUMN"] in col]
        self.stratification_df = self.data[self.info["STRATIFICATION_COLUMN"]].copy(deep=True)
        for key in ["USELESS_COLUMN", "STRATIFICATION_COLUMN"]:
            try:
                column_to_drop_list += info[key]
            except KeyError:
                pass
        for col in column_to_drop_list:
            try:
                inputs.drop(col, inplace=True, axis=1)
            except KeyError:
                logger.warning("Tried to drop \"{}\" but it wasn't found in axis.".format(col))
        ids = inputs.index
        raw_data = np.asarray(inputs)
        return ids, raw_data

    @staticmethod
    def generate_column_names(result):
        if result.shape[1] == 1:
            return ['value']  # regression setting
        else:
            names = []
            for i in range(result.shape[1]):
                names.append('probability_' + str(i))  # classification setting
            return names

    def retrieve_preprocess(self):
        if self.preprocess_path is not None:
            try:
                if isinstance(self.preprocess_path, list):
                    for item in self.preprocess_path:
                        os.system(' '.join(['gsutil -m', 'cp', item, self.local_path]))
                else:
                    os.system(' '.join(['gsutil -m', 'cp', self.preprocess_path, self.local_path]))
            except:
                raise ValueError("Preprocess file not found")

    def retrieve_model(self):
        # Fetch model & preprocess from GCS
        try:
            os.system(' '.join(['gsutil -m', 'cp', self.model_path, self.local_path]))
        except:
            raise ValueError("Model file not found")

    def restore_preprocess(self):
        # Restore preprocess
        try:
            self._preprocessor = []
            if isinstance(self.preprocess_path, list):
                for item in self.preprocess_path:
                    with open(os.path.join(self.local_path, item), 'rb') as f:
                        self._preprocessor.append(pickle.load(f))  # TODO: ensure preprocess order
            else:
                with open(os.path.join(self.local_path, self.preprocess_path), 'rb') as f:
                    self._preprocessor.append(pickle.load(f))
        except:
            self._preprocessor = None

    def restore_model(self):
        # Restore model
        try:
            with open(os.path.join(self.local_path, self.model_path.split("/")[-1]), 'rb') as f:
                self._model = pickle.load(f)
        except:
            raise ValueError("Unable to unpickle model file")

    def model_predict_proba(self, inputs):
        return inputs

    def model_predict(self, inputs):
        return inputs

    @staticmethod
    def _neutralize(df, columns, by, proportion=1.0):
        scores = df[columns]
        exposures = df[by].values
        scores = scores - proportion * exposures.dot(np.linalg.pinv(exposures).dot(scores))
        if scores.std().values[0] == 0:
            print("0 standard deviation!")
            x = scores / scores.std()
            x.reset_index().to_csv(os.getcwd() + "/0_std_dev_neutralize.csv", index=False)
            gcp_path = "gs://my-model-bucket-1000/ET/NUMER/217/TEST_NEUTRAL/0_std_dev_neutralize.csv"
            subprocess.check_call(['gsutil', 'cp', os.getcwd() + "/0_std_dev_neutralize.csv", gcp_path])

            scores.to_csv(os.getcwd() + "/0_std_dev_scores.csv", index=False)
            gcp_path = "gs://my-model-bucket-1000/ET/NUMER/217/TEST_NEUTRAL/0_std_dev_scores.csv"
            subprocess.check_call(['gsutil', 'cp', os.getcwd() + "/0_std_dev_scores.csv", gcp_path])

            df[columns].to_csv(os.getcwd() + "/0_std_dev_cols.csv", index=False)
            gcp_path = "gs://my-model-bucket-1000/ET/NUMER/217/TEST_NEUTRAL/0_std_dev_cols.csv"
            subprocess.check_call(['gsutil', 'cp', os.getcwd() + "/0_std_dev_cols.csv", gcp_path])

            DataFrame(exposures).to_csv(os.getcwd() + "/0_std_dev_exposures.csv", index=False)
            gcp_path = "gs://my-model-bucket-1000/ET/NUMER/217/TEST_NEUTRAL/0_std_dev_exposures.csv"
            subprocess.check_call(['gsutil', 'cp', os.getcwd() + "/0_std_dev_exposures.csv", gcp_path])

        return scores / scores.std()

    @staticmethod
    def _normalize(df):
        X = (df.rank(method="first") - 0.5) / len(df)
        return stats.norm.ppf(X)


    def normalize_and_neutralize(self, df, columns, by, proportion=1.0):
        # Convert the scores to a normal distribution
        df[columns] = self._normalize(df[columns])
        df[columns] = self._neutralize(df, columns, by, proportion)
        return df[columns]

    def neutralize_scores(self, scores, proportion=1.0):

        scores.to_csv(os.getcwd() + "/scores.csv", index=False)
        gcp_path = "gs://my-model-bucket-1000/ET/NUMER/217/TEST_NEUTRAL/scores.csv"
        subprocess.check_call(['gsutil', 'cp', os.getcwd() + "/scores.csv", gcp_path])

        # TODO: generalize to multiclass case
        # Verify scores are 1-D, squeeze them otherwise (!!! - NOT GENERAL IN MULTI-CLASS CASE - !!!)
        squeezed_scores = squeeze_proba(scores.set_index('id'), index=True)

        squeezed_scores.to_csv(os.getcwd() + "/squeezed_scores.csv", index=False)
        gcp_path = "gs://my-model-bucket-1000/ET/NUMER/217/TEST_NEUTRAL/squeezed_scores.csv"
        subprocess.check_call(['gsutil', 'cp', os.getcwd() + "/squeezed_scores.csv", gcp_path])

        # Merge in a single dataframe
        features = self.data.columns
        df = self.data.merge(squeezed_scores, left_index=True, right_index=True)\
            .merge(self.stratification_df, left_index=True, right_index=True)

        print("Number of missing values in dataframe is {}".format(df.isna().sum().sum()))

        df.sample(100).to_csv(os.getcwd() + "/sample_df.csv", index=False)
        gcp_path = "gs://my-model-bucket-1000/ET/NUMER/217/TEST_NEUTRAL/sample_df.csv"
        subprocess.check_call(['gsutil', 'cp', os.getcwd() + "/sample_df.csv", gcp_path])

        neutralized_scores = df.groupby(self.info["STRATIFICATION_COLUMN"])\
            .apply(lambda x: self.normalize_and_neutralize(x, ["preds"], features, proportion))

        print("Number of missing values in dataframe is {}".format(neutralized_scores.isna().sum().sum()))

        neutralized_scores.to_csv(os.getcwd() + "/neutralized_scores.csv", index=False)
        gcp_path = "gs://my-model-bucket-1000/ET/NUMER/217/TEST_NEUTRAL/neutralized_scores.csv"
        subprocess.check_call(['gsutil', 'cp', os.getcwd() + "/neutralized_scores.csv", gcp_path])

        scaler = MinMaxScaler()
        scaled_neutralized_scores = DataFrame([neutralized_scores.index, scaler.fit_transform(neutralized_scores).flatten()]).transpose()
        scaled_neutralized_scores.columns = ["id", "scores"]

        scaled_neutralized_scores.to_csv(os.getcwd() + "/scaled_neutralized_scores.csv", index=False)
        gcp_path = "gs://my-model-bucket-1000/ET/NUMER/217/TEST_NEUTRAL/scaled_neutralized_scores.csv"
        subprocess.check_call(['gsutil', 'cp', os.getcwd() + "/scaled_neutralized_scores.csv", gcp_path])

        return scaled_neutralized_scores  # transform back to 0-1


    def run(self):

        # Fetch data
        self.retrieve_data()
        self.retrieve_metadata()
        self.read_info()
        self.read_metadata()
        self.read_data(**self.params['read'])

        # Load model
        self.retrieve_preprocess()
        self.retrieve_model()
        self.restore_preprocess()
        self.restore_model()

        # Score model
        ids, raw_data = self.default_preprocess(self.info, self.data)
        preprocessed_inputs = raw_data

        try:
            for item in self._preprocessor:
                preprocessed_inputs = item.preprocess(preprocessed_inputs)
        except:
            logger.info("No preprocessing applied")

        tmp_file_path = os.path.join(self.local_path, 'results.csv')
        neutral_tmp_file_path = os.path.join(self.local_path, 'neutralized_results.csv')
        if self._use_probabilities:
            logger.info("Predicting probabilities...")
            probabilities = self.model_predict_proba(preprocessed_inputs)
            column_names = self.generate_column_names(probabilities)
            scores = DataFrame(np.concatenate((ids.values.reshape(-1, 1), probabilities), axis=1),
                     columns=['id'] + column_names)
        else:
            logger.info("Predicting values...")
            outputs = self.model_predict(preprocessed_inputs)
            column_names = self.generate_column_names(outputs)
            scores = DataFrame(np.concatenate((ids.values.reshape(-1, 1), outputs), axis=1),
                     columns=['id'] + column_names)

        # Neutralize scores (a.k.a. remove linear exposures with features)
        neutralized_scores = self.neutralize_scores(scores)

        # Write scores
        scores.to_csv(path_or_buf=tmp_file_path, index=False)
        neutralized_scores.to_csv(path_or_buf=neutral_tmp_file_path, index=False)

        # Step 5 - Send results to GCS
        subprocess.check_call(['gsutil', 'cp', tmp_file_path,
                               os.path.join(self._output_dir, 'results.csv')])
        if self._output_dir.startswith("gs://"):
            shards = self._output_dir.split("/")
            prefix = shards[0:-4]
            neutral_folder = ['NEUTRALIZED_' + shards[-4]]
            suffix = shards[-3:]
            full_neutralized_path = os.path.join("/".join(prefix + neutral_folder + suffix), 'neutralized_results.csv')
        else:
            full_neutralized_path = os.path.join("/".join(["/".join(self._output_dir.split("/")[0:-2]),
                                                           'NEUTRALIZED_'+self._output_dir.split("/")[-2]]),
                                                 'neutralized_results.csv')
        subprocess.check_call(['gsutil', 'cp', neutral_tmp_file_path, full_neutralized_path])

        # Clean-up
        rmtree(self.local_path)
