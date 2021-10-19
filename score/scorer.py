import os
import pickle
import subprocess
import numpy as np
from yaml import safe_load
from scipy import stats
from shutil import rmtree
from pandas import DataFrame, read_csv
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
        self.model_metadata = None
        self.model_feature_importance = None
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
        try:
            if result.shape[1] == 1:
                return ['value']  # regression setting
            else:
                names = []
                for i in range(result.shape[1]):
                    names.append('probability_' + str(i))  # classification setting
                return names
        except IndexError:
            return ['value']  # regression setting

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
            remote_model_path = "/".join(self.model_path.split("/")[0:-1])
            os.system(' '.join(['gsutil -m', 'cp -r', remote_model_path, self.local_path]))
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
            with open(os.path.join(self.local_path, self.model_path.split("/")[-2], self.model_path.split("/")[-1]), 'rb') as f:
                self._model = pickle.load(f)
        except:
            raise ValueError("Unable to unpickle model file")

    def read_feature_importances(self):

        model_featimp_path = "".join([self.local_path, self.model_path.split("/")[-2], "/"])
        file_list = [item for item in os.listdir(model_featimp_path)
                     if os.path.isfile(os.path.join(model_featimp_path, item)) and
                     item.endswith('csv') and item.startswith('featimp')]
        if len(file_list) > 1:
            raise ValueError("Found multiple feature importance files. You must provide a single model feature importance file.")
        local_featimp_path = os.path.join(model_featimp_path, file_list[0])
        try:
            self.model_feature_importance = read_csv(local_featimp_path)
        except FileNotFoundError:
            logger.warning("Feature importance not available.")


    def read_model_metadata(self):
        """
        Load model metadata in memory. Raise exception if operation fails.
        """
        model_metadata_path = "".join([self.local_path, self.model_path.split("/")[-2], "/"])
        file_list = [item for item in os.listdir(model_metadata_path)
                     if os.path.isfile(os.path.join(model_metadata_path, item)) and
                     item.endswith('json') and item.startswith('params')]
        if len(file_list) > 1:
            raise ValueError("Found multiple model parameter files. You must provide a single model parameter file.")
        local_params_path = os.path.join(model_metadata_path, file_list[0])
        with open(local_params_path, 'r') as stream:
            self.model_metadata = safe_load(stream)

    def model_predict_proba(self, inputs):
        return inputs

    def model_predict(self, inputs):
        return inputs

    @staticmethod
    def _neutralize(df, columns, by, proportion=1.0):
        df_32 = df.astype('float32', copy=True) # ensure data type compatibility with np.linalg
        scores = df_32[columns]
        exposures = df_32[by].values

        # Very expensive as exposures increase in number
        scores = scores - proportion * exposures.dot(np.linalg.pinv(exposures).dot(scores))
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

        # TODO: generalize to multiclass case
        # Verify scores are 1-D, squeeze them otherwise (!!! - NOT GENERAL IN MULTI-CLASS CASE - !!!)
        squeezed_scores = squeeze_proba(scores.set_index('id'), index=True)

        # Merge in a single dataframe
        self.read_feature_importances()
        # Select features to neutralize on. At most 300 features to neutralize on.
        if self.model_feature_importance is not None:
            relevant_features_df = self.model_feature_importance.loc[self.model_feature_importance['feature_importance'] > 0].sort_values('feature_importance', ascending=False)
            if relevant_features_df.shape[1] <= 300:
                features = list(relevant_features_df['feature_name'])
            else:
                features = list(relevant_features_df['feature_name'].iloc[0:300])
        elif self.data.columns <= 300:
            features = self.data.columns
        else:
            features = self.data.columns[0:300]
        df = self.data.merge(squeezed_scores, left_index=True, right_index=True)\
            .merge(self.stratification_df, left_index=True, right_index=True)

        try:
            neutralized_scores = df.groupby(self.info["STRATIFICATION_COLUMN"])\
                .apply(lambda x: self.normalize_and_neutralize(x, ["preds"], features, proportion))
        except:
            neutralized_scores = df.groupby(self.info["STRATIFICATION_COLUMN"]) \
                .apply(lambda x: self.normalize_and_neutralize(x, ["value"], features, proportion))

        scaler = MinMaxScaler()
        scaled_neutralized_scores = DataFrame([neutralized_scores.index, scaler.fit_transform(neutralized_scores).flatten()]).transpose()
        scaled_neutralized_scores.columns = ["id", "scores"]

        return scaled_neutralized_scores  # transform back to 0-1


    def run(self):

        # Fetch data
        self.retrieve_data()
        self.retrieve_metadata()
        self.read_info()
        self.read_metadata()
        self.retrieve_preprocess()
        self.retrieve_model()
        self.read_model_metadata()
        strat_column = self.info["STRATIFICATION_COLUMN"] if "STRATIFICATION_COLUMN" in self.info.keys() else []
        cols_to_read = [self.info["ID_COLUMN"]] + strat_column + self.model_metadata['features']
        self.params['read']['columns'] = cols_to_read # read only feature on which the model was trained on
        self.read_data(**self.params['read'])

        # Load model
        self.restore_preprocess()
        self.restore_model()

        # Score model
        ids, preprocessed_inputs = self.default_preprocess(self.info, self.data)

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
            del preprocessed_inputs
            column_names = self.generate_column_names(probabilities)
            scores = DataFrame(np.concatenate((ids.values.reshape(-1, 1), probabilities), axis=1),
                     columns=['id'] + column_names)
            del probabilities
        else:
            logger.info("Predicting values...")
            outputs = self.model_predict(preprocessed_inputs)
            del preprocessed_inputs
            column_names = self.generate_column_names(outputs)
            scores = DataFrame(np.concatenate((ids.values.reshape(-1, 1), outputs.reshape(-1, 1)), axis=1),
                     columns=['id'] + column_names)
            scores['value'] =scores['value'].astype(float)
            del outputs

        del ids

        # Neutralize scores (a.k.a. remove linear exposures with features)
        logger.info("Neutralizing scores...")
        neutralized_scores = self.neutralize_scores(scores)

        # Write scores
        logger.info("Writing scores...")
        scores.to_csv(path_or_buf=tmp_file_path, index=False)
        neutralized_scores.to_csv(path_or_buf=neutral_tmp_file_path, index=False)

        # Step 5 - Send results to GCS
        subprocess.check_call(['gsutil', 'cp', tmp_file_path,
                               os.path.join(self._output_dir, 'results.csv')])
        if self._output_dir.startswith("gs://"):
            shards = self._output_dir.split("/")
            prefix = shards[0:-2]
            neutral_folder = ['NEUTRALIZED_' + shards[-2]] #TODO: fix this path
            suffix = [''] # shards[-3:]
            full_neutralized_path = os.path.join("/".join(prefix + neutral_folder + suffix), 'neutralized_results.csv')
        else:
            full_neutralized_path = os.path.join("/".join(["/".join(self._output_dir.split("/")[0:-2]),
                                                           'NEUTRALIZED_'+self._output_dir.split("/")[-2]]),
                                                 'neutralized_results.csv')
        subprocess.check_call(['gsutil', 'cp', neutral_tmp_file_path, full_neutralized_path])

        # Clean-up
        rmtree(self.local_path)
