from utils.helpers import (
    generate_ngram_results,
    extract_weighted_keywords,
    add_features_to_df,
)
from utils.feature_extractor import FeatureExtractor
import gensim
from gensim.models.doc2vec import Doc2Vec
import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import json
import pickle


class AutoSAS:
    def __init__(self, essay_set, dataset, splits, scaler=None, best_model=None):
        self.essay_set = essay_set
        self.dataset = dataset
        self.splits = splits

        self.scaler = scaler
        self.best_model = best_model

        # init methods
        self.split_dataset()

        # make a folder for everything in this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = f"output/{timestamp}_autosas"

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def split_dataset(self):
        for split_name, ids in self.splits.items():
            int_ids = [int(id) for id in ids]
            self.dataset.loc[self.dataset["Id"].isin(int_ids), "split_custom"] = (
                split_name
            )

    def augment_dataset(self):
        additional_training_data_path = (
            f"augmented_training_data/augmented_training_data_{self.essay_set}.csv"
        )
        additional_training_data = pd.read_csv(additional_training_data_path)
        self.dataset = pd.concat(
            [self.dataset, additional_training_data], ignore_index=True
        )

    def extract_features(self, use_better_spelling=True):
        n_gram_results = generate_ngram_results(self.essay_set, self.dataset, 30)
        # weighted_keywords = extract_weighted_keywords(self.essay_set, self.dataset)
        with open("weighted_keywords.json", "r") as file:
            weighted_keywords = json.load(file)[str(self.essay_set)]

        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
            "word2vec_model/GoogleNews-vectors-negative300.bin", binary=True
        )
        doc2vec_model = Doc2Vec.load(
            "doc2vec_model/doc2vec_wiki_d300_n5_w8_mc50_t12_e10_dbow.model"
        )

        feature_extractor = FeatureExtractor(
            self.essay_set,
            word2vec_model,
            doc2vec_model,
            n_gram_results,
            weighted_keywords,
        )
        add_features_to_df(
            self.dataset,
            feature_extractor,
            use_better_spelling=use_better_spelling,
            add_word2vec=True,
            add_doc2vec=True,
            add_pos=True,
            add_prompt_overlap=True,
            add_weighted_keywords=True,
            add_lexical_overlap=True,
            add_stylistic_features=True,
            add_logical_operators=True,
            add_temporal_features=True,
        )

        with open(f"{self.output_folder}/dataset_with_features.pkl", "wb") as file:
            pickle.dump(self.dataset, file)

    def train_model(
        self,
        features=[
            "word2vec_features",
            "doc2vec_features",
            "pos_features",
            "prompt_overlap_features",
            "weighted_keywords_features",
            "lexical_overlap_features",
            "stylistic_features",
            "logical_operators_features",
            "temporal_features",
        ],
        verbose=False,
    ):
        # prepare data set
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # generate training data
        training_data = self.dataset[(self.dataset["split_custom"] == "train")].copy(
            deep=True
        )
        training_data = training_data.dropna(
            subset=["word2vec_features", "doc2vec_features"]
        )
        X_train = np.vstack(
            training_data[features].apply(lambda x: np.hstack(x), axis=1)
        )
        X_train = self.scaler.fit_transform(X_train)
        y_train = training_data["Score1"]

        num_features = X_train.shape[1]
        print(f"Number of features in X_train: {num_features}")

        # Define hyperparameter ranges
        max_depths = [50, 75, 100, 150, 200]
        n_estimators_list = [50, 75, 100, 150, 200]

        # Initialize variables to store the best model and its score
        best_kappa_score = -1
        best_params = None
        best_clf = None

        # Initialize a list to store results
        results = []

        # Iterate over all combinations of hyperparameters
        for max_depth in max_depths:
            for n_estimators in n_estimators_list:
                # Train model
                clf = RandomForestClassifier(
                    max_depth=max_depth, n_estimators=n_estimators
                )
                clf.fit(X_train, y_train)

                # Validate model
                val_set = self.dataset[(self.dataset["split_custom"] == "valid")].copy(
                    deep=True
                )
                val_set = val_set.dropna(
                    subset=["word2vec_features", "doc2vec_features"]
                )

                X_val = np.vstack(
                    val_set[features].apply(lambda x: np.hstack(x), axis=1).values
                )
                X_val = self.scaler.transform(X_val)
                y_val = val_set["Score1"]

                y_pred = clf.predict(X_val)

                # Calculate evaluation metrics
                conf_matrix = confusion_matrix(y_val, y_pred)
                kappa_score = cohen_kappa_score(
                    y_val, y_pred, weights="quadratic", sample_weight=None
                )
                accuracy = accuracy_score(y_val, y_pred)

                # Print the results for the current hyperparameters
                if verbose:
                    print(f"Max Depth: {max_depth}, N Estimators: {n_estimators}")
                    print(f"Cohen Kappa Score (Weighted): {kappa_score}")
                    print(f"Accuracy: {accuracy}")

                # Append the results to the list
                results.append(
                    {
                        "max_depth": max_depth,
                        "n_estimators": n_estimators,
                        "qwk": kappa_score,
                        "accuracy": accuracy,
                    }
                )

                # Update the best model if the current one is better
                if kappa_score > best_kappa_score:
                    best_kappa_score = kappa_score
                    best_params = (max_depth, n_estimators)
                    best_clf = clf

        # Print the best hyperparameters and their score
        print(f"Essay Set: {self.essay_set}")
        print(
            f"Best Hyperparameters: Max Depth: {best_params[0]}, N Estimators: {best_params[1]}"
        )
        print(f"Best Cohen Kappa Score (Weighted): {best_kappa_score}")

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Save the results to a CSV file
        results_df.to_csv(
            f'{self.output_folder}/{self.essay_set}_validation_set_results_{"_".join(features)}.csv',
            index=False,
        )

        # Pickle the model and the scaler
        with open(
            f"{self.output_folder}/{self.essay_set}_best_model.pkl", "wb"
        ) as model_file:
            pickle.dump(best_clf, model_file)

        with open(
            f"{self.output_folder}/{self.essay_set}_scaler.pkl", "wb"
        ) as scaler_file:
            pickle.dump(self.scaler, scaler_file)

        self.best_model = best_clf

    def evaluate_model(
        self,
        features=[
            "word2vec_features",
            "doc2vec_features",
            "pos_features",
            "prompt_overlap_features",
            "weighted_keywords_features",
            "lexical_overlap_features",
            "stylistic_features",
            "logical_operators_features",
            "temporal_features",
        ],
    ):
        # Ensure that the best model has been set
        if not hasattr(self, "best_model") or self.best_model is None:
            raise ValueError(
                "Best model has not been set. Please run the training process first."
            )

        # Validate model on test set
        test_set = self.dataset[(self.dataset["split_custom"] == "test")].copy(
            deep=True
        )
        test_set = test_set.dropna(subset=["word2vec_features", "doc2vec_features"])

        X_test = np.vstack(
            test_set[features].apply(lambda x: np.hstack(x), axis=1).values
        )
        X_test = self.scaler.transform(X_test)
        y_test = test_set["Score1"]

        # Make predictions on the test set
        y_pred_test = self.best_model.predict(X_test)

        # Calculate evaluation metrics
        test_conf_matrix = confusion_matrix(y_test, y_pred_test)
        test_kappa_score = cohen_kappa_score(y_test, y_pred_test, weights="quadratic")
        test_accuracy = accuracy_score(y_test, y_pred_test)

        # Print the evaluation results
        print("Evaluation on Test Set:")
        print(f"Cohen Kappa Score (Weighted): {test_kappa_score}")
        print(f"Accuracy: {test_accuracy}")
        print(f"Confusion Matrix:\n{test_conf_matrix}")

        # Convert evaluation results to a DataFrame
        evaluation_results_df = pd.DataFrame(
            {
                "Metric": ["Cohen Kappa Score (Weighted)", "Accuracy"],
                "Value": [test_kappa_score, test_accuracy],
            }
        )

        # Save the evaluation results to a CSV file
        evaluation_results_df.to_csv(
            f"{self.output_folder}/{self.essay_set}_test_set_results.csv",
            index=False,
        )
