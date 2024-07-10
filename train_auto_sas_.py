from utils.auto_sas import AutoSAS

import argparse
import pandas as pd
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AutoSAS model.")
    parser.add_argument("--essay_set", type=int, help="Essay set number.")
    parser.add_argument("--input_data_file", type=str, help="Input data CSV file.")
    parser.add_argument("--input_splits_file", type=str, help="Input splits file.")
    parser.add_argument(
        "--augment_dataset",
        type=bool,
        default=True,
        help="Whether to augment the dataset.",
    )
    parser.add_argument(
        "--use_better_spelling",
        type=bool,
        default=True,
        help="Whether to use better spelling for feature extraction.",
    )
    args = parser.parse_args()

    # Load the dataset
    dataset = pd.read_csv(args.input_data_file)

    # Load the splits
    with open(args.input_splits_file, "rb") as f:
        splits = pickle.load(f)

    # Initialize AutoSAS
    auto_sas = AutoSAS(args.essay_set, dataset, splits)

    # Optionally augment the dataset
    if args.augment_dataset:
        auto_sas.augment_dataset()

    auto_sas.extract_features(use_better_spelling=args.use_better_spelling)

    auto_sas.train_model()

    auto_sas.evaluate_model()
