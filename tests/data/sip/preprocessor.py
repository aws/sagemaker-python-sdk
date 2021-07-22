import argparse
import os
import warnings

import subprocess

subprocess.call(["pip", "install", "sagemaker-experiments"])

import pandas as pd
import numpy as np
import tarfile

from smexperiments.tracker import Tracker

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)

columns = [
    "turbine_id",
    "turbine_type",
    "wind_speed",
    "rpm_blade",
    "oil_temperature",
    "oil_level",
    "temperature",
    "humidity",
    "vibrations_frequency",
    "pressure",
    "wind_direction",
    "breakdown",
]

if __name__ == "__main__":

    # Read the arguments passed to the script.
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    args, _ = parser.parse_known_args()

    # Tracking specific parameter value during job.
    tracker = Tracker.load()
    tracker.log_parameter("train-test-split-ratio", args.train_test_split_ratio)

    print("Received arguments {}".format(args))

    # Read input data into a Pandas dataframe.
    input_data_path = os.path.join("/opt/ml/processing/input", "windturbine_raw_data_header.csv")
    print("Reading input data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path)
    df.columns = columns

    # Replacing certain null values.
    df["turbine_type"] = df["turbine_type"].fillna("HAWT")
    tracker.log_parameter("default-turbine-type", "HAWT")

    df["oil_temperature"] = df["oil_temperature"].fillna(37.0)
    tracker.log_parameter("default-oil-temperature", 37.0)

    # Defining one-hot encoders.
    transformer = make_column_transformer(
        (["turbine_id", "turbine_type", "wind_direction"], OneHotEncoder(sparse=False)),
        remainder="passthrough",
    )

    X = df.drop("breakdown", axis=1)
    y = df["breakdown"]

    featurizer_model = transformer.fit(X)
    features = featurizer_model.transform(X)
    labels = LabelEncoder().fit_transform(y)

    # Splitting.
    split_ratio = args.train_test_split_ratio
    print("Splitting data into train and validation sets with ratio {}".format(split_ratio))
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=split_ratio, random_state=0
    )

    print("Train features shape after preprocessing: {}".format(X_train.shape))
    print("Validation features shape after preprocessing: {}".format(X_val.shape))

    # Saving outputs.
    train_features_output_path = os.path.join("/opt/ml/processing/train", "train_features.csv")
    train_labels_output_path = os.path.join("/opt/ml/processing/train", "train_labels.csv")

    val_features_output_path = os.path.join("/opt/ml/processing/val", "val_features.csv")
    val_labels_output_path = os.path.join("/opt/ml/processing/val", "val_labels.csv")

    print("Saving training features to {}".format(train_features_output_path))
    pd.DataFrame(X_train).to_csv(train_features_output_path, header=False, index=False)

    print("Saving validation features to {}".format(val_features_output_path))
    pd.DataFrame(X_val).to_csv(val_features_output_path, header=False, index=False)

    print("Saving training labels to {}".format(train_labels_output_path))
    pd.DataFrame(y_train).to_csv(train_labels_output_path, header=False, index=False)

    print("Saving validation labels to {}".format(val_labels_output_path))
    pd.DataFrame(y_val).to_csv(val_labels_output_path, header=False, index=False)

    # Saving model.
    model_path = os.path.join("/opt/ml/processing/model", "model.joblib")
    model_output_path = os.path.join("/opt/ml/processing/model", "model.tar.gz")

    print("Saving featurizer model to {}".format(model_output_path))
    joblib.dump(featurizer_model, model_path)
    tar = tarfile.open(model_output_path, "w:gz")
    tar.add(model_path, arcname="model.joblib")
    tar.close()

    tracker.close()
