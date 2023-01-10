from __future__ import absolute_import

import argparse
import joblib
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression


# inference functions ---------------
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


if __name__ == "__main__":

    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="candybar_train.csv")
    parser.add_argument("--test-file", type=str, default="candybar_test.csv")

    args, _ = parser.parse_known_args()

    print("reading data")
    X_train = pd.read_csv(os.path.join(args.train, args.train_file))
    y_train = pd.read_csv(os.path.join(args.test, args.test_file))

    # train
    print("training model")
    model = LogisticRegression()

    X_train = X_train.iloc[:, 1:]
    y_train = y_train.iloc[:, 1:]

    model.fit(X_train, y_train)

    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print("model persisted at " + path)
