# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

warnings.filterwarnings(action="ignore", category=DataConversionWarning)


columns = [
    "age",
    "education",
    "major industry code",
    "class of worker",
    "num persons worked for employer",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "income",
]
class_labels = [" - 50000.", " 50000+."]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    args, _ = parser.parse_known_args()

    input_data_path = os.path.join("/opt/ml/processing/input", "census-income.csv")

    df = pd.read_csv(input_data_path)
    df = pd.DataFrame(data=df, columns=columns)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.replace(class_labels, [0, 1], inplace=True)

    negative_examples, positive_examples = np.bincount(df["income"])

    split_ratio = args.train_test_split_ratio
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("income", axis=1), df["income"], test_size=split_ratio, random_state=0
    )

    preprocess = make_column_transformer(
        (
            ["age", "num persons worked for employer"],
            KBinsDiscretizer(encode="onehot-dense", n_bins=10),
        ),
        (["capital gains", "capital losses", "dividends from stocks"], StandardScaler()),
        (["education", "major industry code", "class of worker"], OneHotEncoder(sparse=False)),
    )
    train_features = preprocess.fit_transform(X_train)
    test_features = preprocess.transform(X_test)

    train_features_output_path = os.path.join("/opt/ml/processing/train", "train_features.csv")
    train_labels_output_path = os.path.join("/opt/ml/processing/train", "train_labels.csv")

    test_features_output_path = os.path.join("/opt/ml/processing/test", "test_features.csv")
    test_labels_output_path = os.path.join("/opt/ml/processing/test", "test_labels.csv")

    pd.DataFrame(train_features).to_csv(train_features_output_path, header=False, index=False)

    pd.DataFrame(test_features).to_csv(test_features_output_path, header=False, index=False)

    y_train.to_csv(train_labels_output_path, header=False, index=False)

    y_test.to_csv(test_labels_output_path, header=False, index=False)
