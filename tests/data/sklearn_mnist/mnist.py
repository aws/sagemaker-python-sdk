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
from __future__ import print_function, absolute_import

import argparse
import numpy as np
import os

import joblib
from sklearn import svm


def preprocess_mnist(raw, withlabel, ndim, scale, image_dtype, label_dtype, rgb_format):
    images = raw["x"]
    if ndim == 2:
        images = images.reshape(-1, 28, 28)
    elif ndim == 3:
        images = images.reshape(-1, 1, 28, 28)
        if rgb_format:
            images = np.broadcast_to(images, (len(images), 3) + images.shape[2:])

    elif ndim != 1:
        raise ValueError("invalid ndim for MNIST dataset")
    images = images.astype(image_dtype)
    images *= scale / 255.0

    if withlabel:
        labels = raw["y"].astype(label_dtype)
        return images, labels
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args = parser.parse_args()

    train_file = np.load(os.path.join(args.train, "train.npz"))
    test_file = np.load(os.path.join(args.test, "test.npz"))

    preprocess_mnist_options = {
        "withlabel": True,
        "ndim": 1,
        "scale": 1.0,
        "image_dtype": np.float32,
        "label_dtype": np.int32,
        "rgb_format": False,
    }

    # Preprocess MNIST data
    train_images, train_labels = preprocess_mnist(train_file, **preprocess_mnist_options)
    test_images, test_labels = preprocess_mnist(test_file, **preprocess_mnist_options)

    # Set up a Support Vector Machine classifier to predict digit from images
    clf = svm.SVC(gamma=0.001, C=100.0, max_iter=args.epochs)

    # Fit the SVM classifier with the images and the corresponding labels
    clf.fit(train_images, train_labels)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
