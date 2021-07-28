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

import numpy as np
import os
import tensorflow as tf


def estimator_fn(run_config, hyperparameters):
    input_tensor_name = hyperparameters.get("input_tensor_name", "inputs")
    learning_rate = hyperparameters.get("learning_rate", 0.05)
    feature_columns = [tf.feature_column.numeric_column(input_tensor_name, shape=[4])]
    return tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        optimizer=tf.train.AdagradOptimizer(learning_rate=learning_rate),
        n_classes=3,
        config=run_config,
    )


def serving_input_fn(hyperparameters):
    input_tensor_name = hyperparameters["input_tensor_name"]
    feature_spec = {input_tensor_name: tf.FixedLenFeature(dtype=tf.float32, shape=[4])}
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()


def train_input_fn(training_dir, hyperparameters):
    """Returns input function that would feed the model during training"""
    return _generate_input_fn(training_dir, "iris_training.csv", hyperparameters)


def eval_input_fn(training_dir, hyperparameters):
    """Returns input function that would feed the model during evaluation"""
    return _generate_input_fn(training_dir, "iris_test.csv", hyperparameters)


def _generate_input_fn(training_dir, training_filename, hyperparameters):
    input_tensor_name = hyperparameters["input_tensor_name"]

    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=os.path.join(training_dir, training_filename),
        target_dtype=np.int,
        features_dtype=np.float32,
    )

    return tf.estimator.inputs.numpy_input_fn(
        x={input_tensor_name: np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True,
    )()
