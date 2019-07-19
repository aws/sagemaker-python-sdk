# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.training.rmsprop import RMSPropOptimizer

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10
NUM_DATA_BATCHES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000 * NUM_DATA_BATCHES
BATCH_SIZE = 128
INPUT_TENSOR_NAME = "inputs_input"  # needs to match the name of the first layer + "_input"


def keras_model_fn(hyperparameters):
    """keras_model_fn receives hyperparameters from the training job and returns a compiled keras model.
    The model will be transformed into a TensorFlow Estimator before training and it will be saved in a
    TensorFlow Serving SavedModel at the end of training.
    Args:
        hyperparameters: The hyperparameters passed to the SageMaker TrainingJob that runs your TensorFlow
                         training script.
    Returns: A compiled Keras model
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="same", name="inputs", input_shape=(HEIGHT, WIDTH, DEPTH)))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation("softmax"))

    opt = RMSPropOptimizer(
        learning_rate=hyperparameters["learning_rate"], decay=hyperparameters["decay"]
    )

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model


def serving_input_fn(hyperpameters):
    inputs = {INPUT_TENSOR_NAME: tf.placeholder(tf.float32, [None, 32, 32, 3])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def train_input_fn(training_dir, hyperparameters):
    return _generate_synthetic_data(tf.estimator.ModeKeys.TRAIN, batch_size=BATCH_SIZE)


def eval_input_fn(training_dir, hyperparameters):
    return _generate_synthetic_data(tf.estimator.ModeKeys.EVAL, batch_size=BATCH_SIZE)


def _generate_synthetic_data(mode, batch_size):
    input_shape = [batch_size, HEIGHT, WIDTH, DEPTH]
    images = tf.truncated_normal(
        input_shape, dtype=tf.float32, stddev=1e-1, name="synthetic_images"
    )
    labels = tf.random_uniform(
        [batch_size, NUM_CLASSES],
        minval=0,
        maxval=NUM_CLASSES - 1,
        dtype=tf.float32,
        name="synthetic_labels",
    )

    images = tf.contrib.framework.local_variable(images, name="images")
    labels = tf.contrib.framework.local_variable(labels, name="labels")

    return {INPUT_TENSOR_NAME: images}, labels
