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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

import resnet_model
import tensorflow as tf

INPUT_TENSOR_NAME = "inputs"
SIGNATURE_NAME = "serving_default"

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10
NUM_DATA_BATCHES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000 * NUM_DATA_BATCHES
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
RESNET_SIZE = 32
BATCH_SIZE = 1

# Scale the learning rate linearly with the batch size. When the batch size is
# 128, the learning rate should be 0.05.
_INITIAL_LEARNING_RATE = 0.05 * BATCH_SIZE / 128
_MOMENTUM = 0.9

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4

_BATCHES_PER_EPOCH = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE


def model_fn(features, labels, mode, params):
    """Model function for CIFAR-10."""

    inputs = features[INPUT_TENSOR_NAME]
    tf.summary.image("images", inputs, max_outputs=6)

    network = resnet_model.cifar10_resnet_v2_generator(RESNET_SIZE, NUM_CLASSES)

    inputs = tf.reshape(inputs, [-1, HEIGHT, WIDTH, DEPTH])

    logits = network(inputs, mode == tf.estimator.ModeKeys.TRAIN)

    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {SIGNATURE_NAME: tf.estimator.export.PredictOutput(predictions)}
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, export_outputs=export_outputs
        )

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name="cross_entropy")
    tf.summary.scalar("cross_entropy", cross_entropy)

    # Add weight decay to the loss.
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        boundaries = [int(_BATCHES_PER_EPOCH * epoch) for epoch in [100, 150, 200]]
        values = [_INITIAL_LEARNING_RATE * decay for decay in [1, 0.1, 0.01, 0.001]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values
        )

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name="learning_rate")
        tf.summary.scalar("learning_rate", learning_rate)

        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=_MOMENTUM)

        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions["classes"])
    metrics = {"accuracy": accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name="train_accuracy")
    tf.summary.scalar("train_accuracy", accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=metrics
    )


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
        dtype=tf.int32,
        name="synthetic_labels",
    )

    images = tf.contrib.framework.local_variable(images, name="images")
    labels = tf.contrib.framework.local_variable(labels, name="labels")

    return {INPUT_TENSOR_NAME: images}, labels


def input_fn(serialized_data, content_type):
    return pickle.loads(serialized_data)
