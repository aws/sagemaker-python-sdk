# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import argparse
import numpy as np
import os
import json
import tensorflow as tf

tf_major, tf_minor, _ = tf.__version__.split(".")
if int(tf_major) > 2 or (int(tf_major) == 2 and int(tf_minor) >= 6):
    import tensorflow_io as tfio
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Flatten

"""
This script uses custom loops to train Mnist model and saves the checkpoints using 
checkpoint manager.
"""


# define a model
class LeNet(tf.keras.Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = Conv2D(
            filters=16,
            kernel_size=3,
            padding="valid",
            strides=(2, 2),
            input_shape=(None, 28, 28, 1),
            data_format="channels_last",
            trainable=True,
        )

        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(
            filters=16,
            kernel_size=3,
            strides=(2, 2),
            data_format="channels_last",
            padding="valid",
            trainable=True,
        )
        self.bn2 = BatchNormalization()
        self.flatten = Flatten()
        self.fc = Dense(10, trainable=True)

    def call(self, x):
        x = tf.reshape(x, (-1, 28, 28, 1))
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


@tf.function
def train_step(x, y, net, optimizer, loss_summary, accuracy_summary):
    """
    x: input
    y: true label
    net: model object
    optim: optimizer
    loss_summary: summary writer for loss
    acc_summary: summary writer for accuracy
    """
    with tf.GradientTape() as tape:
        z = net(x)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=y, y_pred=z, from_logits=True, axis=-1
        )
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))

    # instrument loss
    loss_summary(loss)

    # instrument accuracy
    accuracy_summary(y, z)
    return


@tf.function
def eval_step(x, y, net, loss_summary, accuracy_summary):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    z = net(x)

    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true=y, y_pred=z, from_logits=True, axis=-1
    )
    loss = tf.reduce_mean(loss)

    loss_summary(loss)
    accuracy_summary(y, z)
    return


def load_data(data_dir):
    """Load training and eval dataset"""
    x, y = np.load(os.path.join(data_dir, "train_data.npy")), np.load(
        os.path.join(data_dir, "train_labels.npy")
    )

    vx, vy = np.load(os.path.join(data_dir, "eval_data.npy")), np.load(
        os.path.join(data_dir, "eval_labels.npy")
    )

    print("==== train tensor shape ====")
    print(x.shape, y.shape)

    print("==== eval tensor shape ====")
    print(vx.shape, vy.shape)
    # x.shape = (1000, 784), y.shape = (1000, )

    x, y = x.astype(np.float32), y.astype(np.int32)
    vx, vy = vx.astype(np.float32), vy.astype(np.int32)
    x /= 255.0
    vx /= 255.0

    dtrain = tf.data.Dataset.from_tensor_slices((x, y))
    dtrain = dtrain.map(lambda x, y: (tf.reshape(x, (28, 28, 1)), y))
    dtrain = dtrain.shuffle(10000).batch(512)

    deval = tf.data.Dataset.from_tensor_slices((vx, vy))
    deval = deval.map(lambda x, y: (tf.reshape(x, (28, 28, 1)), y))
    deval = deval.batch(10)
    return dtrain, deval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-checkpoint-steps", type=int, default=200)
    parser.add_argument("--throttle-secs", type=int, default=60)
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--export-model-during-training", type=bool, default=False)
    return parser.parse_args()


def main(args):
    if args.model_dir.startswith("s3://"):
        os.environ["S3_REGION"] = "us-west-2"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        os.environ["S3_USE_HTTPS"] = "1"

    net = LeNet()
    net.build(input_shape=(None, 28, 28, 1))

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=net)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, args.model_dir, max_to_keep=11, checkpoint_name="model.ckpt"
    )

    dtrain, deval = load_data(args.train)
    num_epochs = args.epochs
    for i in range(num_epochs):
        for x, y in dtrain:
            train_step(x, y, net, optimizer, train_loss, train_accuracy)

        for x, y in deval:
            eval_step(x, y, net, test_loss, test_accuracy)

        print(
            f"Epoch {i+1}",
            f"Train Loss: {train_loss.result()}",
            f"Train Accuracy: {train_accuracy.result()}",
            f"Test Loss: {test_loss.result()}",
            f"Test Accuracy: {test_accuracy.result()}",
        )

        if args.current_host == args.hosts[0]:
            ckpt_manager.save()
            if int(tf_major) > 2 or (int(tf_major) == 2 and int(tf_minor) >= 16):
                net.export("/opt/ml/model/1")
            else:
                net.save("/opt/ml/model/1")


if __name__ == "__main__":
    main(parse_args())
