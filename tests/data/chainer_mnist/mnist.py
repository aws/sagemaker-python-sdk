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
from __future__ import print_function

import argparse
import os

import chainer
from chainer import serializers, training
from chainer.datasets import tuple_dataset
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import numpy as np

import sagemaker_containers


class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def _preprocess_mnist(raw, withlabel, ndim, scale, image_dtype, label_dtype, rgb_format):
    images = raw["x"][-100:]
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
        labels = raw["y"][-100:].astype(label_dtype)
        return tuple_dataset.TupleDataset(images, labels)
    else:
        return images


if __name__ == "__main__":
    env = sagemaker_containers.training_env()

    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument("--units", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--frequency", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--model-dir", type=str, default=env.model_dir)

    parser.add_argument("--train", type=str, default=env.channel_input_dirs["train"])
    parser.add_argument("--test", type=str, default=env.channel_input_dirs["test"])

    parser.add_argument("--num-gpus", type=int, default=env.num_gpus)

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

    train = _preprocess_mnist(train_file, **preprocess_mnist_options)
    test = _preprocess_mnist(test_file, **preprocess_mnist_options)

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP(args.units, 10))

    if chainer.cuda.available:
        chainer.cuda.get_device_from_id(0).use()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=args.alpha)
    optimizer.setup(model)

    # Load the MNIST dataset
    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size, repeat=False, shuffle=False)

    # Set up a trainer
    device = 0 if chainer.cuda.available else -1  # -1 indicates CPU, 0 indicates first GPU device.
    if chainer.cuda.available:

        def device_name(device_intra_rank):
            return "main" if device_intra_rank == 0 else str(device_intra_rank)

        devices = {device_name(device): device for device in range(args.num_gpus)}
        updater = training.updater.ParallelUpdater(
            train_iter,
            optimizer,
            # The device of the name 'main' is used as a "master", while others are
            # used as slaves. Names other than 'main' are arbitrary.
            devices=devices,
        )
    else:
        updater = training.updater.StandardUpdater(train_iter, optimizer, device=device)

    # Write output files to output_data_dir.
    # These are zipped and uploaded to S3 output path as output.tar.gz.
    trainer = training.Trainer(updater, (args.epochs, "epoch"), out=env.output_data_dir)

    # Evaluate the model with the test dataset for each epoch

    trainer.extend(extensions.Evaluator(test_iter, model, device=device))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph("main/loss"))

    # Take a snapshot for each specified epoch
    trainer.extend(extensions.snapshot(), trigger=(args.frequency, "epoch"))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ["main/loss", "validation/main/loss"], "epoch", file_name="loss.png"
            )
        )
        trainer.extend(
            extensions.PlotReport(
                ["main/accuracy", "validation/main/accuracy"], "epoch", file_name="accuracy.png"
            )
        )

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(
        extensions.PrintReport(
            [
                "epoch",
                "main/loss",
                "validation/main/loss",
                "main/accuracy",
                "validation/main/accuracy",
                "elapsed_time",
            ]
        )
    )

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Run the training
    trainer.run()

    serializers.save_npz(os.path.join(args.model_dir, "model.npz"), model)


def model_fn(model_dir):
    model = L.Classifier(MLP(1000, 10))
    serializers.load_npz(os.path.join(model_dir, "model.npz"), model)
    return model.predictor
