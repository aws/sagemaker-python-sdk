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
import logging
import os

import chainer
from chainer import serializers, training
from chainer.datasets import tuple_dataset
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import chainermn
import numpy as np
import sagemaker_containers

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    return images


if __name__ == "__main__":
    env = sagemaker_containers.training_env()

    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--communicator", type=str, default="pure_nccl")
    parser.add_argument("--frequency", type=int, default=20)
    parser.add_argument("--units", type=int, default=1000)

    parser.add_argument("--model-dir", type=str)
    parser.add_argument("--output-data-dir", type=str, default=env.output_data_dir)
    parser.add_argument("--host", type=str, default=env.current_host)
    parser.add_argument("--num-gpus", type=int, default=env.num_gpus)

    parser.add_argument("--train", type=str, default=env.channel_input_dirs["train"])
    parser.add_argument("--test", type=str, default=env.channel_input_dirs["test"])

    args = parser.parse_args()

    train_file = np.load(os.path.join(args.train, "train.npz"))
    test_file = np.load(os.path.join(args.test, "test.npz"))

    logger.info("Current host: {}".format(args.host))

    communicator = "naive" if args.num_gpus == 0 else args.communicator

    comm = chainermn.create_communicator(communicator)
    device = comm.intra_rank if args.num_gpus > 0 else -1

    print("==========================================")
    print("Using {} communicator".format(comm))
    print("Num unit: {}".format(args.units))
    print("Num Minibatch-size: {}".format(args.batch_size))
    print("Num epoch: {}".format(args.epochs))
    print("==========================================")

    model = L.Classifier(MLP(args.units, 10))
    if device >= 0:
        chainer.cuda.get_device(device).use()

    # Create a multi node optimizer from a standard Chainer optimizer.
    optimizer = chainermn.create_multi_node_optimizer(chainer.optimizers.Adam(), comm)
    optimizer.setup(model)

    preprocess_mnist_options = {
        "withlabel": True,
        "ndim": 1,
        "scale": 1.0,
        "image_dtype": np.float32,
        "label_dtype": np.int32,
        "rgb_format": False,
    }

    train_dataset = _preprocess_mnist(train_file, **preprocess_mnist_options)
    test_dataset = _preprocess_mnist(test_file, **preprocess_mnist_options)

    train_iter = chainer.iterators.SerialIterator(train_dataset, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(
        test_dataset, args.batch_size, repeat=False, shuffle=False
    )

    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epochs, "epoch"), out=args.output_data_dir)

    # Create a multi node evaluator from a standard Chainer evaluator.
    evaluator = extensions.Evaluator(test_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator)

    # Some display and output extensions are necessary only for one worker.
    # (Otherwise, there would just be repeated outputs.)
    if comm.rank == 0:
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
        trainer.extend(extensions.snapshot(), trigger=(args.frequency, "epoch"))
        trainer.extend(extensions.dump_graph("main/loss"))
        trainer.extend(extensions.LogReport())
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
        trainer.extend(extensions.ProgressBar())

    trainer.run()

    # only save the model in the master node
    if args.host == env.hosts[0]:
        serializers.save_npz(os.path.join(env.model_dir, "model.npz"), model)


def model_fn(model_dir):
    model = L.Classifier(MLP(1000, 10))
    serializers.load_npz(os.path.join(model_dir, "model.npz"), model)
    return model.predictor
