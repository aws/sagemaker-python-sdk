import logging
import time

import mxnet as mx
import numpy as np
from mxnet import gluon, autograd
from mxnet.gluon import nn

logger = logging.getLogger(__name__)


def train(channel_input_dirs, hyperparameters, **kwargs):
    # SageMaker passes num_cpus, num_gpus and other args we can use to tailor training to
    # the current container environment, but here we just use simple cpu context.
    ctx = mx.cpu()

    # retrieve the hyperparameters we set in notebook (with some defaults)
    batch_size = hyperparameters.get("batch_size", 100)
    epochs = hyperparameters.get("epochs", 10)
    learning_rate = hyperparameters.get("learning_rate", 0.1)
    momentum = hyperparameters.get("momentum", 0.9)
    log_interval = hyperparameters.get("log_interval", 100)

    training_data = channel_input_dirs["training"]

    # load training and validation data
    # we use the gluon.data.vision.MNIST class because of its built in mnist pre-processing logic,
    # but point it at the location where SageMaker placed the data files, so it doesn't download them again.
    train_data = get_train_data(training_data, batch_size)
    val_data = get_val_data(training_data, batch_size)

    # define the network
    net = define_network()

    # Collect all parameters from net and its children, then initialize them.
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    # Trainer is for updating parameters with gradient.
    trainer = gluon.Trainer(
        net.collect_params(), "sgd", {"learning_rate": learning_rate, "momentum": momentum}
    )
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(epochs):
        # reset data iterator and metric at begining of epoch.
        metric.reset()
        btic = time.time()
        for i, (data, label) in enumerate(train_data):
            # Copy data to ctx if necessary
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # Start recording computation graph with record() section.
            # Recorded graphs can then be differentiated with backward.
            with autograd.record():
                output = net(data)
                L = loss(output, label)
                L.backward()
            # take a gradient step with batch_size equal to data.shape[0]
            trainer.step(data.shape[0])
            # update metric at last.
            metric.update([label], [output])

            if i % log_interval == 0 and i > 0:
                name, acc = metric.get()
                logger.info(
                    "[Epoch %d Batch %d] Training: %s=%f, %f samples/s"
                    % (epoch, i, name, acc, batch_size / (time.time() - btic))
                )

            btic = time.time()

        name, acc = metric.get()
        logger.info("[Epoch %d] Training: %s=%f" % (epoch, name, acc))

        name, val_acc = test(ctx, net, val_data)
        logger.info("[Epoch %d] Validation: %s=%f" % (epoch, name, val_acc))

    return net


def save(net, model_dir):
    # save the model
    y = net(mx.sym.var("data"))
    y.save("%s/model.json" % model_dir)
    net.collect_params().save("%s/model.params" % model_dir)


def define_network():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(128, activation="relu"))
        net.add(nn.Dense(64, activation="relu"))
        net.add(nn.Dense(10))
    return net


def input_transformer(data, label):
    data = data.reshape((-1,)).astype(np.float32) / 255
    return data, label


def get_train_data(data_dir, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.MNIST(data_dir, train=True, transform=input_transformer),
        batch_size=batch_size,
        shuffle=True,
        last_batch="discard",
    )


def get_val_data(data_dir, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.MNIST(data_dir, train=False, transform=input_transformer),
        batch_size=batch_size,
        shuffle=False,
    )


def test(ctx, net, val_data):
    metric = mx.metric.Accuracy()
    for data, label in val_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric.update([label], [output])
    return metric.get()
