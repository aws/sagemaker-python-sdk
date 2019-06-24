from mxnet import gluon


def download_training_data():
    gluon.data.vision.MNIST("./data/training", train=True)
    gluon.data.vision.MNIST("./data/training", train=False)


if __name__ == "__main__":
    download_training_data()
