Amazon SageMaker Python SDK
===========================
Amazon SageMaker Python SDK is an open source library for training and deploying machine-learned models on Amazon SageMaker.

With the SDK, you can train and deploy models using popular deep learning frameworks: **Apache MXNet** and **TensorFlow**. You can also train and deploy models with **algorithms provided by Amazon**, these are scalable implementations of core machine learning algorithms that are optimized for SageMaker and GPU training. If you have **your own algorithms** built into SageMaker-compatible Docker containers, you can train and host models using these as well.

Here you'll find API docs for SageMaker Python SDK. The project home-page is in Github: https://github.com/aws/sagemaker-python-sdk, there you can find the SDK source, installation instructions and a general overview of the library there. 

Overview
----------
The SageMaker Python SDK consists of a few primary interfaces:

.. toctree::
    :maxdepth: 2

    estimators
    predictors
    session
    model

MXNet
----------
A managed environment for MXNet training and hosting on Amazon SageMaker

.. toctree::
    :maxdepth: 2

    sagemaker.mxnet

TensorFlow
----------
A managed environment for TensorFlow training and hosting on Amazon SageMaker

.. toctree::
    :maxdepth: 2

    sagemaker.tensorflow

SageMaker First-Party Algorithms
--------------------------------
Amazon provides implementations of some common machine learning algortithms optimized for GPU architecture and massive datasets.

.. toctree::
    :maxdepth: 2

    kmeans
    pca
    linear_learner
    sagemaker.amazon.amazon_estimator
    factorization_machines
    lda
    ntm
