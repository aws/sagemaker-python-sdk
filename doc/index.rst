###########################
Amazon SageMaker Python SDK
###########################
Amazon SageMaker Python SDK is an open source library for training and deploying machine-learned models on Amazon SageMaker.

With the SDK, you can train and deploy models using popular deep learning frameworks, algorithms provided by Amazon, or your own algorithms built into SageMaker-compatible Docker images.

Here you'll find an overview and API documentation for SageMaker Python SDK. The project homepage is in Github: https://github.com/aws/sagemaker-python-sdk, where you can find the SDK source and installation instructions for the library.

********
Overview
********

.. toctree::
    :maxdepth: 2

    overview

The SageMaker Python SDK consists of a variety classes:

Training:

.. toctree::
    :maxdepth: 1

    estimators
    algorithm
    tuner
    parameter
    automl
    processing
    analytics
    debugger

Inference:

.. toctree::
    :maxdepth: 1

    model
    multi_data_model
    predictors
    transformer
    pipeline
    model_monitor

Utility:

.. toctree::
    :maxdepth: 1

    session
    inputs
    network
    s3

*****
MXNet
*****
A managed environment for MXNet training and hosting on Amazon SageMaker

.. toctree::
    :maxdepth: 1

    using_mxnet

.. toctree::
    :maxdepth: 2

    sagemaker.mxnet

**********
TensorFlow
**********
A managed environment for TensorFlow training and hosting on Amazon SageMaker

.. toctree::
    :maxdepth: 1

    using_tf

.. toctree::
    :maxdepth: 2

    sagemaker.tensorflow

************
Scikit-Learn
************
A managed enrionment for Scikit-learn training and hosting on Amazon SageMaker

.. toctree::
    :maxdepth: 1

    using_sklearn

.. toctree::
    :maxdepth: 2

    sagemaker.sklearn

*******
PyTorch
*******
A managed environment for PyTorch training and hosting on Amazon SageMaker

.. toctree::
    :maxdepth: 1

    using_pytorch

.. toctree::
    :maxdepth: 2

    sagemaker.pytorch

*******
Chainer
*******
A managed environment for Chainer training and hosting on Amazon SageMaker

.. toctree::
    :maxdepth: 1

    using_chainer

.. toctree::
    :maxdepth: 2

    sagemaker.chainer

**********************
Reinforcement Learning
**********************
A managed environment for Reinforcement Learning training and hosting on Amazon SageMaker

.. toctree::
    :maxdepth: 1

    using_rl

.. toctree::
    :maxdepth: 2

    sagemaker.rl

***************
SparkML Serving
***************
A managed environment for SparkML hosting on Amazon SageMaker

.. toctree::
    :maxdepth: 2

    sagemaker.sparkml

********************************
SageMaker First-Party Algorithms
********************************
Amazon provides implementations of some common machine learning algortithms optimized for GPU architecture and massive datasets.

.. toctree::
    :maxdepth: 2

    sagemaker.amazon.amazon_estimator
    factorization_machines
    ipinsights
    kmeans
    knn
    lda
    linear_learner
    ntm
    object2vec
    pca
    randomcutforest

*****************************************
Amazon SageMaker Operators for Kubernetes
*****************************************
Amazon SageMaker Operators for use with Kubernetes.

.. toctree::
    :maxdepth: 2

    amazon_sagemaker_operators_for_kubernetes
    amazon_sagemaker_operators_for_kubernetes_jobs

*********
Workflows
*********
SageMaker APIs to export configurations for creating and managing Airflow workflows.

.. toctree::
    :maxdepth: 1

    using_workflow

.. toctree::
    :maxdepth: 2

    sagemaker.workflow.airflow

*********************************
Amazon SageMaker Model Monitoring
*********************************
You can use Amazon SageMaker Model Monitoring to automatically detect concept drift by monitoring your machine learning models.

.. toctree::
    :maxdepth: 2

    amazon_sagemaker_model_monitoring

*************************
Amazon SageMaker Debugger
*************************
You can use Amazon SageMaker Debugger to automatically detect anomalies while training your machine learning models.

.. toctree::
    :maxdepth: 2

    amazon_sagemaker_debugger

***************************
Amazon SageMaker Processing
***************************
You can use Amazon SageMaker Processing to perform data processing tasks such as data pre- and post-processing, feature engineering, data validation, and model evaluation

.. toctree::
    :maxdepth: 2

    amazon_sagemaker_processing
