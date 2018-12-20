
===================================
AWS SageMaker Estimators and Models
===================================

Amazon SageMaker provides several built-in machine learning algorithms that you can use for a variety of problem types.

The full list of algorithms is available on the AWS website: https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html

SageMaker Python SDK includes Estimator wrappers for the AWS K-means, Principal Components Analysis(PCA), Linear Learner, Factorization Machines, Latent Dirichlet Allocation(LDA), Neural Topic Model(NTM), Random Cut Forest algorithms, k-nearest neighbors (k-NN), Object2Vec, and IP Insights.

Definition and usage
~~~~~~~~~~~~~~~~~~~~
Estimators that wrap Amazon's built-in algorithms define algorithm's hyperparameters with defaults. When a default is not possible you need to provide the value during construction, e.g.:

- ``KMeans`` Estimator requires parameter ``k`` to define number of clusters
- ``PCA`` Estimator requires parameter ``num_components`` to define number of principal components

Interaction is identical as any other Estimators. There are additional details about how data is specified.

Input data format
^^^^^^^^^^^^^^^^^
Please note that Amazon's built-in algorithms are working best with protobuf ``recordIO`` format.
The data is expected to be available in S3 location and depending on algorithm it can handle dat in multiple data channels.

This package offers support to prepare data into required fomrat and upload data to S3.
Provided class ``RecordSet`` captures necessary details like S3 location, number of records, data channel and is expected as input parameter when calling ``fit()``.

Function ``record_set`` is available on algorithms objects to make it simple to achieve the above.
It takes 2D numpy array as input, uploads data to S3 and returns ``RecordSet`` objects. By default it uses ``train`` data channel and no labels but can be specified when called.

Please find an example code snippet for illustration:

.. code:: python

    from sagemaker import PCA
    pca_estimator = PCA(role='SageMakerRole', train_instance_count=1, train_instance_type='ml.m4.xlarge', num_components=3)

    import numpy as np
    records = pca_estimator.record_set(np.arange(10).reshape(2,5))

    pca_estimator.fit(records)


Predictions support
~~~~~~~~~~~~~~~~~~~
Calling inference on deployed Amazon's built-in algorithms requires specific input format. By default, this library creates a predictor that allows to use just numpy data.
Data is converted so that ``application/x-recordio-protobuf`` input format is used. Received response is deserialized from the protobuf and provided as result from the ``predict`` call.
