.. _sdp_api_docs_launch_training_job:

Launch a Distributed Training Job Using the SageMaker Python SDK
================================================================

To use the SageMaker distributed data parallel library with the SageMaker Python SDK,
you will need the following:

-  A TensorFlow or PyTorch training script that is
   adapted to use the distributed data parallel library. Make sure you read through
   the previous topic at
   :ref:`sdp_api_docs`, which includes instructions on how to modify your script and
   framework-specific examples.
-  Your input data must be in an S3 bucket or in FSx in the AWS region
   that you will use to launch your training job. If you use the Jupyter
   notebooks provided, create a SageMaker notebook instance in the same
   region as the bucket that contains your input data. For more
   information about storing your training data, refer to
   the `SageMaker Python SDK data
   inputs <https://sagemaker.readthedocs.io/en/stable/overview.html#use-file-systems-as-training-inputs>`__ documentation.

When you define
a :class:`sagemaker.tensorflow.estimator.TensorFlow` or :class:`sagemaker.pytorch.estimator.PyTorch` estimator,
you must select ``smdistributed`` and then ``dataparallel`` as your ``distribution`` strategy.

.. code:: python

   distribution = { "smdistributed": { "dataparallel": { "enabled": True } } }

.. seealso::

  To learn more, see `Step 2: Launch a SageMaker Distributed Training Job Using the SageMaker Python SDK
  <https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-use-api.html>`_
  in the *Amazon SageMaker Developer Guide*.

We recommend you use one of the example notebooks as your template to launch a training job. When
you use an example notebook you’ll need to swap your training script with the one that came with the
notebook and modify any input functions as necessary. For instructions on how to get started using a
Jupyter Notebook example, see `Distributed Training Jupyter Notebook Examples
<https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training-notebook-examples.html>`_.

Once you have launched a training job, you can monitor it using CloudWatch. To learn more, see
`Monitor and Analyze Training Jobs Using Metrics
<https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html>`_.

After you train a model, you can see how to deploy your trained model to an endpoint for inference by
following one of the `example notebooks for deploying a model
<https://sagemaker-examples.readthedocs.io/en/latest/inference/index.html>`_.
For more information, see `Deploy Models for Inference
<https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html>`_.
