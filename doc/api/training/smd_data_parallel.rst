###################################
Distributed data parallel
###################################

SageMaker's distributed data parallel library extends SageMaker’s training
capabilities on deep learning models with near-linear scaling efficiency,
achieving fast time-to-train with minimal code changes.

- optimizes your training job for AWS network infrastructure and EC2 instance topology.
- takes advantage of gradient update to communicate between nodes with a custom AllReduce algorithm.

When training a model on a large amount of data, machine learning practitioners
will often turn to distributed training to reduce the time to train.
In some cases, where time is of the essence,
the business requirement is to finish training as quickly as possible or at
least within a constrained time period.
Then, distributed training is scaled to use a cluster of multiple nodes,
meaning not just multiple GPUs in a computing instance, but multiple instances
with multiple GPUs. As the cluster size increases, so does the significant drop
in performance. This drop in performance is primarily caused the communications
overhead between nodes in a cluster.

.. important::
   The distributed data parallel library only supports training jobs using CUDA 11. When you define a PyTorch or TensorFlow
   ``Estimator`` with ``dataparallel`` parameter ``enabled`` set to ``True``,
   it uses CUDA 11. When you extend or customize your own training image
   you must use a CUDA 11 base image. See
   `SageMaker Python SDK's distributed data parallel library APIs
   <https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-use-api.html#data-parallel-use-python-skd-api>`__
   for more information.

.. rubric:: Customize your training script

To customize your own training script, you will need the following:

.. raw:: html

   <div data-section-style="5" style="">

-  You must provide TensorFlow / PyTorch training scripts that are
   adapted to use the distributed data parallel library.
-  Your input data must be in an S3 bucket or in FSx in the AWS region
   that you will use to launch your training job. If you use the Jupyter
   notebooks provided, create a SageMaker notebook instance in the same
   region as the bucket that contains your input data. For more
   information about storing your training data, refer to
   the `SageMaker Python SDK data
   inputs <https://sagemaker.readthedocs.io/en/stable/overview.html#use-file-systems-as-training-inputs>`__ documentation.

.. raw:: html

   </div>

Use the API guides for each framework to see
examples of training scripts that can be used to convert your training scripts.
Then use one of the example notebooks as your template to launch a training job.
You’ll need to swap your training script with the one that came with the
notebook and modify any input functions as necessary.
Once you have launched a training job, you can monitor it using CloudWatch.

Then you can see how to deploy your trained model to an endpoint by
following one of the example notebooks for deploying a model. Finally,
you can follow an example notebook to test inference on your deployed
model.



.. toctree::
   :maxdepth: 2

   smd_data_parallel_pytorch
   smd_data_parallel_tensorflow
