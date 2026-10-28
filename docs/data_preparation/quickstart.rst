Quickstart
==========

Prerequisites
-------------

Install the SageMaker Python SDK:

.. code-block:: bash

   pip install --upgrade sagemaker

Before running the examples below, ensure your ``~/.config/sagemaker/config.yaml`` includes at minimum:

.. code-block:: yaml

   SageMaker:
     PythonSDK:
       Modules:
         DataPreparer:
           GlobalDefaults:
             RoleArn: arn:aws:iam::123456789012:role/SageMakerExecutionRole
         Session:
           DefaultS3Bucket: my-sagemaker-bucket

DataPreparer is available in **us-east-1** and **us-west-2** only. All S3 buckets
must reside in the same region as the SageMaker endpoint.

See :doc:`configuration` for the full config reference.

Synthetic Data Generation
--------------------------

.. code-block:: python

   from sagemaker.ai_registry import DataPreparer, DatasetJobConfig, JobCategory
   from sagemaker.core.helper.session_helper import Session

   session = Session()
   preparer = DataPreparer(sagemaker_session=session)

   config = DatasetJobConfig(
       dataset_name="my-dataset",
       generation_config={
           "DatasetSpecification": {
               "SpecS3Uri": "s3://my-bucket/specs/task_description.txt",
           },
           "NumSamples": 100,
       },
   )
   dataset = preparer.generate(config)
   print(dataset.arn, dataset.source)

``DatasetSpecification.SpecS3Uri`` points to a plain-text file describing the data you want generated.
The job blocks until completion by default (``wait=True``).

Data Quality Evaluation
------------------------

.. code-block:: python

   from sagemaker.ai_registry import DataPreparer, DatasetJobConfig, JobCategory
   from sagemaker.core.helper.session_helper import Session

   session = Session()
   preparer = DataPreparer(sagemaker_session=session)

   config = DatasetJobConfig(
       input_data_config=[{
           "DatasetArn": "arn:aws:sagemaker:us-west-2:123456789012:hub-content/MyHub/DataSet/my-dataset/1.0.0",
       }],
       evaluation_name="quality-check-v1",
       output_data_config={"S3OutputPath": "s3://my-bucket/dq-output/"},
   )
   dataset = preparer.evaluate(config)
   print(dataset.arn, dataset.job_arn)

``evaluate`` returns the evaluated ``DataSet``; the scorecard and per-record scores are written
under ``S3OutputPath`` (see :doc:`dq_tutorial`).

Data Transformation
--------------------

.. code-block:: python

   from sagemaker.ai_registry import DataPreparer, DatasetJobConfig, JobCategory
   from sagemaker.core.helper.session_helper import Session

   session = Session()
   preparer = DataPreparer(sagemaker_session=session)

   config = DatasetJobConfig(
       input_data_config=[{
           "DatasetArn": "arn:aws:sagemaker:us-west-2:123456789012:hub-content/MyHub/DataSet/my-dataset/1.0.0",
       }],
       transformation_config={
           "Operations": [{
               "Operator": "TextQualityFilter",
               "Config": {
                   "FilterType": "CharacterLength",
                   "TextField": "prompt",
                   "Action": "drop",
                   "MinLength": 170,
               },
           }],
       },
   )
   datasets = preparer.transform(config)
   for ds in datasets:
       print(ds.arn, ds.source)

``transform`` returns a list, not a single ``DataSet``: operators such as ``Split`` emit one
output per partition, while single-output operators return a list of length 1. The outputs are
published as new versions of hub content, in the same hub as the input.

Operator names and their ``Config`` keys are passed through to the service as declared in the
Hub schema, so the SDK does not validate them field-by-field — an unknown operator or a
misspelled config key is rejected by the service, not at submission time.

Discovering Available Fields
-----------------------------

Use ``print_schema`` to see all accepted fields for a job category:

.. code-block:: python

   from sagemaker.ai_registry import DatasetJobConfig, JobCategory

   DatasetJobConfig.print_schema(JobCategory.SYNTHETIC_DATA_GENERATION)
   DatasetJobConfig.print_schema(JobCategory.DATA_QUALITY_EVALUATION)
   DatasetJobConfig.print_schema(JobCategory.DATA_TRANSFORMATION)

Use ``from_template`` to get a pre-filled config with placeholder values:

.. code-block:: python

   config = DatasetJobConfig.from_template(JobCategory.SYNTHETIC_DATA_GENERATION)
   print(config)  # shows <SET_ME> for required fields
