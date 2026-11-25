DataPreparer Schema Reference
==============================

.. note::
   This page reflects the Hub-published ``JobConfigSchema`` at SDK build time.
   For live inspection of the current schema, use:

   .. code-block:: python

      DatasetJobConfig.print_schema(JobCategory.SYNTHETIC_DATA_GENERATION)
      DatasetJobConfig.print_schema(JobCategory.DATA_QUALITY_EVALUATION)
      DatasetJobConfig.print_schema(JobCategory.DATA_TRANSFORMATION)

Overview
--------

``DataPreparer`` validates user payloads against a dynamically-fetched JSON schema
from the AI Registry Hub. The schema defines all available fields for each job category.

Usage
-----

.. code-block:: python

   from sagemaker.ai_registry import DataPreparer, DatasetJobConfig, JobCategory

   # Discover fields
   schema = DatasetJobConfig.schema(JobCategory.SYNTHETIC_DATA_GENERATION)
   DatasetJobConfig.print_schema(JobCategory.SYNTHETIC_DATA_GENERATION)

   # Minimal SDG call
   result = DataPreparer.generate(
       dataset=DatasetJobConfig(
           dataset_name="my-dataset",
           generation_config={
               "DatasetSpecification": {"SpecS3Uri": "s3://bucket/spec.txt"},
               "NumSamples": 500,
           },
       ),
       logs=True,
   )

   # Minimal DQ call: the default metrics for the dataset's format
   result = DataPreparer.evaluate(
       dataset=DatasetJobConfig(
           input_data_config=[{"DatasetArn": "arn:aws:sagemaker:..."}],
           evaluation_config={"Operations": [{"Operator": "default_metrics"}]},
           output_data_config={"S3OutputPath": "s3://bucket/dq-output/"},
       ),
       logs=True,
   )

   # Minimal DT call. Operators are passed through to the service as declared in
   # the schema, so ``Operator`` and ``Config`` are not validated field-by-field
   # by the SDK. Returns a list — Split emits one DataSet per output partition,
   # single-output operators a list of length 1.
   datasets = DataPreparer.transform(
       dataset=DatasetJobConfig(
           input_data_config=[{"DatasetArn": "arn:aws:sagemaker:..."}],
           transformation_config={
               "Operations": [
                   {
                       "Operator": "TextQualityFilter",
                       "Config": {
                           "FilterType": "CharacterLength",
                           "TextField": "prompt",
                           "Action": "drop",
                           "MinLength": 170,
                       },
                   }
               ],
           },
       ),
       logs=True,
   )

API Reference
-------------

.. autoclass:: sagemaker.ai_registry.data_preparer.preparer.DataPreparer
   :members: generate, evaluate, transform

.. autoclass:: sagemaker.ai_registry.data_preparer.configs.DatasetJobConfig
   :members: from_template, schema, print_schema, required_fields, to_job_config_document, dynamic_model_fields, dynamic_model_json_schema, resolved_schema_version, resolved_schema

.. autoclass:: sagemaker.ai_registry.data_preparer.constants.JobCategory
   :members:

.. autoclass:: sagemaker.ai_registry.data_preparer.constants.JobStatus
   :members:

Exceptions
----------

.. automodule:: sagemaker.ai_registry.data_preparer.exceptions
   :members:
