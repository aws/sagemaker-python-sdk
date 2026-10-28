API Reference
=============

DataPreparer
------------

.. autoclass:: sagemaker.ai_registry.data_preparer.preparer.DataPreparer
   :members: generate, evaluate, transform, stop
   :undoc-members:

DatasetJobConfig
----------------

.. autoclass:: sagemaker.ai_registry.data_preparer.configs.DatasetJobConfig
   :members: from_template, schema, print_schema, required_fields, to_job_config_document, dynamic_model_fields, dynamic_model_json_schema, resolved_schema_version, resolved_schema
   :undoc-members:

DataSet
-------

.. autoclass:: sagemaker.ai_registry.dataset.DataSet
   :members:
   :undoc-members:

JobCategory
-----------

.. autoclass:: sagemaker.ai_registry.data_preparer.constants.JobCategory
   :members:
   :undoc-members:

JobStatus
---------

.. autoclass:: sagemaker.ai_registry.data_preparer.constants.JobStatus
   :members:
   :undoc-members:

Auto-labeling options
---------------------

Typed per-task options for ``DataPreparer.generate``, passed as
``DatasetJobConfig(dataset_options=...)``.

.. automodule:: sagemaker.ai_registry.data_preparer.auto_labeling.shapes
   :members:
   :undoc-members:

LabelingSummary
---------------

.. autoclass:: sagemaker.ai_registry.data_preparer.auto_labeling.summary.LabelingSummary
   :members: from_dataset
   :undoc-members:

Exceptions
----------

.. automodule:: sagemaker.ai_registry.data_preparer.exceptions
   :members:
   :undoc-members:
