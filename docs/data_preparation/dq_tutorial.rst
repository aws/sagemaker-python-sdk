Data Quality Evaluation Tutorial
=================================

Evaluate dataset quality using LLM-as-Judge metrics via the DataPreparer API.

Prerequisites
-------------

- Install the SageMaker Python SDK: ``pip install --upgrade sagemaker``
- ``~/.config/sagemaker/config.yaml`` with RoleArn and DefaultS3Bucket (see :doc:`configuration`)
- DataPreparer is available in **us-east-1** and **us-west-2** only
- A dataset: either from a prior SDG run (``dataset.arn``) or your own JSONL file

Step 1: Setup
-------------

.. code-block:: python

   from sagemaker.core.helper.session_helper import Session
   from sagemaker.ai_registry import DataPreparer, DatasetJobConfig, JobCategory

   session = Session()
   preparer = DataPreparer(sagemaker_session=session)

Step 2: Discover Available Fields
---------------------------------

.. code-block:: python

   DatasetJobConfig.print_schema(JobCategory.DATA_QUALITY_EVALUATION)

Step 3: Import Your Dataset (BYOD)
-----------------------------------

If you already have a ``DatasetArn`` from a prior SDG run, skip to Step 4.

For bring-your-own-data (BYOD), upload to S3 and import into the Hub:

.. code-block:: python

   import json
   import boto3
   from sagemaker.ai_registry.air_hub import AIRHub

   hub_name = AIRHub.get_hub_name()
   sm_client = session.sagemaker_client

   # Upload your JSONL to S3
   # aws s3 cp my_dataset.jsonl s3://my-bucket/byod/my_dataset.jsonl

   import_response = sm_client.import_hub_content(
       HubName=hub_name,
       HubContentName="my-dataset",
       HubContentType="DataSet",
       HubContentVersion="0.0.1",
       DocumentSchemaVersion="4.0.0",
       HubContentDocument=json.dumps({
           "DatasetFileFormat": "jsonl",
           "DatasetS3Bucket": "my-bucket",
           "DatasetS3Prefix": "byod/my_dataset.jsonl",
       }),
   )
   dataset_arn = import_response["HubContentArn"]
   print(f"DatasetArn: {dataset_arn}")

Step 4: Configure and Run Evaluation
-------------------------------------

.. code-block:: python

   eval_config = DatasetJobConfig(
       input_data_config=[{"DatasetArn": dataset_arn}],
       evaluation_name="my-dataset-eval",
       output_data_config={"S3OutputPath": "s3://my-bucket/dq-output/"},
   )

   result = preparer.evaluate(eval_config)
   print("Evaluated DataSet ARN:", result.arn)
   print("Job ARN:", result.job_arn)

``evaluate`` returns the evaluated (input) ``DataSet``; the scorecard and per-record scores
are written under ``S3OutputPath``. If you omit ``output_data_config``, the SDK fills
``S3OutputPath`` from your configured defaults (see :doc:`configuration`).

.. note::

   Evaluation typically takes 5–15 minutes depending on dataset size. During
   execution, the SDK displays a live status panel showing elapsed time and
   CloudWatch log lines.

The SDK automatically previews 10 scored records after completion.

Step 5: Interpret Results
-------------------------

Each job writes its outputs to its own folder under ``S3OutputPath``, named
``<job-name>-<timestamp>/``:

- **Aggregate report**: ``aggregate/quality_report.json`` — summary scores
- **Per-record scores**: ``record_level/evaluated_data.jsonl`` — each record with metrics

.. code-block:: python

   import json, boto3
   from urllib.parse import urlparse

   output = urlparse("s3://my-bucket/dq-output/")
   job_name = result.job_arn.split("/")[-1]
   s3 = boto3.client("s3")

   # Locate this job's output folder, then read the aggregate report
   listing = s3.list_objects_v2(
       Bucket=output.netloc, Prefix=output.path.lstrip("/") + job_name + "-"
   )
   report_key = next(
       obj["Key"] for obj in listing["Contents"]
       if obj["Key"].endswith("aggregate/quality_report.json")
   )
   report = json.loads(s3.get_object(Bucket=output.netloc, Key=report_key)["Body"].read())
   print(json.dumps(report, indent=2))

Metrics
~~~~~~~

LLM-as-Judge metrics (scored 0.0–1.0 with explanations):

.. list-table::
   :header-rows: 1

   * - Metric
     - What it measures
   * - ``dq_relevance``
     - Does the response address the prompt?
   * - ``dq_accuracy``
     - Is the response factually correct?
   * - ``dq_completeness``
     - Does it fully answer the question?
   * - ``dq_readability``
     - Is it clear and well-written?
   * - ``dq_harmlessness``
     - Is it free of harmful content?
   * - ``dq_reasoning_complexity``
     - How much reasoning is demonstrated?
   * - ``dq_linguistic_complexity``
     - How sophisticated is the language?
   * - ``dq_task_complexity``
     - How complex is the task itself?
   * - ``dq_knowledge_depth``
     - What level of knowledge is required?

Deterministic metrics (computed without LLM):

- ``dq_word_length`` — word count
- ``dq_character_length`` — character count
- ``dq_content_structures`` — structural analysis (prose, code, lists)

Next Steps
----------

- Filter low-quality records (score < threshold) before fine-tuning
- :doc:`dt_tutorial` — Apply operators to clean, deduplicate, split or reshape the dataset
- Re-run with different metric configurations
- Chain with SDG: generate → evaluate → iterate on spec → regenerate

Troubleshooting
---------------

**DataPreparerJobFailedError with "HUB_FINALIZATION_FAILED"**
  Verify your RoleArn has ``sagemaker:ImportHubContent`` permissions (see
  :doc:`configuration`) and that the output S3 bucket is in the same region as
  the endpoint.

**Format validation failures**
  Ensure your dataset is valid JSONL (one JSON object per line). Each line must
  parse independently. Use ``DocumentSchemaVersion="4.0.0"`` when importing.

**Region mismatch errors**
  All S3 URIs must be in the same region as the SageMaker endpoint (us-east-1 or
  us-west-2 during preview).
