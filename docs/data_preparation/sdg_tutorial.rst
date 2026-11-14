Synthetic Data Generation Tutorial
===================================

Generate a synthetic sentiment classification dataset using the DataPreparer API.

Prerequisites
-------------

- Install the SageMaker Python SDK: ``pip install --upgrade sagemaker``
- ``~/.config/sagemaker/config.yaml`` with RoleArn and DefaultS3Bucket (see :doc:`configuration`)
- DataPreparer is available in **us-east-1** and **us-west-2** only
- A ``task_description.txt`` spec file uploaded to S3 (see below)

Writing a Task Specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The spec file (``task_description.txt``) is a plain-text file that describes the
synthetic data you want generated. Requirements:

- **Filename**: Must be named ``task_description.txt``
- **Format**: Plain text (UTF-8)
- **Content**: A natural-language description of the dataset — include the task type
  (e.g. classification, QA, summarization), desired columns/fields, domain, and any
  constraints on the output.

Example ``task_description.txt``:

.. code-block:: text

   Generate a sentiment classification dataset for product reviews.
   Each record should have:
   - "text": a realistic product review (1-3 sentences)
   - "label": one of "positive", "negative", or "neutral"
   Cover electronics, clothing, and food categories evenly.

Step 1: Setup
-------------

.. code-block:: python

   from sagemaker.core.helper.session_helper import Session
   from sagemaker.ai_registry import DataPreparer, DatasetJobConfig, JobCategory

   session = Session()
   preparer = DataPreparer(sagemaker_session=session)

   # Verify config
   _cfg = session.sagemaker_config or {}
   _dp = _cfg.get("SageMaker", {}).get("PythonSDK", {}).get("Modules", {}).get("DataPreparer", {})
   role_arn = _dp.get("GlobalDefaults", {}).get("RoleArn") or _cfg.get("SageMaker", {}).get("GlobalDefaults", {}).get("RoleArn", "")
   print(f"Role: {role_arn}")
   print(f"Bucket: {session.default_bucket()}")

Step 2: Discover Available Fields
---------------------------------

.. code-block:: python

   DatasetJobConfig.print_schema(JobCategory.SYNTHETIC_DATA_GENERATION)

This prints the full schema tree showing every configurable field.

Step 3: Configure and Generate Samples
--------------------------------------

.. code-block:: python

   sample_config = DatasetJobConfig(
       dataset_name="sentiment-classification-sample",
       generation_config={
           "DatasetSpecification": {
               "SpecS3Uri": "s3://my-bucket/input/task_description.txt",
           },
           "NumSamples": 10,
       },
   )

   sample_dataset = preparer.generate(sample_config)
   print("DataSet ARN:", sample_dataset.arn)
   print("Source:", sample_dataset.source)

.. note::

   The first generation takes several minutes (typically 5–15 min). The service
   writes and optimizes *code* (generation.py + verifiers.py) that produces your
   data. It iterates internally until samples consistently pass quality checks.
   Subsequent runs reuse this optimized code and complete much faster.
   Full-scale generation (e.g. 500 samples) may take up to ~2 hours.

   During execution, the SDK displays a live status panel showing elapsed time,
   CloudWatch log lines, and sample-count progress (e.g. ``Samples: 3 / 10 (30%)``).

.. tip::

   **Contextual Data Generation (CDG)**: To generate data grounded in reference
   documents, add an ``InputDataS3Uri`` pointing to a folder of source files
   (supported formats: csv, doc, html, json, jsonl, md, pdf, txt; max 1000 files,
   10 MB each):

   .. code-block:: python

      generation_config={
          "DatasetSpecification": {
              "SpecS3Uri": "s3://my-bucket/input/task_description.txt",
              "InputDataS3Uri": "s3://my-bucket/input/reference_docs/",
          },
          "NumSamples": 10,
      }

   Without ``InputDataS3Uri``, the service performs Non-Contextual Data Generation
   (NCDG) using only the spec.

After completion, the SDK automatically previews 10 records from the output.

Step 4: Review and Refine (Optional)
------------------------------------

The service produces code artifacts during the sample run:

.. code-block:: python

   # source format is "{job_root}/output/" — extract the job root to find code artifacts
   code_artifacts_uri = sample_dataset.source.rstrip("/").removesuffix("/output") + "/generation_code/"
   print(f"Code artifacts at: {code_artifacts_uri}")
   # Contains: generation.py, verifiers.py

To iterate on quality:

1. Download and modify ``generation.py`` and/or ``verifiers.py``
2. Upload modified files back to the same S3 path
3. Re-run with the code artifacts:

.. code-block:: python

   refined_config = DatasetJobConfig(
       dataset_name="sentiment-classification-refined",
       generation_config={
           "DatasetSpecification": {
               "CodeArtifacts": {
                   "GenerationCodeS3Uri": code_artifacts_uri + "generation.py",
                   "VerifiersCodeS3Uri": code_artifacts_uri + "verifiers.py",
               },
           },
           "NumSamples": 10,
       },
   )
   refined_dataset = preparer.generate(refined_config)

.. warning::

   Once code artifacts are modified, they become the source of truth for
   generation logic — not the original spec.

Step 5: Full Generation
-----------------------

Once satisfied with sample quality, generate at scale:

.. code-block:: python

   full_config = DatasetJobConfig(
       dataset_name="sentiment-classification",
       generation_config={
           "DatasetSpecification": {
               "CodeArtifacts": {
                   "GenerationCodeS3Uri": code_artifacts_uri + "generation.py",
                   "VerifiersCodeS3Uri": code_artifacts_uri + "verifiers.py",
               },
           },
           "NumSamples": 500,
       },
   )
   full_dataset = preparer.generate(full_config)
   print("Final DataSet ARN:", full_dataset.arn)

The output is JSONL (one JSON object per line) at ``full_dataset.source``.

Next Steps
----------

- :doc:`dq_tutorial` — Evaluate the quality of your generated dataset
- :doc:`dt_tutorial` — Deduplicate, split, reshape or filter it with transformation operators
- Use ``full_dataset.arn`` as input to ModelTrainer for fine-tuning
- **Iterate**: generate → evaluate with DQ → refine spec or code artifacts → regenerate

Troubleshooting
---------------

**DataPreparerJobFailedError with "HUB_FINALIZATION_FAILED"**
  The Hub could not register the output dataset. Verify your RoleArn has
  ``sagemaker:ImportHubContent`` permissions (see :doc:`configuration`) and that
  the output S3 bucket is in the same region as the endpoint.

**Region mismatch errors**
  All S3 URIs (spec, input data, output) must be in the same region as the
  SageMaker endpoint (us-east-1 or us-west-2 during preview).

**Job takes longer than expected**
  First-time generation includes code optimization (5–15 min for samples). Full
  generation at 500 samples may take ~2 hours. Check the live status panel for
  progress.
