Data Transformation Tutorial
=============================

Transform an existing dataset by applying operators — deduplication, splitting,
merging, schema conversion, quality filters, or your own code — with the
DataPreparer API.

Data Transformation differs from the other two categories in two ways worth
knowing up front:

- It operates on a dataset that **already exists**, referenced by ARN, rather
  than generating one from a specification.
- ``transform()`` returns a **list** of ``DataSet`` objects, because a single job
  can publish more than one output.

Prerequisites
-------------

- Install the SageMaker Python SDK: ``pip install --upgrade sagemaker``
- ``~/.config/sagemaker/config.yaml`` with RoleArn and DefaultS3Bucket (see :doc:`configuration`)
- An IAM role with the permissions in :doc:`configuration` — Data Transformation
  needs ``sagemaker:ImportHubContent`` and S3 access, including
  ``s3:GetBucketLocation``
- DataPreparer is available in **us-east-1** and **us-west-2** only
- An existing dataset registered as a ``DataSet``, whose ARN you will pass as
  input. Bring your own data with ``DataSet.create()``, or use the output of an
  SDG job (see :doc:`sdg_tutorial`).

Step 1: Setup
-------------

.. code-block:: python

   from sagemaker.core.helper.session_helper import Session
   from sagemaker.ai_registry import DataPreparer, DatasetJobConfig, JobCategory

   session = Session()
   preparer = DataPreparer(sagemaker_session=session)

   print(f"Bucket: {session.default_bucket()}")

Step 2: Discover Available Operators
------------------------------------

.. code-block:: python

   DatasetJobConfig.print_schema(JobCategory.DATA_TRANSFORMATION)

This prints the full schema tree, including the operator catalog and the
per-operator ``Config`` keys the service currently accepts.

.. important::

   ``print_schema()`` shows value-level constraints that the SDK **displays but
   does not enforce** — patterns, length bounds, and numeric ranges. Operator
   ``Config`` payloads in particular are passed straight through: ``Operator`` is
   a bare string and ``Config`` is an open dictionary, so a misspelled operator
   name or config key is not caught locally. See :doc:`index` for what client-side
   validation does and does not cover, and `Troubleshooting`_ below for how such
   typos surface.

Step 3: Run Your First Transformation
-------------------------------------

This example filters out rows whose ``prompt`` field is shorter than 170
characters:

.. code-block:: python

   dataset = DatasetJobConfig(
       job_category=JobCategory.DATA_TRANSFORMATION,
       input_data_config=[{"DatasetArn": input_arn}],
       output_data_config={"S3OutputPath": "s3://my-bucket/dt-output/"},
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

   results = preparer.transform(dataset)

   for ds in results:
       print("DataSet ARN:", ds.arn)
       print("Source:", ds.source)

The output is published as a new hub content version — the input dataset is never
modified. ``ds.source`` points at the resulting JSONL object in S3.

.. note::

   ``transform()`` always returns a list, even for single-output operators. Index
   into it (``results[0]``) or iterate; do not treat the return value as a single
   ``DataSet``.

.. important::

   ``Operations`` accepts exactly **one** entry per job. The published schema
   advertises ``maxItems=10``, but the service enforces a limit of 1 — a job
   with two operations fails with ``"TransformationConfig structure invalid"``.
   For multi-step pipelines, chain jobs: submit one operator per job and pass
   the output dataset ARN as the next job's input (see `Chaining Jobs`_ below).

Operator Catalog
----------------

Each operator takes its own ``Config`` schema. The examples below show the keys
the service accepts, with the constraints that are easy to get wrong.

TextQualityFilter
~~~~~~~~~~~~~~~~~

Drops or flags rows based on a quality heuristic. See Step 3 for a full example.

ExactDedup
~~~~~~~~~~

Collapses rows whose text field matches exactly.

.. code-block:: python

   {"Operator": "ExactDedup", "Config": {"TextField": "prompt", "Normalize": False}}

``TextField`` defaults to ``"text"``, so set it explicitly if your field is named
anything else. The ``Config`` schema accepts only ``TextField`` and ``Normalize``
and rejects anything else.

.. warning::

   Leave ``Normalize`` at ``False``. It passes validation — the schema declares it
   as a boolean — but the Unicode normalization behind it is only available on the
   infrastructure the service uses for large inputs, so for a dataset of ordinary
   size the job fails while running. The failure reason is the generic
   "Transformation task failed" text and does not mention ``Normalize``.

FuzzyDedup
~~~~~~~~~~

Also collapses near-duplicates, using MinHash/Jaccard similarity.

.. code-block:: python

   {
       "Operator": "FuzzyDedup",
       "Config": {
           "TextField": "prompt",
           "JaccardThreshold": 0.8,
           "NumPerm": 128,
           "NgramSize": 5,
       },
   }

Where ``ExactDedup`` keeps a row that differs by a single word, ``FuzzyDedup``
absorbs it. ``FuzzyDedup`` declares no required fields, so a misspelled key is
the only way this payload can fail validation — and it will, by name.

Split
~~~~~

Partitions one dataset into several. This is the operator that makes
``transform()`` return more than one ``DataSet``.

.. code-block:: python

   {"Operator": "Split", "Config": {"Ratios": {"train": 0.8, "test": 0.2}, "Seed": 42}}

``Ratios`` is required, must have at least two entries, and must sum to ``1.0``.
Every partition is guaranteed at least one record, and the last partition takes
the remainder. Outputs are named after the job and the ratio key — for example
``{job_name}-train`` and ``{job_name}-test`` — and carry the input's file format
forward.

Merge
~~~~~

Concatenates datasets. This is the only operator that requires two inputs.

.. code-block:: python

   {"Operator": "Merge", "Config": {}}

Pass exactly two entries in ``input_data_config``. Two is both the floor and the
ceiling: the service rejects a single input, and the published job-config schema
declares a maximum of two.

.. warning::

   ``Merge`` declares no config properties, so **any** key fails the job — the
   name does not matter. The empty dictionary is required, not a placeholder. The
   failure names both the operator and the key you supplied.

SchemaConversion
~~~~~~~~~~~~~~~~

Rewrites every row into a different dataset format.

.. code-block:: python

   {"Operator": "SchemaConversion", "Config": {"TargetFormat": "OpenAI_Messages"}}

``TargetFormat`` must differ from the input dataset's declared format —
converting a format to itself is rejected while the job runs, not at submission,
and the failure reason is generic. Conversion reshapes rows without dropping any.

SchemaFilter
~~~~~~~~~~~~

Drops rows that do not conform to an expected format — useful for a dataset that
mixes formats.

.. code-block:: python

   transformation_config={
       "Operations": [{
           "Operator": "SchemaFilter",
           "Config": {"ExpectedFormat": "HF_Prompt_Completion"},
       }],
   }

``SchemaFilter`` accepts a dataset that mixes formats, which is precisely the input this
operator exists to clean up; no extra setting is needed.

BYOC
~~~~

Runs transformation code you supply.

.. code-block:: python

   {"Operator": "BYOC", "Config": {"TransformationCodeS3Uri": "s3://my-bucket/code/transformer.py"}}

Your module must define a function named exactly:

.. code-block:: python

   def transform_dataset(input_path, output_path):
       # read input_path, write output_path
       return {"records_in": ..., "records_out": ...}   # stats dict

The code runs in a restricted Python sandbox, so standard library access is
limited and arbitrary imports are not available.

CodeGen
~~~~~~~

Has a model write the transformation code from a natural-language specification,
then optionally executes it.

.. code-block:: python

   {
       "Operator": "CodeGen",
       "Config": {
           "SpecificationS3Uri": "s3://my-bucket/specs/transform_spec.md",
           "AutoExecute": True,
       },
   }

Because the code is model-generated, results are less deterministic than the
other operators. Write the specification to be unambiguous, and re-run before
treating an unexpected result as a defect. This operator requires
``bedrock:InvokeModel`` on the role (see :doc:`configuration`).

The generated code is uploaded alongside the job's output, as
``transformer.py`` under ``{S3OutputPath}/{job_name}/``. Neither ``transform()``
nor the returned ``DataSet`` exposes that URI, but the dataset's ``source`` is
its sibling, so it can be derived without knowing the job name:

.. code-block:: python

   results = preparer.transform(codegen_config)
   code_uri = results[0].source.rsplit("/", 1)[0] + "/transformer.py"

Reading it is the only way to see what the model actually did. It is also the
way to make a CodeGen result reproducible: pass that URI as a ``BYOC``
``TransformationCodeS3Uri`` and subsequent runs execute fixed code instead of
regenerating it.

.. warning::

   ``AutoExecute`` defaults to generating a **preview only**. With
   ``AutoExecute: False`` the job still succeeds and still publishes a
   ``DataSet``, but its source is ``sample_output.jsonl`` — the generated code
   run over a small sample of the input (10 rows by default), not over the
   whole dataset. Only ``AutoExecute: True`` executes the code over the full
   input and writes ``dataset.jsonl``. Nothing in the job status distinguishes
   the two, so check ``source`` before using a CodeGen output as training data.

Working With Multiple Outputs
-----------------------------

``Split`` publishes one ``DataSet`` per partition, so the list length matches the
number of ratio keys:

.. code-block:: python

   results = preparer.transform(split_config)
   assert len(results) == 2
   train, test = results
   print(train.arn, test.arn)

Chaining Jobs
-------------

A job's output datasets are full, first-class datasets, so they can be fed
straight into another Data Transformation job by ARN — no re-import required.
Splitting a corpus, transforming the halves, and recombining them is a supported
path:

.. code-block:: python

   # 1. Split
   split_results = preparer.transform(split_config)
   split_arns = [ds.arn for ds in split_results]

   # 2. Merge the halves back together
   merge_config = DatasetJobConfig(
       job_category=JobCategory.DATA_TRANSFORMATION,
       input_data_config=[{"DatasetArn": arn} for arn in split_arns],
       output_data_config={"S3OutputPath": "s3://my-bucket/dt-output/"},
       transformation_config={"Operations": [{"Operator": "Merge", "Config": {}}]},
   )
   merged = preparer.transform(merge_config)

Payload Limits
--------------

- ``InputDataConfig`` holds at most **2** entries.
- ``Operations`` holds exactly **1** entry — chain jobs, as shown above, rather
  than stacking operators in one job.

.. warning::

   ``print_schema()`` advertises ``operations`` as ``maxItems=10``. The service
   currently accepts only **1**, and a job with two operations fails validation
   with a ``TransformationConfig structure invalid`` message. Do not size a
   pipeline on the published number.

Next Steps
----------

- :doc:`dq_tutorial` — Evaluate the quality of a transformed dataset
- :doc:`configuration` — Role permissions and defaults resolution
- Use an output ``DataSet`` ARN as input to ModelTrainer for fine-tuning

Troubleshooting
---------------

**Job fails with an unknown operator name**
  The operator name reached the service unchallenged — neither the SDK nor the
  published schema constrains it to the catalog. The failure names the operator
  you supplied and lists the supported ones. Check spelling and capitalization
  against ``print_schema()``.

**Job fails naming a Config key**
  A key not in that operator's schema was supplied; most operator schemas reject
  unknown properties. The failure message names both the operator and the
  offending key — for example, ``Threshold`` instead of FuzzyDedup's
  ``JaccardThreshold``.

.. note::

   Both of the above fail **asynchronously**. ``CreateJob`` accepts the request
   and the job is persisted, then validation runs and the job terminates as
   ``Failed`` with the reason. So these surface as a job failure raised from
   waiting, not as a rejected API call. No compute is provisioned, because
   validation precedes it.

**"Merge requires at least 2 input datasets"**
  ``Merge`` needs exactly two entries in ``input_data_config``.

**Rows dropped unexpectedly, or the job fails validating input**
  Operators expect the dataset to match its declared format. To clean a deliberately
  mixed dataset, run ``SchemaFilter`` first, as its own job, and transform its output.

**Deduplication had no effect**
  ``TextField`` defaults to ``"text"``. If your field has another name, the
  operator found nothing to compare.

**DataPreparerJobFailedError with "HUB_FINALIZATION_FAILED"**
  The Hub could not register the output dataset. Verify your RoleArn has
  ``sagemaker:ImportHubContent`` permissions (see :doc:`configuration`) and that
  the output S3 bucket is in the same region as the endpoint.

**Job fails during output validation with an S3 error**
  The role is likely missing ``s3:GetBucketLocation``, which is needed
  separately from object read access. See :doc:`configuration`.

**Region mismatch errors**
  All S3 URIs and datasets must be in the same region as the SageMaker endpoint
  (us-east-1 or us-west-2 during preview).

**Retrying a failed job**
  Jobs are immutable and ``JobName`` must be unique per account and region. There
  is no retry in place — submit a new job. Note that IAM changes take roughly 30
  seconds to propagate and do not revive a job that already terminated.
