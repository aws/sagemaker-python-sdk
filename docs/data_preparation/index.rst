Data Preparation
================

.. important::

   **🆕 V3 Exclusive Feature** — Data Preparation with DataPreparer is available
   only in SageMaker Python SDK V3.

``DataPreparer`` provides a unified API for three data preparation workflows:

- **Synthetic Data Generation (SDG)** — generate training datasets from a natural-language
  task specification.
- **Data Quality Evaluation (DQ)** — evaluate an existing dataset against quality metrics
  and produce a scorecard.
- **Data Transformation (DT)** — run a pipeline of operators (dedup, split, merge, schema
  conversion, quality filters, custom code) over an existing dataset.

These share the same entrypoint class and configuration pattern, differing in the method called
(``generate``, ``evaluate`` or ``transform``) and the fields inside
:class:`~sagemaker.ai_registry.data_preparer.configs.DatasetJobConfig`. SDG and DQ each return a
single ``DataSet``; DT returns a list, since operators such as ``Split`` emit more than one.

Key Concepts
------------

**DataPreparer**
  The entrypoint class. Supports both instance-method and classmethod call styles::

    # Instance style
    preparer = DataPreparer(sagemaker_session=session)
    preparer.generate(config)

    # Classmethod style
    DataPreparer.generate(config, sagemaker_session=session)

**DatasetJobConfig**
  A configuration object built at submission time from a JSON schema published in the AI
  Registry Hub. Use ``DatasetJobConfig.from_template()`` to discover required fields for a
  category, and ``DatasetJobConfig.print_schema()`` to see the full schema including
  constraints the SDK reports but does not enforce (see **Schema-driven validation** below).

**JobCategory**
  Enum identifying the job category. The categories documented here are
  ``SYNTHETIC_DATA_GENERATION``, ``DATA_QUALITY_EVALUATION`` and ``DATA_TRANSFORMATION``.

**DataSet**
  The result object returned by a completed job. Provides the dataset ARN, S3 source location,
  and metadata.

**Schema-driven validation**
  ``DatasetJobConfig`` fetches the latest schema from the Hub at submission time and builds its
  model from it, so field names are not hard-coded in the SDK.

  Client-side checks are deliberately shallow. The SDK enforces:

  - **Required fields** — a missing required field fails before the job is submitted.
  - **Primitive types** — a string where an integer is expected fails.
  - **Nested structure** — objects and lists must have the declared shape.

  The SDK does **not** enforce value-level constraints, even though
  ``print_schema()`` displays them: ``pattern``, ``minLength``/``maxLength``,
  ``minItems``/``maxItems``, and numeric bounds are all passed through to the service.
  Neither are per-operator ``Config`` payloads for Data Transformation validated
  field-by-field. Violations of these surface as a service-side job failure rather than a
  local exception, so read ``FailureReason`` on the failed job when a config the SDK
  accepted did not run.

SDG Workflow
~~~~~~~~~~~~

1. **Sample** — provide a task description (``SpecS3Uri``) and request a small sample
   (e.g. 10 records) to verify the generation direction.
2. **Refine** — adjust the spec or parameters based on sample output.
3. **Full generation** — submit the final config with the desired ``NumSamples``.

DQ Workflow
~~~~~~~~~~~

1. **Import** — reference an existing dataset by ARN.
2. **Evaluate** — submit a ``DataQualityEvaluation`` job that scores each record.
3. **Scorecard** — inspect the output scorecard and per-record quality annotations.

DT Workflow
~~~~~~~~~~~

1. **Import** — reference an existing dataset by ARN, as ``input_data_config``.
2. **Declare operations** — list the operators to apply in ``transformation_config``. Each
   entry is an ``Operator`` name plus its ``Config``; both are passed through to the service
   as declared in the Hub schema, so the SDK does not validate operator configs field-by-field.
3. **Transform** — submit a ``DataTransformation`` job. The output is published as a new version
   of hub content, and ``transform`` returns the resulting ``DataSet`` objects — one per output,
   so ``Split`` yields several and single-output operators a list of length 1.

.. note::

   ``Operations`` accepts exactly **one** entry per job. The published schema advertises
   ``maxItems=10``, but the service enforces a limit of 1. For multi-step pipelines, chain
   jobs: submit one operator per job and use the output dataset ARN as the next job's input.

See :doc:`dt_tutorial` for the operator catalog and a worked example.

Intelligent Defaults
~~~~~~~~~~~~~~~~~~~~

``DataPreparer`` resolves configuration values through a layered chain:

1. Explicit user values in ``DatasetJobConfig``
2. Per-category config in ``SageMakerConfig`` (``Modules.DataPreparer.Job.<Category>``)
3. Global defaults (``Modules.DataPreparer.GlobalDefaults``)
4. Well-known fallbacks (``RoleArn``, ``OutputDataConfig.OutputS3Uri``)
5. Schema-declared defaults

This means most users only need to set a ``RoleArn`` and ``DefaultS3Bucket`` in their
config file to get started.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   sdg_tutorial
   dq_tutorial
   dt_tutorial
   configuration
   api_reference
   /training/data_preparer
