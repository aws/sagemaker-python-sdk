Configuration
=============

``DataPreparer`` uses the SageMaker Python SDK's unified configuration system
(``SageMakerConfig``) to resolve defaults so you can run jobs with minimal code.

Config File Location
--------------------

The SDK searches for ``config.yaml`` in the following order:

1. Path set in the ``SAGEMAKER_USER_CONFIG_OVERRIDE`` environment variable (user config)
   or ``SAGEMAKER_ADMIN_CONFIG_OVERRIDE`` (admin config)
2. ``~/.config/sagemaker/config.yaml`` (user-level, via ``platformdirs``)
3. ``/etc/xdg/sagemaker/config.yaml`` (admin-level, via ``platformdirs``)

.. note::

   The legacy path ``~/.sagemaker/config.yaml`` is only used for local-mode
   configuration and is **not** searched for general SDK config.

``Session()`` loads the config automatically — no extra code required.

Required Keys
-------------

At minimum, configure a role and default S3 bucket:

.. code-block:: yaml

   SageMaker:
     PythonSDK:
       Modules:
         DataPreparer:
           GlobalDefaults:
             RoleArn: arn:aws:iam::123456789012:role/SageMakerExecutionRole
         Session:
           DefaultS3Bucket: my-sagemaker-bucket

.. important::

   During the preview period, DataPreparer is available in **us-east-1** and
   **us-west-2** only. All S3 buckets referenced in job configs must reside in
   the same region as the SageMaker endpoint.

``RoleArn``
  The IAM role assumed by the service to run jobs. Resolved via:

  1. ``Modules.DataPreparer.GlobalDefaults.RoleArn``
  2. ``SageMaker.TrainingJob.RoleArn`` (shared fallback)
  3. ``get_execution_role()`` (notebook/Studio environments)

``DefaultS3Bucket``
  Used to construct the default ``OutputDataConfig.OutputS3Uri`` when none is
  explicitly provided. Configured at ``SageMaker.PythonSDK.Modules.Session.DefaultS3Bucket``.

Role Permissions
----------------

Data preparation jobs run in the service's account and reach back into yours by
assuming the role in ``RoleArn``. Both halves of that role must be set up: the
trust policy decides *who may assume it*, and the permissions policy decides
*what it may then do*. A gap in either one fails the job, and the two failures
look different — see :ref:`dataprep-permission-failures` below.

Trust policy
~~~~~~~~~~~~

The role must trust the SageMaker service principal:

.. code-block:: json

   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Principal": {
           "Service": "sagemaker.amazonaws.com"
         },
         "Action": "sts:AssumeRole"
       }
     ]
   }

Permissions policy
~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": ["s3:GetObject", "s3:PutObject"],
         "Resource": "arn:aws:s3:::my-bucket/*"
       },
       {
         "Effect": "Allow",
         "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
         "Resource": "arn:aws:s3:::my-bucket"
       },
       {
         "Effect": "Allow",
         "Action": "sagemaker:ImportHubContent",
         "Resource": [
           "arn:aws:sagemaker:us-west-2:123456789012:hub/MyHub",
           "arn:aws:sagemaker:us-west-2:123456789012:hub-content/MyHub/DataSet/*"
         ]
       },
       {
         "Effect": "Allow",
         "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
         "Resource": "*"
       }
     ]
   }

What each grant is for:

``s3:GetObject`` / ``s3:PutObject``
  Read the input dataset and write job output. Note the ``/*`` suffix — these
  are object-level actions and do not apply to the bucket ARN itself.

``s3:ListBucket``
  Enumerate objects under the input prefix.

``s3:GetBucketLocation``
  Easy to miss, and its absence is not obvious: the job proceeds through
  transformation and then fails during output validation, because resolving a
  bucket to its region requires this action separately from reading objects.

``sagemaker:ImportHubContent``
  Register the output dataset as hub content. The action is
  ``ImportHubContent``, not ``CreateHubContent``. Scope it to both the hub ARN
  and the ``hub-content/<hub>/DataSet/*`` ARN — the call touches both.

  The hub in question is the one the **input** dataset lives in. Data
  Transformation publishes each output as a new version alongside its input, so
  that hub — not a hub of your choosing — is the one the role must cover.

  .. warning::

     This is the one grant whose absence does not fail the job. Publishing is
     best-effort: if the role cannot ``ImportHubContent`` on the input's hub, the
     job still reports ``Completed`` with no ``FailureReason``, the transformed
     file still lands in S3, and only ``OutputDatasetArns`` is missing from
     ``DescribeJob``. The SDK raises ``DataPreparerJobError`` with the message
     ``Completed job missing OutputDatasetArns in JobConfigDocument``. Read that
     as "the role is missing ``ImportHubContent`` on the input dataset's hub",
     not as a transformation failure.

``bedrock:InvokeModel`` / ``bedrock:InvokeModelWithResponseStream``
  Required for Synthetic Data Generation and Data Quality Evaluation, which
  invoke a model. Data Transformation needs these only if the pipeline includes
  an operator that invokes a model; purely mechanical operators (``Split``,
  ``Merge``, ``FuzzyDedup``, schema conversion) do not.

.. _dataprep-permission-failures:

Diagnosing permission failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A missing trust policy and a missing permissions policy surface as the same
generic ``FailureReason``, so distinguish them by looking up the ``AssumeRole``
event for your role in CloudTrail:

- ``AccessDenied`` on ``AssumeRole`` — the **trust policy** is the problem; the
  service never obtained credentials.
- ``AssumeRole`` succeeded — the **permissions policy** is the problem; the
  service acted as your role and was denied a specific action.
- **No ``FailureReason`` at all, and the job says ``Completed``** — if
  ``OutputDatasetArns`` is absent from ``DescribeJob``, the role is missing
  ``sagemaker:ImportHubContent`` on the input dataset's hub. Nothing else
  produces a successful job with no published output.

Two further points worth knowing before you re-run:

- IAM changes take roughly 30 seconds to propagate. A job submitted immediately
  after a policy edit can still fail on the old policy.
- Jobs are immutable and ``JobName`` must be unique within an account and
  region. There is no retry in place, and fixing IAM does not revive a job that
  already terminated — submit a new job under a new name.

Optional Keys
-------------

Per-category job defaults
~~~~~~~~~~~~~~~~~~~~~~~~~~

Override defaults for a specific job category:

.. code-block:: yaml

   SageMaker:
     PythonSDK:
       Modules:
         DataPreparer:
           Job:
             SyntheticDataGeneration:
               OutputDataConfig:
                 OutputS3Uri: s3://my-bucket/sdg-output/
             DataQualityEvaluation:
               OutputDataConfig:
                 OutputS3Uri: s3://my-bucket/dq-output/

Resolution Chain
----------------

For every field in the job config schema, ``DataPreparer`` resolves values in this order:

1. **Explicit value** — set directly in ``DatasetJobConfig(...)``
2. **Per-category config** — ``Modules.DataPreparer.Job.<Category>.<path>``
3. **Global defaults** — ``Modules.DataPreparer.GlobalDefaults.<path>``
4. **Well-known fallbacks** — hard-coded for ``RoleArn`` and ``OutputDataConfig.OutputS3Uri``
5. **Schema default** — declared in the Hub-published JSON schema

The first non-None value wins. This means you can set broad defaults globally and
override them per-category or per-call as needed.

Verifying Your Config
---------------------

.. code-block:: python

   from sagemaker.core.config.config_manager import SageMakerConfig

   cfg = SageMakerConfig()
   role = cfg.resolve_value_from_config(
       config_path="SageMaker.PythonSDK.Modules.DataPreparer.GlobalDefaults.RoleArn"
   )
   print(f"Resolved RoleArn: {role}")
