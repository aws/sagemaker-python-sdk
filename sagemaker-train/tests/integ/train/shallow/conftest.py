# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Fixtures for the shallow (submit-then-stop) training-job suite.

Inherits ``sagemaker_session``, ``ensure_default_region`` and the adaptive-retry
configuration from the parent ``tests/integ/train/conftest.py`` and
``tests/integ/conftest.py``; only fixtures specific to shallow submission live
here.

Everything here is session- or module-scoped and idempotent: these tests run
in parallel across xdist workers, so any fixture creating an AWS-side artifact must
tolerate a dozen workers racing to create the same thing.
"""

from __future__ import absolute_import

import json
import logging
import os

import pytest

logger = logging.getLogger(__name__)

# Uploaded once and reused. A tiny object is enough: the backend's role-assuming
# validators check that the S3 prefix resolves, not what it contains.
_TRAIN_DATA_KEY = "shallow-integ-test/train/data.jsonl"
_VALIDATION_DATA_KEY = "shallow-integ-test/validation/data.jsonl"

_SAMPLE_RECORDS = [
    {
        "messages": [
            {"role": "user", "content": [{"text": "What is 2+2?"}]},
            {"role": "assistant", "content": [{"text": "4"}]},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": [{"text": "Capital of France?"}]},
            {"role": "assistant", "content": [{"text": "Paris"}]},
        ]
    },
]


def _ensure_object(sagemaker_session, key):
    """Upload the sample dataset at ``key`` if absent; return its S3 URI.

    Idempotent so concurrent xdist workers converge instead of colliding. The
    object is intentionally left behind: it is a few hundred bytes and reusing
    it removes an upload from every subsequent run.
    """
    bucket = sagemaker_session.default_bucket()
    s3 = sagemaker_session.boto_session.client("s3")

    response = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
    if response.get("KeyCount", 0) == 0:
        body = "\n".join(json.dumps(record) for record in _SAMPLE_RECORDS)
        s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))
        logger.info("Uploaded shallow-test fixture data to s3://%s/%s", bucket, key)

    return f"s3://{bucket}/{key}"


@pytest.fixture(autouse=True, scope="session")
def bundled_service_model():
    """Point botocore at the service model bundled in ``sagemaker-core/sample``.

    Some request fields this suite exercises are not in the public botocore model
    yet -- ``ServerlessJobConfig.SequenceLength`` is the current example. Without
    this, botocore rejects the request client-side with

        Unknown parameter in ServerlessJobConfig: "SequenceLength"

    and the test fails before reaching the service, which tells us nothing about
    whether the payload is acceptable. Verified against AWS: setting AWS_DATA_PATH
    adds ``SequenceLength`` to the shape.

    Session-scoped and autouse because botocore caches loaded models per client;
    setting this after a client exists would not take effect. Mirrors the
    ``setup_aws_data_path`` fixture in ``test_recipe_override_integration.py``,
    which solves the same problem for the client-side recipe tests.
    """
    # tests/integ/train/shallow/conftest.py -> repo root is five levels up.
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    )
    sample_path = os.path.join(repo_root, "sagemaker-core", "sample")

    previous = os.environ.get("AWS_DATA_PATH")
    if os.path.isdir(sample_path):
        os.environ["AWS_DATA_PATH"] = sample_path
        logger.info("Using bundled service model at %s", sample_path)
    else:
        # Don't fail the run: on an installed-package layout the bundled model may
        # not be present, and only the few tests using unreleased fields break.
        logger.warning("Bundled service model not found at %s", sample_path)

    yield

    if previous is None:
        os.environ.pop("AWS_DATA_PATH", None)
    else:
        os.environ["AWS_DATA_PATH"] = previous


@pytest.fixture(scope="module")
def train_data_uri(sagemaker_session):
    """S3 URI of a real, existing training-data prefix."""
    return _ensure_object(sagemaker_session, _TRAIN_DATA_KEY)


@pytest.fixture(scope="module")
def validation_data_uri(sagemaker_session):
    """S3 URI of a real, existing validation-data prefix."""
    return _ensure_object(sagemaker_session, _VALIDATION_DATA_KEY)


@pytest.fixture(scope="module")
def nova_train_data_uri(sagemaker_session_us_east_1):
    """Training data in us-east-1, for Nova-only paths (e.g. data mixing).

    Nova models are exercised in us-east-1 in this repo (see the
    ``sagemaker_session_us_east_1`` fixture in the parent conftest), and an S3
    prefix must be in the same region as the job that reads it -- so this cannot
    reuse ``train_data_uri``, which lives in the default region's bucket.
    """
    return _ensure_object(sagemaker_session_us_east_1, _TRAIN_DATA_KEY)


@pytest.fixture(scope="module")
def reward_scored_data_uri():
    """Dataset the RLVR reward functions can actually score.

    The reward-function tests cannot use ``train_data_uri``. Verified against AWS:
    before submitting, the SDK *invokes* the reward function over sample records
    and fails the call if they do not score --

        OSS reward function returned non-200 status code: 500.
        Body: {"error": "GSM8k scoring failed: 'list' object has no attribute 'strip'"}

    The pre-provisioned reward functions expect GSM8k-shaped records, so this
    reuses the same dataset the deep RLVR suite uses rather than this suite's
    generic chat-format fixture.
    """
    return "s3://mc-flows-sdk-testing/input_data/rlvr-rlaif-test-data/train_285.jsonl"


@pytest.fixture(scope="module")
def reward_evaluator(sagemaker_session):
    """An existing AI Registry Evaluator object, if present; skip otherwise.

    Look-up only, for the same reason as ``reward_lambda_arn``: the deep suite's
    fixture will *create* an evaluator (and wait for it), which is a durable
    registry write this suite should not make.
    """
    from sagemaker.ai_registry.evaluator import Evaluator

    name = "test-integ-rlvr-trainer"
    try:
        return Evaluator.get(name, sagemaker_session=sagemaker_session)
    except Exception:
        pytest.skip(f"Evaluator {name!r} not present; skipping")


@pytest.fixture(scope="module")
def reward_lambda_arn(sagemaker_session):
    """ARN of the OSS reward-function Lambda, if it already exists.

    The parent train conftest creates this Lambda on demand
    (``oss_lambda_arn``), including an IAM role and a 15-second propagation
    sleep. This suite only looks it up: creating IAM roles and Lambdas is a
    durable side effect that a fast PR-gate suite should not perform. Skips when
    absent, so the account state decides rather than the test.
    """
    client = sagemaker_session.boto_session.client("lambda")
    name = "pysdk-integ-test-sm-train-oss-reward-fn"
    try:
        return client.get_function(FunctionName=name)["Configuration"]["FunctionArn"]
    except Exception:
        pytest.skip(f"Reward-function Lambda {name!r} not present; skipping")


@pytest.fixture(scope="module")
def mlflow_arn(sagemaker_session):
    """ARN of an existing, ready MLflow app; skip if the account has none.

    Deliberately does NOT create one. The parent train conftest's
    ``mlflow_resource_arn`` fixture will create and delete an app if none exists,
    which takes minutes and provisions a durable resource -- far too heavy for a
    suite whose whole point is to be cheap. Here a missing app just skips the two
    tests that need an ARN; the experiment/run-name path is covered unconditionally.
    """
    client = sagemaker_session.boto_session.client("sagemaker")
    try:
        # Not a paginatable operation ("Operation cannot be paginated:
        # list_mlflow_apps"), so call it directly rather than via get_paginator.
        summaries = client.list_mlflow_apps().get("Summaries", [])
    except Exception as e:
        pytest.skip(f"Could not list MLflow apps: {e}")

    for app in summaries:
        if app.get("Status") in ("Created", "Updated"):
            return app["Arn"]

    pytest.skip("No ready MLflow app in this account; skipping ARN-based test")


@pytest.fixture(scope="module")
def output_path(sagemaker_session):
    """S3 prefix for training output.

    Nothing is ever written here -- the jobs are stopped long before they upload
    artifacts -- but the backend validates the output location, so it must be a
    real, writable prefix.
    """
    return f"s3://{sagemaker_session.default_bucket()}/shallow-integ-test/output/"


@pytest.fixture(scope="module")
def nonexistent_data_uri(sagemaker_session):
    """S3 URI, in a real bucket, that does not exist.

    Used by negative tests to prove input validation actually reaches S3 rather
    than being skipped.
    """
    bucket = sagemaker_session.default_bucket()
    return f"s3://{bucket}/shallow-integ-test/definitely-not-here-04c1f9/"


@pytest.fixture(scope="module")
def execution_role(sagemaker_session):
    """The validated training execution role for this account.

    Resolved through the SDK's own resolver so these tests exercise the same
    role-discovery path real users hit, and so a broken/unassumable default role
    surfaces here rather than as a confusing per-test PassRole failure.
    """
    from sagemaker.train.defaults import TrainDefaults

    return TrainDefaults.get_role(role=None, sagemaker_session=sagemaker_session)


@pytest.fixture(scope="module")
def account_id(sagemaker_session):
    """Caller's AWS account id, for building ARNs in negative tests."""
    return sagemaker_session.boto_session.client("sts").get_caller_identity()["Account"]


@pytest.fixture(scope="module")
def region(sagemaker_session):
    """Region under test, for building ARNs and region-sensitive assertions."""
    return sagemaker_session.boto_session.region_name
