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
"""Names of the long-lived resources the integration tests reuse.

Some integ tests need a resource that is too slow or expensive to create on every
run: a fine-tuned model package, a completed training job to attach to, an MLflow
app, a private hub holding evaluators. Each account the tests run in keeps one copy
of each, named after that account::

    do-not-delete-<account-id>-<what>

so the same test code finds the right copy in whichever account runs it, and no
account ID, ARN or bucket name is written into the tests. The ``do-not-delete-``
prefix (and the ``pysdk-ci-protected=true`` tag the fixtures carry) also tells the
account's resource cleanup job to leave them alone.

Datasets live under one prefix of the account's default bucket::

    s3://sagemaker-<region>-<account-id>/do-not-delete-<account-id>-fixtures/datasets/

To run these tests in a new account, create the fixtures there first. A lookup that
cannot find one logs a warning and returns a placeholder naming the missing resource,
so only the tests that need it fail.

An identical copy lives at sagemaker-serve/tests/integ/integ_resources.py; keep them in
sync. (The submodules' test trees are separate and cannot import each other.)
"""

from __future__ import absolute_import

import fcntl
import functools
import json
import logging
import os
import tempfile

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

PREFIX = "do-not-delete"
DEFAULT_REGION = "us-west-2"
EVALUATOR_VERSION = "0.0.2"


def region() -> str:
    """Region the fixtures are looked up in."""
    return os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or DEFAULT_REGION


@functools.lru_cache(maxsize=None)
def account_id() -> str:
    """The AWS account the tests run in."""
    return boto3.client("sts", region_name=region(), config=_RETRY_CONFIG).get_caller_identity()[
        "Account"
    ]


# Lookups below run when test modules are imported (tests keep configuration in module
# constants). Under ``pytest -n auto`` every xdist worker imports every module, so an
# uncached lookup would be dozens of concurrent List* calls, which those APIs throttle --
# and a collection error on one worker fails the whole session. Each lookup is therefore
# made once per machine and session: the first process to need it calls the API (with
# adaptive retries) and writes the answer to a cache file keyed by the xdist run, and the
# other workers read it. Outside xdist the cache is per process.
_RETRY_CONFIG = Config(retries={"max_attempts": 10, "mode": "adaptive"})


def _cache_path() -> str:
    run = os.environ.get("PYTEST_XDIST_TESTRUNUID") or str(os.getpid())
    return os.path.join(tempfile.gettempdir(), f"pysdk-integ-resources-{run}.json")


def _cached(key: str, resolve) -> str:
    """Resolve ``key`` once per test run across processes, via a lock-protected file."""
    path = _cache_path()
    with open(path + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            try:
                with open(path) as f:
                    cache = json.load(f)
            except (OSError, ValueError):
                cache = {}
            if key not in cache:
                cache[key] = resolve()
                with open(path, "w") as f:
                    json.dump(cache, f)
            return cache[key]
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _missing(resource_type: str, wanted: str) -> str:
    """Placeholder for a fixture a lookup could not find.

    Lookups run at import time (tests keep their configuration in module constants), so
    raising here would turn one missing fixture into a collection error for every test
    in the module, including ones that never use it. Instead the tests that do use it
    get this value and fail on it, with the missing name in the message.
    """
    logger.warning(
        "Integ fixture %s %r not found in %s/%s", resource_type, wanted, account_id(), region()
    )
    return f"{wanted}-NOT-FOUND-in-{account_id()}"


def name(what: str) -> str:
    """``do-not-delete-<account>-<what>``: the name of a reused fixture."""
    return f"{PREFIX}-{account_id()}-{what}"


def arn(service: str, resource: str) -> str:
    return f"arn:aws:{service}:{region()}:{account_id()}:{resource}"


# --- S3 ---------------------------------------------------------------------------


def default_bucket() -> str:
    return f"sagemaker-{region()}-{account_id()}"


def fixtures_prefix() -> str:
    return name("fixtures")


def dataset_uri(key: str) -> str:
    """URI of a fixture dataset, e.g. ``dataset_uri("eval/zc_test.jsonl")``."""
    return f"s3://{default_bucket()}/{fixtures_prefix()}/datasets/{key}"


def output_uri(key: str) -> str:
    """Scratch output location for test runs (not a fixture; the cleanup job ignores S3)."""
    return f"s3://{default_bucket()}/integ-test-output/{key.strip('/')}/"


# Datasets the tests read. Paths are relative to dataset_uri().
SFT_DATASET = "input_data/sft/sample_data_256_final.jsonl"
DPO_DATASET = "input_data/dpo/preference_dataset_train_256.jsonl"
RLVR_RLAIF_DATASET = "input_data/rlvr-rlaif-test-data/train_285.jsonl"
NOVA_SFT_DATASET = "input_data/sft-nova/sft_8_samples.jsonl"
PT_DATASET = "pt-test-data/train_llama31.jsonl"
EVAL_ZC_DATASET = "eval/zc_test.jsonl"
EVAL_GEN_QA_DATASET = "eval/gen_qa.jsonl"
MTRL_PROMPTS = "gsm8k_small/prompts.parquet"


# --- SageMaker --------------------------------------------------------------------


def hub_name() -> str:
    """Private hub holding the evaluators and test base models."""
    return name("hub")


def hub_content_arn(content_type: str, content_name: str, version: str) -> str:
    return arn("sagemaker", f"hub-content/{hub_name()}/{content_type}/{content_name}/{version}")


def evaluator_arn(evaluator: str) -> str:
    """Hub JsonDoc evaluator, e.g. ``rlvr-test-rf``, ``rlaif-test-prompt``, ``eval-lambda-test``."""
    return hub_content_arn("JsonDoc", evaluator, EVALUATOR_VERSION)


def fixture_model_package_group() -> str:
    """Group whose version 1 is the pinned fine-tuned model package. Never written to."""
    return name("ft-fixture")


def fixture_model_package_arn() -> str:
    return arn("sagemaker", f"model-package/{fixture_model_package_group()}/1")


def output_model_package_group() -> str:
    """Group tests register what they train into."""
    return name("ft-models")


def output_model_package_group_arn() -> str:
    return arn("sagemaker", f"model-package-group/{output_model_package_group()}")


def sft_training_job_name() -> str:
    """Completed serverless LoRA SFT job on llama-3.2-1b-instruct."""
    return name("sft-llama32-1b")


def pytorch_training_job_name() -> str:
    """Completed CPU training job with CloudWatch logs."""
    return name("pytorch-smoke")


def pytorch_training_job_arn() -> str:
    return arn("sagemaker", f"training-job/{pytorch_training_job_name()}")


def mtrl_job_name() -> str:
    """Completed AgentRFT (MTRL) job."""
    return name("mtrl-gsm8k")


def mtrl_model_package_group() -> str:
    """Output group of the MTRL fixture job."""
    return name("mtrl-fixture")


@functools.lru_cache(maxsize=None)
def mlflow_app_arn() -> str:
    """ARN of the fixture MLflow app. The service picks the ID, so look it up by name."""
    want = name("mlflow")

    def resolve():
        sm = boto3.client("sagemaker", region_name=region(), config=_RETRY_CONFIG)
        for page in sm.get_paginator("list_mlflow_apps").paginate():
            for app in page.get("Summaries", []):
                if app.get("Name") == want:
                    return app["Arn"]
        return _missing("mlflow-app", want)

    return _cached(f"mlflow-app/{region()}/{want}", resolve)


# --- Lambda -----------------------------------------------------------------------
# The SDK requires "SageMaker" in some Lambda names, so those fixtures keep it.


def rlvr_reward_function_name() -> str:
    return name("rlvr-oss-reward-fn")


def rlvr_reward_function_arn() -> str:
    return arn("lambda", f"function:{rlvr_reward_function_name()}")


def evaluator_function_arn() -> str:
    return arn("lambda", f"function:{name('SageMaker-eval-fn')}")


def agent_connector_function_arn() -> str:
    """Lambda that forwards MTRL rollouts to the gsm8k AgentCore runtime."""
    return arn("lambda", f"function:{name('SageMaker-agent-connector')}")


# --- Bedrock AgentCore ------------------------------------------------------------

GSM8K_AGENT_RUNTIME = "sagemaker_rft_prod_gsm8k_streaming"


@functools.lru_cache(maxsize=None)
def agent_runtime_id(runtime_name: str = GSM8K_AGENT_RUNTIME) -> str:
    """ID of an AgentCore runtime, which the service suffixes; looked up by exact name."""

    def resolve():
        client = boto3.client(
            "bedrock-agentcore-control", region_name=region(), config=_RETRY_CONFIG
        )
        token = None
        while True:
            response = client.list_agent_runtimes(**({"nextToken": token} if token else {}))
            for runtime in response.get("agentRuntimes", []):
                if runtime.get("agentRuntimeName") == runtime_name:
                    return runtime["agentRuntimeId"]
            token = response.get("nextToken")
            if not token:
                return _missing("runtime", runtime_name)

    return _cached(f"agentcore-runtime/{region()}/{runtime_name}", resolve)


def agent_runtime_arn(runtime_name: str = GSM8K_AGENT_RUNTIME) -> str:
    return f"arn:aws:bedrock-agentcore:{region()}:{account_id()}:runtime/{agent_runtime_id(runtime_name)}"
