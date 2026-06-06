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
from __future__ import absolute_import

import json
import os
import pathlib
from datetime import datetime, timedelta, timezone

import boto3
import pytest
from filelock import FileLock
from botocore.config import Config
from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME
from sagemaker.jumpstart.hub.hub import Hub
from sagemaker.session import Session
from tests.integ.sagemaker.jumpstart.constants import (
    ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID,
    ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME,
    HUB_NAME_PREFIX,
    JUMPSTART_TAG,
)

from sagemaker.jumpstart.types import (
    HubContentType,
)


from tests.integ.sagemaker.jumpstart.utils import (
    get_test_artifact_bucket,
    get_test_suite_id,
    get_sm_session,
    with_exponential_backoff,
)


# Only delete leftover hubs from previous test runs that are older than this many
# hours. This guards against deleting a hub that another concurrent test run (or
# xdist worker) is actively using.
STALE_HUB_AGE_HOURS = 3


def _setup(test_suite_id=None, test_hub_name=None):
    print("Setting up...")
    test_suite_id = test_suite_id or get_test_suite_id()
    test_hub_name = test_hub_name or f"{HUB_NAME_PREFIX}{test_suite_id}"
    test_hub_description = "PySDK Integ Test Private Hub"

    os.environ.update({ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID: test_suite_id})
    os.environ.update({ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME: test_hub_name})

    # Create a private hub to use for the test session
    hub = Hub(hub_name=test_hub_name, sagemaker_session=get_sm_session())

    # Check if hub already exists before creating
    try:
        hub.describe()
        print(f"Hub {test_hub_name} already exists, reusing it.")
    except Exception:
        # Hub doesn't exist, create it
        try:
            hub.create(description=test_hub_description)
            print(f"Created new hub: {test_hub_name}")
        except Exception as e:
            if "ResourceLimitExceeded" in str(e):
                print("Hub limit reached. Cleaning up old hubs...")
                _cleanup_old_hubs(get_sm_session())
                # Retry creating the hub
                hub.create(description=test_hub_description)
                print(f"Created new hub after cleanup: {test_hub_name}")
            else:
                raise


def _teardown(test_suite_id=None, test_hub_name=None):
    print("Tearing down...")

    test_cache_bucket = get_test_artifact_bucket()

    test_suite_id = test_suite_id or os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]

    test_hub_name = test_hub_name or os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME]

    boto3_session = boto3.Session(region_name=JUMPSTART_DEFAULT_REGION_NAME)

    sagemaker_client = boto3_session.client(
        "sagemaker",
        config=Config(retries={"max_attempts": 10, "mode": "standard"}),
    )

    sagemaker_session = Session(boto_session=boto3_session, sagemaker_client=sagemaker_client)

    search_endpoints_result = sagemaker_client.search(
        Resource="Endpoint",
        SearchExpression={
            "Filters": [
                {"Name": f"Tags.{JUMPSTART_TAG}", "Operator": "Equals", "Value": test_suite_id}
            ]
        },
    )

    endpoint_names = [
        endpoint_info["Endpoint"]["EndpointName"]
        for endpoint_info in search_endpoints_result["Results"]
    ]
    endpoint_config_names = [
        endpoint_info["Endpoint"]["EndpointConfigName"]
        for endpoint_info in search_endpoints_result["Results"]
    ]
    model_names = list(
        filter(
            lambda elt: elt is not None,
            [
                sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)[
                    "ProductionVariants"
                ][0].get("ModelName")
                for endpoint_config_name in endpoint_config_names
            ],
        )
    )

    inference_component_names = []
    for endpoint_name in endpoint_names:
        for (
            inference_component_name
        ) in sagemaker_session.list_and_paginate_inference_component_names_associated_with_endpoint(
            endpoint_name=endpoint_name
        ):
            inference_component_names.append(inference_component_name)

    # delete inference components for test-suite-tagged endpoints
    for inference_component_name in inference_component_names:
        sagemaker_session.delete_inference_component(
            inference_component_name=inference_component_name, wait=True
        )

    # delete test-suite-tagged endpoints
    for endpoint_name in endpoint_names:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

    # delete endpoint configs for test-suite-tagged endpoints
    for endpoint_config_name in endpoint_config_names:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)

    # delete models for test-suite-tagged endpoints
    for model_name in model_names:
        sagemaker_client.delete_model(ModelName=model_name)

    # delete test artifact/cache s3 folder
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(test_cache_bucket)
    bucket.objects.filter(Prefix=test_suite_id + "/").delete()

    # delete private hubs
    _delete_hubs(sagemaker_session, test_hub_name)


def _cleanup_old_hubs(sagemaker_session, active_hub_name=None):
    """Clean up stale test hubs from previous runs to free up resources.

    Only deletes hubs that are clearly stale (older than ``STALE_HUB_AGE_HOURS``)
    so that hubs actively in use by the current test run or by concurrent xdist
    workers are never removed. The hub for the current run (``active_hub_name``)
    is always preserved.
    """
    try:
        active_hub_name = active_hub_name or os.environ.get(ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=STALE_HUB_AGE_HOURS)

        response = sagemaker_session.list_hubs()
        for hub in response.get("HubSummaries", []):
            hub_name = hub["HubName"]
            if not hub_name.startswith(HUB_NAME_PREFIX):
                continue
            if hub_name == active_hub_name:
                continue

            creation_time = hub.get("CreationTime")
            # Only delete hubs we can confirm are older than the cutoff. If the
            # creation time is unavailable, err on the side of keeping the hub.
            if creation_time is None:
                continue
            if creation_time.tzinfo is None:
                creation_time = creation_time.replace(tzinfo=timezone.utc)
            if creation_time >= cutoff:
                continue

            try:
                print(f"Deleting stale hub: {hub_name}")
                _delete_hubs(sagemaker_session, hub_name)
            except Exception as e:
                print(f"Failed to delete hub {hub_name}: {e}")
    except Exception as e:
        print(f"Failed to cleanup old hubs: {e}")


def _delete_hubs(sagemaker_session, hub_name):
    # list and delete all hub contents first
    try:
        list_hub_content_response = sagemaker_session.list_hub_contents(
            hub_name=hub_name, hub_content_type=HubContentType.MODEL_REFERENCE.value
        )
        for model in list_hub_content_response["HubContentSummaries"]:
            _delete_hub_contents(sagemaker_session, hub_name, model)

        sagemaker_session.delete_hub(hub_name)
    except Exception as e:
        if "ResourceNotFound" in str(e):
            print(f"Hub {hub_name} does not exist, skipping deletion.")
        else:
            raise


@with_exponential_backoff()
def _delete_hub_contents(sagemaker_session, hub_name, model):
    sagemaker_session.delete_hub_content_reference(
        hub_name=hub_name,
        hub_content_type=HubContentType.MODEL_REFERENCE.value,
        hub_content_name=model["HubContentName"],
    )


def _hub_state_root(config):
    """Return the run-level tmp dir shared by the xdist controller and workers.

    The controller's basetemp is the run root (e.g. ``.../pytest-N``) while each
    worker's basetemp is a ``popen-gw*`` subdir of it. Normalizing to the run
    root gives every process the same location for the shared state file.

    Works across pytest versions: prefers the ``TempPathFactory`` attached as
    ``config._tmp_path_factory`` and falls back to the legacy ``_tmpdirhandler``.
    """
    factory = getattr(config, "_tmp_path_factory", None)
    if factory is not None:
        basetemp = pathlib.Path(str(factory.getbasetemp()))
    else:
        basetemp = pathlib.Path(str(config._tmpdirhandler.getbasetemp()))

    if basetemp.name.startswith("popen-gw"):
        return basetemp.parent
    return basetemp


@pytest.fixture(scope="session", autouse=True)
def setup(request):
    """Ensure a single shared private hub exists for the whole test run.

    Under pytest-xdist every worker is a separate process, so a naive
    ``scope="session"`` fixture would create one hub per worker. With high
    parallelism (e.g. ``-n 120``) that quickly exhausts the per-account private
    hub limit (100). All workers therefore coordinate through a lock file and a
    shared JSON state file: the first worker creates the hub, the rest reuse it.

    The hub is intentionally NOT deleted from a worker finalizer. xdist
    distributes tests dynamically, so a worker can finish its whole session (and
    run its finalizers) before another worker even reaches its first hub test;
    reference counting in that finalizer would delete the hub out from under
    workers still using it ("Hub ... does not exist" failures). Teardown instead
    runs exactly once, after all workers finish, in ``pytest_sessionfinish`` on
    the controller process.
    """
    root_tmp_dir = _hub_state_root(request.config)
    state_file = root_tmp_dir / "jumpstart_hub_state.json"
    lock_file = root_tmp_dir / "jumpstart_hub_state.json.lock"

    with FileLock(str(lock_file)):
        if state_file.is_file():
            state = json.loads(state_file.read_text())
        else:
            test_suite_id = get_test_suite_id()
            test_hub_name = f"{HUB_NAME_PREFIX}{test_suite_id}"
            _setup(test_suite_id=test_suite_id, test_hub_name=test_hub_name)
            state = {
                "test_suite_id": test_suite_id,
                "test_hub_name": test_hub_name,
            }
            state_file.write_text(json.dumps(state))

    # Ensure this worker's environment points at the shared hub.
    os.environ.update({ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID: state["test_suite_id"]})
    os.environ.update({ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME: state["test_hub_name"]})


def pytest_sessionfinish(session, exitstatus):
    """Tear down the shared hub once, after all xdist workers have finished.

    xdist workers carry a ``workerinput`` attribute on their config; only the
    controller (or a non-xdist run, which has no workerinput) performs teardown.
    Running here guarantees no worker is still using the hub.
    """
    if hasattr(session.config, "workerinput"):
        return  # xdist worker: the controller handles teardown.

    root_tmp_dir = _hub_state_root(session.config)
    state_file = root_tmp_dir / "jumpstart_hub_state.json"
    lock_file = root_tmp_dir / "jumpstart_hub_state.json.lock"

    with FileLock(str(lock_file)):
        if not state_file.is_file():
            return
        state = json.loads(state_file.read_text())
        try:
            _teardown(
                test_suite_id=state["test_suite_id"],
                test_hub_name=state["test_hub_name"],
            )
        finally:
            state_file.unlink()
