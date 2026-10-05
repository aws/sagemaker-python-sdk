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
"""Integration test for the inference deployment step types.

Runs a single pipeline chaining ``EndpointConfigStep`` ->
``EndpointStep`` -> ``InferenceComponentStep`` end-to-end against the
real service: an inference-component-style endpoint config (no model
name on the variant, execution role on the config), an endpoint, and an
inference component carrying the container specification.

This test provisions a real endpoint instance for its duration; all
resources are deleted in the ``finally`` block.
"""

from __future__ import absolute_import

import os
import time
import uuid

import pytest

from sagemaker import image_uris
from sagemaker.workflow.endpoint_step import EndpointConfigStep, EndpointStep
from sagemaker.workflow.inference_component_step import InferenceComponentStep
from sagemaker.workflow.pipeline import Pipeline
from tests.integ import DATA_DIR

INSTANCE_TYPE = "ml.m5.xlarge"
EXECUTION_TIMEOUT_SECONDS = 45 * 60
POLL_SECONDS = 30


# sagemaker_session and pipeline_session come from tests/conftest.py, and role from
# the workflow conftest. They build their sessions on a boto3.Session carrying an
# explicit region, which a bare Session() would not have: CI runs the integ suite
# with AWS_DEFAULT_REGION unset.


def test_deployment_steps_execute_end_to_end(sagemaker_session, pipeline_session, role):
    """Chained EndpointConfig -> Endpoint -> InferenceComponent pipeline run."""
    stamp = uuid.uuid4().hex[:8]
    config_name = f"integ-deploy-cfg-{stamp}"
    endpoint_name = f"integ-deploy-ep-{stamp}"
    component_name = f"integ-deploy-ic-{stamp}"
    pipeline_name = f"integ-deploy-{stamp}"
    model_name = f"integ-deploy-model-{stamp}"

    # An inference component must reference a model that actually serves
    # ``/ping``, so use the XGBoost serving image with the checked-in churn
    # model artifact (the same artifact the transform-job integ test uses).
    # The model is created outside the pipeline: the step under test is
    # InferenceComponentStep, not model creation.
    region = sagemaker_session.boto_region_name
    image_uri = image_uris.retrieve("xgboost", region, "0.90-1")
    sm_client = sagemaker_session.sagemaker_client
    # Everything that creates a resource sits inside the try, so a failure part way
    # through still reaches cleanup. The model and the uploaded artifact are created
    # before the pipeline exists, so pipeline stays None until it does.
    pipeline = None
    try:
        model_data_url = sagemaker_session.upload_data(
            path=os.path.join(
                DATA_DIR, "experiment", "transform_job_materials", "xgb_model.tar.gz"
            ),
            key_prefix=f"integ-deploy/{stamp}",
        )
        sagemaker_session.create_model(
            name=model_name,
            role=role,
            container_defs={"Image": image_uri, "ModelDataUrl": model_data_url},
        )

        config_step_args = pipeline_session.create_endpoint_config(
            name=config_name,
            production_variants=[
                {
                    "VariantName": "AllTraffic",
                    "InstanceType": INSTANCE_TYPE,
                    "InitialInstanceCount": 1,
                    "ManagedInstanceScaling": {
                        "Status": "ENABLED",
                        "MinInstanceCount": 1,
                        "MaxInstanceCount": 1,
                    },
                    "RoutingConfig": {"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"},
                }
            ],
            role=role,
        )
        config_step = EndpointConfigStep(name="CreateConfig", step_args=config_step_args)

        # The service appends an execution-unique suffix to names created by these
        # steps, so downstream steps must reference the *created* resource via step
        # properties rather than the requested name.
        endpoint_step_args = pipeline_session.create_endpoint(
            endpoint_name=endpoint_name, config_name=config_step.properties.EndpointConfigName
        )
        endpoint_step = EndpointStep(
            name="CreateEndpoint", step_args=endpoint_step_args, depends_on=[config_step]
        )

        component_step_args = pipeline_session.create_inference_component(
            inference_component_name=component_name,
            endpoint_name=endpoint_step.properties.EndpointName,
            variant_name="AllTraffic",
            specification={
                "ModelName": model_name,
                "ComputeResourceRequirements": {
                    "NumberOfCpuCoresRequired": 1.0,
                    "MinMemoryRequiredInMb": 1024,
                },
            },
            runtime_config={"CopyCount": 1},
        )
        component_step = InferenceComponentStep(
            name="CreateComponent", step_args=component_step_args, depends_on=[endpoint_step]
        )

        pipeline = Pipeline(
            name=pipeline_name,
            steps=[config_step, endpoint_step, component_step],
            sagemaker_session=pipeline_session,
        )
        pipeline.upsert(role_arn=role)
        execution = pipeline.start()

        deadline = time.time() + EXECUTION_TIMEOUT_SECONDS
        status = None
        while time.time() < deadline:
            status = execution.describe()["PipelineExecutionStatus"]
            if status not in ("Executing", "Stopping"):
                break
            time.sleep(POLL_SECONDS)

        if status != "Succeeded":
            steps = sm_client.list_pipeline_execution_steps(
                PipelineExecutionArn=execution.arn,
            )["PipelineExecutionSteps"]
            details = "\n".join(
                f"{s['StepName']}: {s.get('StepStatus')} {s.get('FailureReason', '')}"
                for s in steps
            )
            pytest.fail(f"Pipeline execution ended in status {status}. Steps:\n{details}")

        # Resolve the actual server-side names (the steps suffix them).
        actual_config = _resolve(
            sm_client.list_endpoint_configs(NameContains=stamp)["EndpointConfigs"],
            "EndpointConfigName",
        )
        actual_endpoint = _resolve(
            sm_client.list_endpoints(NameContains=stamp)["Endpoints"], "EndpointName"
        )
        actual_component = _resolve(
            sm_client.list_inference_components(NameContains=stamp)["InferenceComponents"],
            "InferenceComponentName",
        )

        assert (
            sm_client.describe_endpoint_config(EndpointConfigName=actual_config)[
                "EndpointConfigName"
            ]
            == actual_config
        )
        assert sm_client.describe_endpoint(EndpointName=actual_endpoint)["EndpointStatus"] == (
            "InService"
        )
        component_desc = sm_client.describe_inference_component(
            InferenceComponentName=actual_component
        )
        assert component_desc["EndpointName"] == actual_endpoint
    finally:
        _cleanup(sagemaker_session, sm_client, stamp, pipeline, model_name)


def _resolve(items, key):
    """Return the single matching resource name from a list_* response."""
    assert len(items) == 1, f"expected exactly one {key} for this run, got {items}"
    return items[0][key]


def _cleanup(sagemaker_session, sm_client, stamp, pipeline, model_name):
    """Delete every resource this run created, in dependency order."""
    for component in sm_client.list_inference_components(NameContains=stamp)["InferenceComponents"]:
        name = component["InferenceComponentName"]
        try:
            sm_client.delete_inference_component(InferenceComponentName=name)
            _wait_component_deleted(sm_client, name)
        except Exception:  # noqa: BLE001 -- best-effort cleanup
            pass
    for endpoint in sm_client.list_endpoints(NameContains=stamp)["Endpoints"]:
        try:
            sm_client.delete_endpoint(EndpointName=endpoint["EndpointName"])
        except Exception:  # noqa: BLE001
            pass
    for config in sm_client.list_endpoint_configs(NameContains=stamp)["EndpointConfigs"]:
        try:
            sm_client.delete_endpoint_config(EndpointConfigName=config["EndpointConfigName"])
        except Exception:  # noqa: BLE001
            pass
    try:
        pipeline.delete()
    except Exception:  # noqa: BLE001
        pass
    try:
        sm_client.delete_model(ModelName=model_name)
    except Exception:  # noqa: BLE001
        pass
    try:
        sagemaker_session.boto_session.client("s3").delete_object(
            Bucket=sagemaker_session.default_bucket(),
            Key=f"integ-deploy/{stamp}/xgb_model.tar.gz",
        )
    except Exception:  # noqa: BLE001
        pass


def _wait_component_deleted(sm_client, component_name, timeout_seconds=10 * 60):
    """The endpoint cannot be deleted until its inference component is gone."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            sm_client.describe_inference_component(InferenceComponentName=component_name)
        except Exception:
            return
        time.sleep(15)
