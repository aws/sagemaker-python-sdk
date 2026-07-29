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
"""End-to-end: deploy a speculative-decoding / kernel-tuning model as an
Inference Component via ``ModelBuilder``.

Why this test exists
--------------------
Speculative-decoding (SD) and kernel-tuning (KT) optimized models host **two or
more model artifacts on one endpoint** — the base weights plus a draft model
(SD) or tuned-kernel artifacts (KT). On the model container these arrive as
``AdditionalModelDataSources``. Inference Components are the hosting surface
customers use to pack such multi-artifact models onto an endpoint, so
"deploy the optimized model as an Inference Component" is a first-class path.

``ModelBuilder`` deploys as an Inference Component when ``deploy()`` is given an
``inference_config=ResourceRequirements(...)``. This test builds a model
carrying the SD/KT ``AdditionalModelDataSources`` shape and deploys it that way,
asserting the Inference Component reaches ``InService`` — i.e. hosting accepts
and loads the multi-source model. It guards two failure modes: an Inference
Component that rejects the multi-source model at deploy time, and a deploy path
that silently collapses the additional sources into a single primary source
(the post-deploy shape assertion catches the latter).

Cost / scope
------------
Producing a genuine optimized ModelPackage inline is infeasible in CI (SD/KT
optimization requires substantial multi-GPU training capacity). So this test
builds a model that carries the SD/KT ``AdditionalModelDataSources`` shape
directly (base + draft channels) — the shape is the deploy-time contract under
test, not the specific optimized weights. Marked ``slow_test`` +
``gpu_intensive`` (one GPU Inference Component endpoint).
"""
from __future__ import absolute_import

import logging
import time
import uuid

import pytest

from sagemaker.core.enums import EndpointType
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.inference_config import ResourceRequirements
from sagemaker.core.resources import Endpoint, EndpointConfig, InferenceComponent, Model
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.core.training.configs import Compute

logger = logging.getLogger(__name__)

# Small, ungated, chat-templated model. Its readable S3 artifact stands in for
# the optimized recommendation's base + draft channels, and it fits a
# single-GPU g6.2xlarge Inference Component.
MODEL_ID = "huggingface-reasoning-qwen3-06b"
INSTANCE_TYPE = "ml.g6.2xlarge"
# Right-sized for a 0.6B on a single L4; a too-large request overflows the host
# and the IC never leaves Creating.
IC_MIN_MEMORY_MB = 4096
IC_NUM_ACCELERATORS = 1


def _additional_model_data_sources(s3_uri):
    """The AdditionalModelDataSources shape an SD/KT optimized model carries:
    base weights + a draft/tuned channel. The same readable artifact backs both
    — the IC deploy path's acceptance of the *shape* is what is under test, not
    the specific weights."""

    def _src(uri):
        return {
            "S3DataSource": {
                "S3Uri": uri,
                "S3DataType": "S3Prefix",
                "CompressionType": "None",
            }
        }

    return [
        {"ChannelName": "base_model", **_src(s3_uri)},
        {"ChannelName": "draft_model", **_src(s3_uri)},
    ]


@pytest.mark.slow_test
@pytest.mark.gpu_intensive
def test_deploy_sdkt_model_as_inference_component():
    """A model carrying SD/KT AdditionalModelDataSources deploys as an
    Inference Component and reaches InService via ``ModelBuilder.deploy``."""
    logger.info("Starting SD/KT deploy-as-Inference-Component integration test...")

    unique_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    role = get_execution_role(sagemaker_session=Session())
    src_model_name = f"air-sdkt-src-{unique_id}"
    ic_model_name = f"air-sdkt-icmodel-{unique_id}"
    endpoint_name = f"air-sdkt-ic-ep-{unique_id}"
    ic_name = f"air-sdkt-ic-{unique_id}"

    source_model = None
    endpoint = None

    try:
        # Build a source model just to obtain a readable image + S3 artifact for
        # this account/region (JumpStart resolves the container + weights).
        from sagemaker.core.jumpstart.configs import JumpStartConfig

        source_mb = ModelBuilder.from_jumpstart_config(
            jumpstart_config=JumpStartConfig(model_id=MODEL_ID),
            compute=Compute(instance_type=INSTANCE_TYPE),
            role_arn=role,
        )
        source_model = source_mb.build(model_name=src_model_name)
        src_container = Model.get(model_name=src_model_name)
        primary = getattr(src_container, "primary_container", None) or getattr(
            src_container, "containers", [None]
        )[0]
        base_s3 = _extract_s3_uri(primary)
        assert base_s3, f"Could not resolve a readable S3 artifact for {MODEL_ID}"
        logger.info("Resolved base artifact: %s", base_s3)

        # Build a model that carries the SD/KT additional sources, then deploy it
        # as an Inference Component (inference_config=ResourceRequirements routes
        # deploy() to the INFERENCE_COMPONENT_BASED path).
        ic_mb = ModelBuilder(model_path=base_s3, role_arn=role)
        ic_mb.additional_model_data_sources = _additional_model_data_sources(base_s3)
        ic_mb.build(model_name=ic_model_name)

        endpoint = ic_mb.deploy(
            endpoint_name=endpoint_name,
            inference_config=ResourceRequirements(
                requests={
                    "num_accelerators": IC_NUM_ACCELERATORS,
                    "memory": IC_MIN_MEMORY_MB,
                    "copies": 1,
                },
            ),
            inference_component_name=ic_name,
            instance_type=INSTANCE_TYPE,
            initial_instance_count=1,
            wait=True,
        )
        logger.info("Deploy returned; endpoint=%s ic=%s", endpoint_name, ic_name)

        # The Inference Component — referencing a model with base + draft
        # AdditionalModelDataSources — must be InService. This is the assertion
        # that turns red if hosting rejects the multi-source shape at
        # create/deploy time.
        ic = InferenceComponent.get(inference_component_name=ic_name)
        assert ic.inference_component_status == "InService", (
            f"Inference Component {ic_name} did not reach InService: "
            f"{ic.inference_component_status} / "
            f"{getattr(ic, 'failure_reason', None)}"
        )

        # And the model the IC references still carries the SD/KT additional
        # sources — i.e. we validated the multi-source path, not a silently
        # collapsed single-source one.
        referenced = ic.specification.model_name
        deployed_model = Model.get(model_name=referenced)
        channels = _additional_channel_names(deployed_model)
        assert {"base_model", "draft_model"}.issubset(channels), (
            f"IC model {referenced} lost its SD/KT additional sources "
            f"(channels={channels}); the deploy path must not collapse them."
        )
        logger.info("SD/KT model InService on IC with channels: %s", channels)

    finally:
        _delete_quietly(
            lambda: InferenceComponent.get(inference_component_name=ic_name),
            f"InferenceComponent {ic_name}",
            wait_gone=True,
        )
        _delete_quietly(
            lambda: Endpoint.get(endpoint_name=endpoint_name),
            f"Endpoint {endpoint_name}",
        )
        _delete_quietly(
            lambda: EndpointConfig.get(endpoint_config_name=endpoint_name),
            f"EndpointConfig {endpoint_name}",
        )
        _delete_quietly(
            lambda: Model.get(model_name=ic_model_name),
            f"Model {ic_model_name}",
        )
        if source_model:
            _delete_quietly(lambda: source_model, f"Model {src_model_name}")


def _extract_s3_uri(container):
    """Pull the S3 artifact URI off a resolved container (ModelDataSource or the
    legacy ModelDataUrl), tolerating the resource-object attribute shape."""
    if container is None:
        return None
    mds = getattr(container, "model_data_source", None)
    if mds is not None:
        s3 = getattr(mds, "s3_data_source", None)
        if s3 is not None and getattr(s3, "s3_uri", None):
            return s3.s3_uri
    return getattr(container, "model_data_url", None)


def _additional_channel_names(model):
    """Return the set of AdditionalModelDataSources channel names on a model."""
    primary = getattr(model, "primary_container", None) or getattr(
        model, "containers", [None]
    )[0]
    if primary is None:
        return set()
    sources = getattr(primary, "additional_model_data_sources", None) or []
    names = set()
    for s in sources:
        name = getattr(s, "channel_name", None)
        if name is None and isinstance(s, dict):
            name = s.get("ChannelName")
        if name:
            names.add(name)
    return names


def _delete_quietly(resource_factory, label, wait_gone=False):
    """Best-effort delete; log and continue on any failure. When ``wait_gone``,
    block until the resource is gone (an endpoint can't be deleted while it
    still hosts an Inference Component)."""
    try:
        resource = resource_factory()
        resource.delete()
        if wait_gone:
            waited = 0
            while waited < 10 * 60:
                try:
                    resource_factory()
                    time.sleep(20)
                    waited += 20
                except Exception:
                    break
        logger.info("Deleted %s", label)
    except Exception as exc:
        logger.warning("Failed to delete %s: %s", label, exc)
