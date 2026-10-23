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
"""``ModelBuilder.deploy`` takes its endpoint options by name and rejects an unknown one.

Before this change ``data_capture_config``, ``endpoint_type``, and ``inference_component_name``
travelled through ``**kwargs`` and only the core endpoint path read them, so a fine-tuned or
Nova deploy dropped ``data_capture_config`` without a word. ``fallback_instance_types`` and
``tracing_config`` were not read anywhere, and a misspelled option was a silent no-op that
deployed an endpoint without it.
"""

from __future__ import absolute_import

import inspect
import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

from sagemaker.core.enums import EndpointType
from sagemaker.core.helper.session_helper import (
    Session,
    _production_variant_instance_types,
    production_variant,
)
from sagemaker.core.inference_config import ResourceRequirements, ServerlessInferenceConfig
from sagemaker.core.model_monitor.data_capture_config import DataCaptureConfig
from sagemaker.core.resources import Endpoint
from sagemaker.core.shapes import ProductionVariant
from sagemaker.core.utils.utils import Unassigned
from sagemaker.serve.model_builder import (
    DEPLOY_KWARGS,
    ModelBuilder,
    _EndpointOptions,
    _instance_pools,
    _with_instance_pools,
)
from sagemaker.serve.utils.types import ModelServer

ROLE_ARN = "arn:aws:iam::123456789012:role/TestRole"
REGION = "us-east-2"
IMAGE_URI = "123456789012.dkr.ecr.us-east-2.amazonaws.com/inference:latest"
TRACING = {"EnableTracing": True, "SamplingPercentage": 10}


def _session():
    session = Mock()
    session.boto_region_name = REGION
    session.default_bucket.return_value = "test-bucket"
    session.default_bucket_prefix = None
    session.config = {}
    session.sagemaker_config = {}
    session.settings = Mock(include_jumpstart_tags=False)
    session.settings._local_download_dir = None
    session.sagemaker_client = MagicMock()
    session.boto_session = Mock()
    session.boto_session.region_name = REGION
    session.endpoint_in_service_or_not.return_value = False
    return session


def _make_builder():
    builder = ModelBuilder(
        model=Mock(),
        model_server=ModelServer.TORCHSERVE,
        image_uri=IMAGE_URI,
        s3_model_data_url="s3://test-bucket/model.tar.gz",
        role_arn=ROLE_ARN,
        instance_type="ml.g5.xlarge",
        sagemaker_session=_session(),
    )
    builder.region = REGION
    builder.built_model = Mock(model_name="built-model")
    builder.model_name = "built-model"
    builder.model_server = None
    return builder


def _data_capture_config():
    return DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=25,
        destination_s3_uri="s3://test-bucket/capture",
        sagemaker_session=_session(),
    )


def _core_path_patches(builder):
    return [
        patch.object(ModelBuilder, "_is_model_customization", return_value=False),
        patch.object(ModelBuilder, "_resolve_model_source_id", return_value=None),
        patch.object(builder, "_wait_for_endpoint"),
        patch("sagemaker.serve.model_builder.Endpoint.get", return_value=Mock(spec=Endpoint)),
    ]


class TestDeploySignature(unittest.TestCase):
    """The options are named parameters and an unknown kwarg raises."""

    def test_deploy_names_the_endpoint_options(self):
        parameters = inspect.signature(ModelBuilder.deploy).parameters

        for name in (
            "data_capture_config",
            "tracing_config",
            "fallback_instance_types",
            "endpoint_type",
            "inference_component_name",
        ):
            self.assertIn(name, parameters)
            self.assertIsNone(parameters[name].default)

    def test_deploy_rejects_an_unknown_kwarg_before_any_service_call(self):
        builder = _make_builder()

        with pytest.raises(TypeError, match="unexpected keyword argument 'fallback_instance_type'"):
            builder.deploy(endpoint_name="ep", fallback_instance_type=["ml.g6.xlarge"])

        builder.sagemaker_session.endpoint_from_production_variants.assert_not_called()

    def test_deploy_kwargs_lists_every_name_a_deploy_path_reads(self):
        for name in ("volume_size", "kms_key", "serializer", "async_inference_config", "role"):
            self.assertIn(name, DEPLOY_KWARGS)
        for name in ("data_capture_config", "endpoint_type", "inference_component_name", "tags"):
            self.assertNotIn(name, DEPLOY_KWARGS)


class TestEndpointOptionsValidation(unittest.TestCase):
    """``_EndpointOptions.validate`` rejects a combination no deploy path can apply."""

    def test_data_capture_on_an_inference_component_endpoint_raises(self):
        options = _EndpointOptions(data_capture_config=_data_capture_config())

        with pytest.raises(ValueError, match="rejects data capture on an inference-component"):
            options.validate(ResourceRequirements(requests={"memory": 1024, "copies": 1}))

    def test_data_capture_config_type_is_checked(self):
        with pytest.raises(TypeError, match="DataCaptureConfig"):
            _EndpointOptions(data_capture_config={"EnableCapture": True}).validate(None)

    def test_fallback_instance_types_rules(self):
        with pytest.raises(ValueError, match="serverless"):
            _EndpointOptions(fallback_instance_types=["ml.g6.xlarge"]).validate(
                ServerlessInferenceConfig()
            )
        with pytest.raises(ValueError, match="at most 4"):
            _EndpointOptions(
                fallback_instance_types=[f"ml.g6.{size}xlarge" for size in (1, 2, 4, 8, 12)]
            ).validate(None)
        with pytest.raises(ValueError, match="repeats"):
            _EndpointOptions(fallback_instance_types=["ml.g6.xlarge", "ml.g6.xlarge"]).validate(
                None
            )
        with pytest.raises(TypeError, match="list of instance type names"):
            _EndpointOptions(fallback_instance_types="ml.g6.xlarge").validate(None)

    def test_endpoint_type_must_match_the_inference_config(self):
        with pytest.raises(ValueError, match="requires inference_config=ResourceRequirements"):
            _EndpointOptions(endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED).validate(None)
        with pytest.raises(ValueError, match="conflicts with a ResourceRequirements"):
            _EndpointOptions(endpoint_type=EndpointType.MODEL_BASED).validate(
                ResourceRequirements(requests={"memory": 1024, "copies": 1})
            )
        with pytest.raises(TypeError, match="EndpointType"):
            _EndpointOptions(endpoint_type="MODEL_BASED").validate(None)

    def test_as_deploy_kwargs_carries_only_the_set_options(self):
        options = _EndpointOptions(fallback_instance_types=["ml.g6.xlarge"], tracing_config=TRACING)

        self.assertEqual(
            options.as_deploy_kwargs(),
            {"tracing_config": TRACING, "fallback_instance_types": ["ml.g6.xlarge"]},
        )
        self.assertEqual(_EndpointOptions().as_deploy_kwargs(), {})


class TestInstancePools(unittest.TestCase):
    """``fallback_instance_types`` becomes ``InstancePools`` with the primary type first."""

    def test_instance_pools_put_the_primary_type_at_priority_one(self):
        self.assertEqual(
            _instance_pools("ml.g5.xlarge", ["ml.g6.xlarge", "ml.g5.xlarge", "ml.g6e.xlarge"]),
            [
                {"InstanceType": "ml.g5.xlarge", "Priority": 1},
                {"InstanceType": "ml.g6.xlarge", "Priority": 2},
                {"InstanceType": "ml.g6e.xlarge", "Priority": 3},
            ],
        )
        with pytest.raises(ValueError, match="requires an instance_type"):
            _instance_pools(None, ["ml.g6.xlarge"])

    def test_production_variant_carries_pools_in_place_of_the_instance_type(self):
        pools = _instance_pools("ml.g5.xlarge", ["ml.g6.xlarge"])

        variant = production_variant(
            "built-model", "ml.g5.xlarge", 2, volume_size=64, instance_pools=pools
        )

        self.assertEqual(variant["InstancePools"], pools)
        self.assertNotIn("InstanceType", variant)
        self.assertEqual(variant["InitialInstanceCount"], 2)
        self.assertEqual(variant["VolumeSizeInGB"], 64)
        self.assertEqual(
            _production_variant_instance_types(variant), ["ml.g5.xlarge", "ml.g6.xlarge"]
        )
        plain = production_variant("built-model", "ml.g5.xlarge", 1)
        self.assertEqual(plain["InstanceType"], "ml.g5.xlarge")
        self.assertNotIn("InstancePools", plain)

    def test_with_instance_pools_moves_a_core_variant_to_pools(self):
        variant = ProductionVariant(
            variant_name="AllTraffic",
            model_name="built-model",
            instance_type="ml.g5.xlarge",
            initial_instance_count=1,
        )

        pooled = _with_instance_pools(variant, ["ml.g6.xlarge"])

        self.assertEqual(
            [(pool.instance_type, pool.priority) for pool in pooled.instance_pools],
            [("ml.g5.xlarge", 1), ("ml.g6.xlarge", 2)],
        )
        self.assertIsInstance(pooled.instance_type, Unassigned)
        self.assertIs(_with_instance_pools(variant, None), variant)


class TestSessionTracingConfig(unittest.TestCase):
    """``Session`` sends ``TracingConfig`` only when the installed service model has it."""

    def _session(self, members):
        boto_session = Mock(region_name=REGION)
        client = MagicMock()
        client.meta.service_model.operation_model.return_value.input_shape.members = members
        session = Session(boto_session=boto_session, sagemaker_client=client)
        return session, client

    def test_tracing_config_reaches_create_endpoint_config(self):
        session, client = self._session({"TracingConfig": object(), "ProductionVariants": object()})
        variants = [production_variant("built-model", "ml.g5.xlarge", 1)]

        with patch.object(session, "_append_sagemaker_config_tags", return_value=[]):
            session.create_endpoint_config(
                name="cfg", production_variants=variants, tracing_config_dict=TRACING
            )

        request = client.create_endpoint_config.call_args.kwargs
        self.assertEqual(request["TracingConfig"], TRACING)
        self.assertEqual(request["EndpointConfigName"], "cfg")

    def test_tracing_config_raises_when_the_service_model_lacks_it(self):
        session, client = self._session({"ProductionVariants": object()})
        variants = [production_variant("built-model", "ml.g5.xlarge", 1)]

        with patch.object(session, "_append_sagemaker_config_tags", return_value=[]):
            with pytest.raises(ValueError, match="TracingConfig is not a CreateEndpointConfig"):
                session.create_endpoint_config(
                    name="cfg", production_variants=variants, tracing_config_dict=TRACING
                )

        client.create_endpoint_config.assert_not_called()


class TestCoreEndpointPath(unittest.TestCase):
    """The single-model path forwards every option to the endpoint configuration."""

    def test_options_reach_the_endpoint_config_request(self):
        builder = _make_builder()
        capture = _data_capture_config()
        seen = {}

        def _create(**kwargs):
            seen.update(kwargs)
            return "ep"

        builder.sagemaker_session.endpoint_from_production_variants = Mock(side_effect=_create)
        patches = _core_path_patches(builder)
        for active in patches:
            active.start()
        try:
            builder.deploy(
                endpoint_name="ep",
                wait=False,
                data_capture_config=capture,
                tracing_config=TRACING,
                fallback_instance_types=["ml.g6.xlarge", "ml.g6e.xlarge"],
                endpoint_type=EndpointType.MODEL_BASED,
            )
        finally:
            for active in patches:
                active.stop()

        self.assertEqual(seen["data_capture_config_dict"], capture._to_request_dict())
        self.assertEqual(seen["tracing_config_dict"], TRACING)
        variant = seen["production_variants"][0]
        self.assertEqual(
            variant["InstancePools"],
            [
                {"InstanceType": "ml.g5.xlarge", "Priority": 1},
                {"InstanceType": "ml.g6.xlarge", "Priority": 2},
                {"InstanceType": "ml.g6e.xlarge", "Priority": 3},
            ],
        )
        self.assertNotIn("InstanceType", variant)

    def test_default_path_sends_no_tracing_and_keeps_the_instance_type(self):
        builder = _make_builder()
        seen = {}

        def _create(**kwargs):
            seen.update(kwargs)
            return "ep"

        builder.sagemaker_session.endpoint_from_production_variants = Mock(side_effect=_create)
        patches = _core_path_patches(builder)
        for active in patches:
            active.start()
        try:
            builder.deploy(endpoint_name="ep", wait=False)
        finally:
            for active in patches:
                active.stop()

        self.assertIsNone(seen["tracing_config_dict"])
        self.assertIsNone(seen["data_capture_config_dict"])
        self.assertEqual(seen["production_variants"][0]["InstanceType"], "ml.g5.xlarge")

    def test_inference_component_name_names_the_component(self):
        builder = _make_builder()
        builder.sagemaker_session.endpoint_from_production_variants = Mock(return_value="ep")
        patches = _core_path_patches(builder) + [
            patch(
                "sagemaker.serve.model_builder.session_helper.production_variant",
                return_value={"VariantName": "AllTraffic"},
            )
        ]
        for active in patches:
            active.start()
        try:
            builder.deploy(
                endpoint_name="ep",
                wait=False,
                inference_config=ResourceRequirements(requests={"memory": 1024, "copies": 1}),
                inference_component_name="my-component",
            )
        finally:
            for active in patches:
                active.stop()

        created = builder.sagemaker_session.create_inference_component.call_args.kwargs
        self.assertEqual(created["inference_component_name"], "my-component")
        self.assertEqual(builder.inference_component_name, "my-component")

    def test_update_endpoint_creates_a_config_from_the_production_variant(self):
        builder = _make_builder()
        builder.sagemaker_session.create_endpoint_config = Mock(return_value="cfg")
        patches = _core_path_patches(builder)
        for active in patches:
            active.start()
        try:
            builder.deploy(
                endpoint_name="existing", update_endpoint=True, wait=False, tracing_config=TRACING
            )
        finally:
            for active in patches:
                active.stop()

        request = builder.sagemaker_session.create_endpoint_config.call_args.kwargs
        self.assertEqual(request["production_variants"][0]["InstanceType"], "ml.g5.xlarge")
        self.assertEqual(request["tracing_config_dict"], TRACING)
        builder.sagemaker_session.update_endpoint.assert_called_once_with("existing", "cfg")


class TestCoreResourcePaths(unittest.TestCase):
    """The fine-tuned, restricted, and Nova paths apply the options or raise."""

    def test_nova_deploy_applies_data_capture_and_instance_pools(self):
        builder = _make_builder()
        capture = _data_capture_config()
        seen = {}

        def _create(**kwargs):
            seen.update(kwargs)

        with (
            patch.object(ModelBuilder, "_base_model_name", return_value="nova-micro"),
            patch.object(ModelBuilder, "_fetch_model_package", return_value=None),
            patch("sagemaker.serve.model_builder.EndpointConfig.create", side_effect=_create),
            patch(
                "sagemaker.serve.model_builder.Endpoint.create", return_value=Mock(spec=Endpoint)
            ),
        ):
            builder._deploy_nova_model(
                endpoint_name="ep",
                wait=False,
                endpoint_options=_EndpointOptions(
                    data_capture_config=capture, fallback_instance_types=["ml.g6.xlarge"]
                ),
            )

        self.assertEqual(seen["data_capture_config"].initial_sampling_percentage, 25)
        self.assertEqual(seen["data_capture_config"].destination_s3_uri, "s3://test-bucket/capture")
        variant = seen["production_variants"][0]
        self.assertEqual(
            [(pool.instance_type, pool.priority) for pool in variant.instance_pools],
            [("ml.g5.xlarge", 1), ("ml.g6.xlarge", 2)],
        )

    def test_nova_deploy_rejects_an_inference_component_name(self):
        builder = _make_builder()

        with patch.object(ModelBuilder, "_is_nova_model", return_value=True):
            with pytest.raises(ValueError, match="creates no inference component"):
                builder._deploy_model_customization(
                    endpoint_name="ep",
                    wait=False,
                    endpoint_options=_EndpointOptions(inference_component_name="ic"),
                )

    def test_fine_tuned_inference_component_deploy_rejects_data_capture(self):
        builder = _make_builder()

        with (
            patch.object(ModelBuilder, "_is_nova_model", return_value=False),
            patch.object(ModelBuilder, "_fetch_model_package", return_value=Mock()),
            patch(
                "sagemaker.serve.utils.model_package_utils.is_restricted_model_package",
                return_value=False,
            ),
            patch.object(ModelBuilder, "_fetch_peft", return_value=None),
            patch.object(ModelBuilder, "_prepare_reused_model_customization_deployment_state"),
        ):
            with pytest.raises(ValueError, match="rejects data capture"):
                builder._deploy_model_customization(
                    endpoint_name="ep",
                    wait=False,
                    inference_config=ResourceRequirements(requests={"memory": 1024, "copies": 1}),
                    endpoint_options=_EndpointOptions(data_capture_config=_data_capture_config()),
                )

    def test_tracing_config_on_a_core_resource_path_goes_through_the_session(self):
        builder = _make_builder()
        builder.sagemaker_session.create_endpoint_config = Mock(return_value="ep")

        with (
            patch.object(ModelBuilder, "_base_model_name", return_value="nova-micro"),
            patch.object(ModelBuilder, "_fetch_model_package", return_value=None),
            patch("sagemaker.serve.model_builder.EndpointConfig.create") as core_create,
            patch(
                "sagemaker.serve.model_builder.Endpoint.create", return_value=Mock(spec=Endpoint)
            ),
        ):
            builder._deploy_nova_model(
                endpoint_name="ep",
                wait=False,
                endpoint_options=_EndpointOptions(tracing_config=TRACING),
            )

        core_create.assert_not_called()
        request = builder.sagemaker_session.create_endpoint_config.call_args.kwargs
        self.assertEqual(request["tracing_config_dict"], TRACING)
        self.assertEqual(request["production_variants"][0]["ModelName"], "built-model")

    def test_recommendation_deploy_rejects_the_endpoint_options(self):
        builder = _make_builder()
        builder._recommendation_job = Mock()

        with patch.object(ModelBuilder, "_deploy_recommendation") as deploy_recommendation:
            with pytest.raises(ValueError, match="cannot be applied by a recommendation deploy"):
                builder.deploy(fallback_instance_types=["ml.g6.xlarge"])

        deploy_recommendation.assert_not_called()
