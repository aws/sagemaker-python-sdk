"""Unit tests for deploying ModelBuilder endpoints onto reserved capacity.

Covers ``capacity_reservation_config`` on ``ModelBuilder.deploy()``, which binds the
endpoint's production variant to a SageMaker training plan (Flexible Training Plans)
or an ODCR-backed ML reservation.
"""

import inspect
import logging
import unittest
from unittest.mock import Mock, patch

import pytest

from sagemaker.core.enums import EndpointType
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.inference_config import ResourceRequirements
from sagemaker.core.resources import Endpoint
from sagemaker.core.shapes import ProductionVariantCapacityReservationConfig
from sagemaker.serve.model_builder import ModelBuilder, _capacity_reservation_config_shape
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.core.inference_config import ServerlessInferenceConfig

ROLE_ARN = "arn:aws:iam::123456789012:role/TestRole"
PLAN_ARN = "arn:aws:sagemaker:us-west-2:123456789012:training-plan/p5-for-endpoints"
EXPECTED_REQUEST = {
    "CapacityReservationPreference": "capacity-reservations-only",
    "MlReservationArn": PLAN_ARN,
}


def _plan_config():
    return ProductionVariantCapacityReservationConfig(
        capacity_reservation_preference="capacity-reservations-only",
        ml_reservation_arn=PLAN_ARN,
    )


class _CoreEndpointTestBase(unittest.TestCase):
    def setUp(self):
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-west-2"
        self.mock_session.endpoint_in_service_or_not = Mock(return_value=False)
        self.mock_session.create_endpoint_config = Mock(return_value="new-endpoint-config")

        patches = [
            patch("sagemaker.serve.model_builder.Endpoint.get", return_value=Mock(spec=Endpoint)),
            patch.object(ModelBuilder, "_wait_for_endpoint"),
            patch.object(
                ModelBuilder, "_resolve_inference_ami_version", side_effect=lambda **kw: None
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _builder(self):
        builder = ModelBuilder(
            model=Mock(),
            role_arn=ROLE_ARN,
            sagemaker_session=self.mock_session,
            mode=Mode.SAGEMAKER_ENDPOINT,
        )
        builder.built_model = Mock()
        builder.built_model.model_name = "test-model"
        builder.model_name = "test-model"
        builder.inference_component_name = None  # normally set during build()
        return builder

    def _deployed_variant(self):
        call = self.mock_session.endpoint_from_production_variants.call_args
        return call.kwargs["production_variants"][0]


class TestDeployCoreEndpointCapacityReservation(_CoreEndpointTestBase):
    def test_model_based_endpoint_gets_capacity_reservation_config(self):
        self._builder()._deploy_core_endpoint(
            instance_type="ml.p5.48xlarge",
            initial_instance_count=1,
            endpoint_name="ftp-endpoint",
            capacity_reservation_config=_plan_config(),
        )

        variant = self._deployed_variant()
        assert variant["InstanceType"] == "ml.p5.48xlarge"
        assert variant["CapacityReservationConfig"] == EXPECTED_REQUEST

    def test_dict_config_is_accepted(self):
        self._builder()._deploy_core_endpoint(
            instance_type="ml.p5.48xlarge",
            initial_instance_count=1,
            endpoint_name="ftp-endpoint",
            capacity_reservation_config={"MlReservationArn": PLAN_ARN},
        )

        assert self._deployed_variant()["CapacityReservationConfig"] == {
            "MlReservationArn": PLAN_ARN
        }

    def test_no_config_leaves_variant_unchanged(self):
        self._builder()._deploy_core_endpoint(
            instance_type="ml.m5.large",
            initial_instance_count=1,
            endpoint_name="plain-endpoint",
        )

        assert "CapacityReservationConfig" not in self._deployed_variant()

    def test_inference_component_endpoint_gets_capacity_reservation_config(self):
        resources = Mock(spec=ResourceRequirements)
        resources.num_cpus = None
        resources.copy_count = 1
        resources.get_compute_resource_requirements.return_value = {
            "NumberOfAcceleratorDevicesRequired": 8
        }

        self._builder()._deploy_core_endpoint(
            instance_type="ml.p5.48xlarge",
            initial_instance_count=1,
            endpoint_name="ftp-ic-endpoint",
            endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
            resources=resources,
            capacity_reservation_config=_plan_config(),
        )

        variant = self._deployed_variant()
        assert "ModelName" not in variant
        assert variant["CapacityReservationConfig"] == EXPECTED_REQUEST
        self.mock_session.create_inference_component.assert_called_once()

    def test_update_endpoint_creates_config_from_variant(self):
        self._builder()._deploy_core_endpoint(
            instance_type="ml.p5.48xlarge",
            initial_instance_count=2,
            endpoint_name="existing-endpoint",
            update_endpoint=True,
            capacity_reservation_config=_plan_config(),
        )

        call = self.mock_session.create_endpoint_config.call_args
        variant = call.kwargs["production_variants"][0]
        assert variant["ModelName"] == "test-model"
        assert variant["InitialInstanceCount"] == 2
        assert variant["CapacityReservationConfig"] == EXPECTED_REQUEST
        assert call.kwargs["name"].startswith("test-model")
        self.mock_session.update_endpoint.assert_called_once_with(
            "existing-endpoint", "new-endpoint-config"
        )
        self.mock_session.endpoint_from_production_variants.assert_not_called()

    def test_update_endpoint_calls_match_real_session_signatures(self):
        """Regression: the update path once passed kwargs create_endpoint_config doesn't take.

        A plain Mock accepts any arguments, so bind each recorded call to the real
        Session method signature to catch drift between the caller and the session API.
        """
        self._builder()._deploy_core_endpoint(
            instance_type="ml.m5.large",
            initial_instance_count=1,
            endpoint_name="existing-endpoint",
            update_endpoint=True,
        )

        for method_name in ("create_endpoint_config", "update_endpoint"):
            call = getattr(self.mock_session, method_name).call_args
            assert call is not None, f"{method_name} was not called"
            inspect.signature(getattr(Session, method_name)).bind(
                self.mock_session, *call.args, **call.kwargs
            )

    def test_existing_endpoint_ic_deploy_warns_capacity_reservation_ignored(self):
        self.mock_session.endpoint_in_service_or_not = Mock(return_value=True)
        resources = Mock(spec=ResourceRequirements)
        resources.num_cpus = None
        resources.copy_count = 1
        resources.get_compute_resource_requirements.return_value = {}

        with self.assertLogs("sagemaker", level=logging.WARNING) as logs:
            self._builder()._deploy_core_endpoint(
                instance_type="ml.p5.48xlarge",
                initial_instance_count=1,
                endpoint_name="existing-ic-endpoint",
                endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
                resources=resources,
                capacity_reservation_config=_plan_config(),
            )

        assert any("already exists" in line for line in logs.output)
        self.mock_session.endpoint_from_production_variants.assert_not_called()
        self.mock_session.create_inference_component.assert_called_once()

    def test_serverless_with_capacity_reservation_raises(self):
        with pytest.raises(ValueError, match="not supported for serverless"):
            self._builder()._deploy_core_endpoint(
                serverless_inference_config=ServerlessInferenceConfig(),
                capacity_reservation_config=_plan_config(),
            )
        self.mock_session.endpoint_from_production_variants.assert_not_called()

    def test_invalid_config_raises_before_creating_resources(self):
        with pytest.raises(ValueError, match="Unsupported capacity_reservation_config keys"):
            self._builder()._deploy_core_endpoint(
                instance_type="ml.p5.48xlarge",
                initial_instance_count=1,
                capacity_reservation_config={"ml_reservation_arn": PLAN_ARN},
            )
        self.mock_session.endpoint_from_production_variants.assert_not_called()


class TestDeployForwardsCapacityReservation(unittest.TestCase):
    def setUp(self):
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-west-2"

    def _builder(self):
        builder = ModelBuilder(
            model=Mock(),
            role_arn=ROLE_ARN,
            sagemaker_session=self.mock_session,
            mode=Mode.SAGEMAKER_ENDPOINT,
        )
        builder.built_model = Mock()
        return builder

    @patch.object(ModelBuilder, "_resolve_model_source_id", return_value=None)
    @patch.object(ModelBuilder, "_is_model_customization", return_value=False)
    @patch.object(ModelBuilder, "_deploy", return_value=Mock(spec=Endpoint))
    def test_deploy_forwards_config_to_deploy_path(self, mock_deploy, *_):
        config = _plan_config()

        self._builder().deploy(instance_type="ml.p5.48xlarge", capacity_reservation_config=config)

        assert mock_deploy.call_args.kwargs["capacity_reservation_config"] is config

    @patch.object(ModelBuilder, "_resolve_model_source_id", return_value=None)
    @patch.object(ModelBuilder, "_is_model_customization", return_value=False)
    @patch.object(ModelBuilder, "_deploy_for_ic", return_value=Mock(spec=Endpoint))
    def test_deployables_ic_path_forwards_config(self, mock_deploy_for_ic, *_):
        config = _plan_config()
        builder = self._builder()
        builder._deployables = {
            "InferenceComponents": [
                {"Name": "ic-1", "ResourceRequirements": Mock(), "Model": Mock()},
                {"Name": "ic-2", "ResourceRequirements": Mock(), "Model": Mock()},
            ]
        }

        builder.deploy(
            endpoint_name="multi-ic-endpoint",
            instance_type="ml.p5.48xlarge",
            capacity_reservation_config=config,
        )

        assert mock_deploy_for_ic.call_count == 2
        for call in mock_deploy_for_ic.call_args_list:
            assert call.kwargs["capacity_reservation_config"] is config

    @patch.object(ModelBuilder, "_resolve_model_source_id", return_value=None)
    @patch.object(ModelBuilder, "_is_model_customization", return_value=False)
    @patch.object(ModelBuilder, "_deploy_for_ic", return_value=Mock(spec=Endpoint))
    def test_deployables_ic_path_without_config_is_unchanged(self, mock_deploy_for_ic, *_):
        builder = self._builder()
        builder._deployables = {
            "InferenceComponents": [
                {"Name": "ic-1", "ResourceRequirements": Mock(), "Model": Mock()}
            ]
        }

        builder.deploy(endpoint_name="multi-ic-endpoint", instance_type="ml.p5.48xlarge")

        assert "capacity_reservation_config" not in mock_deploy_for_ic.call_args.kwargs

    @patch.object(ModelBuilder, "_deploy")
    def test_deploy_rejects_invalid_config_up_front(self, mock_deploy):
        with pytest.raises(TypeError):
            self._builder().deploy(
                instance_type="ml.p5.48xlarge", capacity_reservation_config=PLAN_ARN
            )
        mock_deploy.assert_not_called()


class TestCapacityReservationShapeHelper:
    def test_dict_becomes_shape(self):
        shape = _capacity_reservation_config_shape(dict(EXPECTED_REQUEST))
        assert isinstance(shape, ProductionVariantCapacityReservationConfig)
        assert shape.capacity_reservation_preference == "capacity-reservations-only"
        assert shape.ml_reservation_arn == PLAN_ARN

    def test_none_and_empty_return_none(self):
        assert _capacity_reservation_config_shape(None) is None
        assert _capacity_reservation_config_shape({}) is None


class TestNovaDeployCapacityReservation(unittest.TestCase):
    @patch("sagemaker.serve.model_builder.Endpoint.create")
    @patch("sagemaker.serve.model_builder.EndpointConfig.create")
    def test_nova_variant_gets_capacity_reservation_config(
        self, mock_config_create, mock_endpoint_create
    ):
        builder = ModelBuilder.__new__(ModelBuilder)
        builder.built_model = Mock()
        builder.built_model.model_name = "nova-model"
        builder.instance_type = "ml.p5.48xlarge"
        builder._tags = None
        builder._base_model_name = Mock(return_value="nova-lite")
        builder._fetch_model_package = Mock(return_value=None)
        mock_endpoint_create.return_value = Mock(spec=Endpoint)

        builder._deploy_nova_model(
            endpoint_name="nova-endpoint",
            wait=False,
            capacity_reservation_config=_plan_config(),
        )

        variant = mock_config_create.call_args.kwargs["production_variants"][0]
        assert variant.capacity_reservation_config.ml_reservation_arn == PLAN_ARN


class TestDeployForIcExistingComponent(unittest.TestCase):
    @patch("sagemaker.serve.model_builder.Endpoint.get", return_value=Mock(spec=Endpoint))
    def test_updating_existing_ic_warns_capacity_reservation_ignored(self, _):
        builder = ModelBuilder.__new__(ModelBuilder)
        builder.sagemaker_session = Mock()
        builder.region = "us-west-2"
        builder._does_ic_exist = Mock(return_value=True)
        builder._update_inference_component = Mock()

        with self.assertLogs("sagemaker", level=logging.WARNING) as logs:
            builder._deploy_for_ic(
                ic_data={"Name": "ic-1", "ResourceRequirements": Mock(), "Model": Mock()},
                endpoint_name="existing-endpoint",
                capacity_reservation_config=_plan_config(),
            )

        assert any("existing inference component ic-1" in line for line in logs.output)
        builder._update_inference_component.assert_called_once()
