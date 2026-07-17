"""Unit tests for JumpStart default instance type resolution in ModelBuilder.

Regression tests for the bug where ``__post_init__`` ran
``_initialize_compute_config()`` before ``self.region`` was assigned,
causing the JumpStart spec-default lookup to fail with a silently
swallowed AttributeError and every JumpStart model to default to
``ml.m5.large`` instead of the spec's default instance type.
"""

import json
import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sagemaker.serve.model_builder import ModelBuilder

JS_MODEL_ID = "openai-reasoning-gpt-oss-120b"
SPEC_DEFAULT_INSTANCE_TYPE = "ml.g7e.2xlarge"
TEST_ROLE = "arn:aws:iam::123456789012:role/TestRole"

# Production-shaped JumpStart model spec, extracted from the master-v2 branch's
# tests/unit/sagemaker/jumpstart/constants.py BASE_SPEC fixture.
_SPEC_FIXTURE_PATH = pathlib.Path(__file__).parent / "data" / "jumpstart_base_spec.json"
# BASE_SPEC's real default is ml.m5.large, which collides with the buggy code's
# hardcoded fallback and cannot discriminate pass from fail. Override the
# discriminating field (and keep it in the supported list) so a resolved spec
# default is distinguishable from the fallback.
_FIXTURE_DEFAULT_INSTANCE_TYPE = "ml.g5.12xlarge"


def _load_fixture_spec_dict():
    spec = json.loads(_SPEC_FIXTURE_PATH.read_text())
    spec["default_inference_instance_type"] = _FIXTURE_DEFAULT_INSTANCE_TYPE
    # Exclude ml.m5.large (the buggy code's hardcoded fallback) from the
    # supported list so membership assertions discriminate red from green.
    spec["supported_inference_instance_types"] = [_FIXTURE_DEFAULT_INSTANCE_TYPE] + [
        t
        for t in spec.get("supported_inference_instance_types", [])
        if t != "ml.m5.large"
    ]
    return spec


def _get_fixture_spec(*args, **kwargs):
    """side_effect for JumpStartModelsAccessor.get_model_specs, mirroring the
    master-v2 get_spec_from_base_spec test convention."""
    from sagemaker.core.jumpstart.types import JumpStartModelSpecs

    return JumpStartModelSpecs(_load_fixture_spec_dict())



def _mock_session():
    """Mock sagemaker session following the package's shared-setup convention."""
    mock_session = Mock()
    mock_session.boto_region_name = "us-west-2"
    mock_session.config = {}
    mock_session.sagemaker_config = {}
    mock_session.boto_session = Mock()
    mock_session.boto_session.region_name = "us-west-2"
    return mock_session


def _build_jumpstart_builder(get_deploy_kwargs_mock=None, **builder_kwargs):
    """Construct a ModelBuilder for a JumpStart model with the spec lookup mocked.

    Returns (builder, get_deploy_kwargs_mock) so tests can assert on the lookup call.
    """
    if get_deploy_kwargs_mock is None:
        get_deploy_kwargs_mock = Mock(
            return_value=SimpleNamespace(instance_type=SPEC_DEFAULT_INSTANCE_TYPE)
        )

    with patch(
        "sagemaker.serve.model_builder_utils._ModelBuilderUtils._is_jumpstart_model_id",
        return_value=True,
    ), patch(
        "sagemaker.serve.model_builder_utils.get_deploy_kwargs",
        get_deploy_kwargs_mock,
    ), patch.object(
        ModelBuilder, "_initialize_jumpstart_config"
    ):
        builder = ModelBuilder(
            model=JS_MODEL_ID,
            role_arn=TEST_ROLE,
            sagemaker_session=_mock_session(),
            **builder_kwargs,
        )
    return builder, get_deploy_kwargs_mock


class TestJumpStartDefaultInstanceTypeResolution(unittest.TestCase):
    """Construction must resolve the JumpStart spec default instance type."""

    def test_construction_resolves_spec_default_instance_type(self):
        """No instance_type passed: builder must adopt the spec default, not a hardcoded guess."""
        builder, _ = _build_jumpstart_builder()

        self.assertEqual(builder.instance_type, SPEC_DEFAULT_INSTANCE_TYPE)
        self.assertFalse(builder._user_provided_instance_type)

    def test_spec_lookup_receives_resolved_region(self):
        """Region must be resolved before the spec lookup runs (init-order regression guard)."""
        builder, get_deploy_kwargs_mock = _build_jumpstart_builder()

        get_deploy_kwargs_mock.assert_called_once()
        _, called_kwargs = get_deploy_kwargs_mock.call_args
        self.assertEqual(called_kwargs.get("region"), "us-west-2")
        self.assertEqual(called_kwargs.get("model_id"), JS_MODEL_ID)

    def test_region_attribute_exists_before_compute_config(self):
        """self.region must exist by the time instance-type detection runs."""
        seen = {}
        original = ModelBuilder._get_default_instance_type

        def spy(self):
            seen["has_region"] = hasattr(self, "region")
            seen["region"] = getattr(self, "region", None)
            return original(self)

        with patch.object(ModelBuilder, "_get_default_instance_type", spy):
            _build_jumpstart_builder()

        self.assertTrue(seen.get("has_region"), "self.region missing during instance-type detection")
        self.assertEqual(seen.get("region"), "us-west-2")

    def test_user_provided_instance_type_wins_and_skips_lookup(self):
        """Explicit instance_type must be honored without consulting the spec."""
        builder, get_deploy_kwargs_mock = _build_jumpstart_builder(
            instance_type="ml.p5.48xlarge"
        )

        self.assertEqual(builder.instance_type, "ml.p5.48xlarge")
        self.assertTrue(builder._user_provided_instance_type)
        get_deploy_kwargs_mock.assert_not_called()

    def test_spec_lookup_failure_logs_warning_and_falls_back(self):
        """A genuine lookup failure must be logged, not silently swallowed."""
        failing = Mock(side_effect=RuntimeError("spec fetch exploded"))

        with self.assertLogs("sagemaker.core.utils.utils", level="WARNING") as logs:
            builder, _ = _build_jumpstart_builder(get_deploy_kwargs_mock=failing)

        self.assertEqual(builder.instance_type, "ml.m5.large")  # documented fallback
        self.assertTrue(
            any("spec fetch exploded" in line for line in logs.output),
            f"expected swallowed exception in warning log, got: {logs.output}",
        )


class TestDefaultInstanceTypePropagatesToDeploy(unittest.TestCase):
    """The resolved spec default must flow into the deploy chain unchanged."""

    def test_deploy_passes_spec_default_to_internal_deploy(self):
        """deploy() with no instance_type must hand the spec default to _deploy."""
        builder, _ = _build_jumpstart_builder()
        builder.built_model = Mock()

        with patch.object(ModelBuilder, "_is_model_customization", return_value=False), patch.object(
            ModelBuilder, "_deploy", return_value=Mock()
        ) as mock_deploy:
            builder.deploy(endpoint_name="test-endpoint")

        mock_deploy.assert_called_once()
        _, called_kwargs = mock_deploy.call_args
        self.assertEqual(called_kwargs.get("instance_type"), SPEC_DEFAULT_INSTANCE_TYPE)

    def test_deploy_core_endpoint_passes_instance_type_to_production_variant(self):
        """_deploy_core_endpoint must place the instance type into the ProductionVariant
        used for CreateEndpointConfig/CreateEndpoint."""
        builder, _ = _build_jumpstart_builder()
        builder.built_model = Mock()
        builder.built_model.model_name = "test-model"
        builder.model_name = "test-model"
        builder.sagemaker_session.endpoint_in_service_or_not = Mock(return_value=False)

        with patch(
            "sagemaker.serve.model_builder.session_helper.production_variant",
            return_value={"VariantName": "AllTraffic"},
        ) as mock_pv:
            try:
                builder._deploy_core_endpoint(
                    instance_type=builder.instance_type,
                    initial_instance_count=1,
                    endpoint_name="test-endpoint",
                    wait=False,
                )
            except Exception:
                # Downstream endpoint creation uses mocks; we only assert the
                # ProductionVariant handoff below.
                pass

        mock_pv.assert_called_once()
        called_args, called_kwargs = mock_pv.call_args
        passed_instance_type = (
            called_kwargs.get("instance_type")
            if "instance_type" in called_kwargs
            else called_args[1]
        )
        self.assertEqual(passed_instance_type, SPEC_DEFAULT_INSTANCE_TYPE)


class TestSpecFixtureDrivenResolution(unittest.TestCase):
    """End-to-end resolution against a production-shaped spec fixture.

    Unlike the classes above, these tests do NOT mock get_deploy_kwargs.
    They patch only the spec source (JumpStartModelsAccessor.get_model_specs,
    the master-v2 test convention) with a realistic spec document, so the
    real chain runs: _get_default_instance_type ->
    get_deploy_kwargs -> _add_instance_type_to_kwargs ->
    instance_types.retrieve_default -> spec.default_inference_instance_type.
    """

    def _build(self, **builder_kwargs):
        with patch(
            "sagemaker.core.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs",
            side_effect=_get_fixture_spec,
        ), patch(
            "sagemaker.serve.model_builder_utils._ModelBuilderUtils._is_jumpstart_model_id",
            return_value=True,
        ), patch.object(
            ModelBuilder, "_initialize_jumpstart_config"
        ):
            return ModelBuilder(
                model="pytorch-ic-mobilenet-v2",  # model_id inside the fixture
                role_arn=TEST_ROLE,
                sagemaker_session=_mock_session(),
                **builder_kwargs,
            )

    def test_spec_document_default_resolves_through_real_machinery(self):
        """The fixture's default_inference_instance_type must be resolved
        by the real retrieve_default chain, not any hardcoded value."""
        builder = self._build()

        self.assertEqual(builder.instance_type, _FIXTURE_DEFAULT_INSTANCE_TYPE)

    def test_resolved_default_is_in_spec_supported_list(self):
        """The resolved type must come from the spec's supported instance types."""
        builder = self._build()

        spec = _load_fixture_spec_dict()
        self.assertIn(builder.instance_type, spec["supported_inference_instance_types"])

    def test_explicit_instance_type_still_wins_over_spec_document(self):
        builder = self._build(instance_type="ml.p4d.24xlarge")

        self.assertEqual(builder.instance_type, "ml.p4d.24xlarge")


if __name__ == "__main__":
    unittest.main()
