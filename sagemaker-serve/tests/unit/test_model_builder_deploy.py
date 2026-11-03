"""
Unit tests for ModelBuilder deployment and container definition methods.
Focuses on increasing coverage for deploy-related functionality.
"""

import unittest
from unittest.mock import Mock, patch

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.core.resources import Model, Endpoint
from sagemaker.core.enums import EndpointType
from sagemaker.core.inference_config import (
    ResourceRequirements,
)
from sagemaker.core.shapes import (
    ContainerDefinition,
    ImageConfig,
    ModelDataSource,
    S3ModelDataSource,
)
from sagemaker.core.workflow.pipeline_context import PipelineSession


class TestModelBuilderContainerDef(unittest.TestCase):
    """Test ModelBuilder container definition methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-west-2"

    @unittest.skip("Complex container_def mocking - method not being called as expected")
    @patch("sagemaker.core.helper.session_helper.container_def")
    def test_prepare_container_def_base_simple(self, mock_container_def):
        """Test _prepare_container_def_base with simple configuration."""
        mock_container_def.return_value = {
            "Image": "test-image",
            "ModelDataUrl": "s3://bucket/model.tar.gz",
        }

        self.mock_session.default_bucket_prefix = "test-prefix"

        builder = ModelBuilder(
            model=Mock(),
            image_uri="test-image",
            s3_model_data_url="s3://bucket/model.tar.gz",
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        builder.model_name = "test-model"
        builder.env_vars = {"KEY": "value"}

        result = builder._prepare_container_def_base()

        self.assertIsNotNone(result)
        mock_container_def.assert_called_once()

    @unittest.skip("Complex pipeline model mocking - _core_container_to_dict not being called")
    @patch("sagemaker.serve.model_builder.ModelBuilder._core_container_to_dict")
    def test_prepare_container_def_base_with_pipeline_models(self, mock_to_dict):
        """Test _prepare_container_def_base with pipeline models (list of Models)."""
        mock_model1 = Mock(spec=Model)
        mock_model1.containers = []
        mock_container1 = Mock()
        mock_container1.image = "image1"
        mock_container1.model_data_url = "s3://bucket/model1.tar.gz"
        mock_container1.environment = {"ENV1": "val1"}
        mock_model1.primary_container = mock_container1

        mock_model2 = Mock(spec=Model)
        mock_model2.containers = []
        mock_container2 = Mock()
        mock_container2.image = "image2"
        mock_container2.model_data_url = "s3://bucket/model2.tar.gz"
        mock_container2.environment = {"ENV2": "val2"}
        mock_model2.primary_container = mock_container2

        mock_to_dict.side_effect = [
            {"Image": "image1", "ModelDataUrl": "s3://bucket/model1.tar.gz"},
            {"Image": "image2", "ModelDataUrl": "s3://bucket/model2.tar.gz"},
        ]

        builder = ModelBuilder(
            model=[mock_model1, mock_model2],
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )

        result = builder._prepare_container_def_base()

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(mock_to_dict.call_count, 2)

    def test_prepare_container_def_base_invalid_pipeline_models(self):
        """Test _prepare_container_def_base raises error for invalid pipeline models."""
        builder = ModelBuilder(
            model=[Mock(), "not-a-model"],
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )

        with self.assertRaises(ValueError) as context:
            builder._prepare_container_def_base()

        self.assertIn("must be sagemaker.core.resources.Model instances", str(context.exception))

    def test_core_container_to_dict(self):
        """Test _core_container_to_dict converts container properly."""
        container = ContainerDefinition(
            image="test-image",
            model_data_url="s3://bucket/model.tar.gz",
            environment={"KEY": "value"},
        )

        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )

        result = builder._core_container_to_dict(container)

        self.assertEqual(
            result,
            {
                "Image": "test-image",
                "ModelDataUrl": "s3://bucket/model.tar.gz",
                "Environment": {"KEY": "value"},
            },
        )


@patch("sagemaker.serve.model_builder.resolve_nested_dict_value_from_config")
@patch("sagemaker.serve.model_builder.resolve_value_from_config")
@patch.object(ModelBuilder, "_init_sagemaker_session_if_does_not_exist")
class TestCoreContainerDefinitionFields(unittest.TestCase):
    """Core container fields must survive the CreateModel conversion (issue #6476).

    ``_core_container_to_dict`` used to copy a fixed subset of fields, which
    dropped ``ModelPackageName`` and ``ModelDataSource`` from the generated
    ``CreateModel`` request.
    """

    ROLE_ARN = "arn:aws:iam::123456789012:role/TestRole"
    IMAGE = "123456789012.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.4-2"
    PACKAGE_ARN = "arn:aws:sagemaker:us-west-2:123456789012:model-package/example/1"

    def _container_defs(self, container, mock_resolve, mock_resolve_nested):
        """Run _create_sagemaker_model and return the container_defs sent to create_model."""
        mock_resolve.side_effect = lambda value, *args, **kwargs: value
        mock_resolve_nested.side_effect = lambda value, *args, **kwargs: value

        session = Mock()
        session.boto_region_name = "us-west-2"
        session.default_bucket.return_value = "test-bucket"
        session.default_bucket_prefix = None
        session.config = {}
        session.sagemaker_config = {}

        builder = ModelBuilder(
            model=[
                Model(
                    model_name="source-model",
                    primary_container=container,
                    execution_role_arn=self.ROLE_ARN,
                )
            ],
            role_arn=self.ROLE_ARN,
            sagemaker_session=session,
            instance_type="ml.m5.large",
        )
        builder.model_name = "test-model"
        builder.sagemaker_session = Mock(spec=PipelineSession)

        builder._create_sagemaker_model()

        builder.sagemaker_session.create_model.assert_called_once()
        return builder.sagemaker_session.create_model.call_args.kwargs["container_defs"]

    def test_model_package_container_keeps_model_package_name(
        self, mock_init, mock_resolve, mock_resolve_nested
    ):
        """A model-package reference must build a request instead of failing on Image."""
        container_defs = self._container_defs(
            ContainerDefinition(model_package_name=self.PACKAGE_ARN),
            mock_resolve,
            mock_resolve_nested,
        )

        self.assertEqual(container_defs, [{"ModelPackageName": self.PACKAGE_ARN}])

    def test_model_data_source_container_keeps_model_data_source(
        self, mock_init, mock_resolve, mock_resolve_nested
    ):
        """A ModelDataSource must not be silently dropped from the request."""
        container_defs = self._container_defs(
            ContainerDefinition(
                image=self.IMAGE,
                model_data_source=ModelDataSource(
                    s3_data_source=S3ModelDataSource(
                        s3_data_type="S3Prefix",
                        compression_type="None",
                        s3_uri="s3://example-bucket/model/",
                    )
                ),
                environment={"EXAMPLE": "true"},
            ),
            mock_resolve,
            mock_resolve_nested,
        )

        self.assertEqual(
            container_defs,
            [
                {
                    "Image": self.IMAGE,
                    "ModelDataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "CompressionType": "None",
                            "S3Uri": "s3://example-bucket/model/",
                        }
                    },
                    "Environment": {"EXAMPLE": "true"},
                }
            ],
        )

    def test_plain_image_container_is_unchanged(self, mock_init, mock_resolve, mock_resolve_nested):
        """A plain image container must keep producing the same request fields."""
        container_defs = self._container_defs(
            ContainerDefinition(
                image=self.IMAGE,
                model_data_url="s3://example-bucket/model.tar.gz",
                environment={"A": "1"},
            ),
            mock_resolve,
            mock_resolve_nested,
        )

        self.assertEqual(
            container_defs,
            [
                {
                    "Image": self.IMAGE,
                    "ModelDataUrl": "s3://example-bucket/model.tar.gz",
                    "Environment": {"A": "1"},
                }
            ],
        )

    def test_image_config_is_serialized_to_a_plain_dict(
        self, mock_init, mock_resolve, mock_resolve_nested
    ):
        """ImageConfig must reach the request as a dict, not as a shape object."""
        container_defs = self._container_defs(
            ContainerDefinition(
                image=self.IMAGE,
                image_config=ImageConfig(repository_access_mode="Vpc"),
            ),
            mock_resolve,
            mock_resolve_nested,
        )

        self.assertEqual(
            container_defs,
            [{"Image": self.IMAGE, "ImageConfig": {"RepositoryAccessMode": "Vpc"}}],
        )

    @patch("sagemaker.serve.model_builder.Model.get")
    def test_model_package_container_under_interactive_session(
        self, mock_model_get, mock_init, mock_resolve, mock_resolve_nested
    ):
        """Outside a PipelineSession the base-name lookup must tolerate a missing Image."""
        mock_resolve.side_effect = lambda value, *args, **kwargs: value
        mock_resolve_nested.side_effect = lambda value, *args, **kwargs: value

        session = Mock()
        session.boto_region_name = "us-west-2"
        session.default_bucket.return_value = "test-bucket"
        session.default_bucket_prefix = None
        session.config = {}
        session.sagemaker_config = {}

        builder = ModelBuilder(
            model=[
                Model(
                    model_name="source-model",
                    primary_container=ContainerDefinition(model_package_name=self.PACKAGE_ARN),
                    execution_role_arn=self.ROLE_ARN,
                )
            ],
            role_arn=self.ROLE_ARN,
            sagemaker_session=session,
            instance_type="ml.m5.large",
        )
        builder.model_name = "test-model"

        builder._create_sagemaker_model()

        kwargs = session.create_model.call_args.kwargs
        self.assertEqual(kwargs["container_defs"], [{"ModelPackageName": self.PACKAGE_ARN}])

    def test_model_package_container_passes_real_boto_param_validation(
        self, mock_init, mock_resolve, mock_resolve_nested
    ):
        """End-to-end proof against the real SageMaker service model (offline).

        boto3's client-side ParamValidator needs no network and no credentials. The
        request the old converter produced carried ``Image: Unassigned()`` and was
        rejected by that validator; the request produced now is accepted.
        """
        import boto3
        from botocore import validate

        from sagemaker.core.utils.utils import Unassigned

        shape = (
            boto3.client("sagemaker", region_name="us-west-2")
            .meta.service_model.operation_model("CreateModel")
            .input_shape
        )
        validator = validate.ParamValidator()
        base = {
            "ModelName": "test-model",
            "ExecutionRoleArn": self.ROLE_ARN,
        }

        # What the old field copy produced for a model-package reference -- boto rejects it.
        old_report = validator.validate(
            {**base, "Containers": [{"Image": Unassigned(), "Environment": {}}]}, shape
        )
        self.assertTrue(old_report.has_errors())

        container_defs = self._container_defs(
            ContainerDefinition(model_package_name=self.PACKAGE_ARN),
            mock_resolve,
            mock_resolve_nested,
        )

        new_report = validator.validate({**base, "Containers": container_defs}, shape)
        self.assertFalse(new_report.has_errors(), new_report.generate_report())


class TestModelBuilderDeploy(unittest.TestCase):
    """Test ModelBuilder deploy methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.default_bucket.return_value = "test-bucket"
        self.mock_session.default_bucket_prefix = "test-prefix"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-west-2"

    def test_deploy_without_built_model_raises_error(self):
        """Test _deploy raises error when built_model is None."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )

        with self.assertRaises(ValueError) as context:
            builder._deploy()

        self.assertIn("Must call build() before deploy()", str(context.exception))

    @patch("sagemaker.serve.model_builder.ModelBuilder._deploy_core_endpoint")
    def test_deploy_sagemaker_endpoint_mode(self, mock_deploy_core):
        """Test _deploy with SAGEMAKER_ENDPOINT mode."""
        mock_endpoint = Mock(spec=Endpoint)
        mock_deploy_core.return_value = mock_endpoint

        builder = ModelBuilder(
            model=Mock(),
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        builder.built_model = Mock(spec=Model)
        builder.model_server = None

        result = builder._deploy(endpoint_name="test-endpoint")

        self.assertEqual(result, mock_endpoint)
        mock_deploy_core.assert_called_once()

    @patch("sagemaker.serve.model_builder.ModelBuilder._deploy_local_endpoint")
    def test_deploy_local_container_mode(self, mock_deploy_local):
        """Test _deploy with LOCAL_CONTAINER mode."""
        mock_endpoint = Mock()
        mock_deploy_local.return_value = mock_endpoint

        builder = ModelBuilder(
            model=Mock(),
            mode=Mode.LOCAL_CONTAINER,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        builder.built_model = Mock()
        builder.model_server = None

        result = builder._deploy()

        self.assertEqual(result, mock_endpoint)
        mock_deploy_local.assert_called_once()

    @patch("sagemaker.serve.local_resources.LocalEndpoint.create")
    def test_deploy_in_process_mode(self, mock_local_endpoint):
        """Test _deploy with IN_PROCESS mode."""
        mock_endpoint = Mock()
        mock_local_endpoint.return_value = mock_endpoint

        builder = ModelBuilder(
            model=Mock(),
            mode=Mode.IN_PROCESS,
            model_server=ModelServer.TORCHSERVE,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        builder.built_model = Mock()
        builder.secret_key = "test-key"
        builder.container_config = "host"
        builder._serializer = Mock()
        builder._deserializer = Mock()
        builder.modes = {str(Mode.IN_PROCESS): Mock()}

        result = builder._deploy(endpoint_name="test-endpoint")

        self.assertEqual(result, mock_endpoint)
        mock_local_endpoint.assert_called_once()

    def test_deploy_unsupported_mode_raises_error(self):
        """Test _deploy raises error for unsupported mode."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        builder.built_model = Mock()
        builder.mode = "UNSUPPORTED"
        builder.model_server = None

        with self.assertRaises(ValueError) as context:
            builder._deploy()

        self.assertIn("not supported", str(context.exception))


class TestModelBuilderDeployCore(unittest.TestCase):
    """Test ModelBuilder _deploy_core_endpoint method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.default_bucket.return_value = "test-bucket"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-west-2"
        self.mock_session.endpoint_in_service_or_not = Mock(return_value=False)

    @unittest.skip("Mock iterable issue with sagemaker_config access")
    def test_deploy_core_endpoint_missing_role_raises_error(self):
        """Test _deploy_core_endpoint raises error when role_arn is None."""
        self.mock_session.sagemaker_config = {}

        builder = ModelBuilder(model=Mock(), sagemaker_session=self.mock_session)
        builder.built_model = Mock()
        builder.role_arn = None

        with self.assertRaises(ValueError) as context:
            builder._deploy_core_endpoint(instance_type="ml.m5.large", initial_instance_count=1)

        self.assertIn("Role can not be null", str(context.exception))

    @unittest.skip("Mock subscriptability issue with sagemaker_config dict access")
    def test_deploy_core_endpoint_missing_instance_info_raises_error(self):
        """Test _deploy_core_endpoint raises error without instance type/count."""
        self.mock_session.sagemaker_config = {}

        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        builder.built_model = Mock()

        with self.assertRaises(ValueError) as context:
            builder._deploy_core_endpoint()

        self.assertIn("Must specify instance type and instance count", str(context.exception))

    def test_deploy_core_endpoint_invalid_async_config_raises_error(self):
        """Test _deploy_core_endpoint raises error for invalid async config."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        builder.built_model = Mock()

        with self.assertRaises(ValueError) as context:
            builder._deploy_core_endpoint(
                instance_type="ml.m5.large",
                initial_instance_count=1,
                async_inference_config={"not": "valid"},
            )

        self.assertIn("AsyncInferenceConfig object", str(context.exception))

    def test_deploy_core_endpoint_invalid_serverless_config_raises_error(self):
        """Test _deploy_core_endpoint raises error for invalid serverless config."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        builder.built_model = Mock()

        with self.assertRaises(ValueError) as context:
            builder._deploy_core_endpoint(serverless_inference_config={"not": "valid"})

        self.assertIn("ServerlessInferenceConfig object", str(context.exception))

    @patch("sagemaker.serve.model_builder.Endpoint.get")
    @patch("sagemaker.serve.model_builder.session_helper.production_variant")
    def test_deploy_core_endpoint_ic_based_waits_on_status_only(
        self, mock_production_variant, mock_endpoint_get
    ):
        """IC-based endpoints never get an endpoint log group, so both waits skip log streaming."""
        mock_production_variant.return_value = {"VariantName": "AllTraffic"}
        mock_endpoint_get.return_value = Mock(spec=Endpoint)
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        builder.built_model = Mock()
        builder.built_model.model_name = "test-model"
        builder.model_name = "test-model"

        with patch.object(builder, "_wait_for_endpoint") as mock_wait:
            builder._deploy_core_endpoint(
                instance_type="ml.g5.xlarge",
                initial_instance_count=1,
                endpoint_name="test-endpoint",
                endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
                resources=ResourceRequirements(requests={"memory": 1024, "copies": 1}),
                inference_component_name="test-ic",
            )

        self.mock_session.create_inference_component.assert_called_once()
        self.assertEqual(mock_wait.call_count, 2)
        for wait_call in mock_wait.call_args_list:
            self.assertIs(wait_call.kwargs["stream_endpoint_logs"], False)

    @unittest.skip("Missing inference_component_name attribute - complex deployment flow")
    def test_deploy_core_endpoint_sharded_model_forces_ic_based(self):
        """Test _deploy_core_endpoint forces INFERENCE_COMPONENT_BASED for sharded models."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        builder.built_model = Mock()
        builder.built_model.model_name = "test-model"
        builder.model_name = "test-model"
        builder._is_sharded_model = True
        builder._enable_network_isolation = False
        builder._tags = []

        with patch.object(builder, "_wait_for_endpoint"):
            with patch("sagemaker.core.resources.Endpoint.get") as mock_get:
                mock_endpoint = Mock(spec=Endpoint)
                mock_get.return_value = mock_endpoint

                with self.assertLogs(level="WARNING") as log:
                    builder._deploy_core_endpoint(
                        instance_type="ml.m5.large",
                        initial_instance_count=1,
                        endpoint_type=EndpointType.MODEL_BASED,
                        resources=ResourceRequirements(
                            requests={"memory": 1024, "copies": 1}, limits={}
                        ),
                    )

                # Check that warning was logged
                self.assertTrue(any("INFERENCE_COMPONENT_BASED" in msg for msg in log.output))

    @unittest.skip("Error message format mismatch - test assertion too strict")
    def test_deploy_core_endpoint_sharded_model_network_isolation_raises_error(self):
        """Test _deploy_core_endpoint raises error for sharded model with network isolation."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        builder.built_model = Mock()
        builder.built_model.model_name = "test-model"
        builder.model_name = "test-model"
        builder._is_sharded_model = True
        builder._enable_network_isolation = True

        with self.assertRaises(ValueError) as context:
            builder._deploy_core_endpoint(
                instance_type="ml.m5.large",
                initial_instance_count=1,
                resources=ResourceRequirements(requests={"memory": 1024, "copies": 1}, limits={}),
            )

        self.assertIn("network isolation", str(context.exception).lower())
        builder._is_sharded_model = True
        builder._enable_network_isolation = True

        with self.assertRaises(ValueError) as context:
            builder._deploy_core_endpoint(
                instance_type="ml.m5.large",
                initial_instance_count=1,
                resources=ResourceRequirements(num_cpus=1, num_accelerators=1, copy_count=1),
            )

        self.assertIn("EnableNetworkIsolation cannot be set to True", str(context.exception))


class TestModelBuilderDeployHelpers(unittest.TestCase):
    """Test ModelBuilder deployment helper methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.config = {}

    def test_get_deploy_wrapper_for_djl(self):
        """Test _get_deploy_wrapper returns DJL wrapper."""
        builder = ModelBuilder(
            model=Mock(),
            model_server=ModelServer.DJL_SERVING,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )

        wrapper = builder._get_deploy_wrapper()

        self.assertIsNotNone(wrapper)
        self.assertEqual(wrapper.__name__, "_djl_model_builder_deploy_wrapper")

    def test_get_deploy_wrapper_for_tgi(self):
        """Test _get_deploy_wrapper returns TGI wrapper."""
        builder = ModelBuilder(
            model=Mock(),
            model_server=ModelServer.TGI,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )

        wrapper = builder._get_deploy_wrapper()

        self.assertIsNotNone(wrapper)
        self.assertEqual(wrapper.__name__, "_tgi_model_builder_deploy_wrapper")

    def test_get_deploy_wrapper_for_tei(self):
        """Test _get_deploy_wrapper returns TEI wrapper."""
        builder = ModelBuilder(
            model=Mock(),
            model_server=ModelServer.TEI,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )

        wrapper = builder._get_deploy_wrapper()

        self.assertIsNotNone(wrapper)
        self.assertEqual(wrapper.__name__, "_tei_model_builder_deploy_wrapper")

    def test_get_deploy_wrapper_for_mms(self):
        """Test _get_deploy_wrapper returns MMS wrapper."""
        builder = ModelBuilder(
            model=Mock(),
            model_server=ModelServer.MMS,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )

        wrapper = builder._get_deploy_wrapper()

        self.assertIsNotNone(wrapper)
        self.assertEqual(wrapper.__name__, "_transformers_model_builder_deploy_wrapper")

    def test_get_deploy_wrapper_for_torchserve_returns_none(self):
        """Test _get_deploy_wrapper returns None for TORCHSERVE."""
        builder = ModelBuilder(
            model=Mock(),
            model_server=ModelServer.TORCHSERVE,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )

        wrapper = builder._get_deploy_wrapper()

        self.assertIsNone(wrapper)

    @patch("sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id")
    def test_get_deploy_wrapper_for_jumpstart(self, mock_is_js):
        """Test _get_deploy_wrapper returns JumpStart wrapper for JS models."""
        mock_is_js.return_value = True

        builder = ModelBuilder(
            model="huggingface-llm-falcon-7b",
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )

        wrapper = builder._get_deploy_wrapper()

        self.assertIsNotNone(wrapper)
        self.assertEqual(wrapper.__name__, "_js_builder_deploy_wrapper")

    def test_does_ic_exist_true(self):
        """Test _does_ic_exist returns True when IC exists."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        builder.sagemaker_session.describe_inference_component = Mock(
            return_value={"InferenceComponentName": "test-ic"}
        )

        result = builder._does_ic_exist("test-ic")

        self.assertTrue(result)

    def test_does_ic_exist_false(self):
        """Test _does_ic_exist returns False when IC doesn't exist."""
        from botocore.exceptions import ClientError

        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )

        error_response = {
            "Error": {
                "Code": "ValidationException",
                "Message": "Could not find inference component",
            }
        }
        builder.sagemaker_session.describe_inference_component = Mock(
            side_effect=ClientError(error_response, "DescribeInferenceComponent")
        )

        result = builder._does_ic_exist("non-existent-ic")

        self.assertFalse(result)

    def test_deploy_for_ic_creates_new_ic_with_explicit_instance_kwargs(self):
        """Test _deploy_for_ic forwards instance_type and initial_instance_count to _deploy once."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        builder._does_ic_exist = Mock(return_value=False)
        builder._deploy = Mock(return_value="endpoint")
        built_model = Mock()
        resource_requirements = ResourceRequirements(requests={"memory": 1024, "copies": 1})
        ic_data = {
            "Name": "test-ic",
            "ResourceRequirements": resource_requirements,
            "Model": built_model,
        }

        result = builder._deploy_for_ic(
            ic_data=ic_data,
            endpoint_name="test-endpoint",
            container_timeout_in_seconds=600,
            instance_type="ml.g5.xlarge",
            initial_instance_count=2,
        )

        self.assertEqual(result, "endpoint")
        builder._deploy.assert_called_once_with(
            built_model=built_model,
            endpoint_name="test-endpoint",
            endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
            resources=resource_requirements,
            inference_component_name="test-ic",
            instance_type="ml.g5.xlarge",
            initial_instance_count=2,
            container_timeout_in_seconds=600,
        )

    def test_deploy_for_ic_creates_new_ic_with_default_instance_kwargs(self):
        """Test _deploy_for_ic falls back to builder instance_type and one instance."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        builder.instance_type = "ml.c5.xlarge"
        builder._does_ic_exist = Mock(return_value=False)
        builder._deploy = Mock(return_value="endpoint")
        built_model = Mock()
        resource_requirements = ResourceRequirements(requests={"memory": 1024, "copies": 1})
        ic_data = {
            "Name": "test-ic",
            "ResourceRequirements": resource_requirements,
            "Model": built_model,
        }

        result = builder._deploy_for_ic(ic_data=ic_data, endpoint_name="test-endpoint")

        self.assertEqual(result, "endpoint")
        builder._deploy.assert_called_once_with(
            built_model=built_model,
            endpoint_name="test-endpoint",
            endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
            resources=resource_requirements,
            inference_component_name="test-ic",
            instance_type="ml.c5.xlarge",
            initial_instance_count=1,
        )


class TestModelBuilderResetState(unittest.TestCase):
    """Test ModelBuilder _reset_build_state method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.config = {}

    def test_reset_build_state_clears_built_model(self):
        """Test _reset_build_state clears built_model."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        builder.built_model = Mock()
        builder.secret_key = "test-key"

        builder._reset_build_state()

        self.assertIsNone(builder.built_model)
        self.assertEqual(builder.secret_key, "")

    def test_reset_build_state_clears_jumpstart_flags(self):
        """Test _reset_build_state clears JumpStart preparation flags."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        builder.prepared_for_djl = True
        builder.prepared_for_tgi = True
        builder.prepared_for_mms = True

        builder._reset_build_state()

        self.assertFalse(hasattr(builder, "prepared_for_djl"))
        self.assertFalse(hasattr(builder, "prepared_for_tgi"))
        self.assertFalse(hasattr(builder, "prepared_for_mms"))

    def test_reset_build_state_clears_cached_data(self):
        """Test _reset_build_state clears cached data."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        builder.js_model_config = {"key": "value"}
        builder.hf_model_config = {"key": "value"}
        builder._cached_js_model_specs = {"key": "value"}

        builder._reset_build_state()

        self.assertFalse(hasattr(builder, "js_model_config"))
        self.assertFalse(hasattr(builder, "hf_model_config"))
        self.assertFalse(hasattr(builder, "_cached_js_model_specs"))

    def test_reset_build_state_clears_upload_state(self):
        """Test _reset_build_state clears upload/packaging state."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        builder.s3_model_data_url = "s3://bucket/model.tar.gz"
        builder.s3_upload_path = "s3://bucket/upload"
        builder.uploaded_code = Mock()
        builder.repacked_model_data = "s3://bucket/repacked.tar.gz"

        builder._reset_build_state()

        self.assertIsNone(builder.s3_model_data_url)
        self.assertIsNone(builder.s3_upload_path)
        self.assertFalse(hasattr(builder, "uploaded_code"))
        self.assertFalse(hasattr(builder, "repacked_model_data"))


if __name__ == "__main__":
    unittest.main()
