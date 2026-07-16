"""
Unit tests for private hub artifact resolution fix.

Tests two defects:
1. from_jumpstart_config sets hub_name after __init__ already ran
   _initialize_jumpstart_config(), leaving hub_arn as None.
2. _build_for_jumpstart does not forward hub_arn to get_init_kwargs,
   so model data resolves from the public catalog instead of the private hub.
"""

import unittest
from unittest.mock import Mock, patch

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.core.training.configs import Compute
from sagemaker.core.jumpstart.configs import JumpStartConfig


MOCK_ROLE_ARN = "arn:aws:iam::123456789012:role/SageMakerRole"
MOCK_HUB_NAME = "my-private-hub"
MOCK_HUB_ARN = "arn:aws:sagemaker:us-east-1:123456789012:hub/my-private-hub"
MOCK_MODEL_ID = "huggingface-llm-phi-4-mini-instruct"
MOCK_MODEL_VERSION = "1.1.0"


def _mock_session():
    """Create a mock session that won't trigger real AWS calls."""
    session = Mock()
    session.boto_region_name = "us-east-1"
    session.sagemaker_config = None
    session.boto_session = Mock()
    session.boto_session.region_name = "us-east-1"
    return session


# Common patch to prevent __init__ from making real S3/API calls during
# instance type auto-detection and model ID validation.
_PATCH_IS_JS = patch.object(ModelBuilder, "_is_jumpstart_model_id", return_value=False)


class TestFromJumpStartConfigHubArnDerivation(unittest.TestCase):
    """Test that from_jumpstart_config correctly derives hub_arn from hub_name."""

    @_PATCH_IS_JS
    @patch("sagemaker.serve.model_builder._retrieve_model_deploy_kwargs", return_value={})
    @patch("sagemaker.core.jumpstart.utils.validate_model_id_and_get_type", return_value=None)
    @patch(
        "sagemaker.core.jumpstart.hub.utils.generate_hub_arn_for_init_kwargs",
        return_value=MOCK_HUB_ARN,
    )
    def test_hub_arn_derived_when_hub_name_set(
        self, mock_generate_arn, mock_validate, mock_deploy_kwargs, mock_is_js
    ):
        """hub_arn must be derived after hub_name is assigned in from_jumpstart_config."""
        js_config = JumpStartConfig(
            model_id=MOCK_MODEL_ID,
            model_version=MOCK_MODEL_VERSION,
            hub_name=MOCK_HUB_NAME,
        )

        mb = ModelBuilder.from_jumpstart_config(
            jumpstart_config=js_config,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=_mock_session(),
        )

        # The key assertion: hub_arn is populated, proving _initialize_jumpstart_config
        # ran after hub_name was set in from_jumpstart_config
        self.assertEqual(mb.hub_name, MOCK_HUB_NAME)
        self.assertEqual(mb.hub_arn, MOCK_HUB_ARN)
        # generate_hub_arn_for_init_kwargs must have been called with the hub_name
        mock_generate_arn.assert_called()

    @_PATCH_IS_JS
    @patch("sagemaker.serve.model_builder._retrieve_model_deploy_kwargs", return_value={})
    @patch(
        "sagemaker.core.jumpstart.hub.utils.generate_hub_arn_for_init_kwargs",
        return_value=MOCK_HUB_ARN,
    )
    @patch("sagemaker.core.jumpstart.utils.validate_model_id_and_get_type", return_value=None)
    def test_hub_arn_populated_end_to_end(
        self, mock_validate, mock_generate_arn, mock_deploy_kwargs, mock_is_js
    ):
        """End-to-end: hub_arn is correctly populated when hub_name is specified."""
        js_config = JumpStartConfig(
            model_id=MOCK_MODEL_ID,
            model_version=MOCK_MODEL_VERSION,
            hub_name=MOCK_HUB_NAME,
        )

        mb = ModelBuilder.from_jumpstart_config(
            jumpstart_config=js_config,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=_mock_session(),
        )

        self.assertEqual(mb.hub_name, MOCK_HUB_NAME)
        self.assertEqual(mb.hub_arn, MOCK_HUB_ARN)
        mock_generate_arn.assert_called()

    @_PATCH_IS_JS
    @patch("sagemaker.serve.model_builder._retrieve_model_deploy_kwargs", return_value={})
    @patch("sagemaker.core.jumpstart.utils.validate_model_id_and_get_type", return_value=None)
    def test_hub_arn_is_none_when_no_hub_name(self, mock_validate, mock_deploy_kwargs, mock_is_js):
        """hub_arn should remain None when hub_name is not provided."""
        js_config = JumpStartConfig(
            model_id=MOCK_MODEL_ID,
            model_version=MOCK_MODEL_VERSION,
        )

        mb = ModelBuilder.from_jumpstart_config(
            jumpstart_config=js_config,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=_mock_session(),
        )

        self.assertIsNone(mb.hub_name)
        self.assertIsNone(mb.hub_arn)


class TestBuildForJumpStartForwardsHubArn(unittest.TestCase):
    """Test that _build_for_jumpstart forwards hub_arn to get_init_kwargs."""

    def setUp(self):
        self.mock_session = _mock_session()

    @_PATCH_IS_JS
    @patch("sagemaker.core.jumpstart.utils.validate_model_id_and_get_type", return_value=None)
    @patch("sagemaker.core.jumpstart.factory.utils.get_init_kwargs")
    @patch("sagemaker.serve.model_builder.ModelBuilder._create_model")
    @patch("sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode")
    def test_hub_arn_forwarded_to_get_init_kwargs(
        self, mock_prepare, mock_create, mock_get_kwargs, mock_validate, mock_is_js
    ):
        """get_init_kwargs must receive hub_arn so model data resolves via private hub."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = (
            "763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.27.0-lmi10.0.0-cu124"
        )
        mock_init_kwargs.env = {}
        mock_init_kwargs.model_data = {
            "S3DataSource": {
                "S3Uri": "s3://my-private-hub-bucket/artifacts/model.tar.gz",
                "S3DataType": "S3Prefix",
                "CompressionType": "None",
            }
        }
        mock_init_kwargs.enable_network_isolation = None
        mock_get_kwargs.return_value = mock_init_kwargs

        mock_model = Mock()
        mock_create.return_value = mock_model

        builder = ModelBuilder(
            model=MOCK_MODEL_ID,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            mode=Mode.SAGEMAKER_ENDPOINT,
        )
        builder._optimizing = False
        builder.hub_name = MOCK_HUB_NAME
        builder.hub_arn = MOCK_HUB_ARN
        builder.model_version = MOCK_MODEL_VERSION

        builder._build_for_jumpstart()

        # Verify hub_arn was passed to get_init_kwargs
        mock_get_kwargs.assert_called_once()
        call_kwargs = mock_get_kwargs.call_args
        actual_hub_arn = call_kwargs.kwargs.get("hub_arn") or call_kwargs[1].get("hub_arn")
        self.assertEqual(actual_hub_arn, MOCK_HUB_ARN)

    @_PATCH_IS_JS
    @patch("sagemaker.core.jumpstart.utils.validate_model_id_and_get_type", return_value=None)
    @patch("sagemaker.core.jumpstart.factory.utils.get_init_kwargs")
    @patch("sagemaker.serve.model_builder.ModelBuilder._create_model")
    @patch("sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode")
    def test_hub_arn_none_when_no_private_hub(
        self, mock_prepare, mock_create, mock_get_kwargs, mock_validate, mock_is_js
    ):
        """When no private hub is configured, hub_arn should be None (public catalog)."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = (
            "763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.27.0-lmi10.0.0-cu124"
        )
        mock_init_kwargs.env = {}
        mock_init_kwargs.model_data = "s3://jumpstart-cache-prod-us-east-1/models/model.tar.gz"
        mock_init_kwargs.enable_network_isolation = None
        mock_get_kwargs.return_value = mock_init_kwargs

        mock_model = Mock()
        mock_create.return_value = mock_model

        builder = ModelBuilder(
            model=MOCK_MODEL_ID,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            mode=Mode.SAGEMAKER_ENDPOINT,
        )
        builder._optimizing = False
        builder.model_version = MOCK_MODEL_VERSION

        builder._build_for_jumpstart()

        # Verify hub_arn is NOT passed when no private hub (public catalog)
        mock_get_kwargs.assert_called_once()
        call_kwargs = mock_get_kwargs.call_args
        actual_hub_arn = call_kwargs.kwargs.get("hub_arn")
        self.assertIsNone(actual_hub_arn)

    @_PATCH_IS_JS
    @patch("sagemaker.core.jumpstart.utils.validate_model_id_and_get_type", return_value=None)
    @patch("sagemaker.core.jumpstart.factory.utils.get_init_kwargs")
    @patch("sagemaker.serve.model_builder.ModelBuilder._create_model")
    @patch("sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode")
    def test_private_hub_resolves_non_public_model_data(
        self, mock_prepare, mock_create, mock_get_kwargs, mock_validate, mock_is_js
    ):
        """With hub_arn set, model_data should resolve to private hub bucket, not public cache."""
        private_s3_uri = "s3://my-private-hub-bucket/hub-content/artifacts/model.tar.gz"
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = (
            "763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.27.0-lmi10.0.0-cu124"
        )
        mock_init_kwargs.env = {}
        mock_init_kwargs.model_data = {
            "S3DataSource": {
                "S3Uri": private_s3_uri,
                "S3DataType": "S3Prefix",
                "CompressionType": "None",
            }
        }
        mock_init_kwargs.enable_network_isolation = None
        mock_get_kwargs.return_value = mock_init_kwargs

        mock_model = Mock()
        mock_create.return_value = mock_model

        builder = ModelBuilder(
            model=MOCK_MODEL_ID,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            mode=Mode.SAGEMAKER_ENDPOINT,
        )
        builder._optimizing = False
        builder.hub_name = MOCK_HUB_NAME
        builder.hub_arn = MOCK_HUB_ARN
        builder.model_version = MOCK_MODEL_VERSION

        builder._build_for_jumpstart()

        # Confirm model data does NOT point to public JumpStart cache
        self.assertNotIn("jumpstart-cache-prod", builder.s3_model_data_url)
        self.assertEqual(builder.s3_model_data_url, private_s3_uri)


class TestDetectJumpStartImageForwardsHubArn(unittest.TestCase):
    """Test that _detect_jumpstart_image forwards hub_arn to get_init_kwargs."""

    def setUp(self):
        self.mock_session = _mock_session()

    @_PATCH_IS_JS
    @patch("sagemaker.core.jumpstart.utils.validate_model_id_and_get_type", return_value=None)
    @patch("sagemaker.serve.model_builder_utils.get_init_kwargs")
    def test_hub_arn_forwarded_in_detect_jumpstart_image(
        self, mock_get_kwargs, mock_validate, mock_is_js
    ):
        """_detect_jumpstart_image must pass hub_arn so private hub images resolve correctly."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = (
            "763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.27.0-lmi10.0.0-cu124"
        )
        mock_init_kwargs.get = lambda k: mock_init_kwargs.image_uri if k == "image_uri" else None
        mock_get_kwargs.return_value = mock_init_kwargs

        builder = ModelBuilder(
            model=MOCK_MODEL_ID,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
        )
        builder.hub_arn = MOCK_HUB_ARN
        builder.model_version = MOCK_MODEL_VERSION

        builder._detect_jumpstart_image()

        mock_get_kwargs.assert_called_once()
        call_kwargs = mock_get_kwargs.call_args
        actual_hub_arn = call_kwargs.kwargs.get("hub_arn") or call_kwargs[1].get("hub_arn")
        self.assertEqual(actual_hub_arn, MOCK_HUB_ARN)


class TestEndToEndPrivateHubFlow(unittest.TestCase):
    """Integration-style test: from_jumpstart_config with hub_name -> _build_for_jumpstart."""

    @_PATCH_IS_JS
    @patch("sagemaker.core.jumpstart.factory.utils.get_init_kwargs")
    @patch("sagemaker.serve.model_builder.ModelBuilder._create_model")
    @patch("sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode")
    @patch("sagemaker.serve.model_builder._retrieve_model_deploy_kwargs", return_value={})
    @patch("sagemaker.core.jumpstart.utils.validate_model_id_and_get_type", return_value=None)
    @patch(
        "sagemaker.core.jumpstart.hub.utils.generate_hub_arn_for_init_kwargs",
        return_value=MOCK_HUB_ARN,
    )
    def test_from_jumpstart_config_then_build_uses_private_hub(
        self,
        mock_generate_arn,
        mock_validate,
        mock_deploy_kwargs,
        mock_prepare,
        mock_create,
        mock_get_kwargs,
        mock_is_js,
    ):
        """Full flow: from_jumpstart_config with hub_name -> build -> hub_arn passed through."""
        private_s3_uri = "s3://private-hub-bucket/content/model.tar.gz"
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = (
            "763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.27.0-lmi10.0.0-cu124"
        )
        mock_init_kwargs.env = {}
        mock_init_kwargs.model_data = {
            "S3DataSource": {
                "S3Uri": private_s3_uri,
                "S3DataType": "S3Prefix",
                "CompressionType": "None",
            }
        }
        mock_init_kwargs.enable_network_isolation = None
        mock_get_kwargs.return_value = mock_init_kwargs

        mock_model = Mock()
        mock_create.return_value = mock_model

        js_config = JumpStartConfig(
            model_id=MOCK_MODEL_ID,
            model_version=MOCK_MODEL_VERSION,
            hub_name=MOCK_HUB_NAME,
        )

        mb = ModelBuilder.from_jumpstart_config(
            jumpstart_config=js_config,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=_mock_session(),
            compute=Compute(instance_type="ml.g5.xlarge"),
        )

        # Verify hub_arn was derived
        self.assertEqual(mb.hub_arn, MOCK_HUB_ARN)
        self.assertEqual(mb.hub_name, MOCK_HUB_NAME)

        # Now trigger build
        mb.mode = Mode.SAGEMAKER_ENDPOINT
        mb._optimizing = False
        mb._build_for_jumpstart()

        # Verify hub_arn was forwarded to get_init_kwargs
        mock_get_kwargs.assert_called_once()
        call_kwargs = mock_get_kwargs.call_args
        actual_hub_arn = call_kwargs.kwargs.get("hub_arn") or call_kwargs[1].get("hub_arn")
        self.assertEqual(actual_hub_arn, MOCK_HUB_ARN)

        # Verify model data points to private hub, not public cache
        self.assertEqual(mb.s3_model_data_url, private_s3_uri)
        self.assertNotIn("jumpstart-cache-prod", mb.s3_model_data_url)


MOCK_MODEL_REFERENCE_ARN = (
    "arn:aws:sagemaker:us-east-1:123456789012:hub-content/"
    "my-private-hub/ModelReference/huggingface-llm-phi-4-mini-instruct/1.1.0"
)
MOCK_HUB_CONTENT_NAME = "my-team-phi4-mini"
MOCK_IMAGE_URI = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.27.0-lmi10.0.0-cu124"
)


def _init_kwargs_mock(model_reference_arn):
    """Build a get_init_kwargs return value with an explicit model_reference_arn.

    Uses an explicit value (string or None) rather than relying on Mock
    auto-attributes, which are always truthy.
    """
    mock_init_kwargs = Mock()
    mock_init_kwargs.image_uri = MOCK_IMAGE_URI
    mock_init_kwargs.env = {}
    mock_init_kwargs.model_data = {
        "S3DataSource": {
            "S3Uri": "s3://jumpstart-cache-prod-us-east-1/artifacts/model/",
            "S3DataType": "S3Prefix",
            "CompressionType": "None",
        }
    }
    mock_init_kwargs.enable_network_isolation = None
    mock_init_kwargs.model_reference_arn = model_reference_arn
    return mock_init_kwargs


class TestModelReferenceArnPropagation(unittest.TestCase):
    """Regression tests: model_reference_arn resolved by get_init_kwargs must be
    propagated onto the builder so the container definition attaches
    S3DataSource.HubAccessConfig.HubContentArn in the CreateModel call.

    Without this propagation, the execution role is forced to have
    s3:GetObject on the public jumpstart-cache-prod bucket, defeating
    private hub brokered access. (Missed by PR #5985.)
    """

    def setUp(self):
        self.mock_session = _mock_session()

    def _build(self, model_reference_arn, hub_arn=MOCK_HUB_ARN, hub_content_name=None):
        with _PATCH_IS_JS, patch(
            "sagemaker.core.jumpstart.utils.validate_model_id_and_get_type",
            return_value=None,
        ), patch(
            "sagemaker.core.jumpstart.factory.utils.get_init_kwargs"
        ) as mock_get_kwargs, patch(
            "sagemaker.serve.model_builder.ModelBuilder._create_model"
        ) as mock_create, patch(
            "sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode"
        ):
            mock_get_kwargs.return_value = _init_kwargs_mock(model_reference_arn)
            mock_create.return_value = Mock()

            builder = ModelBuilder(
                model=MOCK_MODEL_ID,
                role_arn=MOCK_ROLE_ARN,
                sagemaker_session=self.mock_session,
                mode=Mode.SAGEMAKER_ENDPOINT,
            )
            builder._optimizing = False
            builder.model_version = MOCK_MODEL_VERSION
            if hub_arn:
                builder.hub_name = MOCK_HUB_NAME
                builder.hub_arn = hub_arn
            if hub_content_name:
                builder.hub_content_name = hub_content_name

            builder._build_for_jumpstart()
            return builder, mock_get_kwargs

    def test_model_reference_arn_propagated_from_init_kwargs(self):
        """The ARN resolved by the factory must land on the builder instance.

        This is the test that would have caught the v3.15.1-v3.16.0 regression:
        hub_arn was forwarded to get_init_kwargs (PR #5985), but the resulting
        model_reference_arn was dropped, so CreateModel never got HubAccessConfig.
        """
        builder, _ = self._build(model_reference_arn=MOCK_MODEL_REFERENCE_ARN)

        self.assertEqual(
            getattr(builder, "model_reference_arn", None),
            MOCK_MODEL_REFERENCE_ARN,
            "model_reference_arn from get_init_kwargs was not propagated to the "
            "builder; the container definition will be missing "
            "S3DataSource.HubAccessConfig.HubContentArn",
        )

    def test_model_reference_arn_absent_for_public_catalog(self):
        """No private hub: model_reference_arn must remain unset (public path unchanged)."""
        builder, _ = self._build(model_reference_arn=None, hub_arn=None)

        self.assertIsNone(getattr(builder, "model_reference_arn", None))


class TestContainerDefAttachesHubAccessConfig(unittest.TestCase):
    """The downstream attachment point: container_def() must inject
    HubAccessConfig.HubContentArn into the S3DataSource when a
    model_reference_arn is provided. Pure function, no AWS calls.
    """

    def test_hub_access_config_attached(self):
        from sagemaker.core.helper.session_helper import container_def

        c_def = container_def(
            MOCK_IMAGE_URI,
            model_data_url={
                "S3DataSource": {
                    "S3Uri": "s3://jumpstart-cache-prod-us-east-1/artifacts/model/",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            model_reference_arn=MOCK_MODEL_REFERENCE_ARN,
        )

        self.assertEqual(
            c_def["ModelDataSource"]["S3DataSource"]["HubAccessConfig"],
            {"HubContentArn": MOCK_MODEL_REFERENCE_ARN},
        )

    def test_no_hub_access_config_without_model_reference_arn(self):
        from sagemaker.core.helper.session_helper import container_def

        c_def = container_def(
            MOCK_IMAGE_URI,
            model_data_url={
                "S3DataSource": {
                    "S3Uri": "s3://jumpstart-cache-prod-us-east-1/artifacts/model/",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
        )

        self.assertNotIn("HubAccessConfig", c_def["ModelDataSource"]["S3DataSource"])


class TestHubContentNameSupport(unittest.TestCase):
    """Aliased hub content references: when the private hub content reference
    is named differently from the public model_id, the SDK must resolve hub
    content by its actual name (hub_content_name), not the model_id.
    """

    def setUp(self):
        self.mock_session = _mock_session()

    def test_hub_content_name_used_as_lookup_model_id(self):
        """When hub_content_name is set, get_init_kwargs must receive it as model_id."""
        helper = TestModelReferenceArnPropagation()
        helper.mock_session = self.mock_session
        _, mock_get_kwargs = helper._build(
            model_reference_arn=MOCK_MODEL_REFERENCE_ARN,
            hub_content_name=MOCK_HUB_CONTENT_NAME,
        )

        call_kwargs = mock_get_kwargs.call_args.kwargs
        self.assertEqual(call_kwargs.get("model_id"), MOCK_HUB_CONTENT_NAME)

    def test_model_id_used_when_no_hub_content_name(self):
        """Without hub_content_name, the model_id is used for the hub lookup."""
        helper = TestModelReferenceArnPropagation()
        helper.mock_session = self.mock_session
        _, mock_get_kwargs = helper._build(model_reference_arn=MOCK_MODEL_REFERENCE_ARN)

        call_kwargs = mock_get_kwargs.call_args.kwargs
        self.assertEqual(call_kwargs.get("model_id"), MOCK_MODEL_ID)

    @_PATCH_IS_JS
    @patch("sagemaker.serve.model_builder._retrieve_model_deploy_kwargs", return_value={})
    @patch("sagemaker.core.jumpstart.utils.validate_model_id_and_get_type", return_value=None)
    @patch(
        "sagemaker.core.jumpstart.hub.utils.generate_hub_arn_for_init_kwargs",
        return_value=MOCK_HUB_ARN,
    )
    def test_from_jumpstart_config_threads_hub_content_name(
        self, mock_generate_arn, mock_validate, mock_deploy_kwargs, mock_is_js
    ):
        """JumpStartConfig.hub_content_name must be threaded onto the builder."""
        js_config = JumpStartConfig(
            model_id=MOCK_MODEL_ID,
            model_version=MOCK_MODEL_VERSION,
            hub_name=MOCK_HUB_NAME,
            hub_content_name=MOCK_HUB_CONTENT_NAME,
        )

        mb = ModelBuilder.from_jumpstart_config(
            jumpstart_config=js_config,
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=_mock_session(),
        )

        self.assertEqual(
            getattr(mb, "hub_content_name", None), MOCK_HUB_CONTENT_NAME
        )


class TestCreateModelContainerDefinition(unittest.TestCase):
    """Full-chain test: after a private hub build, the container definition
    that the SDK sends to the CreateModel API must carry
    S3DataSource.HubAccessConfig.HubContentArn.

    This exercises the real path end to end at the unit level:
    _build_for_jumpstart -> self.model_reference_arn ->
    _prepare_container_def_base -> container_def -> CreateModel payload.
    No individual link is mocked between the builder attribute and the
    final dict. This is the payload-shape assertion that, had it existed,
    would have caught the regression that shipped in v3.15.1-v3.16.0.
    """

    def setUp(self):
        self.mock_session = _mock_session()
        # _prepare_container_def_base consults session config lookups that
        # iterate the config object; a bare Mock is not iterable.
        self.mock_session.config = None

    def _container_def_after_build(self, model_reference_arn, hub_arn=MOCK_HUB_ARN):
        helper = TestModelReferenceArnPropagation()
        helper.mock_session = self.mock_session
        builder, _ = helper._build(
            model_reference_arn=model_reference_arn, hub_arn=hub_arn
        )
        # A JumpStart hub deploy has no custom inference code to upload;
        # ensure the code-upload branch (which would call S3) is not taken.
        builder.source_dir = None
        builder.dependencies = None
        builder.entry_point = None
        builder.git_config = None
        # _prepare_container_def_base builds the exact container definition
        # dict passed to the CreateModel API as PrimaryContainer/Containers.
        return builder._prepare_container_def_base()

    def test_create_model_container_def_includes_hub_access_config(self):
        """Private hub build: CreateModel payload must include HubAccessConfig."""
        c_def = self._container_def_after_build(
            model_reference_arn=MOCK_MODEL_REFERENCE_ARN
        )

        self.assertIn("ModelDataSource", c_def)
        s3_data_source = c_def["ModelDataSource"]["S3DataSource"]
        self.assertEqual(
            s3_data_source.get("HubAccessConfig"),
            {"HubContentArn": MOCK_MODEL_REFERENCE_ARN},
            "The container definition sent to CreateModel is missing "
            "HubAccessConfig; the execution role would need s3:GetObject on "
            "the public JumpStart cache bucket, defeating private hub "
            "brokered access.",
        )

    def test_create_model_container_def_no_hub_access_config_for_public(self):
        """Public catalog build: CreateModel payload must NOT include HubAccessConfig."""
        c_def = self._container_def_after_build(
            model_reference_arn=None, hub_arn=None
        )

        self.assertIn("ModelDataSource", c_def)
        self.assertNotIn(
            "HubAccessConfig", c_def["ModelDataSource"]["S3DataSource"]
        )


if __name__ == "__main__":
    unittest.main()
