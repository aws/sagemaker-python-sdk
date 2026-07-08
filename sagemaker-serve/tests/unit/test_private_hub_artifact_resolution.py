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


if __name__ == "__main__":
    unittest.main()
