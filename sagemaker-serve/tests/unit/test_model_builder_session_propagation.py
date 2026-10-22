"""Unit tests for session propagation on ModelBuilder's model-reload paths (issue #6459).

``build()`` creates the SageMaker Model with the configured ``sagemaker_session``
and then reloads it with ``Model.get()``. A session-less ``Model.get()`` builds its
own boto3 session, so the DescribeModel call is signed with the default
credentials: cross-account builds create the model in the target account and then
fail to describe it. The same applies to the Model lookups used for resource reuse.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import boto3

from sagemaker.core.resources import Base, Model
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer

ROLE_ARN = "arn:aws:iam::111111111111:role/TargetAccountSageMakerRole"
REGION = "us-east-2"
IMAGE_URI = "111111111111.dkr.ecr.us-east-2.amazonaws.com/inference:latest"
MODEL_NAME = "modelbuilder-session-propagation"
MODEL_ARN = f"arn:aws:sagemaker:{REGION}:111111111111:model/{MODEL_NAME}"

# Two real, distinguishable boto3 sessions. CONFIGURED_BOTO_SESSION is what the
# caller hands to ModelBuilder (the target account); DEFAULT_BOTO_SESSION stands
# in for the process default session (a different account).
CONFIGURED_BOTO_SESSION = boto3.Session(
    region_name=REGION,
    aws_access_key_id="configured-key",
    aws_secret_access_key="configured-secret",
)
DEFAULT_BOTO_SESSION = boto3.Session(
    region_name=REGION,
    aws_access_key_id="default-key",
    aws_secret_access_key="default-secret",
)


def _configured_session():
    """A sagemaker_session wired to CONFIGURED_BOTO_SESSION (the target account)."""
    session = Mock()
    session.boto_region_name = REGION
    session.boto_session = CONFIGURED_BOTO_SESSION
    session.default_bucket.return_value = "test-bucket"
    session.default_bucket_prefix = None
    session.config = {}
    session.sagemaker_config = {}
    session.settings = Mock()
    session.settings.include_jumpstart_tags = False
    session.settings._local_download_dir = None
    session.sagemaker_client = MagicMock()
    return session


def _make_builder(session):
    builder = ModelBuilder(
        model=Mock(),
        model_server=ModelServer.TORCHSERVE,
        role_arn=ROLE_ARN,
        sagemaker_session=session,
        instance_type="ml.m5.large",
    )
    builder.model_name = MODEL_NAME
    builder.region = REGION
    return builder


@patch("sagemaker.serve.model_builder.resolve_nested_dict_value_from_config")
@patch("sagemaker.serve.model_builder.resolve_value_from_config")
@patch.object(ModelBuilder, "_init_sagemaker_session_if_does_not_exist")
@patch.object(ModelBuilder, "_ensure_base_name_if_needed")
@patch.object(ModelBuilder, "_prepare_container_def")
class TestCreateSageMakerModelReloadSession(unittest.TestCase):
    """The reload after CreateModel must use the configured session."""

    def _run(self, session, mocks):
        mock_prepare, mock_resolve, mock_resolve_nested = mocks
        mock_prepare.return_value = {"Image": IMAGE_URI, "Environment": {}}
        mock_resolve.side_effect = lambda value, *args, **kwargs: value
        mock_resolve_nested.side_effect = lambda value, *args, **kwargs: value

        builder = _make_builder(_configured_session())
        if session is not None:
            builder.sagemaker_session = session
        return builder, builder._create_sagemaker_model()

    def test_reload_passes_configured_session_to_model_get(
        self, mock_prepare, mock_base_name, mock_init, mock_resolve, mock_resolve_nested
    ):
        reloaded = Mock(model_name=MODEL_NAME, model_arn=MODEL_ARN)

        with patch(
            "sagemaker.serve.model_builder.Model.get", return_value=reloaded
        ) as mock_model_get:
            builder, result = self._run(None, (mock_prepare, mock_resolve, mock_resolve_nested))

        builder.sagemaker_session.create_model.assert_called_once()
        mock_model_get.assert_called_once_with(
            model_name=MODEL_NAME,
            session=CONFIGURED_BOTO_SESSION,
            region=REGION,
        )
        self.assertIs(result, reloaded)

    def test_reload_describe_client_is_built_from_configured_session(
        self, mock_prepare, mock_base_name, mock_init, mock_resolve, mock_resolve_nested
    ):
        """DescribeModel must be issued by a client built from the configured session.

        Model.get() resolves its client through Base.get_sagemaker_client; with no
        session that helper creates one from the ambient credential chain, which is
        the default account.
        """
        captured = {}

        def fake_get_client(session=None, region_name=None, service_name="sagemaker"):
            captured["session"] = session
            captured["region_name"] = region_name
            client = MagicMock()
            client.describe_model.return_value = {"ModelName": MODEL_NAME}
            return client

        with patch.object(Base, "get_sagemaker_client", side_effect=fake_get_client):
            _builder, result = self._run(None, (mock_prepare, mock_resolve, mock_resolve_nested))

        self.assertIs(captured["session"], CONFIGURED_BOTO_SESSION)
        self.assertIsNot(captured["session"], DEFAULT_BOTO_SESSION)
        self.assertEqual(captured["region_name"], REGION)
        self.assertIsInstance(result, Model)
        self.assertEqual(result.model_name, MODEL_NAME)

    def test_pipeline_session_build_still_returns_without_reloading(
        self, mock_prepare, mock_base_name, mock_init, mock_resolve, mock_resolve_nested
    ):
        """A PipelineSession build returns before the reload and must stay that way."""
        with patch("sagemaker.serve.model_builder.Model.get") as mock_model_get:
            _builder, result = self._run(
                Mock(spec=PipelineSession), (mock_prepare, mock_resolve, mock_resolve_nested)
            )

        self.assertIsNone(result)
        mock_model_get.assert_not_called()


class TestModelLookupSession(unittest.TestCase):
    """The Model lookups used for resource reuse must use the configured session."""

    def setUp(self):
        self.session = _configured_session()
        self.builder = _make_builder(self.session)
        self.found_model = Mock(model_name=MODEL_NAME, model_arn=MODEL_ARN)

    def test_find_reusable_model_passes_configured_session(self):
        with (
            patch.object(
                ModelBuilder, "_resolve_model_source_id", return_value="s3://bucket/model"
            ),
            patch(
                "sagemaker.serve.model_reuse.find_sagemaker_model_arn_by_tag",
                return_value=MODEL_ARN,
            ),
            patch(
                "sagemaker.serve.model_builder.Model.get", return_value=self.found_model
            ) as mock_model_get,
        ):
            result = self.builder._find_reusable_model()

        mock_model_get.assert_called_once_with(
            model_name=MODEL_NAME,
            session=CONFIGURED_BOTO_SESSION,
            region=REGION,
        )
        self.assertIs(result, self.found_model)

    def test_get_model_for_endpoint_passes_configured_session(self):
        client = self.session.sagemaker_client
        client.describe_endpoint.return_value = {"EndpointConfigName": "test-config"}
        client.describe_endpoint_config.return_value = {
            "ProductionVariants": [{"ModelName": MODEL_NAME}]
        }

        with patch(
            "sagemaker.serve.model_builder.Model.get", return_value=self.found_model
        ) as mock_model_get:
            result = self.builder._get_model_for_endpoint("test-endpoint")

        mock_model_get.assert_called_once_with(
            model_name=MODEL_NAME,
            session=CONFIGURED_BOTO_SESSION,
            region=REGION,
        )
        self.assertIs(result, self.found_model)


if __name__ == "__main__":
    unittest.main()
