"""Unit tests for tagging the Model that ``ModelBuilder.build()`` creates (issue #6501).

``build()`` creates a SageMaker Model resource and sends ``tags=format_tags(self._tags)``,
but ``_tags`` was only ever set by ``deploy()``. Any caller that builds without deploying
-- a pipeline ``ModelStep``, batch transform, evaluation, an inference pipeline -- created
an untagged Model with no public way to fix it.
"""

import inspect
import unittest
from unittest.mock import MagicMock, Mock, patch

from sagemaker.core.shapes import Tag
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer

ROLE_ARN = "arn:aws:iam::123456789012:role/TestRole"
REGION = "us-east-2"
IMAGE_URI = "123456789012.dkr.ecr.us-east-2.amazonaws.com/inference:latest"
MODEL_NAME = "modelbuilder-build-tags"
BOTO_TAGS = [{"Key": "project", "Value": "example"}]


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
    return session


def _make_builder():
    builder = ModelBuilder(
        model=Mock(),
        model_server=ModelServer.TORCHSERVE,
        image_uri=IMAGE_URI,
        s3_model_data_url="s3://test-bucket/model.tar.gz",
        role_arn=ROLE_ARN,
        instance_type="ml.m5.large",
        sagemaker_session=_session(),
    )
    builder.model_name = MODEL_NAME
    builder.region = REGION
    return builder


class TestBuildAcceptsTags(unittest.TestCase):
    """``build()`` must expose a public way to tag the Model it creates."""

    def test_build_has_a_tags_parameter(self):
        self.assertIn("tags", inspect.signature(ModelBuilder.build).parameters)

    def _build_with(self, **kwargs):
        """Run build() with model creation stubbed, returning the tags it would send."""
        builder = _make_builder()
        seen = {}

        def _capture(*args, **call_kwargs):
            seen["tags"] = builder._tags
            builder.built_model = Mock(model_arn=f"arn:aws:sagemaker:{REGION}:1:model/x")
            return builder.built_model

        with patch.object(
            ModelBuilder, "_build_single_modelbuilder", side_effect=_capture
        ), patch.object(ModelBuilder, "_get_serve_setting"):
            builder.build(**kwargs)

        return builder, seen.get("tags")

    def test_boto_style_tags_reach_the_build_path(self):
        _, tags = self._build_with(tags=BOTO_TAGS)

        self.assertEqual(tags, BOTO_TAGS)

    def test_core_tag_objects_are_normalised_to_boto_keys(self):
        """A v3 ``Tag`` must reach CreateModel as {"Key", "Value"}, not {"key", "value"}."""
        _, tags = self._build_with(tags=[Tag(key="project", value="example")])

        self.assertEqual(tags, BOTO_TAGS)

    def test_mapping_form_is_expanded(self):
        _, tags = self._build_with(tags={"project": "example"})

        self.assertEqual(tags, BOTO_TAGS)

    def test_build_tags_merge_with_add_tags(self):
        builder = _make_builder()
        builder.add_tags({"Key": "team", "Value": "ml-platform"})
        seen = {}

        def _capture(*args, **kwargs):
            seen["tags"] = list(builder._tags)
            builder.built_model = Mock(model_arn=f"arn:aws:sagemaker:{REGION}:1:model/x")
            return builder.built_model

        with patch.object(
            ModelBuilder, "_build_single_modelbuilder", side_effect=_capture
        ), patch.object(ModelBuilder, "_get_serve_setting"):
            builder.build(tags=BOTO_TAGS)

        self.assertEqual(
            seen["tags"],
            [{"Key": "team", "Value": "ml-platform"}, {"Key": "project", "Value": "example"}],
        )

    def test_omitting_tags_leaves_them_unset(self):
        _, tags = self._build_with()

        self.assertIsNone(tags)


@patch("sagemaker.serve.model_builder.resolve_nested_dict_value_from_config")
@patch("sagemaker.serve.model_builder.resolve_value_from_config")
@patch.object(ModelBuilder, "_init_sagemaker_session_if_does_not_exist")
@patch.object(ModelBuilder, "_ensure_base_name_if_needed")
@patch.object(ModelBuilder, "_prepare_container_def")
class TestCreateModelRequestCarriesTags(unittest.TestCase):
    """The tags must survive into the CreateModel request, pipeline capture included."""

    def _create_model_kwargs(self, mocks, tags):
        mock_prepare, mock_resolve, mock_resolve_nested = mocks
        mock_prepare.return_value = {"Image": IMAGE_URI, "Environment": {}}
        mock_resolve.side_effect = lambda value, *args, **kwargs: value
        mock_resolve_nested.side_effect = lambda value, *args, **kwargs: value

        builder = _make_builder()
        # A PipelineSession captures the request instead of calling CreateModel,
        # which is the ModelStep path the defect was reported against.
        builder.sagemaker_session = Mock(spec=PipelineSession)
        builder._tags = tags

        builder._create_sagemaker_model()

        builder.sagemaker_session.create_model.assert_called_once()
        return builder.sagemaker_session.create_model.call_args.kwargs

    def test_pipeline_create_model_request_has_tags(
        self, mock_prepare, mock_base_name, mock_init, mock_resolve, mock_resolve_nested
    ):
        kwargs = self._create_model_kwargs(
            (mock_prepare, mock_resolve, mock_resolve_nested), BOTO_TAGS
        )

        self.assertEqual(kwargs["tags"], BOTO_TAGS)
