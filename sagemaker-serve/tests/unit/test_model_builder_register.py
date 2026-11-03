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
"""Tests for ModelBuilder.register() argument handling."""

from __future__ import absolute_import

from unittest.mock import MagicMock, patch

import pytest

from sagemaker.core.model_metrics import MetricsSource as CompatMetricsSource
from sagemaker.core.model_metrics import ModelMetrics as CompatModelMetrics
from sagemaker.core.shapes import MetricsSource, ModelMetrics, ModelQuality
from sagemaker.serve.model_builder import ModelBuilder

IMAGE_URI = "123456789012.dkr.ecr.us-west-2.amazonaws.com/img:latest"
MODEL_DATA_URL = "s3://bucket/model.tar.gz"
METRICS_S3_URI = "s3://bucket/evaluation.json"

EXPECTED_MODEL_METRICS = {
    "ModelQuality": {
        "Statistics": {
            "ContentType": "application/json",
            "S3Uri": METRICS_S3_URI,
        }
    }
}


@pytest.fixture(autouse=True)
def _ambient_execution_role():
    """Stand in a resolved serving role.

    ModelBuilder resolves a default serving role at construction when none is
    supplied; these tests build with a mock session and no role.
    """
    with patch(
        "sagemaker.serve.model_builder.resolve_and_validate_role",
        return_value="arn:aws:iam::123456789012:role/ambient-execution-role",
    ):
        yield


@pytest.fixture
def builder():
    """A ModelBuilder wired to a mock session, ready to register()."""
    session = MagicMock()
    # register() is telemetry-wrapped and reads sagemaker_config.
    session.sagemaker_config = {}
    session.boto_region_name = "us-west-2"
    mb = ModelBuilder(
        image_uri=IMAGE_URI,
        s3_model_data_url=MODEL_DATA_URL,
        sagemaker_session=session,
        instance_type="ml.m5.large",
    )
    mb.framework = None
    mb.framework_version = None
    return mb


def _register(mb, **kwargs):
    """Run register() offline and return the args passed to the registry helper."""
    with patch(
        "sagemaker.serve.model_builder.create_model_package_from_containers",
        return_value={"ModelPackageArn": "arn:aws:sagemaker:us-west-2:1:model-package/g/1"},
    ), patch(
        "sagemaker.serve.model_builder.update_container_with_inference_params",
        side_effect=lambda **kw: kw.get("container_def"),
    ), patch.object(
        mb, "_prepare_container_def", return_value={"Image": IMAGE_URI}
    ), patch(
        "sagemaker.serve.model_builder.get_model_package_args", return_value={}
    ) as mock_args:
        mb.register(model_package_group_name="test-group", **kwargs)
    return mock_args.call_args


class TestRegisterContentAndResponseTypes:
    """content_types / response_types are optional and must not raise.

    They are declared with None defaults, so omitting them must leave them out
    of the request rather than raising AttributeError.
    """

    def test_register_without_content_and_response_types(self, builder):
        """Omitting both must succeed and pass None through."""
        call_args = _register(builder)

        assert call_args.args[0] is None
        assert call_args.args[1] is None

    def test_register_without_content_and_response_types_leaves_builder_clean(self, builder):
        """Omitting both must not invent attributes on the builder."""
        _register(builder)

        assert not hasattr(builder, "content_types")
        assert not hasattr(builder, "response_types")

    def test_register_with_content_and_response_types(self, builder):
        """Supplying both must forward them to the registry helper."""
        call_args = _register(
            builder,
            content_types=["application/json"],
            response_types=["text/csv"],
        )

        assert call_args.args[0] == ["application/json"]
        assert call_args.args[1] == ["text/csv"]

    def test_register_reuses_types_from_earlier_registration(self, builder):
        """A later call that omits the types must reuse the earlier values."""
        _register(
            builder,
            content_types=["application/json"],
            response_types=["text/csv"],
        )

        call_args = _register(builder)

        assert call_args.args[0] == ["application/json"]
        assert call_args.args[1] == ["text/csv"]


class TestRegisterModelMetrics:
    """register() must accept the ModelMetrics type its signature declares."""

    def test_register_with_core_shape_model_metrics(self, builder):
        """sagemaker.core.shapes.ModelMetrics is the declared type and must work."""
        model_metrics = ModelMetrics(
            model_quality=ModelQuality(
                statistics=MetricsSource(
                    content_type="application/json",
                    s3_uri=METRICS_S3_URI,
                )
            )
        )

        call_args = _register(builder, model_metrics=model_metrics)

        assert call_args.kwargs["model_metrics"] is model_metrics

    def test_register_with_compatibility_model_metrics(self, builder):
        """The older compatibility class must keep working."""
        model_metrics = CompatModelMetrics(
            model_statistics=CompatMetricsSource(
                content_type="application/json",
                s3_uri=METRICS_S3_URI,
            )
        )

        call_args = _register(builder, model_metrics=model_metrics)

        assert call_args.kwargs["model_metrics"] is model_metrics


class TestRegisterModelMetricsRequestShape:
    """The serialized request must carry the CreateModelPackage ModelMetrics shape."""

    def _capture_request(self, mb, **kwargs):
        """Run register() without stubbing the registry helper and return the request."""
        with patch(
            "sagemaker.serve.model_builder.create_model_package_from_containers",
            return_value={"ModelPackageArn": "arn:aws:sagemaker:us-west-2:1:model-package/g/1"},
        ) as mock_create, patch(
            "sagemaker.serve.model_builder.update_container_with_inference_params",
            side_effect=lambda **kw: kw.get("container_def"),
        ), patch.object(
            mb, "_prepare_container_def", return_value={"Image": IMAGE_URI}
        ):
            mb.register(model_package_group_name="test-group", **kwargs)
        return mock_create.call_args.kwargs

    def test_core_shape_model_metrics_serializes_to_request_shape(self, builder):
        """The declared core shape must produce the documented ModelMetrics structure."""
        model_metrics = ModelMetrics(
            model_quality=ModelQuality(
                statistics=MetricsSource(
                    content_type="application/json",
                    s3_uri=METRICS_S3_URI,
                )
            )
        )

        request = self._capture_request(builder, model_metrics=model_metrics)

        assert request["model_metrics"] == EXPECTED_MODEL_METRICS

    def test_compatibility_model_metrics_serializes_to_request_shape(self, builder):
        """The compatibility class must keep producing the same ModelQuality block."""
        model_metrics = CompatModelMetrics(
            model_statistics=CompatMetricsSource(
                content_type="application/json",
                s3_uri=METRICS_S3_URI,
            )
        )

        request = self._capture_request(builder, model_metrics=model_metrics)

        assert request["model_metrics"]["ModelQuality"] == EXPECTED_MODEL_METRICS["ModelQuality"]

    def test_omitted_types_are_absent_from_request(self, builder):
        """Omitted content/response types must not appear in the request at all."""
        request = self._capture_request(builder)

        assert "content_types" not in request
        assert "response_types" not in request
