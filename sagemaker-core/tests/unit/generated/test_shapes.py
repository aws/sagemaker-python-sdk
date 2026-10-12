import ast
import datetime
import unittest
from unittest.mock import MagicMock, patch

from pydantic import BaseModel, ValidationError

import os
from sagemaker.core.resources import Base as ResourceBase, Endpoint
from sagemaker.core.shapes import Base, AdditionalS3DataSource, DataCaptureConfigSummary
from sagemaker.core.utils.utils import Unassigned

# Use the installed package location
import sagemaker.core.shapes

FILE_NAME = os.path.join(
    os.path.dirname(os.path.abspath(sagemaker.core.shapes.__file__)), "shapes.py"
)


class TestGeneratedShape(unittest.TestCase):
    def test_generated_shapes_have_pydantic_enabled(self):
        # This test ensures that all main shapes inherit Base which inherits BaseModel, thereby forcing pydantic validiation  # noqa: E501
        assert issubclass(Base, BaseModel)
        assert (
            self._fetch_number_of_classes_in_file_not_inheriting_a_class(FILE_NAME, "Base") == 1
        )  # 1 Because Base class itself does not inherit

    def test_pydantic_validation_for_generated_class_success(self):
        additional_s3_data_source = AdditionalS3DataSource(
            s3_data_type="filestring", s3_uri="s3/uri"
        )
        assert isinstance(additional_s3_data_source.s3_data_type, str)
        assert isinstance(additional_s3_data_source.s3_uri, str)
        assert isinstance(additional_s3_data_source.compression_type, Unassigned)

    def test_pydantic_validation_for_generated_class_success_with_optional_attributes_provided(
        self,
    ):
        additional_s3_data_source = AdditionalS3DataSource(
            s3_data_type="filestring", s3_uri="s3/uri", compression_type="zip"
        )
        assert isinstance(additional_s3_data_source.s3_data_type, str)
        assert isinstance(additional_s3_data_source.s3_uri, str)
        assert isinstance(additional_s3_data_source.compression_type, str)

    def test_pydantic_validation_for_generated_class_throws_error_for_incorrect_input(
        self,
    ):
        with self.assertRaises(ValidationError):
            AdditionalS3DataSource(s3_data_type="str", s3_uri=12)

    def _fetch_number_of_classes_in_file_not_inheriting_a_class(
        self, filepath: str, base_class_name: str
    ):
        count = 0
        with open(filepath, "r") as file:
            tree = ast.parse(file.read(), filename=filepath)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    if not any(base_class.id == base_class_name for base_class in node.bases):
                        count = count + 1
        return count


class TestDataCaptureConfigSummaryOptionalKmsKeyId(unittest.TestCase):
    """DescribeEndpoint omits DataCaptureConfig.KmsKeyId when data capture is enabled
    without a customer-managed KMS key (issue #5738)."""

    _DESCRIBE_ENDPOINT_RESPONSE = {
        "EndpointName": "my-endpoint",
        "EndpointArn": "arn:aws:sagemaker:us-west-2:111122223333:endpoint/my-endpoint",
        "EndpointConfigName": "my-endpoint-config",
        "EndpointStatus": "InService",
        "CreationTime": datetime.datetime(2026, 1, 1),
        "LastModifiedTime": datetime.datetime(2026, 1, 1),
        "DataCaptureConfig": {
            "EnableCapture": True,
            "CaptureStatus": "Started",
            "CurrentSamplingPercentage": 100,
            "DestinationS3Uri": "s3://my-bucket/data-capture",
        },
    }

    def test_shape_validates_without_kms_key_id(self):
        summary = DataCaptureConfigSummary(
            enable_capture=True,
            capture_status="Started",
            current_sampling_percentage=100,
            destination_s3_uri="s3://my-bucket/data-capture",
        )
        assert isinstance(summary.kms_key_id, Unassigned)

    def test_shape_accepts_kms_key_id(self):
        summary = DataCaptureConfigSummary(
            enable_capture=True,
            capture_status="Started",
            current_sampling_percentage=100,
            destination_s3_uri="s3://my-bucket/data-capture",
            kms_key_id="my-kms-key",
        )
        assert summary.kms_key_id == "my-kms-key"

    def test_endpoint_get_without_kms_key_id(self):
        client = MagicMock()
        client.describe_endpoint.return_value = self._DESCRIBE_ENDPOINT_RESPONSE
        # Endpoint.get() resolves its client via resources.Base.get_sagemaker_client.
        with patch.object(ResourceBase, "get_sagemaker_client", return_value=client):
            endpoint = Endpoint.get("my-endpoint")

        client.describe_endpoint.assert_called_once_with(EndpointName="my-endpoint")
        assert endpoint.data_capture_config.enable_capture is True
        assert endpoint.data_capture_config.destination_s3_uri == "s3://my-bucket/data-capture"
        assert isinstance(endpoint.data_capture_config.kms_key_id, Unassigned)
