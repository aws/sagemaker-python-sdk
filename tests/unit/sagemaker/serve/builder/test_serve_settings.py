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
from __future__ import absolute_import

from unittest.mock import Mock, patch
from unittest import TestCase
import os

from tests.unit import DATA_DIR

from sagemaker.config import load_sagemaker_config
from sagemaker.session_settings import SessionSettings
from sagemaker.serve.builder.serve_settings import _ServeSettings

REGION = "us-west-2"
ROLE_ARN = "role_arn"
S3_MODEL_DATA_URI = "s3uri"
INSTANCE_TYPE = "ml.c6i.4xlarge"
ENV_VAR = {"key": "value"}


def mock_session():
    session = Mock()
    session.settings = SessionSettings()
    session.boto_region_name = REGION
    session.sagemaker_config = None
    session._append_sagemaker_config_tags.return_value = []
    session.default_bucket_prefix = None

    return session


class ServeSettingsTest(TestCase):
    @patch("sagemaker.serve.builder.serve_settings.Session", return_value=mock_session())
    def test_serve_settings_with_config_file(self, session):
        session().sagemaker_config = load_sagemaker_config(
            additional_config_paths=[os.path.join(DATA_DIR, "serve_resources")]
        )
        serve_settings = _ServeSettings()

        self.assertEqual(
            serve_settings.role_arn, "arn:aws:iam::123456789012:role/service-role/testing-role"
        )

        self.assertEqual(
            serve_settings.role_arn, "arn:aws:iam::123456789012:role/service-role/testing-role"
        )
        self.assertEqual(serve_settings.s3_model_data_url, "s3://testing-s3-bucket")
        self.assertEqual(serve_settings.instance_type, "ml.m5.xlarge")
        self.assertTrue("EnvVarKey" in serve_settings.env_vars)
        self.assertEqual(serve_settings.env_vars.get("EnvVarKey"), "EnvVarValue")
        self.assertEqual(serve_settings.telemetry_opt_out, True)

    @patch("sagemaker.serve.builder.serve_settings.Session", return_value=mock_session())
    def test_serve_settings_with_config_file_overridden(self, session):
        session().sagemaker_config = load_sagemaker_config(
            additional_config_paths=[os.path.join(DATA_DIR, "serve_resources")]
        )
        serve_settings = _ServeSettings(
            role_arn=ROLE_ARN,
            s3_model_data_url=S3_MODEL_DATA_URI,
            instance_type=INSTANCE_TYPE,
            env_vars=ENV_VAR,
        )

        self.assertEqual(serve_settings.role_arn, ROLE_ARN)
        self.assertEqual(serve_settings.s3_model_data_url, S3_MODEL_DATA_URI)
        self.assertEqual(serve_settings.instance_type, INSTANCE_TYPE)
        self.assertTrue("key" in serve_settings.env_vars)
        self.assertEqual(serve_settings.env_vars.get("key"), "value")
        self.assertTrue("EnvVarKey" not in serve_settings.env_vars)
