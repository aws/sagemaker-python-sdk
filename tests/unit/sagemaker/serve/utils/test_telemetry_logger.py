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
import unittest
from unittest.mock import MagicMock, patch
from sagemaker.serve.utils.telemetry_logger import _send_telemetry

mock_session = MagicMock()


class TestTelemetryLogger(unittest.TestCase):
    @patch("sagemaker.serve.utils.telemetry_logger._requests_helper")
    @patch("sagemaker.serve.utils.telemetry_logger._get_accountId")
    def test_log_sucessfully(self, mocked_get_accountId, mocked_request_helper):
        mock_session.boto_session.region_name = "ap-south-1"
        mocked_get_accountId.return_value = "testAccountId"
        _send_telemetry("someStatus", 1, mock_session)
        mocked_request_helper.assert_called_with(
            "https://dev-exp-t-ap-south-1.s3.ap-south-1.amazonaws.com/"
            "telemetry?x-accountId=testAccountId&x-mode=1&x-status=someStatus",
            2,
        )

    @patch("sagemaker.serve.utils.telemetry_logger._get_accountId")
    def test_log_handle_exception(self, mocked_get_accountId):
        mocked_get_accountId.side_effect = Exception("Internal error")
        _send_telemetry("someStatus", 1, mock_session)
        self.assertRaises(Exception)


if __name__ == "__main__":
    unittest.main()
