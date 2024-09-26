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

from unittest import TestCase
from urllib.error import HTTPError
from unittest.mock import Mock, patch
from sagemaker.huggingface.llm_utils import (
    get_huggingface_model_metadata,
    download_huggingface_model_metadata,
)

MOCK_HF_ID = "mock_hf_id"
MOCK_HF_HUB_TOKEN = "mock_hf_hub_token"
MOCK_HF_MODEL_METADATA_JSON = {"mock_key": "mock_value"}


class LlmUtilsTests(TestCase):
    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    def test_huggingface_model_metadata_success(self, mock_json, mock_urllib):
        mock_json.load.return_value = MOCK_HF_MODEL_METADATA_JSON
        ret_json = get_huggingface_model_metadata(MOCK_HF_ID)

        mock_urllib.request.urlopen.assert_called_once_with(
            f"https://huggingface.co/api/models/{MOCK_HF_ID}"
        )
        self.assertEqual(ret_json["mock_key"], "mock_value")

    @patch("sagemaker.huggingface.llm_utils.urllib")
    @patch("sagemaker.huggingface.llm_utils.json")
    def test_huggingface_model_metadata_gated_success(self, mock_json, mock_urllib):
        mock_json.load.return_value = MOCK_HF_MODEL_METADATA_JSON
        mock_hf_model_metadata_url = Mock()
        mock_urllib.request.Request.side_effect = mock_hf_model_metadata_url

        ret_json = get_huggingface_model_metadata(MOCK_HF_ID, MOCK_HF_HUB_TOKEN)

        mock_urllib.request.Request.assert_called_once_with(
            f"https://huggingface.co/api/models/{MOCK_HF_ID}",
            None,
            {"Authorization": "Bearer " + MOCK_HF_HUB_TOKEN},
        )
        self.assertEqual(ret_json["mock_key"], "mock_value")

    @patch("sagemaker.huggingface.llm_utils.urllib")
    def test_huggingface_model_metadata_unauthorized_exception(self, mock_urllib):
        mock_urllib.request.urlopen.side_effect = HTTPError(
            code=401, msg="Unauthorized", url=None, hdrs=None, fp=None
        )
        with self.assertRaises(ValueError) as context:
            get_huggingface_model_metadata(MOCK_HF_ID)

        expected_error_msg = (
            "Trying to access a gated/private HuggingFace model without valid credentials. "
            "Please provide a HUGGING_FACE_HUB_TOKEN in env_vars"
        )
        self.assertEquals(expected_error_msg, str(context.exception))

    @patch("sagemaker.huggingface.llm_utils.urllib")
    def test_huggingface_model_metadata_general_exception(self, mock_urllib):
        mock_urllib.request.urlopen.side_effect = TimeoutError("timed out")
        with self.assertRaises(ValueError) as context:
            get_huggingface_model_metadata(MOCK_HF_ID)

        expected_error_msg = (
            f"Did not find model metadata for the following HuggingFace Model ID {MOCK_HF_ID}"
        )
        self.assertEquals(expected_error_msg, str(context.exception))

    @patch("huggingface_hub.snapshot_download")
    def test_download_huggingface_model_metadata(self, mock_snapshot_download):
        mock_snapshot_download.side_effect = None

        download_huggingface_model_metadata(MOCK_HF_ID, "local_path", MOCK_HF_HUB_TOKEN)

        mock_snapshot_download.assert_called_once_with(
            repo_id=MOCK_HF_ID, local_dir="local_path", token=MOCK_HF_HUB_TOKEN
        )

    @patch("importlib.util.find_spec")
    def test_download_huggingface_model_metadata_ex(self, mock_find_spec):
        mock_find_spec.side_effect = lambda *args, **kwargs: False

        self.assertRaisesRegex(
            ImportError,
            "Unable to import huggingface_hub, check if huggingface_hub is installed",
            lambda: download_huggingface_model_metadata(
                MOCK_HF_ID, "local_path", MOCK_HF_HUB_TOKEN
            ),
        )
