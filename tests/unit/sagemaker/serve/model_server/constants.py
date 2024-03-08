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

MOCK_MODEL_PATH = "/path/to/mock/model/dir"
MOCK_CODE_DIR = "/path/to/mock/model/dir/code"
MOCK_JUMPSTART_ID = "mock_llm_js_id"
MOCK_TMP_DIR = "tmp123456"
MOCK_COMPRESSED_MODEL_DATA_STR = (
    "s3://jumpstart-cache/to/infer-prepack-huggingface-llm-falcon-7b-bf16.tar.gz"
)
MOCK_UNCOMPRESSED_MODEL_DATA_STR = "s3://jumpstart-cache/to/artifacts/inference-prepack/v1.0.1/"
MOCK_UNCOMPRESSED_MODEL_DATA_STR_FOR_DICT = (
    "s3://jumpstart-cache/to/artifacts/inference-prepack/v1.0.1/dict/"
)
MOCK_UNCOMPRESSED_MODEL_DATA_DICT = {
    "S3DataSource": {
        "S3Uri": MOCK_UNCOMPRESSED_MODEL_DATA_STR_FOR_DICT,
        "S3DataType": "S3Prefix",
        "CompressionType": "None",
    }
}
MOCK_INVALID_MODEL_DATA_DICT = {}
