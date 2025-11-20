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
"""S3 utilities for SageMaker."""
from __future__ import absolute_import

# Re-export from client
from sagemaker.core.s3.client import (  # noqa: F401
    S3Uploader,
    S3Downloader,
    parse_s3_url,
    is_s3_url,
    s3_path_join,
    determine_bucket_and_prefix,
)

# Re-export from utils (these are duplicated but kept for compatibility)
from sagemaker.core.s3.utils import (  # noqa: F401
    parse_s3_url as parse_s3_url_utils,
    is_s3_url as is_s3_url_utils,
    s3_path_join as s3_path_join_utils,
    determine_bucket_and_prefix as determine_bucket_and_prefix_utils,
)

__all__ = [
    "S3Uploader",
    "S3Downloader",
    "parse_s3_url",
    "is_s3_url",
    "s3_path_join",
    "determine_bucket_and_prefix",
]
