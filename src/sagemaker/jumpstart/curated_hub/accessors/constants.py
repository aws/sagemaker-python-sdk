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
"""This module accessors for the SageMaker JumpStart Curated Hub."""
from __future__ import absolute_import

PRIVATE_MODEL_UNCOMPRESSED_TRAINING_ARTIFACT_S3_SUFFIX = "train/"
PRIVATE_MODEL_TRAINING_ARTIFACT_TARBALL_S3_SUFFIX = "train/model.tar.gz"
PRIVATE_MODEL_TRAINING_SCRIPT_S3_SUFFIX = "train/sourcedir.tar.gz"
PRIVATE_MODEL_UNCOMPRESSED_HOSTING_ARTIFACT_S3_SUFFIX = "host/"
PRIVATE_MODEL_HOSTING_ARTIFACT_S3_TARBALL_SUFFIX = "host/model.tar.gz"
PRIVATE_MODEL_HOSTING_SCRIPT_S3_SUFFIX = "host/sourcedir.tar.gz"
PRIVATE_MODEL_INFERENCE_NOTEBOOK_S3_SUFFIX = "host/notebook.ipynb"

UNCOMPRESSED_ARTIFACTS_VALUE = "S3Prefix"
