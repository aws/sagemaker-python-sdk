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

from sagemaker.s3 import parse_s3_url


def get_model_tarball_full_uri_from_base_uri(base_uri: str, training_job_name: str) -> str:
    return "/".join(
        [
            base_uri,
            training_job_name,
            "output",
            "model.tar.gz",
        ]
    )


def get_full_hyperparameters(
    base_hyperparameters: dict, job_name: str, model_artifacts_uri: str
) -> dict:

    bucket, key = parse_s3_url(model_artifacts_uri)
    return {
        **base_hyperparameters,
        "sagemaker_job_name": job_name,
        "model-artifact-bucket": bucket,
        "model-artifact-key": key,
    }
