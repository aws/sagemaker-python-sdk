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

import pytest
import os

import sagemaker.amazon.pca
from sagemaker.utils import unique_name_from_base
from tests.integ import datasets, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name
from tests.integ.kms_utils import get_or_create_kms_key

INPUT_LOCAL_PATH = "tests/data/async_inference_input/async-inference-pca-input.csv"


@pytest.fixture
def training_set():
    return datasets.one_p_mnist()


@pytest.fixture()
def s3_files_kms_key(sagemaker_session):
    return get_or_create_kms_key(sagemaker_session=sagemaker_session)


def test_async_walkthrough(sagemaker_session, cpu_instance_type, training_set, s3_files_kms_key):
    job_name = unique_name_from_base("pca")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        pca = sagemaker.amazon.pca.PCA(
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            num_components=48,
            sagemaker_session=sagemaker_session,
            enable_network_isolation=True,
        )

        pca.algorithm_mode = "randomized"
        pca.subtract_mean = True
        pca.extra_components = 5
        pca.fit(pca.record_set(training_set[0][:100]), job_name=job_name)

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        predictor = pca.deploy(
            endpoint_name=job_name,
            inference_type="async",
            initial_instance_count=1,
            instance_type=cpu_instance_type,
        )

        s3_key_prefix = os.path.join(
            "integ-test-test-async-inference",
            job_name,
        )

        s3_input_path = os.path.join(
            "s3://",
            sagemaker_session.default_bucket(),
            s3_key_prefix,
            "async-inference-pca-input.csv",
        )

        sagemaker_session.upload_data(
            path=INPUT_LOCAL_PATH,
            bucket=sagemaker_session.default_bucket(),
            key_prefix=s3_key_prefix,
            extra_args={"ContentType": "text/csv"},
        )

        assert predictor.predictor_type == "async"
        result = predictor.predict(
            data=s3_input_path,
        )
        assert isinstance(result, str)
        assert result.startswith("s3://" + sagemaker_session.default_bucket())
