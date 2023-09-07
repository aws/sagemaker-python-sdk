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
import time

import sagemaker.amazon.pca
from sagemaker.utils import unique_name_from_base
from sagemaker.async_inference import AsyncInferenceConfig, AsyncInferenceResponse
from sagemaker.predictor_async import AsyncPredictor
from tests.integ import datasets, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name

INPUT_LOCAL_PATH = "tests/data/async_inference_input/async-inference-pca-input.csv"


@pytest.fixture
def training_set():
    return datasets.one_p_mnist()


def test_async_walkthrough(sagemaker_session, cpu_instance_type, training_set):
    job_name = unique_name_from_base("pca")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        pca = sagemaker.amazon.pca.PCA(
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            num_components=48,
            sagemaker_session=sagemaker_session,
        )

        pca.algorithm_mode = "randomized"
        pca.subtract_mean = True
        pca.extra_components = 5
        pca.fit(pca.record_set(training_set[0][:100]), job_name=job_name)

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        predictor_async = pca.deploy(
            endpoint_name=job_name,
            initial_instance_count=1,
            instance_type=cpu_instance_type,
            async_inference_config=AsyncInferenceConfig(),
        )
        assert isinstance(predictor_async, AsyncPredictor)

        data = training_set[0][:5]
        result_no_wait_with_data = predictor_async.predict_async(data=data)
        assert isinstance(result_no_wait_with_data, AsyncInferenceResponse)
        assert result_no_wait_with_data.output_path.startswith(
            "s3://" + sagemaker_session.default_bucket()
        )
        assert result_no_wait_with_data.failure_path.startswith(
            "s3://"
            + sagemaker_session.default_bucket()
            + f"/{sagemaker_session.default_bucket_prefix}"
            + "/async-endpoint-failures/"
        )
        time.sleep(5)
        result_no_wait_with_data = result_no_wait_with_data.get_result()
        assert len(result_no_wait_with_data) == 5
        for record in result_no_wait_with_data:
            assert record.label["projection"] is not None

        result_wait_with_data = predictor_async.predict(data=data)
        assert len(result_wait_with_data) == 5
        for idx, record in enumerate(result_wait_with_data):
            assert record.label["projection"] is not None
            assert record.label["projection"] == result_no_wait_with_data[idx].label["projection"]

        s3_key_prefix = os.path.join(
            "integ-test-test-async-inference",
            job_name,
        )

        input_s3_path = os.path.join(
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

        result_not_wait = predictor_async.predict_async(input_path=input_s3_path)
        assert isinstance(result_not_wait, AsyncInferenceResponse)
        assert result_not_wait.output_path.startswith("s3://" + sagemaker_session.default_bucket())
        assert result_not_wait.failure_path.startswith(
            "s3://"
            + sagemaker_session.default_bucket()
            + f"/{sagemaker_session.default_bucket_prefix}"
            + "/async-endpoint-failures/"
        )
        time.sleep(5)
        result_not_wait = result_not_wait.get_result()
        assert len(result_not_wait) == 5
        for record in result_not_wait:
            assert record.label["projection"] is not None

        result_wait = predictor_async.predict(input_path=input_s3_path)
        assert len(result_wait) == 5
        for idx, record in enumerate(result_wait):
            assert record.label["projection"] is not None
            assert record.label["projection"] == result_not_wait[idx].label["projection"]
