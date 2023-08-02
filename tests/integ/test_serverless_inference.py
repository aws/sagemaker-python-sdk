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

import sagemaker.amazon.pca
from sagemaker.utils import unique_name_from_base
from sagemaker.serverless import ServerlessInferenceConfig
from tests.integ import datasets, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


@pytest.fixture
def training_set():
    return datasets.one_p_mnist()


def test_serverless_walkthrough(sagemaker_session, cpu_instance_type, training_set):
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

    serverless_name = unique_name_from_base("pca-serverless")
    with timeout_and_delete_endpoint_by_name(serverless_name, sagemaker_session):

        predictor_serverless = pca.deploy(
            endpoint_name=serverless_name, serverless_inference_config=ServerlessInferenceConfig()
        )

        result = predictor_serverless.predict(training_set[0][:5])

        assert len(result) == 5
        for record in result:
            assert record.label["projection"] is not None

    # Test out Serverless Provisioned Concurrency endpoint happy case
    serverless_pc_name = unique_name_from_base("pca-serverless-pc")
    with timeout_and_delete_endpoint_by_name(serverless_pc_name, sagemaker_session):

        predictor_serverless_pc = pca.deploy(
            endpoint_name=serverless_pc_name,
            serverless_inference_config=ServerlessInferenceConfig(provisioned_concurrency=1),
        )

        result = predictor_serverless_pc.predict(training_set[0][:5])

        assert len(result) == 5
        for record in result:
            assert record.label["projection"] is not None
