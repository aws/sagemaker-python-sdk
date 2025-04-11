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

import numpy as np

from sagemaker import RandomCutForest, RandomCutForestModel
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.utils import unique_name_from_base
from tests.integ import TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


def test_randomcutforest(sagemaker_session, cpu_instance_type):
    job_name = unique_name_from_base("randomcutforest")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        # Generate a thousand 14-dimensional datapoints.
        feature_num = 14
        train_input = np.random.rand(1000, feature_num)

        rcf = RandomCutForest(
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            num_trees=50,
            num_samples_per_tree=20,
            eval_metrics=["accuracy", "precision_recall_fscore"],
            sagemaker_session=sagemaker_session,
        )

        rcf.fit(records=rcf.record_set(train_input), job_name=job_name)

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        model = RandomCutForestModel(
            rcf.model_data, role="SageMakerRole", sagemaker_session=sagemaker_session
        )
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=job_name)

        predict_input = np.random.rand(1, feature_num)
        result = predictor.predict(predict_input)

        assert len(result) == 1
        for record in result:
            assert record.label["score"] is not None
            assert len(record.label["score"].float32_tensor.values) == 1


def test_randomcutforest_serverless_inference(sagemaker_session, cpu_instance_type):
    job_name = unique_name_from_base("randomcutforest")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        # Generate a thousand 14-dimensional datapoints.
        feature_num = 14
        train_input = np.random.rand(1000, feature_num)

        rcf = RandomCutForest(
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            num_trees=50,
            num_samples_per_tree=20,
            eval_metrics=["accuracy", "precision_recall_fscore"],
            sagemaker_session=sagemaker_session,
        )

        rcf.fit(records=rcf.record_set(train_input), job_name=job_name)

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        model = RandomCutForestModel(
            rcf.model_data, role="SageMakerRole", sagemaker_session=sagemaker_session
        )
        predictor = model.deploy(
            serverless_inference_config=ServerlessInferenceConfig(), endpoint_name=job_name
        )

        predict_input = np.random.rand(1, feature_num)
        result = predictor.predict(predict_input)

        assert len(result) == 1
        for record in result:
            assert record.label["score"] is not None
            assert len(record.label["score"].float32_tensor.values) == 1
