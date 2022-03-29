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

import os

import numpy as np
import pytest

from sagemaker import NTM, NTMModel, Predictor
from sagemaker.amazon.common import read_records
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.utils import unique_name_from_base
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name
from tests.integ.record_set import prepare_record_set_from_local_files


@pytest.mark.release
@pytest.mark.skip(
    reason="This test has always failed, but the failure was masked by a bug. "
    "This test should be fixed. Details in https://github.com/aws/sagemaker-python-sdk/pull/968"
)
def test_ntm(sagemaker_session, cpu_instance_type):
    job_name = unique_name_from_base("ntm")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        data_path = os.path.join(DATA_DIR, "ntm")
        data_filename = "nips-train_1.pbr"

        with open(os.path.join(data_path, data_filename), "rb") as f:
            all_records = read_records(f)

        # all records must be same
        feature_num = int(all_records[0].features["values"].float32_tensor.shape[0])

        ntm = NTM(
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            num_topics=10,
            sagemaker_session=sagemaker_session,
        )

        record_set = prepare_record_set_from_local_files(
            data_path, ntm.data_location, len(all_records), feature_num, sagemaker_session
        )
        ntm.fit(records=record_set, job_name=job_name)

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        model = NTMModel(ntm.model_data, role="SageMakerRole", sagemaker_session=sagemaker_session)
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=job_name)

        predict_input = np.random.rand(1, feature_num)
        result = predictor.predict(predict_input)

        assert len(result) == 1
        for record in result:
            assert record.label["topic_weights"] is not None


def test_ntm_serverless_inference(sagemaker_session, cpu_instance_type):
    job_name = unique_name_from_base("ntm-serverless")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        data_path = os.path.join(DATA_DIR, "ntm")
        data_filename = "nips-train_1.pbr"

        with open(os.path.join(data_path, data_filename), "rb") as f:
            all_records = read_records(f)

        # all records must be same
        feature_num = int(all_records[0].features["values"].float32_tensor.shape[0])

        ntm = NTM(
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            num_topics=10,
            sagemaker_session=sagemaker_session,
        )

        record_set = prepare_record_set_from_local_files(
            data_path, ntm.data_location, len(all_records), feature_num, sagemaker_session
        )
        ntm.fit(records=record_set, job_name=job_name)

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        model = NTMModel(ntm.model_data, role="SageMakerRole", sagemaker_session=sagemaker_session)
        predictor = model.deploy(
            serverless_inference_config=ServerlessInferenceConfig(), endpoint_name=job_name
        )
        assert isinstance(predictor, Predictor)
