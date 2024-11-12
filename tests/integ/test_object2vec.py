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

import pytest

from sagemaker.predictor import Predictor
from sagemaker import Object2Vec, Object2VecModel
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.utils import unique_name_from_base
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name
from tests.integ.record_set import prepare_record_set_from_local_files

FEATURE_NUM = None


@pytest.mark.skip(
    reason="This test has always failed, but the failure was masked by a bug. "
    "This test should be fixed. Details in https://github.com/aws/sagemaker-python-sdk/pull/968"
)
def test_object2vec(sagemaker_session, cpu_instance_type):
    job_name = unique_name_from_base("object2vec")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        data_path = os.path.join(DATA_DIR, "object2vec")
        data_filename = "train.jsonl"

        with open(os.path.join(data_path, data_filename), "r") as f:
            num_records = len(f.readlines())

        object2vec = Object2Vec(
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            epochs=3,
            enc0_max_seq_len=20,
            enc0_vocab_size=45000,
            enc_dim=16,
            num_classes=3,
            negative_sampling_rate=0,
            comparator_list="hadamard,concat,abs_diff",
            tied_token_embedding_weight=False,
            token_embedding_storage_type="dense",
            sagemaker_session=sagemaker_session,
        )

        record_set = prepare_record_set_from_local_files(
            data_path, object2vec.data_location, num_records, FEATURE_NUM, sagemaker_session
        )

        object2vec.fit(records=record_set, job_name=job_name)

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        model = Object2VecModel(
            object2vec.model_data, role="SageMakerRole", sagemaker_session=sagemaker_session
        )
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=job_name)
        assert isinstance(predictor, Predictor)

        predict_input = {"instances": [{"in0": [354, 623], "in1": [16]}]}

        result = predictor.predict(predict_input)

        assert len(result) == 1
        for record in result:
            assert record.label["scores"] is not None


def test_object2vec_serverless_inference(sagemaker_session, cpu_instance_type):
    job_name = unique_name_from_base("object2vec-serverless")

    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        data_path = os.path.join(DATA_DIR, "object2vec")
        data_filename = "train.jsonl"

        with open(os.path.join(data_path, data_filename), "r") as f:
            num_records = len(f.readlines())

        object2vec = Object2Vec(
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            epochs=3,
            enc0_max_seq_len=20,
            enc0_vocab_size=45000,
            enc_dim=16,
            num_classes=3,
            negative_sampling_rate=0,
            comparator_list="hadamard,concat,abs_diff",
            tied_token_embedding_weight=False,
            token_embedding_storage_type="dense",
            sagemaker_session=sagemaker_session,
        )

        record_set = prepare_record_set_from_local_files(
            data_path, object2vec.data_location, num_records, FEATURE_NUM, sagemaker_session
        )

        object2vec.fit(records=record_set, job_name=job_name)

    with timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        model = Object2VecModel(
            object2vec.model_data, role="SageMakerRole", sagemaker_session=sagemaker_session
        )
        predictor = model.deploy(
            serverless_inference_config=ServerlessInferenceConfig(), endpoint_name=job_name
        )
        assert isinstance(predictor, Predictor)
