# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import pickle

import numpy as np
import pytest

import tests.integ
from tests.integ.timeout import timeout_and_delete_endpoint_by_name, timeout

from sagemaker.tensorflow import estimator
from sagemaker.utils import unique_name_from_base

PICKLE_CONTENT_TYPE = "application/python-pickle"


class PickleSerializer(object):
    def __init__(self):
        self.content_type = PICKLE_CONTENT_TYPE

    def __call__(self, data):
        return pickle.dumps(data, protocol=2)


@pytest.mark.canary_quick
@pytest.mark.skipif(
    tests.integ.PYTHON_VERSION != "py2", reason="TensorFlow image supports only python 2."
)
@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.HOSTING_NO_P2_REGIONS
    or tests.integ.test_region() in tests.integ.TRAINING_NO_P2_REGIONS,
    reason="no ml.p2 instances in these regions",
)
def test_cifar(sagemaker_session):
    with timeout(minutes=45):
        script_path = os.path.join(tests.integ.DATA_DIR, "cifar_10", "source")

        dataset_path = os.path.join(tests.integ.DATA_DIR, "cifar_10", "data")

        tensorflow_estimator = estimator.TensorFlow(
            entry_point="resnet_cifar_10.py",
            source_dir=script_path,
            role="SageMakerRole",
            framework_version="1.12",
            training_steps=50,
            evaluation_steps=5,
            train_instance_count=2,
            train_instance_type="ml.p2.xlarge",
            sagemaker_session=sagemaker_session,
            train_max_run=45 * 60,
            base_job_name="test-cifar",
        )

        inputs = tensorflow_estimator.sagemaker_session.upload_data(
            path=dataset_path, key_prefix="data/cifar10"
        )
        job_name = unique_name_from_base("test-tf-cifar")

        tensorflow_estimator.fit(inputs, logs=False, job_name=job_name)
        print("job succeeded: {}".format(tensorflow_estimator.latest_training_job.name))

    endpoint_name = tensorflow_estimator.latest_training_job.name
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        predictor = tensorflow_estimator.deploy(
            initial_instance_count=1, instance_type="ml.p2.xlarge"
        )
        predictor.serializer = PickleSerializer()
        predictor.content_type = PICKLE_CONTENT_TYPE

        data = np.random.randn(32, 32, 3)
        predict_response = predictor.predict(data)
        assert len(predict_response["outputs"]["probabilities"]["floatVal"]) == 10
