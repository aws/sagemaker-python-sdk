# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import numpy as np
import pytest

from sagemaker import RandomCutForest, RandomCutForestModel
from sagemaker.utils import name_from_base
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


@pytest.mark.continuous_testing
def test_randomcutforest(sagemaker_session):
    with timeout(minutes=15):
        # Generate a thousand 14-dimensional datapoints.
        feature_num = 14
        train_input = np.random.rand(1000, feature_num)

        rcf = RandomCutForest(role='SageMakerRole', train_instance_count=1, train_instance_type='ml.c4.xlarge',
                              num_trees=50, num_samples_per_tree=20, sagemaker_session=sagemaker_session,
                              base_job_name='test-randomcutforest')

        rcf.fit(rcf.record_set(train_input))

    endpoint_name = name_from_base('randomcutforest')
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=20):
        model = RandomCutForestModel(rcf.model_data, role='SageMakerRole', sagemaker_session=sagemaker_session)
        predictor = model.deploy(1, 'ml.c4.xlarge', endpoint_name=endpoint_name)

        predict_input = np.random.rand(1, feature_num)
        result = predictor.predict(predict_input)

        assert len(result) == 1
        for record in result:
            assert record.label["score"] is not None
            assert len(record.label["score"].float32_tensor.values) == 1
