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

import tests.integ.lock as lock
from sagemaker.pytorch.estimator import PyTorch
from tests.integ import DATA_DIR


@pytest.mark.local_mode
def test_source_dirs(tmpdir, sagemaker_local_session):
    source_dir = os.path.join(DATA_DIR, "pytorch_source_dirs")
    lib = os.path.join(str(tmpdir), "alexa.py")

    with open(lib, "w") as f:
        f.write("def question(to_anything): return 42")

    # TODO: fails on newer versions of pytorch in call to np.load(BytesIO(stream.read()))
    # "ValueError: Cannot load file containing pickled data when allow_pickle=False"
    estimator = PyTorch(
        entry_point="train.py",
        role="SageMakerRole",
        source_dir=source_dir,
        dependencies=[lib],
        framework_version="0.4",  # hard-code to last known good pytorch for now (see TODO above)
        py_version="py3",
        instance_count=1,
        instance_type="local",
        sagemaker_session=sagemaker_local_session,
    )
    estimator.fit()

    # endpoint tests all use the same port, so we use this lock to prevent concurrent execution
    with lock.lock():
        try:
            predictor = estimator.deploy(initial_instance_count=1, instance_type="local")
            predict_response = predictor.predict([7])
            assert predict_response == [49]
        finally:
            predictor.delete_endpoint()
