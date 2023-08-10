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

import json
import os

import pytest

import sagemaker.utils
import tests.integ as integ
from tests.integ.s3_utils import extract_files_from_s3
from tests.integ.utils import gpu_list, retry_with_instance_list
from sagemaker.tensorflow import TensorFlow
from tests.integ import timeout

from packaging.version import Version

horovod_dir = os.path.join(os.path.dirname(__file__), "..", "data", "horovod")


@pytest.mark.release
def test_hvd_cpu(
    sagemaker_session,
    tensorflow_training_latest_version,
    tensorflow_training_latest_py_version,
    cpu_instance_type,
    tmpdir,
):
    if Version(tensorflow_training_latest_version) >= Version("2.13"):
        pytest.skip("Horovod is deprecated in TensorFlow 2.13 and above")
    _create_and_fit_estimator(
        sagemaker_session,
        tensorflow_training_latest_version,
        tensorflow_training_latest_py_version,
        cpu_instance_type,
        tmpdir,
    )


@pytest.mark.release
@pytest.mark.skipif(
    integ.test_region() in integ.TRAINING_NO_P2_REGIONS
    and integ.test_region() in integ.TRAINING_NO_P3_REGIONS,
    reason="no ml.p2 or ml.p3 instances in this region",
)
@retry_with_instance_list(gpu_list(integ.test_region()))
def test_hvd_gpu(
    sagemaker_session,
    tensorflow_training_latest_version,
    tensorflow_training_latest_py_version,
    tmpdir,
    **kwargs,
):
    if (
        Version(tensorflow_training_latest_version) >= Version("2.12")
        and kwargs["instance_type"] == "ml.p2.xlarge"
    ):
        pytest.skip("P2 instances have been deprecated for sagemaker jobs starting TensorFlow 2.12")
    if Version(tensorflow_training_latest_version) >= Version("2.13"):
        pytest.skip("Horovod is deprecated in TensorFlow 2.13 and above")

    _create_and_fit_estimator(
        sagemaker_session,
        tensorflow_training_latest_version,
        tensorflow_training_latest_py_version,
        kwargs["instance_type"],
        tmpdir,
    )


def read_json(file, tmp):
    with open(os.path.join(tmp, file)) as f:
        return json.load(f)


def _create_and_fit_estimator(sagemaker_session, tf_version, py_version, instance_type, tmpdir):
    job_name = sagemaker.utils.unique_name_from_base("tf-horovod")
    estimator = TensorFlow(
        entry_point=os.path.join(horovod_dir, "hvd_basic.py"),
        role="SageMakerRole",
        instance_count=2,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        py_version=py_version,
        framework_version=tf_version,
        distribution={"mpi": {"enabled": True}},
        disable_profiler=True,
    )

    with timeout.timeout(minutes=integ.TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit(job_name=job_name)

        tmp = str(tmpdir)
        extract_files_from_s3(estimator.model_data, tmp, sagemaker_session)

        for rank in range(2):
            assert read_json("rank-%s" % rank, tmp)["rank"] == rank
