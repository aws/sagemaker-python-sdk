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
import tarfile

import boto3
import pytest
from six.moves.urllib.parse import urlparse

import sagemaker.utils
import tests.integ as integ
from sagemaker.mxnet import MXNet
from tests.integ import timeout

horovod_dir = os.path.join(os.path.dirname(__file__), "..", "data", "horovod")


@pytest.mark.release
def test_hvd_cpu(
    mxnet_training_latest_version,
    mxnet_training_latest_py_version,
    sagemaker_session,
    cpu_instance_type,
    tmpdir,
):
    _create_and_fit_estimator(
        mxnet_training_latest_version,
        mxnet_training_latest_py_version,
        sagemaker_session,
        cpu_instance_type,
        tmpdir,
    )


@pytest.mark.release
@pytest.mark.skipif(
    integ.test_region() in integ.TRAINING_NO_P2_REGIONS, reason="no ml.p2 instances in this region"
)
def test_hvd_gpu(
    mxnet_training_latest_version,
    mxnet_training_latest_py_version,
    sagemaker_session,
    gpu_instance_type,
    tmpdir,
):
    _create_and_fit_estimator(
        mxnet_training_latest_version,
        mxnet_training_latest_py_version,
        sagemaker_session,
        gpu_instance_type,
        tmpdir,
    )


def read_json(file, tmp):
    with open(os.path.join(tmp, file)) as f:
        return json.load(f)


def extract_files_from_s3(s3_url, tmpdir, sagemaker_session):
    parsed_url = urlparse(s3_url)
    s3 = boto3.resource("s3", region_name=sagemaker_session.boto_region_name)

    model = os.path.join(tmpdir, "model")
    s3.Bucket(parsed_url.netloc).download_file(parsed_url.path.lstrip("/"), model)

    with tarfile.open(model, "r") as tar_file:
        tar_file.extractall(tmpdir)


def _create_and_fit_estimator(mxnet_version, py_version, sagemaker_session, instance_type, tmpdir):
    job_name = sagemaker.utils.unique_name_from_base("mx-horovod")
    estimator = MXNet(
        entry_point=os.path.join(horovod_dir, "hvd_mnist_mxnet.py"),
        role="SageMakerRole",
        instance_count=2,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        py_version=py_version,
        framework_version=mxnet_version,
        distribution={"mpi": {"enabled": True}},
        disable_profiler=True,
    )

    with timeout.timeout(minutes=integ.TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit(job_name=job_name)

        tmp = str(tmpdir)
        extract_files_from_s3(estimator.model_data, tmp, sagemaker_session)

        for rank in range(2):
            assert read_json("rank-%s" % rank, tmp)["rank"] == rank
